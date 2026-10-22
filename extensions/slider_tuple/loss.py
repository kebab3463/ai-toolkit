"""The tuple slider loss (F2).

For one batch of B tuples with m sampled members each, arranged in m slots
(slot j holds every tuple's j-th member, highest scale first):

    L = mean_b sum_j w_bj * mean( (v_theta,s_bj(x_t^bj, c_b, t_b) - (eps_b - x^bj))^2 )

with one shared noise eps_b and timestep t_b per tuple and the LoRA at
multiplier s_bj. Optional per-member loss masks M_bj (F3) multiply the
squared error element-wise after being resized to the latent size the way
SDTrainer builds mask_multiplier (nearest interpolate, expanded over
channels) and normalized to mean 1 per member.

Shared residual weight (lambda, ``shared_weight``): with r_bj = target_bj - v_bj
the residual of member j and r_b = sum_j w_bj r_bj / sum_j w_bj the tuple's
shared part, member j trains toward target_bj - (1 - lambda) r_b. lambda = 1 is
the plain loss above. Lower lambda weights the shared residual down against
each member's deviation r_bj - r_b, which carries the slider.

What it changes, with x_j the members' noisy inputs and x their mean: the part
of the LoRA's effect that is even in s, which the shared residual trains toward
the members' common content (NOTES.md, "Run 3 results"), loses that drive; at
lambda = 0 what is left is of order (x_j - x) x (r_j - r), zero for identical
inputs and small for tuple members, which differ only in the body. The part
linear in s keeps its main term (sum_j w s = 0 cancels r there) but also loses
r * sum_j w s_j (x_j - x): the shared residual where the members' inputs
differ. That term is odd in s and of the same order as the slider signal
(damped by (1 - t) at high noise), so lambda < 1 is not free for the slider
either; whether dropping it helps is an experiment. The reported loss is the
plain MSE either way.

Two members at scales +w / -w with unit weights and no masks is exactly
upstream's ``polarity`` loss (toolkit/guidance.py get_guided_loss_polarity),
whose code path this mirrors: same scheduler call, noising, conditioning,
targets and summed (not averaged) member losses. Like polarity it runs the
backward pass itself and returns a detached loss that still requires grad.
"""

from typing import List, Optional, Sequence

import torch

from toolkit.prompt_utils import concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype


def _set_flow_timesteps(sd, train_config, latents):
    # same call (and side effect) as get_guided_loss_polarity
    linear_timesteps = any([
        train_config.linear_timesteps,
        train_config.linear_timesteps2,
        train_config.timestep_type == "linear",
    ])
    timestep_type = "linear" if linear_timesteps else train_config.timestep_type
    sd.noise_scheduler.set_train_timesteps(1000, device=sd.device_torch, timestep_type=timestep_type, latents=latents)


def latent_loss_mask(mask: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) pixel mask -> (B, C, h, w) weights with mean 1 per item."""
    mask = mask.to(latents.device, dtype=torch.float32)
    mask = torch.nn.functional.interpolate(mask, size=latents.shape[-2:])
    mask = mask.expand(-1, latents.shape[1], -1, -1)
    mean = mask.mean(dim=(1, 2, 3), keepdim=True)
    if (mean <= 0).any():
        raise ValueError("slider_tuple: a member's loss mask is zero everywhere")
    return mask / mean


def slider_tuple_loss(
    sd,
    slot_latents: Sequence[torch.Tensor],
    slot_scales: Sequence[Sequence[float]],
    slot_weights: Sequence[Sequence[float]],
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    conditional_embeds,
    pred_kwargs: dict,
    batch,
    train_config,
    backward_per_member: bool = False,
    slot_masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
    member_losses: Optional[List[List[float]]] = None,
    shared_weight: float = 1.0,
) -> torch.Tensor:
    """slot_latents[j]: (B, ...) clean latents of slot j; slot_scales[j][b] and
    slot_weights[j][b]: LoRA multiplier and loss weight of tuple b's member in
    slot j; slot_masks[j]: (B, 1, H, W) pixel loss masks for slot j, or None
    for unmasked. If ``member_losses`` is a list, it receives each slot's
    per-item (masked, unweighted) MSE. ``shared_weight``: lambda above.
    Returns the detached total loss (backward already done)."""
    m = len(slot_latents)
    per_slot: List[Optional[List[float]]] = [None] * m
    dtype = get_torch_dtype(sd.torch_dtype)
    device = sd.device_torch

    with torch.no_grad():
        noise = noise.to(device, dtype=dtype).detach()
        latents = [x.to(device, dtype=dtype).detach() for x in slot_latents]
        targets = [noise for _ in latents]
        if sd.is_flow_matching:
            _set_flow_timesteps(sd, train_config, latents[0])
            targets = [(noise - x).detach() for x in latents]
        noisy = [
            sd.condition_noisy_latents(sd.add_noise(x, noise, timesteps).detach(), batch)
            for x in latents
        ]
        weights = [torch.tensor(w, device=device, dtype=torch.float32) for w in slot_weights]
        masks = [
            None if slot_masks is None or slot_masks[j] is None else latent_loss_mask(slot_masks[j], latents[j])
            for j in range(m)
        ]
        sd.unet.train()
        sd.network.is_active = True

    def predict(slots: List[int]) -> torch.Tensor:
        with torch.no_grad():
            sd.network.multiplier = [float(s) for j in slots for s in slot_scales[j]]
            cat_embeds = concat_prompt_embeds([conditional_embeds] * len(slots))
            cat_latents = torch.cat([noisy[j] for j in slots], dim=0)
            cat_timesteps = torch.cat([timesteps] * len(slots), dim=0)
        return sd.predict_noise(
            latents=cat_latents.to(device, dtype=dtype).detach(),
            conditional_embeddings=cat_embeds.to(device, dtype=dtype).detach(),
            timestep=cat_timesteps,
            guidance_scale=1.0,
            **pred_kwargs,
        )

    def shared_residual(preds: List[torch.Tensor]) -> torch.Tensor:
        """Per tuple, the weighted mean over members of target - prediction (detached)."""
        with torch.no_grad():
            total = sum(weights[j].view(-1, *[1] * (preds[j].dim() - 1)) for j in range(m))
            return sum(
                weights[j].view(-1, *[1] * (preds[j].dim() - 1)) * (targets[j].float() - preds[j].detach().float())
                for j in range(m)
            ) / total

    def slot_loss(j: int, pred: torch.Tensor, shift: Optional[torch.Tensor]):
        """(loss to backpropagate, plain per-item MSE) for slot j; ``shift`` = (1 - lambda) r."""
        target = targets[j].float() if shift is None else targets[j].float() - shift
        mse = torch.nn.functional.mse_loss(pred.float(), target, reduction="none")
        plain = mse if shift is None else (pred.detach().float() - targets[j].float()) ** 2
        if masks[j] is not None:
            mse, plain = mse * masks[j], plain * masks[j]
        dims = list(range(1, mse.dim()))
        per_item = plain.mean(dim=dims)
        per_slot[j] = per_item.detach().tolist()
        return (weights[j] * mse.mean(dim=dims)).mean(), (weights[j] * per_item).mean().detach()

    reshape = shared_weight != 1.0
    if backward_per_member:
        shift = None
        if reshape:  # the shared residual needs every member's prediction first
            with torch.no_grad():
                shift = (1.0 - shared_weight) * shared_residual([predict([j]) for j in range(m)])
        loss = 0.0
        for j in range(m):
            train, plain = slot_loss(j, predict([j]), shift)
            train.backward()
            loss = loss + plain
    else:
        preds = list(torch.chunk(predict(list(range(m))), m, dim=0))
        shift = (1.0 - shared_weight) * shared_residual(preds) if reshape else None
        train, loss = 0.0, 0.0
        for j in range(m):
            t, plain = slot_loss(j, preds[j], shift)
            train, loss = train + t, loss + plain
        train.backward()

    if member_losses is not None:
        member_losses.extend(per_slot)
    loss = loss.detach()
    loss.requires_grad_(True)
    return loss
