"""The tuple slider loss (F2).

For one batch of B tuples with m sampled members each, arranged in m slots
(slot j holds every tuple's j-th member, highest scale first):

    L = mean_b sum_j w_bj * mean( (v_theta,s_bj(x_t^bj, c_b, t_b) - (eps_b - x^bj))^2 )

with one shared noise eps_b and timestep t_b per tuple and the LoRA at
multiplier s_bj. Two members at scales +w / -w with unit weights is exactly
upstream's ``polarity`` loss (toolkit/guidance.py get_guided_loss_polarity),
whose code path this mirrors: same scheduler call, noising, conditioning,
targets and summed (not averaged) member losses. Like polarity it runs the
backward pass itself and returns a detached loss that still requires grad.
"""

from typing import List, Sequence

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
) -> torch.Tensor:
    """slot_latents[j]: (B, ...) clean latents of slot j; slot_scales[j][b] and
    slot_weights[j][b]: LoRA multiplier and loss weight of tuple b's member in
    slot j. Returns the detached total loss (backward already done)."""
    m = len(slot_latents)
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
        sd.unet.train()
        sd.network.is_active = True

    def run(slots: List[int]) -> torch.Tensor:
        with torch.no_grad():
            sd.network.multiplier = [float(s) for j in slots for s in slot_scales[j]]
            cat_embeds = concat_prompt_embeds([conditional_embeds] * len(slots))
            cat_latents = torch.cat([noisy[j] for j in slots], dim=0)
            cat_timesteps = torch.cat([timesteps] * len(slots), dim=0)
        prediction = sd.predict_noise(
            latents=cat_latents.to(device, dtype=dtype).detach(),
            conditional_embeddings=cat_embeds.to(device, dtype=dtype).detach(),
            timestep=cat_timesteps,
            guidance_scale=1.0,
            **pred_kwargs,
        )
        total = 0.0
        for j, pred in zip(slots, torch.chunk(prediction, len(slots), dim=0)):
            mse = torch.nn.functional.mse_loss(pred.float(), targets[j].float(), reduction="none")
            per_item = mse.mean(dim=list(range(1, mse.dim())))
            total = total + (weights[j] * per_item).mean()
        return total

    if backward_per_member:
        loss = 0.0
        for j in range(m):
            slot_loss = run([j])
            slot_loss.backward()
            loss = loss + slot_loss.detach()
    else:
        loss = run(list(range(m)))
        loss.backward()

    loss = loss.detach()
    loss.requires_grad_(True)
    return loss
