"""Anchor preservation loss for image sliders (F4).

ConceptSliderTrainer's anchor idea applied to image-pair / tuple training:
with the LoRA at a random multiplier m in [-s_max, s_max], the prediction for
an unrelated anchor prompt should equal the frozen model's (LoRA off), so the
slider only moves what the tuples move.

    L_anchor = anchor_strength * mean( (v_theta,m(x_t, c_anchor, t) - v_frozen(x_t, c_anchor, t))^2 )

x_t uses the step's noise and timesteps (full timestep range; the F5 window
applies to the slider loss only). Runs its own backward, like the slider loss.
"""

import os
from types import SimpleNamespace
from typing import List, Sequence

import torch

from toolkit.prompt_utils import concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def list_anchor_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise ValueError(f"slider.anchor_images: not a folder: {folder}")
    paths = sorted(
        os.path.join(root, name)
        for root, dirs, files in os.walk(folder)
        for name in files
        if name.lower().endswith(IMAGE_EXTS) and not name.startswith(".")
    )
    if not paths:
        raise ValueError(f"slider.anchor_images: no images in {folder}")
    return paths


def center_crop_geometry(image_size, target_size):
    """Resize-to-cover + center-crop geometry in the attribute names
    members.load_member_tensor reads from a file item."""
    (w, h), (tw, th) = image_size, target_size
    scale = max(tw / w, th / h)
    sw, sh = max(tw, round(w * scale)), max(th, round(h * scale))
    return SimpleNamespace(
        scale_to_width=sw, scale_to_height=sh,
        crop_x=(sw - tw) // 2, crop_y=(sh - th) // 2,
        crop_width=tw, crop_height=th,
        flip_x=False, flip_y=False,
    )


def anchor_loss(
    sd,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    anchor_embeds: Sequence,
    multipliers: Sequence[float],
    strength: float,
    pred_kwargs: dict,
    batch,
) -> tuple:
    """Returns (detached weighted loss, unweighted mse float); backward already done.

    anchor_embeds: one PromptEmbeds per batch item; multipliers: LoRA
    multiplier per batch item."""
    dtype = get_torch_dtype(sd.torch_dtype)
    device = sd.device_torch
    with torch.no_grad():
        latents = latents.to(device, dtype=dtype).detach()
        noise = noise.to(device, dtype=dtype).detach()
        noisy = sd.condition_noisy_latents(sd.add_noise(latents, noise, timesteps).detach(), batch)
        embeds = concat_prompt_embeds(list(anchor_embeds)).to(device, dtype=dtype)

        sd.network.is_active = False  # frozen-model target
        target = sd.predict_noise(
            latents=noisy, conditional_embeddings=embeds, timestep=timesteps, guidance_scale=1.0, **pred_kwargs
        ).detach()
        sd.network.is_active = True
        sd.network.multiplier = [float(m) for m in multipliers]

    pred = sd.predict_noise(
        latents=noisy, conditional_embeddings=embeds, timestep=timesteps, guidance_scale=1.0, **pred_kwargs
    )
    mse = torch.nn.functional.mse_loss(pred.float(), target.float())
    loss = mse * strength
    loss.backward()
    return loss.detach(), mse.item()
