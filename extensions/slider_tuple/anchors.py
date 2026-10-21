"""Anchor preservation loss for image sliders (F4).

ConceptSliderTrainer's anchor idea applied to image-pair / tuple training:
with the LoRA at a random multiplier m in [-s_max, s_max], the prediction for
an unrelated anchor prompt should equal the frozen model's (LoRA off), so the
slider only moves what the tuples move.

    L_anchor = anchor_strength * mean( (v_theta,m(x_t, c_anchor, t) - v_frozen(x_t, c_anchor, t))^2 )

Anchors are prompts (noised from the batch's own latents) and/or images with their
own captions (load_anchor_pool), e.g. photos of men for a women-only slider, so
the LoRA learns to leave them alone at any strength.

x_t uses the step's noise and timesteps (full timestep range; the F5 window
applies to the slider loss only). Runs its own backward, like the slider loss.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Sequence

import torch

from toolkit.prompt_utils import concat_prompt_embeds
from toolkit.train_tools import get_torch_dtype

from .captions import DEFAULT_CAPTION_KEY, read_caption

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


@dataclass(frozen=True)
class AnchorItem:
    """One anchor: a caption with its own image (paired), or a prompt without one (the noisy
    latent then comes from the batch, or from a random image of an uncaptioned anchor folder)."""

    text: str
    image: Optional[str] = None


@dataclass
class AnchorPool:
    items: List[AnchorItem]
    unpaired_images: List[str]  # uncaptioned anchor_images (legacy): drawn at random for prompt-only items

    def describe(self) -> str:
        paired = sum(i.image is not None for i in self.items)
        prompts = len(self.items) - paired
        parts = []
        if paired:
            parts.append(f"{paired} captioned anchor images")
        if prompts:
            source = f"{len(self.unpaired_images)} anchor images" if self.unpaired_images else "the batch latents"
            parts.append(f"{prompts} prompts over {source}")
        return " + ".join(parts)


def load_anchor_pool(prompts: Sequence[str], images_folder: Optional[str],
                     caption_key: str = DEFAULT_CAPTION_KEY) -> AnchorPool:
    """Anchor prompts plus anchor_images. A folder whose images all have a caption next to them
    (<stem>.txt / .json) gives paired anchors, each image with its own caption. A folder with no
    captions is the older form: its images supply the noisy latents for the anchor prompts at
    random. Partly captioned folders are an error."""
    items = [AnchorItem(text=p) for p in prompts]
    unpaired: List[str] = []
    if images_folder:
        images = list_anchor_images(images_folder)
        texts = [read_caption(os.path.dirname(p), os.path.splitext(os.path.basename(p))[0], caption_key)
                 for p in images]
        captioned = [t is not None for t in texts]
        if all(captioned):
            items += [AnchorItem(text=t, image=p) for p, t in zip(images, texts)]
        elif any(captioned):
            missing = [os.path.basename(p) for p, c in zip(images, captioned) if not c]
            raise ValueError(f"slider.anchor_images: {len(missing)} of {len(images)} images have no caption "
                             f"(<name>.txt or .json): {missing[:5]}. Caption all of them (paired anchors) or none.")
        else:
            if not prompts:
                raise ValueError("slider.anchor_images without captions needs anchor_prompts "
                                 "(or caption every image for paired anchors)")
            unpaired = images
    if not items:
        raise ValueError("slider anchors need anchor_prompts or captioned anchor_images")
    return AnchorPool(items, unpaired)


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
