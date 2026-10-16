"""Loading tuple members crop-aligned with their anchor file item (F2).

The dataloader loads only each tuple's anchor image as a FileItemDTO; it picks
the bucket size, crop and flip. Every other member is loaded here with exactly
the same steps as the anchor's bucketed image path
(``ImageProcessingDTOMixin.load_and_process_image``): exif transpose, RGB,
flip_x / flip_y, bicubic resize to scale_to_*, crop, ToTensor, rescale to
[-1, 1]. This generalizes upstream's single ``unconditional_path`` partner
(``load_unconditional_image``) to k members.
"""

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

_to_tensor = transforms.ToTensor()


def crop_key(file_item) -> Tuple:
    return (
        int(file_item.scale_to_width), int(file_item.scale_to_height),
        int(file_item.crop_x), int(file_item.crop_y),
        int(file_item.crop_width), int(file_item.crop_height),
        bool(file_item.flip_x), bool(file_item.flip_y),
    )


def load_member_image(path: str, file_item) -> Image.Image:
    """The member image after the anchor's flip, resize and crop."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if file_item.flip_x:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if file_item.flip_y:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.resize((file_item.scale_to_width, file_item.scale_to_height), Image.BICUBIC)
    return img.crop((
        file_item.crop_x,
        file_item.crop_y,
        file_item.crop_x + file_item.crop_width,
        file_item.crop_y + file_item.crop_height,
    ))


def load_member_mask(path: str, file_item) -> torch.Tensor:
    """(1, H, W) loss mask in [mask_min_value, 1], aligned with the member.

    Mirrors upstream ``load_mask_image`` (RGB -> optional invert -> flips ->
    grayscale -> bicubic resize -> crop -> mask_min_value floor) without its
    random blur augmentation, so masks are deterministic."""
    config = file_item.dataset_config
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if config.invert_mask:
        img = ImageOps.invert(img)
    if file_item.flip_x:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if file_item.flip_y:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.convert("L").resize((file_item.scale_to_width, file_item.scale_to_height), Image.BICUBIC)
    img = img.crop((
        file_item.crop_x,
        file_item.crop_y,
        file_item.crop_x + file_item.crop_width,
        file_item.crop_y + file_item.crop_height,
    ))
    mask = _to_tensor(img)
    return config.mask_min_value + mask * (1.0 - config.mask_min_value)


def load_member_tensor(path: str, file_item) -> torch.Tensor:
    """(3, H, W) in [-1, 1], matching the anchor's dataloader tensor."""
    return _to_tensor(load_member_image(path, file_item)) * 2 - 1


class MemberLatentCache:
    """Member latents kept on CPU, keyed by image path and the anchor's crop.

    Crops are fixed per file item for bucketed datasets, so each member is
    VAE-encoded once per distinct (path, crop, flip)."""

    def __init__(self):
        self._latents: Dict[Tuple, torch.Tensor] = {}

    def __len__(self):
        return len(self._latents)

    def get(self, requests: Sequence[Tuple[str, object]], encode: Callable[[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
        """requests: (image path, anchor file item) pairs. ``encode`` maps a
        (N, 3, H, W) pixel batch to (N, ...) latents; misses with the same crop
        size are encoded together."""
        keys = [(path,) + crop_key(item) for path, item in requests]
        first_miss: Dict[Tuple, int] = {}
        for i, key in enumerate(keys):
            if key not in self._latents:
                first_miss.setdefault(key, i)
        # one encode per crop size (crop_width, crop_height are key[5:7])
        by_size: Dict[Tuple[int, int], List[int]] = {}
        for key, i in first_miss.items():
            by_size.setdefault(key[5:7], []).append(i)
        for idxs in by_size.values():
            pixels = torch.stack([load_member_tensor(*requests[i]) for i in idxs])
            for i, latent in zip(idxs, encode(pixels)):
                self._latents[keys[i]] = latent.detach().to("cpu")
        return [self._latents[k] for k in keys]


def dump_tuple(
    out_path: str,
    images: Sequence[Image.Image],
    labels: Sequence[str],
    max_side: int = 384,
) -> str:
    """Save the members side by side (as cropped for training) with labels,
    plus a row of each member minus the first, to check crop alignment."""
    thumbs = []
    for img in images:
        scale = max_side / max(img.size)
        thumbs.append(img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BICUBIC))
    w, h = thumbs[0].size
    sheet = Image.new("RGB", (w * len(thumbs), h * 2 + 16), "black")
    draw = ImageDraw.Draw(sheet)
    first = np.asarray(thumbs[0], dtype=np.float32)
    for i, (thumb, label) in enumerate(zip(thumbs, labels)):
        sheet.paste(thumb, (i * w, 16))
        diff = np.clip(np.abs(np.asarray(thumb, dtype=np.float32) - first) * 4, 0, 255).astype(np.uint8)
        sheet.paste(Image.fromarray(diff), (i * w, 16 + h))
        draw.text((i * w + 4, 2), label, fill="yellow")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    return out_path
