"""M4 / F2: members load crop-aligned with the anchor file item."""

import os

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps

from extensions.slider_tuple.members import (
    MemberLatentCache,
    dump_tuple,
    load_member_image,
    load_member_tensor,
)
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets

from .stubs import DataModelStub


def noise_image(path, size, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return path


def anchor_item(tmp_path, size=(96, 72), resolution=64):
    """A real FileItemDTO for one image whose bucket needs a resize and a crop."""
    folder = tmp_path / "anchors"
    folder.mkdir()
    path = noise_image(str(folder / "a.png"), size, seed=0)
    loader = get_dataloader_from_datasets(
        [DatasetConfig(folder_path=str(folder), resolution=resolution, num_workers=0, caption_ext="txt")],
        batch_size=1,
        sd=DataModelStub(),
    )
    batch = next(iter(loader))
    item = batch.file_items[0]
    return path, item, batch.tensor[0], loader.dataset.datasets[0].transform


def test_anchor_image_as_member_reproduces_dataloader_tensor(tmp_path):
    path, item, anchor_tensor, _ = anchor_item(tmp_path)
    # the bucket really resized and cropped this image
    assert (item.scale_to_width, item.scale_to_height) != Image.open(path).size
    assert (item.crop_width, item.crop_height) != (item.scale_to_width, item.scale_to_height) or item.crop_x or item.crop_y
    assert torch.equal(load_member_tensor(path, item), anchor_tensor)


def test_flips_match_upstream_image_path(tmp_path):
    path, item, _, transform = anchor_item(tmp_path)
    for flip_x, flip_y in [(True, False), (False, True), (True, True)]:
        item.flip_x, item.flip_y = flip_x, flip_y
        item.load_and_process_image(transform)
        assert torch.equal(load_member_tensor(path, item), item.tensor), (flip_x, flip_y)


def test_member_with_other_content_is_pixel_aligned(tmp_path):
    path, item, anchor_tensor, _ = anchor_item(tmp_path)
    inverted = str(tmp_path / "inverted.png")
    ImageOps.invert(Image.open(path).convert("RGB")).save(inverted)
    member = load_member_tensor(inverted, item)
    # inversion commutes with the resize/crop, so the member is the anchor's negative
    assert member.shape == anchor_tensor.shape
    assert (member + anchor_tensor).abs().max() <= 2 / 255 + 1e-6


def test_latent_cache_encodes_each_member_crop_once(tmp_path):
    path, item, _, _ = anchor_item(tmp_path)
    other = noise_image(str(tmp_path / "b.png"), Image.open(path).size, seed=1)
    calls = []

    def encode(pixels):
        calls.append(pixels.shape[0])
        return pixels.mean(dim=1, keepdim=True)  # stand-in "latent"

    cache = MemberLatentCache()
    first = cache.get([(path, item), (other, item), (path, item)], encode)
    assert calls == [2] and len(cache) == 2
    assert torch.equal(first[0], first[2])
    again = cache.get([(other, item), (path, item)], encode)
    assert calls == [2]  # all hits
    assert torch.equal(again[0], first[1])
    # a different flip is a different crop key
    item.flip_x = not item.flip_x
    cache.get([(path, item)], encode)
    assert calls == [2, 1] and len(cache) == 3


def test_debug_dump_writes_member_strip(tmp_path):
    path, item, _, _ = anchor_item(tmp_path)
    images = [load_member_image(path, item), ImageOps.invert(load_member_image(path, item))]
    out = dump_tuple(str(tmp_path / "dump" / "t.png"), images, ["s=+1", "s=-1"], max_side=64)
    sheet = Image.open(out)
    assert sheet.size == (2 * images[0].width * 64 // max(images[0].size), 2 * images[0].height * 64 // max(images[0].size) + 16)
