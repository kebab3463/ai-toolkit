"""F3: masked slider loss."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from extensions.slider_tuple.loss import latent_loss_mask, slider_tuple_loss
from extensions.slider_tuple.members import load_member_mask
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets

from .stubs import DataModelStub, StubModel, make_batch, make_embeds, make_train_config
from .test_tuple_trainer import first_batch, make_trainer, noise_image, tuple_loss


def pair(seed=5, shape=(1, 4, 8, 8)):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=gen) for _ in range(3)]  # plus, minus, noise


def run(slots, noise, masks=None, sd=None):
    sd = sd or StubModel()
    loss = slider_tuple_loss(
        sd, slot_latents=slots, slot_scales=[[1.0], [-1.0]], slot_weights=[[1.0], [1.0]],
        noise=noise, timesteps=torch.tensor([500.0]), conditional_embeds=make_embeds(1),
        pred_kwargs={}, batch=make_batch(slots[0]), train_config=make_train_config(), slot_masks=masks,
    )
    return loss, sd.lora.weight.grad.clone()


def head_mask(pixels=64, head_rows=16):
    """1 on the body, 0 on a 'head' band at the top (pixel space, 8x the latent)."""
    mask = torch.ones(1, 1, pixels, pixels)
    mask[:, :, :head_rows] = 0.0
    return mask


def test_all_ones_mask_reproduces_unmasked_loss():
    plus, minus, noise = pair()
    ones = torch.ones(1, 1, 64, 64)
    loss_plain, grad_plain = run([plus, minus], noise)
    loss_ones, grad_ones = run([plus, minus], noise, masks=[ones, ones])
    assert loss_ones.item() == pytest.approx(loss_plain.item(), rel=1e-6)
    assert torch.allclose(grad_ones, grad_plain, rtol=1e-6, atol=1e-8)
    # one masked slot, one unmasked slot (None) is also fine
    loss_mixed, _ = run([plus, minus], noise, masks=[ones, None])
    assert loss_mixed.item() == pytest.approx(loss_plain.item(), rel=1e-6)


def test_masked_out_head_latents_do_not_reach_the_lora_gradient():
    plus, minus, noise = pair()
    mask = head_mask()
    plus_other_head = plus.clone()
    plus_other_head[:, :, :2] = torch.randn(1, 4, 2, 8) * 5  # latent rows 0-1 = pixel rows 0-15
    minus_other_head = minus.clone()
    minus_other_head[:, :, :2] = 3.0

    _, grad = run([plus, minus], noise, masks=[mask, mask])
    _, grad_changed_head = run([plus_other_head, minus_other_head], noise, masks=[mask, mask])
    assert torch.allclose(grad, grad_changed_head, rtol=1e-6, atol=1e-8)

    # control: without the mask the same head change moves the gradient
    _, grad_unmasked = run([plus, minus], noise)
    _, grad_unmasked_changed = run([plus_other_head, minus_other_head], noise)
    assert (grad_unmasked - grad_unmasked_changed).abs().max() > 1e-3


def test_latent_mask_resizes_like_sdtrainer_and_normalizes_per_member():
    latents = torch.zeros(2, 4, 8, 8)
    masks = torch.stack([head_mask()[0], torch.full((1, 64, 64), 0.25)])
    weights = latent_loss_mask(masks, latents)
    assert weights.shape == (2, 4, 8, 8)
    assert weights.mean(dim=(1, 2, 3)).tolist() == pytest.approx([1.0, 1.0])
    assert (weights[0, :, :2] == 0).all() and (weights[0, :, 2:] > 0).all()
    # same nearest-neighbor resize SDTrainer uses for mask_multiplier
    expected = torch.nn.functional.interpolate(masks, size=(8, 8)).expand(-1, 4, -1, -1)
    assert torch.allclose(weights[0], expected[0] / expected[0].mean())
    with pytest.raises(ValueError, match="zero everywhere"):
        latent_loss_mask(torch.zeros(1, 1, 64, 64), torch.zeros(1, 4, 8, 8))


@pytest.mark.parametrize("min_value, invert", [(0.0, False), (0.2, False), (0.1, True)])
def test_member_masks_match_upstream_mask_loading(tmp_path, min_value, invert):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir(), masks.mkdir()
    noise_image(str(images / "a.png"), seed=1)
    mask_file = str(masks / "a.png")
    rng = np.random.default_rng(2)
    Image.fromarray(rng.integers(0, 256, (72, 96), dtype=np.uint8)).save(mask_file)
    config = DatasetConfig(folder_path=str(images), mask_path=str(masks), mask_min_value=min_value,
                           invert_mask=invert, resolution=64, num_workers=0)
    loader = get_dataloader_from_datasets([config], batch_size=1, sd=DataModelStub())
    item = loader.dataset.datasets[0].file_list[0]
    for flip_x, flip_y in [(False, False), (True, False), (True, True)]:
        item.flip_x, item.flip_y = flip_x, flip_y
        item.load_mask_image()  # upstream: blur radius is 0 for a 72 px image, so deterministic
        assert torch.allclose(load_member_mask(mask_file, item), item.mask_tensor, atol=1e-6), (flip_x, flip_y)


def test_trainer_applies_manifest_masks(tmp_path):
    root = tmp_path / "data"
    (root / "p01").mkdir(parents=True)
    for i, name in enumerate(["plus", "minus"]):
        noise_image(str(root / "p01" / f"{name}.png"), seed=i)
    Image.new("L", (96, 72), 255).save(root / "p01" / "ones.png")
    body = np.full((72, 96), 255, dtype=np.uint8)
    body[:24] = 0
    Image.fromarray(body).save(root / "p01" / "body.png")

    def manifest(mask_plus, mask_minus, name):
        members = [{"image": "p01/plus.png", "level": 1.0}, {"image": "p01/minus.png", "level": -1.0}]
        for m, mask in zip(members, (mask_plus, mask_minus)):
            if mask:
                m["mask"] = f"p01/{mask}"
        path = root / name
        path.write_text(json.dumps({"id": "p01", "caption": "a person", "members": members}) + "\n")
        return str(path)

    losses = {}
    for name, masks in {"plain": (None, None), "ones": ("ones.png", "ones.png"),
                        "body": ("body.png", "body.png"), "half": ("body.png", None)}.items():
        trainer, config = make_trainer(tmp_path / name, manifest(*masks, f"{name}.jsonl"))
        losses[name] = tuple_loss(trainer, first_batch(trainer, config, 1))[0].item()
    assert losses["ones"] == pytest.approx(losses["plain"], rel=1e-6)
    assert losses["body"] != pytest.approx(losses["plain"], rel=1e-3)
    assert losses["half"] not in (pytest.approx(losses["plain"], rel=1e-3), pytest.approx(losses["body"], rel=1e-3))


def test_different_member_masks_leak_shared_content():
    """With identical member images, matching masks cancel exactly (zero LoRA
    gradient at init); different per-member masks do not."""
    from extensions.slider_tuple.centering import centered_scales

    image = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(3))
    noise = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(4))
    scales = centered_scales([2.0, 1.0, 0.0], [1.0, 1.0, 1.0], kappa=1.0)
    body_a, body_b = head_mask(head_rows=16), head_mask(head_rows=24)

    def grad(masks):
        sd = StubModel()
        with torch.no_grad():
            sd.lora.weight.zero_()
        slider_tuple_loss(
            sd, slot_latents=[image.clone() for _ in scales], slot_scales=[[s] for s in scales],
            slot_weights=[[1.0]] * 3, noise=noise, timesteps=torch.tensor([400.0]), conditional_embeds=make_embeds(1),
            pred_kwargs={}, batch=make_batch(image), train_config=make_train_config(), slot_masks=masks,
        )
        return sd.lora.weight.grad.abs().max().item()

    assert grad([body_a, body_a, body_a]) < 1e-6
    # a mismatch on the member at multiplier 0 (the middle level) is harmless ...
    assert grad([body_a, body_b, body_a]) < 1e-6
    # ... but between members with nonzero multipliers it leaks
    assert grad([body_a, body_a, body_b]) > 1e-3
