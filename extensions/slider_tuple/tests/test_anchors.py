"""M5 / F4: anchor preservation loss."""


import numpy as np
import pytest
import torch
from PIL import Image

from extensions.slider_tuple.anchors import anchor_loss, center_crop_geometry, list_anchor_images
from extensions.slider_tuple.members import load_member_tensor
from extensions.slider_tuple.tuples import SliderTrainConfig

from .stubs import StubModel, make_batch, make_embeds
from .test_tuple_trainer import first_batch, make_manifest, make_trainer, tuple_loss


def inputs(seed=0, batch_size=2):
    gen = torch.Generator().manual_seed(seed)
    latents = torch.randn(batch_size, 4, 8, 8, generator=gen)
    noise = torch.randn(batch_size, 4, 8, 8, generator=gen)
    return latents, noise, torch.tensor([300.0, 700.0][:batch_size])


def run_anchor(sd, multipliers, strength=1.0, seed=0):
    latents, noise, timesteps = inputs(seed, len(multipliers))
    embeds = [make_embeds(1, seed=i) for i in range(len(multipliers))]
    return anchor_loss(sd, latents, noise, timesteps, embeds, multipliers, strength, {}, make_batch(latents))


def test_target_is_the_frozen_model():
    sd = StubModel()
    loss, mse = run_anchor(sd, [0.5, -1.5], strength=2.0)
    latents, noise, timesteps = inputs()
    x_t = sd.add_noise(latents, noise, timesteps)
    with torch.no_grad():
        delta = torch.tensor([0.5, -1.5]).view(-1, 1, 1, 1) * sd.lora(x_t)  # LoRA-on minus LoRA-off
    assert mse == pytest.approx(delta.pow(2).mean().item(), rel=1e-5)
    assert loss.item() == pytest.approx(2.0 * mse, rel=1e-6)
    assert sd.network.is_active and sd.network.multiplier == [0.5, -1.5]


def test_zero_when_the_lora_does_nothing():
    sd = StubModel()
    loss, mse = run_anchor(sd, [0.0, 0.0])
    assert mse == 0.0 and sd.lora.weight.grad.abs().max() == 0
    sd = StubModel()
    with torch.no_grad():
        sd.lora.weight.zero_()
    _, mse = run_anchor(sd, [1.0, -1.0])
    assert mse == 0.0


def test_gradient_steps_reduce_the_anchor_loss():
    sd = StubModel()
    optimizer = torch.optim.SGD(sd.lora_parameters(), lr=0.5)
    history = []
    for step in range(20):
        optimizer.zero_grad()
        _, mse = run_anchor(sd, [0.8, -0.6], seed=step)
        history.append(mse)
        optimizer.step()
    assert history[-1] < 0.2 * history[0]


def test_center_crop_geometry_covers_and_centers(tmp_path):
    g = center_crop_geometry((200, 100), (64, 48))
    assert (g.scale_to_width, g.scale_to_height) == (96, 48)
    assert (g.crop_x, g.crop_y, g.crop_width, g.crop_height) == (16, 0, 64, 48)
    path = tmp_path / "scene.png"
    Image.new("RGB", (200, 100), "green").save(path)
    assert load_member_tensor(str(path), g).shape == (3, 48, 64)
    assert list_anchor_images(str(tmp_path)) == [str(path)]
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no images"):
        list_anchor_images(str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="not a folder"):
        list_anchor_images(str(tmp_path / "missing"))


def anchor_trainer(tmp_path, slider):
    manifest = make_manifest(tmp_path / "data", {"p01": [("after", 1.0), ("before", -1.0)]})
    trainer, config = make_trainer(tmp_path, manifest, slider=slider)
    trainer.slider_anchor_embeds = [make_embeds(1, seed=9), make_embeds(1, seed=10)]
    return trainer, first_batch(trainer, config, 1)


def test_trainer_adds_and_logs_anchor_loss(tmp_path):
    slider = {"anchor_strength": 0.5, "anchor_prompts": ["a forest", "a kitchen"]}
    plain, batch_plain = anchor_trainer(tmp_path / "plain", None)
    slider_only = tuple_loss(plain, batch_plain)[0].item()

    trainer, batch = anchor_trainer(tmp_path / "anchor", slider)
    torch.manual_seed(0)
    total = tuple_loss(trainer, batch)[0]
    mse = trainer.additional_logs["slider/anchor_loss"]
    assert mse > 0
    assert total.item() == pytest.approx(slider_only + 0.5 * mse, rel=1e-5)
    assert total.requires_grad and total.grad_fn is None

    never, batch_never = anchor_trainer(tmp_path / "never", {**slider, "anchor_prob": 0.0})
    tuple_loss(never, batch_never)
    assert "slider/anchor_loss" not in never.additional_logs


def test_anchor_images_replace_batch_latents(tmp_path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    for i in range(2):
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (120, 90, 3), dtype=np.uint8)).save(scenes / f"s{i}.png")
    trainer, batch = anchor_trainer(
        tmp_path, {"anchor_strength": 1.0, "anchor_prompts": ["a forest"], "anchor_images": str(scenes)}
    )
    trainer.slider_anchor_images = list_anchor_images(str(scenes))
    latents = trainer.anchor_latents(batch)
    assert latents.shape == batch.latents.shape
    assert not torch.allclose(latents, batch.latents)
    tuple_loss(trainer, batch)
    assert trainer.additional_logs["slider/anchor_loss"] > 0


@pytest.mark.parametrize(
    "slider, message",
    [
        ({"anchor_strength": 1.0}, "needs anchor_prompts"),
        ({"anchor_strength": -1.0, "anchor_prompts": ["x"]}, ">= 0"),
        ({"anchor_prob": 1.5}, r"in \[0, 1\]"),
        ({"anchor_prompts": ["ok", ""]}, "non-empty strings"),
        ({"anchor_strength": "high"}, "must be a number"),
    ],
)
def test_anchor_option_validation(slider, message):
    with pytest.raises(ValueError, match=message):
        SliderTrainConfig.from_raw(slider)
