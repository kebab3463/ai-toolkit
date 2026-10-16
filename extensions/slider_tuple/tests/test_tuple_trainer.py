"""M4 / F2 end to end on CPU: manifest -> upstream dataloader -> trainer tuple loss."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from extensions.slider_tuple.members import MemberLatentCache, load_member_tensor
from extensions.slider_tuple.scales import slider_options_by_config
from extensions.slider_tuple.SliderTupleTrainer import SliderTupleTrainer
from extensions.slider_tuple.tuples import SliderTrainConfig
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets
from toolkit.guidance import get_guided_loss_polarity

from .stubs import DataModelStub, StubModel, make_batch, make_embeds, make_train_config


def noise_image(path, seed, size=(96, 72)):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def make_manifest(root, tuples):
    """tuples: {id: [(name, scale), ...]} -> tuples.jsonl with one noise image per member."""
    lines = []
    for n, (tid, members) in enumerate(tuples.items()):
        os.makedirs(os.path.join(root, tid), exist_ok=True)
        entry = {"id": tid, "caption": f"photo of person {tid}", "members": []}
        for i, (name, scale) in enumerate(members):
            noise_image(os.path.join(root, tid, f"{name}.png"), seed=10 * n + i)
            entry["members"].append({"image": f"{tid}/{name}.png", "scale": scale})
        lines.append(json.dumps(entry))
    path = os.path.join(root, "tuples.jsonl")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def make_trainer(tmp_path, manifest, slider=None, extra_raw=None):
    raw = {"slider_tuples": manifest, "guidance_type": "slider_tuple", "resolution": 64, **(extra_raw or {})}
    config = DatasetConfig(**raw, num_workers=0)
    trainer = SliderTupleTrainer.__new__(SliderTupleTrainer)
    trainer.dataset_configs = [config]
    trainer.slider_dataset_options = slider_options_by_config([raw], [config])
    trainer.slider_config = SliderTrainConfig.from_raw(slider)
    trainer.slider_tuples_by_anchor = {}
    trainer.slider_member_latents = MemberLatentCache()
    trainer.save_root = str(tmp_path / "run")
    trainer.do_guided_loss = False
    trainer.setup_tuple_datasets()
    trainer.sd = StubModel()
    trainer.train_config = make_train_config()
    trainer.device_torch = torch.device("cpu")
    trainer.step_num = 0
    trainer.additional_logs = {}
    return trainer, config


def first_batch(trainer, config, batch_size):
    loader = get_dataloader_from_datasets([config], batch_size=batch_size, sd=DataModelStub())
    batch = next(iter(loader))
    batch.latents = trainer.sd.encode_images(batch.tensor)  # what process_general_training_batch does
    return batch


def tuple_loss(trainer, batch, seed=0):
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(batch.latents.shape, generator=gen)
    timesteps = torch.randint(1, 1000, (batch.latents.shape[0],), generator=gen).float()
    embeds = make_embeds(batch.latents.shape[0])
    loss = trainer.get_guided_loss(
        noisy_latents=None, conditional_embeds=embeds, match_adapter_assist=False,
        network_weight_list=[1.0] * batch.latents.shape[0], timesteps=timesteps, pred_kwargs={},
        batch=batch, noise=noise, mask_multiplier=None, prior_pred=None,
    )
    return loss, noise, timesteps, embeds


def test_setup_points_dataset_at_anchor_json(tmp_path):
    manifest = make_manifest(tmp_path / "data", {"p01": [("after", 1.0), ("before", -1.0)], "p02": [("a", 0.5), ("b", -0.5)]})
    trainer, config = make_trainer(tmp_path, manifest)
    assert trainer.do_guided_loss
    assert config.dataset_path == os.path.join(trainer.save_root, "slider_tuple", "dataset_0.json")
    with open(config.dataset_path) as f:
        anchors = json.load(f)
    assert {os.path.basename(p): v["caption"] for p, v in anchors.items()} == {
        "after.png": "photo of person p01", "a.png": "photo of person p02"}
    batch = first_batch(trainer, config, batch_size=2)
    assert sorted(batch.get_caption_list()) == ["photo of person p01", "photo of person p02"]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_two_member_tuple_through_trainer_equals_polarity(tmp_path, batch_size):
    tuples = {"p01": [("after", 1.0), ("before", -1.0), ("ref", 0.0)], "p02": [("after", 1.0), ("before", -1.0)]}
    manifest = make_manifest(tmp_path / "data", tuples)
    trainer, config = make_trainer(tmp_path, manifest)
    batch = first_batch(trainer, config, batch_size)
    loss, noise, timesteps, embeds = tuple_loss(trainer, batch)
    assert trainer.sd.network.multiplier == [1.0] * batch_size + [-1.0] * batch_size

    # polarity on the same pair: the anchor as the conditional image, the crop-aligned
    # "before" member as the unconditional image, same noise / timesteps / embeds
    before = torch.stack([
        load_member_tensor(item.path.replace("after.png", "before.png"), item) for item in batch.file_items
    ])
    ref_sd = StubModel()
    ref = get_guided_loss_polarity(
        noisy_latents=None, conditional_embeds=embeds, match_adapter_assist=False,
        network_weight_list=[1.0] * batch_size, timesteps=timesteps, pred_kwargs={},
        batch=make_batch(batch.latents, ref_sd.encode_images(before)), noise=noise, sd=ref_sd,
        train_config=make_train_config(),
    )
    assert loss.item() == pytest.approx(ref.item(), rel=1e-6)
    assert torch.allclose(trainer.sd.lora.weight.grad, ref_sd.lora.weight.grad, rtol=1e-5, atol=1e-7)
    # the scale-0 reference member was never encoded
    assert len(trainer.slider_member_latents) == batch_size


def test_zero_sum_weights_logged_and_member_latents_cached(tmp_path):
    manifest = make_manifest(tmp_path / "data", {"p01": [("m", -0.6), ("p", 1.4)]})
    trainer, config = make_trainer(tmp_path, manifest, slider={"zero_sum_weights": True})
    batch = first_batch(trainer, config, 1)
    tuple_loss(trainer, batch)
    assert trainer.additional_logs["slider/weighted_scale_sum"] == pytest.approx(0.0, abs=1e-12)
    assert trainer.additional_logs["slider/members_per_step"] == 2
    assert len(trainer.slider_member_latents) == 1
    tuple_loss(trainer, batch, seed=1)
    assert len(trainer.slider_member_latents) == 1  # second step reuses the cached member latent

    trainer_plain, config_plain = make_trainer(tmp_path / "plain", manifest)
    tuple_loss(trainer_plain, first_batch(trainer_plain, config_plain, 1))
    assert trainer_plain.additional_logs["slider/weighted_scale_sum"] == pytest.approx(0.8)


def test_members_per_step_and_normalization(tmp_path):
    manifest = make_manifest(tmp_path / "data", {"p01": [("a", 2.0), ("b", 1.0), ("c", -1.0), ("d", -4.0)]})
    trainer, config = make_trainer(
        tmp_path, manifest, slider={"members_per_step": "all", "backward_per_member": True},
        extra_raw={"slider_scale_normalize": "max_abs"},
    )
    batch = first_batch(trainer, config, 1)
    loss, *_ = tuple_loss(trainer, batch)
    assert torch.isfinite(loss)
    assert trainer.additional_logs["slider/members_per_step"] == 4
    # backward_per_member leaves the last slot's multipliers; scales divided by max |s| = 4
    assert trainer.sd.network.multiplier == [-1.0]
    assert [m.scale for m in next(iter(trainer.slider_tuples_by_anchor.values())).members] == [0.5, 0.25, -0.25, -1.0]


def test_debug_dump_written_for_first_steps(tmp_path):
    manifest = make_manifest(tmp_path / "data", {"p01": [("after", 1.0), ("ref", 0.0), ("before", -1.0)]})
    trainer, config = make_trainer(tmp_path, manifest, slider={"debug_dump_steps": 1})
    batch = first_batch(trainer, config, 1)
    tuple_loss(trainer, batch)
    dump = os.path.join(trainer.save_root, "slider_debug", "step000000_p01.png")
    assert os.path.exists(dump)
    trainer.step_num = 1
    tuple_loss(trainer, batch)
    assert not os.path.exists(os.path.join(trainer.save_root, "slider_debug", "step000001_p01.png"))


def test_rejects_mixed_and_misconfigured_datasets(tmp_path):
    manifest = make_manifest(tmp_path / "data", {"p01": [("a", 1.0), ("b", -1.0)]})
    with pytest.raises(ValueError, match="only slider_tuples"):
        make_trainer(tmp_path, manifest, extra_raw={"folder_path": str(tmp_path)})
    with pytest.raises(ValueError, match="buckets must be enabled"):
        make_trainer(tmp_path, manifest, extra_raw={"buckets": False})

    raw_tuples = {"slider_tuples": manifest, "guidance_type": "slider_tuple", "resolution": 64}
    raw_plain = {"folder_path": str(tmp_path / "data" / "p01"), "resolution": 64}
    configs = [DatasetConfig(**raw_tuples), DatasetConfig(**raw_plain)]
    trainer = SliderTupleTrainer.__new__(SliderTupleTrainer)
    trainer.dataset_configs = configs
    trainer.slider_dataset_options = slider_options_by_config([raw_tuples, raw_plain], configs)
    trainer.slider_config = SliderTrainConfig()
    trainer.slider_tuples_by_anchor = {}
    trainer.save_root = str(tmp_path / "run")
    with pytest.raises(ValueError, match="cannot be mixed"):
        trainer.setup_tuple_datasets()


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"slider_tuples": "x.jsonl"}, "requires guidance_type: slider_tuple"),
        ({"guidance_type": "slider_tuple"}, "requires slider_tuples"),
        ({"guidance_type": "slider_tuple", "slider_tuples": "x.jsonl", "slider_scales": True}, "does not apply"),
    ],
)
def test_dataset_option_validation(raw, message):
    with pytest.raises(ValueError, match=message):
        slider_options_by_config([raw], [DatasetConfig(**raw)])


@pytest.mark.parametrize(
    "slider, message",
    [
        ({"members_per_step": 1}, "members_per_step"),
        ({"debug_dump_steps": -1}, "debug_dump_steps"),
        ({"member_per_step": 2}, "unknown keys"),
    ],
)
def test_slider_block_validation(slider, message):
    with pytest.raises(ValueError, match=message):
        SliderTrainConfig.from_raw(slider)
