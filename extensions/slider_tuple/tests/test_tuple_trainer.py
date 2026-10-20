"""F2 end to end on CPU: manifest -> upstream dataloader -> trainer tuple loss."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from extensions.slider_tuple.members import load_member_tensor
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


def make_manifest(root, tuples, original=False):
    """tuples: {id: [(name, level), ...]} -> tuples.jsonl with one noise image per
    member; the first member is the real photo."""
    lines = []
    for n, (tid, members) in enumerate(tuples.items()):
        os.makedirs(os.path.join(root, tid), exist_ok=True)
        entry = {"id": tid, "caption": f"photo of person {tid}", "members": []}
        for i, (name, level) in enumerate(members):
            noise_image(os.path.join(root, tid, f"{name}.png"), seed=10 * n + i)
            entry["members"].append({"image": f"{tid}/{name}.png", "level": level, "role": "real" if i == 0 else "edit"})
        if original:
            noise_image(os.path.join(root, tid, "original.png"), seed=999)
            entry["original"] = f"{tid}/original.png"
        lines.append(json.dumps(entry))
    path = os.path.join(root, "tuples.jsonl")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def make_trainer(tmp_path, manifest, slider=None, extra_raw=None):
    # level_norm none keeps multipliers equal to centered levels unless a test asks otherwise
    slider = {"level_norm": "none", **(slider or {})}
    raw = {"slider_tuples": manifest, "guidance_type": "slider_tuple", "resolution": 64, **(extra_raw or {})}
    config = DatasetConfig(**raw, num_workers=0)
    trainer = SliderTupleTrainer.__new__(SliderTupleTrainer)
    trainer.dataset_configs = [config]
    trainer.slider_dataset_options = slider_options_by_config([raw], [config])
    trainer.init_slider_state(SliderTrainConfig.from_raw(slider))
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
    manifest = make_manifest(tmp_path / "data", {"p01": [("real", 1.0), ("edit", -1.0)], "p02": [("a", 0.5), ("b", -0.5)]})
    trainer, config = make_trainer(tmp_path, manifest)
    assert trainer.do_guided_loss
    assert config.dataset_path == os.path.join(trainer.save_root, "slider_tuple", "dataset_0.json")
    with open(config.dataset_path) as f:
        anchors = json.load(f)
    assert {os.path.basename(p): v["caption"] for p, v in anchors.items()} == {
        "real.png": "photo of person p01", "a.png": "photo of person p02"}
    batch = first_batch(trainer, config, batch_size=2)
    assert sorted(batch.get_caption_list()) == ["photo of person p01", "photo of person p02"]


# acceptance (1): level difference 2 kappa, unit weights == upstream polarity at weight 1
@pytest.mark.parametrize("batch_size", [1, 2])
def test_pair_two_kappa_apart_through_trainer_equals_polarity(tmp_path, batch_size):
    tuples = {"p01": [("real", 3.0), ("edit", 1.0)], "p02": [("real", 0.5), ("edit", -1.5)]}
    manifest = make_manifest(tmp_path / "data", tuples)
    trainer, config = make_trainer(tmp_path, manifest)  # level_norm none: kappa = 1, levels 2 kappa apart
    assert trainer.slider_kappa == 1.0
    batch = first_batch(trainer, config, batch_size)
    loss, noise, timesteps, embeds = tuple_loss(trainer, batch)
    assert trainer.sd.network.multiplier == [1.0] * batch_size + [-1.0] * batch_size

    before = torch.stack([
        load_member_tensor(item.path.replace("real.png", "edit.png"), item) for item in batch.file_items
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
    assert trainer.additional_logs["slider/weighted_scale_sum"] == pytest.approx(0.0, abs=1e-12)


def test_default_kappa_trains_pairs_at_half_unit(tmp_path):
    manifest = make_manifest(tmp_path / "data", {"p01": [("real", 3.0), ("edit", 1.0)]})
    trainer, config = make_trainer(tmp_path, manifest, slider={"level_norm": "median_span"})
    tuple_loss(trainer, first_batch(trainer, config, 1))
    assert trainer.sd.network.multiplier == [0.5, -0.5]
    assert trainer.additional_logs["slider/span"] == pytest.approx(1.0)


def test_three_member_tuple_centers_weights_and_logs_ranks(tmp_path):
    root = tmp_path / "data"
    manifest = make_manifest(root, {"p01": [("real", 1.4), ("mid", 0.9), ("low", -0.2)]})
    with open(manifest) as f:
        entry = json.loads(f.read())
    entry["members"][1]["quality"] = 0.5
    entry["quality"] = 0.8
    with open(manifest, "w") as f:
        f.write(json.dumps(entry) + "\n")
    trainer, config = make_trainer(tmp_path, manifest)
    tuple_loss(trainer, first_batch(trainer, config, 1))
    # closed form: weighted mean 0.66 with weights (1, .5, 1)
    assert trainer.sd.network.multiplier == pytest.approx([0.74, 0.24, -0.86])
    logs = trainer.additional_logs
    assert logs["slider/members_per_step"] == 3
    assert logs["slider/weighted_scale_sum"] == pytest.approx(0.0, abs=1e-12)
    assert logs["slider/span"] == pytest.approx(1.6)
    assert {k for k in logs if k.startswith("slider/loss_rank")} == {"slider/loss_rank0", "slider/loss_rank1", "slider/loss_rank2"}
    assert trainer.slider_plans["p01"].tuple_weight == pytest.approx(0.8)
    # the scale-0 concept is gone: every member trains, and the cache holds the two non-anchor members
    assert len(trainer.slider_member_latents) == 2


def test_subsets_keep_min_span(tmp_path):
    levels = [("a", 3.0), ("b", 2.9), ("c", 1.0), ("d", 0.0)]
    manifest = make_manifest(tmp_path / "data", {"p01": levels})
    trainer, config = make_trainer(tmp_path, manifest, slider={"members_per_step": 2, "min_span": 1.0})
    batch = first_batch(trainer, config, 1)
    for seed in range(20):
        tuple_loss(trainer, batch, seed=seed)
        assert trainer.additional_logs["slider/span"] >= 1.0
        assert trainer.additional_logs["slider/members_per_step"] == 2


# acceptance (5): dropped and capped tuples are reported at startup
def test_startup_reports_dropped_and_capped_tuples(tmp_path, capsys):
    tuples = {f"t{i}": [("r", 1.0), ("e", 0.0)] for i in range(9)}
    tuples["big"] = [("r", 4.0), ("e", 0.0)]
    tuples["flat"] = [("r", 1.0), ("e", 0.9)]
    manifest = make_manifest(tmp_path / "data", tuples)
    trainer, config = make_trainer(tmp_path, manifest, slider={"level_norm": "median_span", "weight_cap_pct": 90})
    out = capsys.readouterr().out
    assert "10 tuples kept, 1 dropped, 1 weight-capped" in out
    assert "dropped flat: span 0.1" in out and "capped big" in out
    with open(config.dataset_path) as f:
        assert len(json.load(f)) == 10  # the dropped tuple is not in the anchor dataset
    assert trainer.slider_plans["big"].tuple_weight < 1.0


def test_center_offset_warns_and_shifts_multipliers(tmp_path, capsys):
    manifest = make_manifest(tmp_path / "data", {"p01": [("real", 1.0), ("edit", -1.0)]})
    trainer, config = make_trainer(tmp_path, manifest, slider={"center_offset": 1.0})
    assert "WARNING center_offset" in capsys.readouterr().out
    tuple_loss(trainer, first_batch(trainer, config, 1))
    assert trainer.sd.network.multiplier == [2.0, 0.0]
    assert trainer.additional_logs["slider/weighted_scale_sum"] == pytest.approx(2.0)


def test_member_latents_cached_across_steps(tmp_path):
    manifest = make_manifest(tmp_path / "data", {"p01": [("real", 1.0), ("edit", -1.0)]})
    trainer, config = make_trainer(tmp_path, manifest)
    batch = first_batch(trainer, config, 1)
    tuple_loss(trainer, batch)
    tuple_loss(trainer, batch, seed=1)
    assert len(trainer.slider_member_latents) == 1


def test_backward_per_member_through_trainer(tmp_path):
    manifest = make_manifest(tmp_path / "data", {"p01": [("a", 2.0), ("b", 1.0), ("c", -1.0), ("d", -4.0)]})
    trainer, config = make_trainer(tmp_path, manifest, slider={"backward_per_member": True})
    loss, *_ = tuple_loss(trainer, first_batch(trainer, config, 1))
    assert torch.isfinite(loss)
    assert trainer.additional_logs["slider/members_per_step"] == 4
    assert trainer.sd.network.multiplier == [-3.5]  # last slot; levels centered on 0.5


# acceptance (6): the debug dump shows all members crop-aligned (plus the original)
def test_debug_dump_written_for_first_steps(tmp_path):
    manifest = make_manifest(tmp_path / "data", {"p01": [("real", 1.0), ("mid", 0.0), ("low", -1.0)]}, original=True)
    trainer, config = make_trainer(tmp_path, manifest, slider={"debug_dump_steps": 1})
    batch = first_batch(trainer, config, 1)
    tuple_loss(trainer, batch)
    dump = os.path.join(trainer.save_root, "slider_debug", "step000000_p01.png")
    assert Image.open(dump).width == 4 * 384  # three members + original side by side
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
    trainer.init_slider_state(SliderTrainConfig())
    trainer.save_root = str(tmp_path / "run")
    with pytest.raises(ValueError, match="cannot be mixed"):
        trainer.setup_tuple_datasets()


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"slider_tuples": "x.jsonl"}, "requires guidance_type: slider_tuple"),
        ({"guidance_type": "slider_tuple"}, "requires slider_tuples"),
        ({"guidance_type": "slider_tuple", "slider_tuples": "x.jsonl", "slider_scales": True}, "does not apply"),
        ({"guidance_type": "slider_tuple", "slider_tuples": "x.jsonl", "slider_scale_normalize": "max_abs"}, "requires slider_scales"),
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
        ({"zero_sum_weights": True}, "unknown keys"),
        ({"level_norm": "mean"}, "level_norm"),
        ({"min_span": -0.1}, "min_span"),
        ({"weight_cap_pct": 0}, "weight_cap_pct"),
    ],
)
def test_slider_block_validation(slider, message):
    with pytest.raises(ValueError, match=message):
        SliderTrainConfig.from_raw(slider)


def test_startup_warns_about_mismatched_member_masks(tmp_path, capsys):
    root = tmp_path / "data"
    manifest = make_manifest(root, {"p01": [("real", 1.0), ("edit", -1.0)], "p02": [("real", 1.0), ("edit", -1.0)]})
    Image.new("L", (96, 72), 255).save(root / "m.png")
    lines = [json.loads(line) for line in open(manifest)]
    lines[0]["mask"] = "m.png"                      # p01: one tuple-level mask -> fine
    lines[1]["members"][0]["mask"] = "m.png"        # p02: only one member masked -> mismatch
    with open(manifest, "w") as f:
        f.write("\n".join(json.dumps(x) for x in lines) + "\n")
    make_trainer(tmp_path, manifest)
    out = capsys.readouterr().out
    assert "WARNING 1 tuples mask their members differently: p02" in out
