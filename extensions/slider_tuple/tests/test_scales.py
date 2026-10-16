"""M3 / F1: per-image slider scales from sidecars and scales.json."""

import json
import os

import pytest
from PIL import Image

from extensions.slider_tuple.scales import (
    SliderDatasetOptions,
    normalize_scales,
    resolve_scales,
    slider_options_by_config,
)
from extensions.slider_tuple.SliderTupleTrainer import SliderTupleTrainer
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets

from .stubs import DataModelStub


def make_images(folder, stems, size=64):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for stem in stems:
        path = os.path.join(folder, f"{stem}.png")
        Image.new("RGB", (size, size)).save(path)
        with open(os.path.join(folder, f"{stem}.txt"), "w") as f:
            f.write("a photo of a person")
        paths.append(path)
    return paths


def write_sidecar(folder, stem, text):
    with open(os.path.join(folder, f"{stem}.scale"), "w") as f:
        f.write(text)


def write_scales_json(folder, data):
    with open(os.path.join(folder, "scales.json"), "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------- resolution order


def test_sidecar_then_json_then_default(tmp_path):
    paths = make_images(tmp_path, ["a", "b", "c"])
    write_sidecar(tmp_path, "a", "-2.5\n")
    write_scales_json(tmp_path, {"b": 1.4})
    scales, sources = resolve_scales(paths, str(tmp_path), default=0.7)
    assert scales == [-2.5, 1.4, 0.7]
    assert sources == {"sidecar": 1, "json": 1, "default": 1}


def test_missing_sidecars_fall_back_without_error(tmp_path):
    paths = make_images(tmp_path, ["a", "b"])
    scales, sources = resolve_scales(paths, str(tmp_path), default=1.0)
    assert scales == [1.0, 1.0]
    assert sources == {"default": 2}


def test_sidecar_and_json_must_agree(tmp_path):
    paths = make_images(tmp_path, ["a"])
    write_sidecar(tmp_path, "a", "0.5")
    write_scales_json(tmp_path, {"a": 0.5})
    assert resolve_scales(paths, str(tmp_path), 1.0)[0] == [0.5]
    write_scales_json(tmp_path, {"a": 0.6})
    with pytest.raises(ValueError, match="sidecar says 0.5 but scales.json says 0.6"):
        resolve_scales(paths, str(tmp_path), 1.0)


@pytest.mark.parametrize("text", ["abc", "", "nan", "inf"])
def test_bad_sidecar_values_raise(tmp_path, text):
    paths = make_images(tmp_path, ["a"])
    write_sidecar(tmp_path, "a", text)
    with pytest.raises(ValueError, match="a.scale"):
        resolve_scales(paths, str(tmp_path), 1.0)


def test_scales_json_must_be_an_object_of_numbers(tmp_path):
    paths = make_images(tmp_path, ["a"])
    write_scales_json(tmp_path, [1.0])
    with pytest.raises(ValueError, match="expected an object"):
        resolve_scales(paths, str(tmp_path), 1.0)
    write_scales_json(tmp_path, {"a": "big"})
    with pytest.raises(ValueError, match="must be a number"):
        resolve_scales(paths, str(tmp_path), 1.0)


def test_scales_json_rejects_ambiguous_stems(tmp_path):
    paths = make_images(tmp_path / "x", ["a"]) + make_images(tmp_path / "y", ["a"])
    write_scales_json(tmp_path, {"a": 1.0})
    with pytest.raises(ValueError, match="match more than one image"):
        resolve_scales(paths, str(tmp_path), 1.0)
    # sidecars are per image, so the same stems are fine without scales.json
    os.remove(tmp_path / "scales.json")
    write_sidecar(tmp_path / "x", "a", "-1")
    write_sidecar(tmp_path / "y", "a", "1")
    assert resolve_scales(paths, str(tmp_path), 0.0)[0] == [-1.0, 1.0]


def test_repeated_paths_are_not_ambiguous(tmp_path):
    # num_repeats lists the same path several times
    paths = make_images(tmp_path, ["a"]) * 3
    write_scales_json(tmp_path, {"a": 0.25})
    assert resolve_scales(paths, str(tmp_path), 1.0)[0] == [0.25] * 3


# ---------------------------------------------------------------- normalization and options


def test_normalize_max_abs_maps_into_unit_range():
    scales, divisor = normalize_scales([-0.6, 1.4, 0.0, 0.7], "max_abs")
    assert divisor == 1.4
    assert scales == pytest.approx([-0.6 / 1.4, 1.0, 0.0, 0.5])


def test_normalize_none_is_identity():
    assert normalize_scales([-2.0, 3.0], "none") == ([-2.0, 3.0], None)


def test_normalize_all_zero_raises():
    with pytest.raises(ValueError, match="nonzero"):
        normalize_scales([0.0, 0.0], "max_abs")


def test_options_default_off_and_validate():
    assert SliderDatasetOptions.from_raw({}) == SliderDatasetOptions(scales=False, normalize="none")
    with pytest.raises(ValueError, match="must be one of"):
        SliderDatasetOptions.from_raw({"slider_scales": True, "slider_scale_normalize": "minmax"})
    with pytest.raises(ValueError, match="requires slider_scales"):
        SliderDatasetOptions.from_raw({"slider_scale_normalize": "max_abs"})


def test_options_pair_with_split_dataset_configs():
    # a resolution list splits one raw dataset into several DatasetConfigs
    raw = [
        {"folder_path": "/data/a", "resolution": [512, 768], "slider_scales": True},
        {"folder_path": "/data/b", "resolution": 512},
    ]
    configs = [DatasetConfig(folder_path="/data/a", resolution=512),
               DatasetConfig(folder_path="/data/a", resolution=768),
               DatasetConfig(folder_path="/data/b", resolution=512)]
    options = slider_options_by_config(raw, configs)
    assert [options[id(c)].scales for c in configs] == [True, True, False]
    assert raw[0]["resolution"] == [512, 768]  # caller's config is not mutated
    with pytest.raises(RuntimeError, match="raw dataset configs"):
        slider_options_by_config(raw, configs[:2])


# ---------------------------------------------------------------- acceptance: batches through the real dataloader


def run_batches(folder, raw_dataset, batch_size, num_workers=0, batches=1):
    """Build the real dataloader, apply scales the way the trainer does, draw batches."""
    config = DatasetConfig(**{**raw_dataset, "num_workers": num_workers})
    loader = get_dataloader_from_datasets([config], batch_size=batch_size, sd=DataModelStub())
    trainer = SliderTupleTrainer.__new__(SliderTupleTrainer)
    trainer.data_loader = loader
    trainer.data_loader_reg = None
    trainer.slider_dataset_options = slider_options_by_config([raw_dataset], [config])
    trainer.apply_slider_scales()
    got = {}
    it = iter(loader)
    for _ in range(batches):
        batch = next(it)
        for item, weight in zip(batch.file_items, batch.get_network_weight_list()):
            got[os.path.basename(item.path)] = weight
    return got


def test_batch_multipliers_equal_sidecar_values(tmp_path):
    make_images(tmp_path, ["m2", "m1", "p1", "ref"])
    write_sidecar(tmp_path, "m2", "-1.8")
    write_sidecar(tmp_path, "m1", "-0.6")
    write_sidecar(tmp_path, "p1", "1.4")  # "ref" has no sidecar -> network_weight
    raw = {"folder_path": str(tmp_path), "resolution": 64, "caption_ext": "txt",
           "network_weight": 0.0, "slider_scales": True}
    got = run_batches(tmp_path, raw, batch_size=4)
    assert got == {"m2.png": -1.8, "m1.png": -0.6, "p1.png": 1.4, "ref.png": 0.0}


def test_batch_multipliers_with_max_abs_normalization(tmp_path):
    make_images(tmp_path, ["m1", "p1"])
    write_scales_json(tmp_path, {"m1": -0.6, "p1": 1.4})
    raw = {"folder_path": str(tmp_path), "resolution": 64, "caption_ext": "txt",
           "slider_scales": True, "slider_scale_normalize": "max_abs"}
    got = run_batches(tmp_path, raw, batch_size=2)
    assert got == pytest.approx({"m1.png": -0.6 / 1.4, "p1.png": 1.0})


def test_scales_reach_dataloader_workers(tmp_path):
    # workers start on first iteration, after the trainer applied the scales
    make_images(tmp_path, ["m1", "p1"])
    write_sidecar(tmp_path, "m1", "-1")
    write_sidecar(tmp_path, "p1", "2")
    raw = {"folder_path": str(tmp_path), "resolution": 64, "caption_ext": "txt", "slider_scales": True}
    got = run_batches(tmp_path, raw, batch_size=1, num_workers=1, batches=2)
    assert got == {"m1.png": -1.0, "p1.png": 2.0}


def test_disabled_dataset_keeps_network_weight(tmp_path):
    make_images(tmp_path, ["a", "b"])
    write_sidecar(tmp_path, "a", "5")  # ignored: slider_scales is off
    raw = {"folder_path": str(tmp_path), "resolution": 64, "caption_ext": "txt", "network_weight": 0.8}
    got = run_batches(tmp_path, raw, batch_size=2)
    assert got == {"a.png": 0.8, "b.png": 0.8}
