"""sweep_grid.py and sweep_from_samples.py: the training previews as a sweep."""

import importlib.util
import json
import os
import shutil

import pytest
import yaml
from PIL import Image

from extensions.slider_tuple.sweep import grid_samples, rows_from_samples

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "experiments", "weight_r3b_rank4.yaml")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_and_rows():
    samples = grid_samples(["a woman", "b woman"], [1, 2], [1.0, -1.0, 0.0])
    assert len(samples) == 12 and samples[0] == {"prompt": "a woman", "seed": 1, "network_multiplier": -1.0}
    rows = rows_from_samples(samples, {i: f"img{i}.png" for i in range(12) if i != 5})
    assert len(rows) == 11 and {r.prompt_idx for r in rows} == {0, 1}
    legacy = rows_from_samples([{"prompt": "x --m 1.5", "seed": 3}], {0: "p.png"})[0]
    assert (legacy.prompt, legacy.m, legacy.seed) == ("x", 1.5, 3)
    with pytest.raises(ValueError, match="needs a seed"):
        rows_from_samples([{"prompt": "x --m 1"}], {0: "p.png"})


def test_sweep_grid_rewrites_a_real_config(tmp_path):
    script = load("sweep_grid")
    held = tmp_path / "held_out.jsonl"
    held.write_text("".join(json.dumps({"id": i, "caption": f'caption "{i}" with: colons'}) + "\n" for i in "ab"))
    config = tmp_path / "c.yaml"
    shutil.copy(CONFIG, config)
    script.main([str(config), "--prompts", str(held)])
    text = config.read_text()
    assert text.startswith("# Run 3b")  # comments kept
    sample = yaml.safe_load(text)["config"]["process"][0]["sample"]
    assert len(sample["samples"]) == 2 * 3 * 7 and "prompts" not in sample
    assert sample["samples"][0] == {"prompt": 'caption "a" with: colons', "seed": 1000, "network_multiplier": -3.0}
    assert sample["format"] == "png" and sample["walk_seed"] is False and sample["sample_every"] == 250
    assert yaml.safe_load(text)["config"]["process"][0]["train"]["skip_first_sample"] is True
    script.main([str(config), "--prompts", str(held), "--seeds", "7", "--multipliers", "0,1,-1"])  # idempotent
    assert len(yaml.safe_load(config.read_text())["config"]["process"][0]["sample"]["samples"]) == 2 * 3


def mean_red(image):
    import numpy as np
    return float(np.asarray(image, dtype=np.float32)[..., 0].mean())


def test_sweep_from_samples_per_step(tmp_path):
    script = load("sweep_from_samples")
    samples = grid_samples(["p one", "p two"], [1, 2], [0.0, 1.0, -1.0])
    run = tmp_path / "output" / "myrun"
    (run / "samples").mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(
        {"job": "extension", "config": {"name": "myrun", "process": [{"sample": {"samples": samples}}]}}))
    for step, slope in ((125, 10), (250, 40)):
        for i, s in enumerate(samples):
            red = int(100 + slope * s["network_multiplier"])
            Image.new("RGB", (16, 16), (red, 0, 0)).save(run / "samples" / f"17000{step}__{step:09d}_{i}.png")
    partial = run / "samples" / "17999__000000375_0.png"  # a step still being sampled
    Image.new("RGB", (16, 16)).save(partial)
    out = tmp_path / "sweeps"
    argv = [str(run), "--out", str(out), "--measure", f"{__name__}:mean_red", "--no-lpips"]
    done = script.main(argv, models=(None, None, None))
    assert [os.path.basename(d) for d in done] == ["myrun_000000125", "myrun_000000250"]
    for step, slope in ((125, 10), (250, 40)):
        with open(out / f"myrun_{step:09d}" / "summary.json") as f:
            assert json.load(f)["positive"]["slope"] == pytest.approx(slope, abs=0.5)
    assert script.main(argv, models=(None, None, None)) == []  # nothing new to measure
