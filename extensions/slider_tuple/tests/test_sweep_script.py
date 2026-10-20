"""F6: slider_sweep.py end to end without a GPU (all images pre-rendered, LPIPS off)."""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "slider_sweep.py")


def load_script():
    spec = importlib.util.spec_from_file_location("slider_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --measure / --face callables for the test (module:function)
def mean_red(image):
    return float(np.asarray(image, dtype=np.float32)[..., 0].mean())


def mean_color(image):
    return np.asarray(image, dtype=np.float32).reshape(-1, 3).mean(axis=0)


def run(script, tmp_path, red_for_m, extra=()):
    out = tmp_path / "sweep"
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a person in a park\na person in an office\n")
    for i in range(2):
        for seed in (1, 2):
            for m in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, -0.5, -1.0]:
                path = script.image_path(str(out), i, seed, m)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                Image.new("RGB", (32, 32), (int(red_for_m(m)), 80, 80)).save(path)
    argv = ["slider_sweep.py", "--lora", "unused.safetensors", "--prompts", str(prompts), "--seeds", "1,2",
            "--measure", f"{__name__}:mean_red", "--no-lpips", "--out", str(out), *extra]
    old = sys.argv
    sys.argv = argv
    try:
        script.main()
    finally:
        sys.argv = old
    with open(out / "summary.json") as f:
        return out, json.load(f)


def test_sweep_outputs_and_linear_measure(tmp_path):
    script = load_script()
    out, summary = run(script, tmp_path, red_for_m=lambda m: 100 + 40 * m)
    assert summary["positive"]["slope"] == pytest.approx(40.0, abs=0.5)
    assert summary["positive"]["r2"] == pytest.approx(1.0, abs=1e-3)
    assert summary["positive"]["spearman"] == pytest.approx(1.0)
    assert summary["args"]["steps"] == 8 and summary["args"]["cfg"] == 1.0 and summary["args"]["mu"] == 1.15
    assert len(open(out / "sweep.csv").read().splitlines()) == 1 + 2 * 2 * 9
    assert sorted(os.listdir(out / "sheets")) == ["prompt_000.png", "prompt_001.png"]
    assert (out / "summary.txt").read_text().startswith("positive range:")


def test_identical_renders_give_no_effect_and_full_face_similarity(tmp_path):
    script = load_script()
    _, summary = run(script, tmp_path, red_for_m=lambda m: 120, extra=["--face", f"{__name__}:mean_color"])
    assert summary["positive"]["slope"] == pytest.approx(0.0, abs=1e-9)
    assert summary["positive"]["min_face_sim"] == pytest.approx(1.0)
    assert summary["flags"] == []


def test_raw_preset_and_prompt_sources(tmp_path):
    script = load_script()
    old = sys.argv
    sys.argv = ["x", "--raw", "--prompts", "p.txt", "--out", "o"]
    try:
        args = script.parse_args()
    finally:
        sys.argv = old
    assert (args.steps, args.cfg, args.mu) == (25, 4.0, None)
    manifest = tmp_path / "held_out.jsonl"
    manifest.write_text(json.dumps({"id": "a", "caption": "photo of a person", "members": []}) + "\n")
    assert script.read_prompts(str(manifest)) == ["photo of a person"]
