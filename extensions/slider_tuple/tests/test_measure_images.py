"""scripts/measure_images.py: grouping, smoothing, picking, and the folder run with stub models."""

import csv
import importlib.util
import os

import numpy as np
import pytest
from PIL import Image

from .test_body_measure import body, keypoints

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "measure_images.py")


def load_script():
    spec = importlib.util.spec_from_file_location("measure_images", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_files_and_smoothing(tmp_path):
    script = load_script()
    for name in ("t1_005.png", "t1_001.png", "t2_010.png", "single.png", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    groups = script.group_files(str(tmp_path))
    assert [i for i, _ in groups["t1"]] == [1, 5] and list(groups) == ["single", "t1", "t2"]
    x = np.arange(20)
    noisy = 1.0 - 0.01 * x + np.where(x % 2, 0.02, -0.02)
    fitted, noise = script.smooth(x, noisy, degree=1)
    assert np.allclose(fitted, 1.0 - 0.01 * x, atol=0.003) and noise == pytest.approx(0.02, abs=0.002)


def test_pick_evenly_spaced_distinct():
    script = load_script()
    smoothed = np.linspace(1.0, 0.7, 31)  # top 1.0 ... bottom 0.7
    targets, chosen = script.pick(smoothed, 1.0, 0.7, 2)
    assert targets == pytest.approx([0.9, 0.8]) and chosen == [10, 20]
    _, chosen = script.pick(np.array([1.0, 0.99, 0.7]), 1.0, 0.7, 2)  # both targets nearest index 1
    assert len(set(chosen)) == 2


def test_folder_run_picks_and_exports(tmp_path):
    script = load_script()
    frames, top, bottom = tmp_path / "frames", tmp_path / "heavy", tmp_path / "thin"
    for folder in (frames, top, bottom):
        folder.mkdir()
    save = lambda path, width: Image.fromarray((body(width) * 255).astype(np.uint8)).convert("RGB").save(path)
    for i in range(1, 30, 2):  # widths 64 -> 36: scores 0.8 -> 0.45 (torso 80 px)
        save(frames / f"tA_{i:03d}.png", int(round(64 - (i - 1))))
    save(top / "tA.png", 64)
    save(bottom / "tA.png", 36)
    stub = (lambda img: (np.asarray(img.convert("L"), np.float32) > 127).astype(np.float32), lambda img: keypoints())
    out = tmp_path / "out"
    groups, picks = script.main([str(frames), "--out", str(out), "--top", str(top), "--bottom", str(bottom),
                                 "--pick", "2", "--preview", "--export"], models=stub)
    assert groups[0]["top"] == pytest.approx(0.8) and groups[0]["bottom"] == pytest.approx(0.45)
    assert [q["rank"] for q in picks] == [1, 2]
    assert all(abs(q["error"]) < 0.02 for q in picks) and picks[0]["index"] < picks[1]["index"]
    assert sorted(os.listdir(out / "export")) == ["mid1", "mid2"] and os.listdir(out / "export" / "mid1") == ["tA.png"]
    assert (out / "preview" / "tA.png").exists()
    # the cache: a second run measures nothing (models that fail would raise)
    def boom(_):
        raise AssertionError("cached images must not be re-measured")
    script.main([str(frames), "--out", str(out), "--top", str(top), "--bottom", str(bottom), "--pick", "2"],
                models=(boom, boom))
    with open(out / "measurements.csv") as f:
        assert len(list(csv.DictReader(f))) == 15 + 2
