"""compose / decompose round trip and scripts/reweight_masks.py."""

import importlib.util
import json
import os

import numpy as np
import pytest
from PIL import Image

from extensions.slider_tuple.manifest import load_manifest
from extensions.slider_tuple.tuple_masks import compose, decompose, to_png_array

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "reweight_masks.py")


def parts():
    body = np.zeros((40, 30), np.float32)
    body[10:40, 8:22] = 1.0
    body[10:40, 7] = 0.5  # a soft edge
    head = np.zeros((40, 30), np.float32)
    head[2:10, 10:20] = 1.0
    return body, head


def test_head_weight_and_decompose_round_trip():
    body, head = parts()
    mask = compose(body, head, background=0.2, head_weight=0.0)
    assert mask[5, 15] == 0.0 and mask[30, 15] == 1.0 and mask[0, 0] == pytest.approx(0.2)
    assert compose(body, head, head_weight=0.15)[5, 15] == pytest.approx(0.15)
    stored = to_png_array(mask).astype(np.float32) / 255.0  # through 8 bits, as on disk
    b, h = decompose(stored)
    assert np.array_equal(h, head) and np.allclose(b, body * (1 - head), atol=0.01)
    with pytest.raises(ValueError, match="head_weight < background"):
        decompose(stored, background=0.2, head_weight=0.3)


def test_reweight_script_writes_new_masks_and_manifest(tmp_path):
    spec = importlib.util.spec_from_file_location("reweight_masks", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    data = tmp_path / "data"
    (data / "masks").mkdir(parents=True)
    body, head = parts()
    Image.fromarray(to_png_array(compose(body, head))).save(data / "masks" / "t1.png")
    for name in ("a", "b"):
        Image.new("RGB", (30, 40)).save(data / f"{name}.png")
    entry = {"id": "t1", "caption": "c", "mask": "masks/t1.png",
             "members": [{"image": "a.png", "level": 1.0}, {"image": "b.png", "level": -1.0}]}
    (data / "train.jsonl").write_text(json.dumps(entry) + "\n")
    out = tmp_path / "bg05_head03" / "train.jsonl"
    script.main([str(data / "train.jsonl"), str(out), "--background", "0.5", "--head-weight", "0.3"])
    tup = load_manifest(str(out))[0]
    new = np.asarray(Image.open(tup.mask), dtype=np.float32) / 255.0
    assert new[5, 15] == pytest.approx(0.3, abs=0.003) and new[0, 0] == pytest.approx(0.5, abs=0.003)
    assert new[30, 15] == pytest.approx(1.0) and new[30, 7] == pytest.approx(0.75, abs=0.01)
    assert os.path.exists(tup.members[0].image)  # member paths rebased to the new manifest
    old = np.asarray(Image.open(data / "masks" / "t1.png"), dtype=np.float32) / 255.0
    assert old[0, 0] == pytest.approx(0.2, abs=0.003)  # originals untouched
    with pytest.raises(SystemExit, match="new manifest"):
        script.main([str(data / "train.jsonl"), str(data / "train.jsonl"), "--background", "0.5", "--head-weight", "0.3"])
