"""One loss mask per tuple: geometry, composition and the generator script (stub models)."""

import importlib.util
import json
import os

import numpy as np
import pytest
from PIL import Image

from extensions.slider_tuple.manifest import load_manifest
from extensions.slider_tuple.tuple_masks import compose, head_ellipse, overlay, union

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "make_tuple_masks.py")


def kps(**points):
    """17 COCO keypoints, all invisible except the named ones: name=(x, y)."""
    names = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder"]
    out = np.zeros((17, 3))
    for name, (x, y) in points.items():
        out[names.index(name)] = (x, y, 5.0)
    return out


def test_head_ellipse_from_ears_covers_the_head_only():
    k = kps(nose=(50, 34), l_eye=(46, 30), r_eye=(54, 30), l_ear=(40, 31), r_ear=(60, 31),
            l_shoulder=(30, 60), r_shoulder=(70, 60))
    head = head_ellipse(k, (100, 100), margin=0.0)
    # width 1.4 * 20 = 28, height 37.8, centered just above the face points
    assert head[31, 50] == 1 and head[18, 50] == 1          # face and hair above it
    assert head[31, 30] == 0 and head[60, 50] == 0          # beside the head, shoulder line
    cols = np.where(head.any(axis=0))[0]
    assert 27 <= cols.max() - cols.min() + 1 <= 29


def test_head_ellipse_fallbacks_and_missing_face():
    by_eyes = head_ellipse(kps(l_eye=(46, 30), r_eye=(54, 30)), (100, 100), margin=0.0)
    assert by_eyes.any(axis=0).sum() in (23, 24, 25)       # 3 x eye distance 8
    by_shoulders = head_ellipse(kps(nose=(50, 30), l_shoulder=(30, 60), r_shoulder=(70, 60)), (100, 100), margin=0.0)
    # max(0.45 x shoulder width 40 = 18, neck 30 / 1.35 = 22.2)
    assert by_shoulders.any(axis=0).sum() in (21, 22, 23)
    assert head_ellipse(kps(l_shoulder=(30, 60), r_shoulder=(70, 60)), (100, 100)) is None  # no face point
    assert head_ellipse(kps(nose=(50, 30)), (100, 100)) is None                           # nothing to size it


def test_head_ellipse_in_profile_is_not_collapsed():
    # profile / back view: both ears and both eyes land on nearly the same point
    k = kps(nose=(52, 30), l_eye=(51, 28), r_eye=(52, 28), l_ear=(48, 29), r_ear=(49, 29),
            l_shoulder=(45, 62), r_shoulder=(55, 62))
    head = head_ellipse(k, (100, 100), margin=0.0)
    width = head.any(axis=0).sum()
    assert width >= 20  # from the neck (~33 px / 1.35), not the 1-3 px ear / eye distances
    low = kps(l_ear=(40, 31), r_ear=(60, 31))
    low[3, 2] = 0.5  # left ear below the score threshold -> no ear distance, no eyes -> None
    assert head_ellipse(low, (100, 100), min_score=1.0) is None


def test_compose_union_and_overlay():
    body_a = np.zeros((4, 4), np.float32); body_a[2:, :2] = 1.0
    body_b = np.zeros((4, 4), np.float32); body_b[2:, 1:3] = 0.5
    head = np.zeros((4, 4), np.float32); head[0, 0] = 1.0
    mask = compose(union([body_a, body_b]), head, background=0.2)
    assert mask[0, 0] == 0.0                                  # head
    assert mask[0, 3] == pytest.approx(0.2)                   # background
    assert mask[3, 0] == pytest.approx(1.0)                   # body in a
    assert mask[3, 2] == pytest.approx(0.6)                   # soft body edge in b
    with pytest.raises(ValueError):
        compose(body_a, head, background=0.5, body_weight=0.4)
    shown = overlay(np.full((4, 4, 3), 200, np.uint8), mask)
    assert shown[0, 0, 0] > shown[0, 0, 1]                    # head tinted red


def load_script():
    spec = importlib.util.spec_from_file_location("make_tuple_masks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_data(root):
    """Two tuples of 3 members (64x64); 'nohead' has no detectable face."""
    lines = []
    for tid in ("p01", "nohead"):
        os.makedirs(root / tid, exist_ok=True)
        members = []
        for i, level in enumerate((2.0, 1.0, 0.0)):
            Image.new("RGB", (64, 64), (40 * i, 0, 0)).save(root / tid / f"m{i}.png")
            members.append({"image": f"{tid}/m{i}.png", "level": level, "mask": f"{tid}/m{i}.png"})
        lines.append(json.dumps({"id": tid, "caption": "a person", "members": members}))
    (root / "tuples.jsonl").write_text("\n".join(lines) + "\n")
    return root / "tuples.jsonl"


def stub_models():
    def matte_fn(img):
        # member i's body: columns 20..(30 + 5 i), i read from the image's red value
        i = img.getpixel((0, 0))[0] // 40
        body = np.zeros((64, 64), np.float32)
        body[20:, 20:30 + 5 * i] = 1.0
        return body

    def keypoints_fn(img):
        return kps(l_ear=(20, 10), r_ear=(30, 10))  # every member: same head (pasted back)

    return matte_fn, keypoints_fn


def test_generator_writes_tuple_masks_and_manifest(tmp_path):
    script = load_script()
    data = tmp_path / "data"
    manifest = make_data(data)
    out = tmp_path / "out" / "tuples_masked.jsonl"
    out.parent.mkdir()
    args = script.parse_args([str(manifest), str(out), "--preview", "--head-margin", "0"])
    matte_fn, keypoints_fn = stub_models()

    missing = script.run(args, matte_fn, keypoints_fn)
    assert missing == []
    tuples = load_manifest(str(out))
    t = tuples[0]
    assert t.mask is not None and not t.mask_mismatch           # one tuple mask, member masks dropped
    mask = np.asarray(Image.open(t.mask), dtype=np.float32) / 255
    assert mask[40, 39] == pytest.approx(1.0, abs=0.01)          # only member 2's body reaches column 39: union
    assert mask[40, 50] == pytest.approx(0.2, abs=0.01)          # background
    assert mask[10, 25] == 0.0                                   # head
    assert os.path.exists(tmp_path / "out" / "masks" / "preview" / "p01.jpg")


def test_generator_reports_tuples_without_a_head_and_uses_head_masks(tmp_path):
    script = load_script()
    manifest = make_data(tmp_path / "data")
    out = tmp_path / "masked.jsonl"
    matte_fn, _ = stub_models()
    missing = script.run(script.parse_args([str(manifest), str(out)]), matte_fn, lambda img: None)
    assert missing == ["p01", "nohead"]
    assert all(t.mask is None for t in load_manifest(str(out)))  # left unmasked, not half-masked

    heads = tmp_path / "heads"
    heads.mkdir()
    head = np.zeros((64, 64), np.uint8); head[:8, :8] = 255
    Image.fromarray(head).save(heads / "p01.png")
    missing = script.run(script.parse_args([str(manifest), str(out), "--head-masks", str(heads)]), matte_fn, lambda img: None)
    assert missing == ["nohead"]
    mask = np.asarray(Image.open(load_manifest(str(out))[0].mask), dtype=np.float32) / 255
    assert mask[0, 0] == 0.0 and mask[10, 25] > 0
