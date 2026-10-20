"""scripts/paste_back.py end to end on synthetic heavy/thin/orig images (stub models)."""

import csv
import importlib.util
import json
import os

import cv2
import numpy as np
import pytest
from PIL import Image

from extensions.slider_tuple.manifest import load_manifest

from .test_paste_back import BODY_CENTER, HEAD_CENTER, HEAD_R, SIZE, ellipse, scene, textured

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "paste_back.py")


def load_script():
    spec = importlib.util.spec_from_file_location("paste_back_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stub_models():
    head_patch = textured(2)[int(HEAD_CENTER[1]) - 12:int(HEAD_CENTER[1]) + 12, int(HEAD_CENTER[0]) - 12:int(HEAD_CENTER[0]) + 12]
    body = np.maximum(ellipse(BODY_CENTER, (70, 100)), 0)

    def matte_fn(img):
        return body.copy()  # generous fixed body region: enough for plumbing

    def keypoints_fn(img):
        # find the head by its texture; ears 2 * 0.71 * HEAD_R apart around it
        scores = cv2.matchTemplate(np.asarray(img), head_patch, cv2.TM_CCOEFF_NORMED)
        _, peak, _, (x, y) = cv2.minMaxLoc(scores)
        if peak < 0.5:
            return None
        cx, cy = x + 12, y + 12
        kps = np.zeros((17, 3))
        kps[3] = (cx - 0.71 * HEAD_R, cy, 5.0)
        kps[4] = (cx + 0.71 * HEAD_R, cy, 5.0)
        kps[0] = (cx, cy, 5.0)
        return kps

    return matte_fn, keypoints_fn


def make_dataset(root):
    for stem, seed in (("a", 10), ("b", 20)):
        original, heavy, *_ = scene(global_shift=(3.0, -2.0), head_shift=(2.5, 1.5), body_w=60, noise_seed=seed)
        _, thin, *_ = scene(global_shift=(-1.0, 1.5), head_shift=(-1.0, 0.5), body_w=35, noise_seed=seed + 1)
        for folder, image in (("orig", original), ("heavy", heavy), ("thin", thin)):
            os.makedirs(root / folder, exist_ok=True)
            Image.fromarray(image).save(root / folder / f"{stem}.png")
    Image.fromarray(textured(5)).save(root / "orig" / "lonely.png")  # no members -> skipped


def test_paste_back_script_writes_a_tuple_dataset(tmp_path):
    script = load_script()
    data, out = tmp_path / "data", tmp_path / "tuples"
    make_dataset(data)
    argv = [str(data), str(out), "--member", "heavy=1", "--member", "thin=-1", "--original", "orig", "--preview"]
    report = script.main(argv, models=stub_models())

    assert {(r["name"], r["member"]) for r in report} == {(s, m) for s in "ab" for m in ("heavy", "thin")}
    assert all(r["head_pasted"] for r in report)
    tuples = load_manifest(str(out / "tuples.jsonl"))
    assert [t.id for t in tuples] == ["a", "b"]
    t = tuples[0]
    assert [m.level for m in t.members] == [1.0, -1.0] and t.caption == "a photo of a woman"
    assert t.original.endswith("original/a.png") and t.mask.endswith("masks/a.png") and not t.mask_mismatch
    assert os.path.exists(out / "preview" / "a.jpg") and os.path.exists(out / "report.csv")
    with open(out / "report.csv") as f:
        rows = list(csv.DictReader(f))
    assert {r["registration"] for r in rows} == {"ecc"}

    heavy = np.asarray(Image.open(out / "heavy" / "a.png"), dtype=np.float32)
    thin = np.asarray(Image.open(out / "thin" / "a.png"), dtype=np.float32)
    away = cv2.dilate(ellipse(BODY_CENTER, (70, 100)).astype(np.uint8), np.ones((41, 41), np.uint8)) == 0
    away &= cv2.dilate(ellipse(HEAD_CENTER, (HEAD_R * 1.6, HEAD_R * 1.6)).astype(np.uint8), np.ones((21, 21), np.uint8)) == 0
    assert np.abs(heavy - thin)[away].mean() < 0.5  # backgrounds are the original's in both

    # re-run with captions: images are reused (the models must not be touched), manifest rewritten
    captions = tmp_path / "captions"
    captions.mkdir()
    (captions / "a.txt").write_text("a woman in a red room\n")

    def boom(_):
        raise AssertionError("models should not run when outputs exist")

    script.main(argv + ["--captions", str(captions)], models=(boom, boom))
    tuples = load_manifest(str(out / "tuples.jsonl"))
    assert [t.caption for t in tuples] == ["a woman in a red room", "a photo of a woman"]

    # JSON captions: --caption-key picks the field; .txt wins over .json
    (captions / "b.json").write_text(json.dumps({"caption_long": "a long caption", "caption_short": "short"}))
    script.main(argv + ["--captions", str(captions), "--caption-key", "caption_short"], models=(boom, boom))
    assert [t.caption for t in load_manifest(str(out / "tuples.jsonl"))] == ["a woman in a red room", "short"]
    with pytest.raises(SystemExit, match="caption_x"):
        script.main(argv + ["--captions", str(captions), "--caption-key", "caption_x"], models=(boom, boom))


def test_member_spec_validation():
    script = load_script()
    with pytest.raises(SystemExit):
        script.parse_args(["r", "o", "--member", "heavy"])
    assert script.parse_args(["r", "o", "--member", "heavy=1.5"]).members == [("heavy", 1.5)]
