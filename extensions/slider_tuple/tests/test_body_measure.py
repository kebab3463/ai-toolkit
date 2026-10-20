"""body_measure.py on synthetic mattes, and the sweep / dataset scripts with stub models."""

import importlib.util
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

from extensions.slider_tuple.body_measure import draw_measurement, measure_body

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keypoints(shoulder_y=40, hip_y=120, cx=100, score=5.0):
    kps = np.zeros((17, 3))
    kps[:, 2] = score
    kps[5], kps[6] = (cx - 20, shoulder_y, score), (cx + 20, shoulder_y, score)
    kps[11], kps[12] = (cx - 15, hip_y, score), (cx + 15, hip_y, score)
    return kps


def body(width, h=240, w=200, cx=100, top=30, bottom=230):
    matte = np.zeros((h, w), np.float32)
    matte[top:bottom, cx - width // 2:cx + width // 2] = 1.0
    return matte


def test_score_is_width_over_torso_length():
    m = measure_body(body(48), keypoints())
    assert m.score == pytest.approx(48 / 80) and m.torso_px == pytest.approx(80)
    assert m.band == (52, 160) and m.widths["waist"] == pytest.approx(0.6)


def test_wider_body_scores_higher_and_scale_does_not_matter():
    assert measure_body(body(64), keypoints()).score > measure_body(body(48), keypoints()).score
    big = np.kron(body(48), np.ones((2, 2), np.float32))  # same body at twice the resolution
    kps = keypoints(); kps[:, :2] *= 2
    assert measure_body(big, kps).score == pytest.approx(measure_body(body(48), keypoints()).score, rel=0.02)


def test_other_people_do_not_count():
    matte = body(48)
    matte[30:230, 160:190] = 1.0  # a second person, not touching
    assert measure_body(matte, keypoints()).score == pytest.approx(48 / 80)


def test_unmeasurable_cases_explain_why():
    assert measure_body(body(48), None).reason == "no person detected"
    hidden = keypoints(); hidden[11, 2] = 0.1
    m = measure_body(body(48), hidden)
    assert m.score is None and "11" in m.reason
    assert "too short" in measure_body(body(48), keypoints(shoulder_y=60, hip_y=65)).reason
    assert draw_measurement(np.zeros((240, 200, 3), np.uint8), m).shape == (240, 200, 3)


# stub "models": the matte is the bright pixels, keypoints fixed
def stub_models():
    return (lambda img: (np.asarray(img.convert("L"), np.float32) > 127).astype(np.float32),
            lambda img: keypoints())


def save_body(path, width):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((body(width) * 255).astype(np.uint8)).convert("RGB").save(path)


def test_measure_tuples_reports_order_and_writes_measured_levels(tmp_path, capsys):
    script = load_script("measure_tuples")
    entries = []
    for i, (heavy, orig, thin) in enumerate([(70, 56, 40), (64, 50, 44), (40, 50, 60)]):  # the last is "wrong"
        tid = f"t{i}"
        save_body(str(tmp_path / "heavy" / f"{tid}.png"), heavy)
        save_body(str(tmp_path / "thin" / f"{tid}.png"), thin)
        save_body(str(tmp_path / "original" / f"{tid}.png"), orig)
        entries.append({"id": tid, "caption": "c", "original": f"original/{tid}.png", "mask": f"masks/{tid}.png",
                        "members": [{"image": f"heavy/{tid}.png", "level": 1.0, "role": "edit"},
                                    {"image": f"thin/{tid}.png", "level": -1.0, "role": "edit"}]})
    manifest = tmp_path / "tuples.jsonl"
    manifest.write_text("".join(json.dumps(e) + "\n" for e in entries))
    out = tmp_path / "measured" / "tuples_measured.jsonl"
    os.makedirs(out.parent)
    script.main([str(manifest), "--out", str(out), "--add-original", "--preview"], models=stub_models())
    printed = capsys.readouterr().out
    assert "measure orders 2/3 tuples" in printed and "original lies between its members in 3/3" in printed
    written = [json.loads(line) for line in out.read_text().splitlines()]
    t0 = written[0]
    assert [m["level"] for m in t0["members"]] == pytest.approx([70 / 80, 40 / 80, 56 / 80])
    assert t0["members"][2]["role"] == "real" and t0["members"][0]["image"] == os.path.join("..", "heavy", "t0.png")
    assert (out.parent / "measurements.csv").exists() and len(os.listdir(out.parent / "measure_preview")) == 3


def test_sweep_builtin_body_measure(tmp_path):
    script = load_script("slider_sweep")
    out = tmp_path / "sweep"
    rows = []
    for m, width in ((0.0, 48), (0.5, 56), (1.0, 64)):
        path = script.image_path(str(out), 0, 1, m)
        save_body(path, width)
        rows.append(script.SweepRow(prompt_idx=0, prompt="p", seed=1, m=m, path=path))
    sys.argv, old = ["x", "--prompts", "p.txt", "--out", str(out), "--measure", "body", "--no-lpips"], sys.argv
    try:
        args = script.parse_args()
    finally:
        sys.argv = old
    lpips_off = None
    script.measure(args, rows, models=(lpips_off, *stub_models()))
    assert [r.measure for r in rows] == pytest.approx([0.6, 0.7, 0.8])
    assert all(r.lpips_bg is None for r in rows)
