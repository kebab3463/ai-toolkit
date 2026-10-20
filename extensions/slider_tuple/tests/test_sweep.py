"""F6: sweep statistics, side-effect metrics and outputs."""

import numpy as np
import pytest
from PIL import Image

from extensions.slider_tuple.sweep import (
    DEFAULT_DIAGNOSTIC,
    DEFAULT_MULTIPLIERS,
    SweepRow,
    contact_sheet,
    cosine,
    fit,
    load_callable,
    masked_lpips,
    read_csv,
    summarize,
    write_csv,
)


def rows_for(measure_fn, lpips_fn=lambda m: 0.0, face_fn=lambda m: 1.0, face_lpips_fn=lambda m: 0.0,
             seeds=(1, 2), path="x.png"):
    rows = []
    for seed in seeds:
        for m in DEFAULT_MULTIPLIERS + DEFAULT_DIAGNOSTIC:
            rows.append(SweepRow(prompt_idx=0, prompt="p", seed=seed, m=m, path=path, measure=measure_fn(m, seed),
                                 lpips_bg=lpips_fn(m), face_lpips=face_lpips_fn(m), face_sim=face_fn(m)))
    return rows


def test_default_multipliers():
    assert DEFAULT_MULTIPLIERS == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    assert DEFAULT_DIAGNOSTIC == [-0.5, -1.0]


def test_fit_recovers_a_line_and_monotonicity():
    ms = [0, 0.5, 1.0, 1.5]
    out = fit(ms, [2.0 + 3.0 * m for m in ms])
    assert out["slope"] == pytest.approx(3.0) and out["intercept"] == pytest.approx(2.0)
    assert out["r2"] == pytest.approx(1.0) and out["spearman"] == pytest.approx(1.0)
    curved = fit(ms, [m ** 3 for m in ms])
    assert curved["r2"] < 1.0 and curved["spearman"] == pytest.approx(1.0)  # monotone but not linear


def test_fit_flat_and_degenerate():
    flat = fit([0, 1, 2], [5.0, 5.0, 5.0])
    assert flat["slope"] == pytest.approx(0.0) and flat["r2"] == 1.0 and flat["spearman"] is None
    assert fit([1.0], [2.0])["slope"] is None


# acceptance: an all-zero LoRA renders the same image at every multiplier
def test_zero_lora_sweep_summarizes_to_no_effect():
    s = summarize(rows_for(lambda m, seed: 3.0 + 0.1 * seed))
    assert s.positive["slope"] == pytest.approx(0.0, abs=1e-12)
    assert s.positive["worst_lpips_bg"] == 0.0 and s.positive["min_face_sim"] == 1.0
    assert s.positive["worst_face_lpips"] == 0.0
    assert s.positive["intercept_minus_m0"] == pytest.approx(0.0, abs=1e-12)
    assert s.flags == []


def test_positive_and_diagnostic_ranges_are_separate():
    s = summarize(rows_for(lambda m, seed: 2.0 * m if m >= 0 else 10.0, lpips_fn=lambda m: abs(m) * 0.01))
    assert s.positive["slope"] == pytest.approx(2.0)  # the odd negative values do not enter the fit
    assert s.diagnostic["measure_at_-1"] == 10.0
    assert s.positive["worst_lpips_bg"] == pytest.approx(0.015)
    assert s.diagnostic["worst_lpips_bg"] == pytest.approx(0.01)


def test_flags_side_effects_worse_at_minus_one():
    s = summarize(rows_for(lambda m, seed: m, lpips_fn=lambda m: 0.2 if m == -1 else 0.02,
                           face_fn=lambda m: 0.6 if m == -1 else 0.95), margin=0.05)
    assert len(s.flags) == 2
    assert "background LPIPS at m=-1" in s.flags[0] and "face similarity at m=-1" in s.flags[1]
    assert summarize(rows_for(lambda m, seed: m, lpips_fn=lambda m: 0.04 if m == -1 else 0.02), margin=0.05).flags == []
    assert any(line.startswith("FLAG:") for line in s.lines())


def test_masked_lpips_ignores_both_person_regions():
    dist = np.ones((4, 4))
    dist[:2] = 5.0  # big change only where the people are
    person_a = np.zeros((4, 4)); person_a[0] = 1.0
    person_b = np.zeros((4, 4)); person_b[1] = 1.0  # outline moved down a row
    assert masked_lpips(dist, person_a, person_b) == 1.0
    assert masked_lpips(dist, np.ones((4, 4)), np.zeros((4, 4))) is None


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 2]) == pytest.approx(0.0)
    assert cosine(None, [1]) is None and cosine([0, 0], [1, 1]) is None


def test_csv_roundtrip_and_contact_sheet(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (64, 48), "red").save(img)
    rows = rows_for(lambda m, seed: m, seeds=(7,), path=str(img))
    rows[0].face_sim = None
    path = write_csv(rows, str(tmp_path / "out" / "sweep.csv"))
    back = read_csv(path)
    assert [(r.m, r.measure, r.face_lpips, r.face_sim) for r in back] == [(r.m, r.measure, r.face_lpips, r.face_sim) for r in rows]
    sheet = Image.open(contact_sheet(back, str(tmp_path / "out" / "sheet.png"), thumb=32))
    assert sheet.size == (32 * 9, (32 + 28) * 1)


def test_load_callable():
    assert load_callable("math:sqrt")(9.0) == 3.0
    with pytest.raises(ValueError, match="module:function"):
        load_callable("math.sqrt")
    with pytest.raises(ValueError, match="not a callable"):
        load_callable("math:pi")


def test_head_crop_box_is_square_padded_and_clipped():
    from extensions.slider_tuple.sweep import head_crop_box
    a = np.zeros((100, 120)); a[10:30, 40:60] = 1
    b = np.zeros((100, 120)); b[14:34, 44:64] = 1  # the head moved a little
    x0, y0, x1, y1 = head_crop_box([a, b], pad=0.0)
    assert (x0, y0, x1, y1) == (40, 10, 64, 34) and x1 - x0 == y1 - y0  # union x 40-63, y 10-33
    assert head_crop_box([np.zeros((10, 10))]) is None
    edge = np.zeros((50, 50)); edge[0:10, 0:10] = 1
    assert head_crop_box([edge], pad=1.0)[:2] == (0, 0)


def test_face_lpips_in_summary_and_flags():
    s = summarize(rows_for(lambda m, seed: m, face_lpips_fn=lambda m: 0.3 if m == -1 else 0.02))
    assert s.positive["worst_face_lpips"] == pytest.approx(0.02)
    assert s.diagnostic["face_lpips_at_-1"] == pytest.approx(0.3)
    assert any("face LPIPS at m=-1" in f for f in s.flags)
