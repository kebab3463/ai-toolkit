"""F2: centered multipliers, weights, subset sampling, filtering and capping."""

import random

import numpy as np
import pytest

from extensions.slider_tuple.centering import (
    centered_scales,
    leverage,
    level_unit,
    member_weights,
    plan_tuples,
    sample_members,
    validate_members_per_step,
    weighted_scale_sum,
)
from extensions.slider_tuple.manifest import SliderTuple, TupleMember


def tup(levels, tid="t", qualities=None, level_vars=None, quality=None):
    qualities = qualities or [None] * len(levels)
    level_vars = level_vars or [None] * len(levels)
    members = [TupleMember(image=f"{tid}_{i}.png", level=a, quality=q, level_var=v)
               for i, (a, q, v) in enumerate(zip(levels, qualities, level_vars))]
    return SliderTuple(id=tid, caption="c", members=members, quality=quality)


# acceptance (3): asymmetric levels and weights match the closed form and sum to zero when weighted
def test_closed_form_for_asymmetric_levels_and_weights():
    levels, weights, kappa = [1.4, 0.9, -0.2], [1.0, 0.5, 1.0], 0.8
    mean = (1.0 * 1.4 + 0.5 * 0.9 + 1.0 * -0.2) / 2.5  # 0.66
    expected = [(a - mean) / kappa for a in levels]
    scales = centered_scales(levels, weights, kappa)
    assert scales == pytest.approx(expected)
    assert scales == pytest.approx([0.925, 0.3, -1.075])
    assert weighted_scale_sum(scales, weights) == pytest.approx(0.0, abs=1e-12)


def test_member_weights_from_quality_and_variance():
    t = tup([1.4, 0.9, -0.2], qualities=[1.0, 0.5, 1.0])
    w = member_weights(t)
    assert w == pytest.approx([1.2, 0.6, 1.2])  # (1, .5, 1) normalized to mean 1
    # centering does not depend on the overall weight scale
    assert centered_scales([1.4, 0.9, -0.2], w, 1.0) == pytest.approx(centered_scales([1.4, 0.9, -0.2], [1, 0.5, 1], 1.0))
    t = tup([2.0, 1.0], qualities=[0.8, None], level_vars=[0.5, 0.25])
    assert member_weights(t) == pytest.approx([1.6 / 2.8, 4.0 / 2.8])  # (0.8 / 0.5, 1 / 0.25) over their mean 2.8


@pytest.mark.parametrize("subset", [[0, 1], [0, 2], [1, 2], [0, 1, 2, 3]])
def test_centering_over_any_sampled_subset_sums_to_zero(subset):
    levels, weights = [3.1, 2.4, 1.6, 0.2], [1.0, 0.3, 2.0, 0.7]
    s = centered_scales([levels[i] for i in subset], [weights[i] for i in subset], kappa=1.3)
    assert weighted_scale_sum(s, [weights[i] for i in subset]) == pytest.approx(0.0, abs=1e-12)


def test_pair_gets_plus_minus_half_span_over_kappa():
    # a pair is the k = 2 case: multipliers +-delta / (2 kappa)
    assert centered_scales([2.5, 1.0], [1.0, 1.0], kappa=1.5) == pytest.approx([0.5, -0.5])
    assert centered_scales([2.0, 0.0], [1.0, 1.0], kappa=1.0) == pytest.approx([1.0, -1.0])


def test_offset_shifts_every_scale():
    s = centered_scales([2.0, 0.0], [1.0, 1.0], kappa=1.0, offset=1.0)
    assert s == pytest.approx([2.0, 0.0])
    assert weighted_scale_sum(s, [1.0, 1.0]) == pytest.approx(2.0)


def test_level_unit_modes():
    ts = [tup([3, 1]), tup([5, 4, 1]), tup([1, 0.5])]  # spans 2, 4, 0.5
    assert level_unit(ts, "median_span") == 2.0
    assert level_unit(ts, "max_span") == 4.0
    assert level_unit(ts, "none") == 1.0
    with pytest.raises(ValueError, match="level_norm"):
        level_unit(ts, "mean_span")


def test_sampled_subsets_respect_min_span():
    t = tup([3.0, 2.9, 2.0, 1.0])
    rng = random.Random(0)
    seen = set()
    for _ in range(300):
        picks = sample_members(t, 2, min_span=1.0, rng=rng)
        levels = [t.members[i].level for i in picks]
        assert levels == sorted(levels, reverse=True) and levels[0] - levels[1] >= 1.0
        seen.add(tuple(picks))
    # every pair with span >= 1 occurs; (0, 1) and (1, 2) span 0.1 and 0.9 and never do
    assert seen == {(0, 2), (0, 3), (1, 3), (2, 3)}
    assert sample_members(t, "all", min_span=10.0) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="no 2-member subset"):
        sample_members(t, 2, min_span=2.5)


@pytest.mark.parametrize("value", [1, 0, 2.5, "two", True])
def test_members_per_step_validation(value):
    with pytest.raises(ValueError):
        validate_members_per_step(value)
    assert validate_members_per_step(3) == 3 and validate_members_per_step("all") == "all"


def test_leverage_is_omega_times_weighted_squared_scales():
    t = tup([2.0, 0.0], quality=0.5)
    assert leverage(t, kappa=1.0) == pytest.approx(0.5 * (1 + 1))


# acceptance (5): dropped and capped tuples are reported at startup
def test_plan_drops_small_spans_and_caps_heavy_tuples():
    tuples = [tup([1.0, 0.0], f"t{i}") for i in range(9)]
    tuples.append(tup([4.0, 0.0], "big"))           # 4x span -> 16x leverage
    tuples.append(tup([1.0, 0.9], "flat"))          # span 0.1 < 0.3 kappa
    tuples.append(tup([1.0, 0.0], "zero", quality=0.0))
    plans, report = plan_tuples(tuples, level_norm="median_span", min_span=0.3, weight_cap_pct=90)
    assert report.kappa == 1.0
    assert [tid for tid, _ in report.dropped] == ["flat", "zero"]
    assert [tid for tid, _ in report.capped] == ["big"]
    assert report.kept == 10 and "big" in plans and "flat" not in plans
    # the capped tuple's leverage lands exactly on the 90th percentile of the kept tuples'
    cap = float(np.percentile([leverage(t, 1.0) for t in tuples[:10]], 90))
    assert leverage(tuples[9], 1.0, omega=plans["big"].tuple_weight) == pytest.approx(cap)
    lines = report.lines()
    assert lines[0].startswith("slider_tuple: kappa = 1; 10 tuples kept, 2 dropped, 1 weight-capped")
    assert any("dropped flat" in line for line in lines) and any("capped big" in line for line in lines)
    assert plans["t0"].ranks == {0: 0, 1: 1}


def test_plan_without_cap_and_all_dropped():
    tuples = [tup([1.0, 0.0], "a"), tup([4.0, 0.0], "b")]
    plans, report = plan_tuples(tuples, level_norm="none", min_span=0.3, weight_cap_pct=100)
    assert report.capped == [] and plans["b"].tuple_weight == 1.0
    with pytest.raises(ValueError, match="every tuple was dropped"):
        plan_tuples(tuples, level_norm="none", min_span=10.0)
