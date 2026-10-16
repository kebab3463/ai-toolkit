"""M4 / F2: member sampling and zero-sum loss weights."""

import random

import pytest

from extensions.slider_tuple.manifest import SliderTuple, TupleMember
from extensions.slider_tuple.sampling import (
    member_weights,
    sample_members,
    validate_members_per_step,
    weighted_scale_sum,
)


def tup(*scales):
    return SliderTuple(id="t", caption="c", members=[TupleMember(image=f"{i}.png", scale=s) for i, s in enumerate(scales)])


def test_samples_only_nonzero_members_in_descending_scale():
    t = tup(-2.0, -1.0, 0.0, 1.0, 2.0)
    rng = random.Random(0)
    for _ in range(200):
        chosen = sample_members(t, 3, rng=rng)
        assert len(chosen) == 3 and 2 not in chosen
        scales = [t.members[i].scale for i in chosen]
        assert scales == sorted(scales, reverse=True)


def test_pair_tuple_trains_both_members_positive_first():
    t = tup(-1.0, 1.0)
    assert sample_members(t, 2) == [1, 0]
    assert sample_members(t, "all") == [1, 0]
    assert sample_members(t, 5) == [1, 0]  # capped at the trainable count


def test_force_opposite_signs_always_mixes_signs():
    t = tup(-1.0, 0.5, 1.0, 1.5, 2.0)
    rng = random.Random(1)
    for _ in range(300):
        scales = [t.members[i].scale for i in sample_members(t, 2, force_opposite_signs=True, rng=rng)]
        assert scales[0] > 0 > scales[-1]
    # without forcing, same-sign pairs do occur
    rng = random.Random(1)
    same = sum(
        min(t.members[i].scale for i in c) > 0
        for c in (sample_members(t, 2, rng=rng) for _ in range(300))
    )
    assert same > 0


def test_force_opposite_signs_falls_back_when_one_sided():
    t = tup(0.5, 1.0, 2.0)
    assert len(sample_members(t, 2, force_opposite_signs=True, rng=random.Random(0))) == 2


@pytest.mark.parametrize("value", [1, 0, -1, 2.5, "two", True])
def test_members_per_step_validation(value):
    with pytest.raises(ValueError):
        validate_members_per_step(value)
    assert validate_members_per_step(3) == 3 and validate_members_per_step("all") == "all"


@pytest.mark.parametrize(
    "scales",
    [(-0.6, 1.4), (1.4, -0.6), (-2.0, -0.5, 1.0), (-1.0, 0.3, 0.6, 2.0), (-1.0, 1.0)],
)
def test_zero_sum_weights_cancel_weighted_scale_sum(scales):
    w = member_weights(scales, [None] * len(scales), zero_sum=True)
    assert weighted_scale_sum(scales, w) == pytest.approx(0.0, abs=1e-12)
    assert sum(w) / len(w) == pytest.approx(1.0)
    assert all(x > 0 for x in w)


def test_zero_sum_scales_the_smaller_side_up():
    w_neg, w_pos = member_weights([-0.6, 1.4], [None, None], zero_sum=True)
    assert (w_neg, w_pos) == pytest.approx((1.4, 0.6))
    # symmetric tuples keep unit weights
    assert member_weights([-1.0, 1.0], [None, None], zero_sum=True) == pytest.approx([1.0, 1.0])


def test_weights_without_zero_sum_or_with_one_sign_are_one():
    assert member_weights([-0.6, 1.4], [None, None], zero_sum=False) == [1.0, 1.0]
    assert member_weights([0.5, 1.4], [None, None], zero_sum=True) == [1.0, 1.0]


def test_explicit_manifest_weights_win():
    assert member_weights([-0.6, 1.4], [2.0, None], zero_sum=True) == [2.0, 1.0]
