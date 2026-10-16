"""Which tuple members train in a step, and with what loss weight (F2)."""

import random as _random
from typing import List, Optional, Sequence, Union

from .manifest import SliderTuple

MembersPerStep = Union[int, str]  # an int >= 2, or "all"


def validate_members_per_step(value) -> MembersPerStep:
    if value == "all":
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"slider members_per_step must be an integer >= 2 or 'all', got {value!r}")
    return value


def sample_members(
    tup: SliderTuple,
    members_per_step: MembersPerStep = 2,
    force_opposite_signs: bool = False,
    rng=_random,
) -> List[int]:
    """Indices of trainable (nonzero-scale) members for one step, ordered by
    descending scale so a +/- pair matches polarity's [positive, negative] order."""
    pool = list(tup.trainable)
    k = len(pool) if members_per_step == "all" else min(members_per_step, len(pool))
    if k == len(pool):
        chosen = pool
    elif force_opposite_signs:
        pos = [i for i in pool if tup.members[i].scale > 0]
        neg = [i for i in pool if tup.members[i].scale < 0]
        if pos and neg:
            chosen = [rng.choice(pos), rng.choice(neg)]
            rest = [i for i in pool if i not in chosen]
            chosen += rng.sample(rest, k - 2)
        else:
            chosen = rng.sample(pool, k)
    else:
        chosen = rng.sample(pool, k)
    return sorted(chosen, key=lambda i: tup.members[i].scale, reverse=True)


def member_weights(scales: Sequence[float], explicit: Sequence[Optional[float]], zero_sum: bool) -> List[float]:
    """Loss weight per sampled member.

    Explicit manifest weights win (missing ones count as 1). Otherwise, with
    zero_sum and both signs present, each sign side gets one weight inversely
    proportional to its total |scale| (the smaller side is scaled up), so
    sum(w_i * s_i) == 0, and the weights are rescaled to mean 1. Otherwise 1.
    """
    if any(w is not None for w in explicit):
        return [1.0 if w is None else float(w) for w in explicit]
    n = len(scales)
    pos_total = sum(s for s in scales if s > 0)
    neg_total = -sum(s for s in scales if s < 0)
    if not zero_sum or pos_total == 0 or neg_total == 0:
        return [1.0] * n
    raw = [1.0 / pos_total if s > 0 else 1.0 / neg_total for s in scales]
    mean = sum(raw) / n
    return [w / mean for w in raw]


def weighted_scale_sum(scales: Sequence[float], weights: Sequence[float]) -> float:
    return sum(w * s for w, s in zip(weights, scales))
