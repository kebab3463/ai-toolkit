"""Centered multipliers, weights and filtering for tuple sliders (F2).

Each member i of a tuple has a measured level a_i and a weight w_i. In a step,
over the members sampled for that step,

    s_i = (a_i - a_w) / kappa + offset,     a_w = sum(w_i a_i) / sum(w_i)

so sum(w_i s_i) = 0 exactly when offset is 0: content shared by the members
cancels from the LoRA update to first order, and weight 0 is the base model.
kappa is a global level unit (median or max within-tuple span) so typical
inference weights are near 1. Member weights w_i = quality_i * (1 / level_var_i
if given), normalized to mean 1 within the tuple; the tuple weight is the
tuple's quality. Tuples whose span is below min_span * kappa are dropped, and
tuple weights are capped so the leverage omega * sum(w_i s_i^2) stays at or
below a percentile of all tuples' leverage.
"""

import itertools
import random as _random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from .manifest import SliderTuple

MembersPerStep = Union[int, str]  # an int >= 2, or "all"
LEVEL_NORMS = ("median_span", "max_span", "none")


def validate_members_per_step(value) -> MembersPerStep:
    if value == "all":
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError(f"slider members_per_step must be an integer >= 2 or 'all', got {value!r}")
    return value


def member_weights(tup: SliderTuple) -> List[float]:
    """w_i = quality_i * (1 / level_var_i if given), normalized to mean 1 over the tuple."""
    raw = []
    for m in tup.members:
        w = 1.0 if m.quality is None else m.quality
        if m.level_var is not None:
            w /= m.level_var
        raw.append(w)
    mean = sum(raw) / len(raw)
    return [w / mean for w in raw] if mean > 0 else [0.0] * len(raw)


def tuple_weight(tup: SliderTuple) -> float:
    return 1.0 if tup.quality is None else tup.quality


def level_unit(tuples: Sequence[SliderTuple], mode: str) -> float:
    """kappa over the training set."""
    if mode not in LEVEL_NORMS:
        raise ValueError(f"slider level_norm must be one of {LEVEL_NORMS}, got {mode!r}")
    if mode == "none":
        return 1.0
    spans = [t.span for t in tuples]
    kappa = float(np.median(spans)) if mode == "median_span" else max(spans)
    if kappa <= 0:
        raise ValueError("slider level_norm: the tuples' level spans are all zero")
    return kappa


def centered_scales(levels: Sequence[float], weights: Sequence[float], kappa: float, offset: float = 0.0) -> List[float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("centered_scales: member weights sum to zero")
    mean = sum(w * a for w, a in zip(weights, levels)) / total
    return [(a - mean) / kappa + offset for a in levels]


def weighted_scale_sum(scales: Sequence[float], weights: Sequence[float]) -> float:
    return sum(w * s for w, s in zip(weights, scales))


def leverage(tup: SliderTuple, kappa: float, omega: float = None) -> float:
    """omega * sum(w_i s_i^2) with every member centered (the tuple's pull on the update)."""
    w = member_weights(tup)
    s = centered_scales([m.level for m in tup.members], w, kappa)
    return (tuple_weight(tup) if omega is None else omega) * sum(wi * si * si for wi, si in zip(w, s))


def sample_members(tup: SliderTuple, members_per_step: MembersPerStep, min_span: float, rng=_random) -> List[int]:
    """Member indices for one step, highest level first. With fewer than all
    members, a uniformly random subset among those whose level span is at least
    ``min_span`` (absolute level units)."""
    n = len(tup.members)
    k = n if members_per_step == "all" else min(members_per_step, n)
    if k == n:
        chosen = list(range(n))
    else:
        levels = [m.level for m in tup.members]
        valid = [c for c in itertools.combinations(range(n), k)
                 if max(levels[i] for i in c) - min(levels[i] for i in c) >= min_span]
        if not valid:
            raise ValueError(f"tuple {tup.id!r}: no {k}-member subset spans at least {min_span:g}")
        chosen = list(rng.choice(valid))
    return sorted(chosen, key=lambda i: -tup.members[i].level)


@dataclass
class TuplePlan:
    """What the trainer uses per tuple: weights, tuple weight after capping, rank per member."""
    member_weights: List[float]
    tuple_weight: float
    ranks: Dict[int, int]


@dataclass
class FilterReport:
    kappa: float
    kept: int
    dropped: List[Tuple[str, str]] = field(default_factory=list)  # (id, reason)
    capped: List[Tuple[str, float]] = field(default_factory=list)  # (id, factor applied to omega)

    def lines(self) -> List[str]:
        out = [f"slider_tuple: kappa = {self.kappa:g}; {self.kept} tuples kept, "
               f"{len(self.dropped)} dropped, {len(self.capped)} weight-capped"]
        out += [f"slider_tuple:   dropped {tid}: {why}" for tid, why in self.dropped]
        out += [f"slider_tuple:   capped {tid}: tuple weight x{f:.3f}" for tid, f in self.capped]
        return out


def plan_tuples(
    tuples: Sequence[SliderTuple],
    level_norm: str = "median_span",
    min_span: float = 0.3,
    weight_cap_pct: float = 95.0,
) -> Tuple[Dict[str, TuplePlan], FilterReport]:
    """kappa, span filtering (min_span in kappa units) and leverage capping over the training set."""
    kappa = level_unit(tuples, level_norm)
    report = FilterReport(kappa=kappa, kept=0)
    kept = []
    for t in tuples:
        if t.span < min_span * kappa:
            report.dropped.append((t.id, f"span {t.span:g} < min_span {min_span:g} x kappa {kappa:g}"))
        elif tuple_weight(t) <= 0 or sum(member_weights(t)) <= 0:
            report.dropped.append((t.id, "zero weight"))
        else:
            kept.append(t)
    report.kept = len(kept)
    if not kept:
        raise ValueError("slider_tuple: every tuple was dropped (see the report above)")

    lev = {t.id: leverage(t, kappa) for t in kept}
    cap = float(np.percentile(list(lev.values()), weight_cap_pct)) if weight_cap_pct < 100 else float("inf")
    plans = {}
    for t in kept:
        omega = tuple_weight(t)
        if lev[t.id] > cap:
            factor = cap / lev[t.id]
            report.capped.append((t.id, factor))
            omega *= factor
        plans[t.id] = TuplePlan(
            member_weights=member_weights(t),
            tuple_weight=omega,
            ranks={i: r for r, i in enumerate(t.by_level)},
        )
    return plans, report
