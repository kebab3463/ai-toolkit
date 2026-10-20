"""Linearity sweep metrics (F6), independent of the renderer so they test on CPU.

A sweep renders prompts x seeds x multipliers. Every image gets:

- ``measure``: the user's attribute measurement (``--measure module:function``,
  PIL image -> float);
- ``lpips_bg``: LPIPS to the same prompt/seed at m = 0, outside the union of
  both images' person masks (a changing body outline is not a side effect);
- ``face_sim``: cosine similarity of face embeddings to m = 0
  (``--face module:function``, PIL image -> 1-D embedding or None).

The summary fits measure against m over the positive range (slope, R^2,
Spearman rho, intercept vs the m = 0 value) and reports the worst side effects;
the negative diagnostic range is summarized separately, and side effects at
m = -1 that are worse than at m = +1 by more than a margin are flagged (the
direction is learned as a whole, so an unwanted component at -m is there at +m
with the opposite sign).
"""

import csv
import importlib
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import stats

DEFAULT_MULTIPLIERS = [round(0.25 * i, 2) for i in range(7)]  # 0 ... 1.5
DEFAULT_DIAGNOSTIC = [-0.5, -1.0]
CSV_FIELDS = ["prompt_idx", "prompt", "seed", "m", "measure", "lpips_bg", "face_sim", "path"]


@dataclass
class SweepRow:
    prompt_idx: int
    prompt: str
    seed: int
    m: float
    path: str
    measure: Optional[float] = None
    lpips_bg: Optional[float] = None
    face_sim: Optional[float] = None


def load_callable(spec: str) -> Callable:
    """'package.module:function' -> the function (the module is imported normally)."""
    if ":" not in spec:
        raise ValueError(f"expected module:function, got {spec!r}")
    module, name = spec.split(":", 1)
    fn = getattr(importlib.import_module(module), name, None)
    if not callable(fn):
        raise ValueError(f"{spec!r} is not a callable")
    return fn


def cosine(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else None


def masked_lpips(distance_map: np.ndarray, person_a: np.ndarray, person_b: np.ndarray) -> Optional[float]:
    """Mean of an LPIPS spatial distance map outside the union of two person mattes (all HxW, mattes in [0, 1])."""
    background = 1.0 - np.maximum(person_a, person_b)
    total = background.sum()
    return float((distance_map * background).sum() / total) if total > 0 else None


def fit(ms: Sequence[float], values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Least-squares line and Spearman rho of values against multipliers."""
    ms, values = np.asarray(ms, dtype=np.float64), np.asarray(values, dtype=np.float64)
    out = {"n": int(len(ms)), "slope": None, "intercept": None, "r2": None, "spearman": None}
    if len(ms) < 2 or np.ptp(ms) == 0:
        return out
    slope, intercept = np.polyfit(ms, values, 1)
    residual = values - (slope * ms + intercept)
    ss_tot = float(((values - values.mean()) ** 2).sum())
    out.update(slope=float(slope), intercept=float(intercept))
    # a flat response is a perfect (zero-slope) fit, not an undefined one
    out["r2"] = 1.0 - float((residual ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    if np.ptp(values) > 0:
        out["spearman"] = float(stats.spearmanr(ms, values).statistic)
    return out


def _worst(values: Sequence[Optional[float]], worst=max) -> Optional[float]:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return worst(values) if values else None


def _at(rows: Sequence[SweepRow], m: float, key: str) -> Optional[float]:
    values = [getattr(r, key) for r in rows if math.isclose(r.m, m) and getattr(r, key) is not None]
    return float(np.mean(values)) if values else None


@dataclass
class SweepSummary:
    positive: Dict[str, Optional[float]] = field(default_factory=dict)
    diagnostic: Dict[str, Optional[float]] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def lines(self) -> List[str]:
        def fmt(v):
            return "n/a" if v is None else f"{v:.4g}"
        out = ["positive range: " + ", ".join(f"{k} {fmt(v)}" for k, v in self.positive.items())]
        out.append("negative diagnostic: " + ", ".join(f"{k} {fmt(v)}" for k, v in self.diagnostic.items()))
        out += [f"FLAG: {f}" for f in self.flags] or ["no flags"]
        return out


def summarize(rows: Sequence[SweepRow], margin: float = 0.05) -> SweepSummary:
    pos = [r for r in rows if r.m >= 0]
    neg = [r for r in rows if r.m < 0]
    s = SweepSummary()

    measured = [r for r in pos if r.measure is not None]
    line = fit([r.m for r in measured], [r.measure for r in measured]) if measured else fit([], [])
    s.positive = {
        "slope": line["slope"], "r2": line["r2"], "spearman": line["spearman"],
        "intercept": line["intercept"], "measure_at_0": _at(pos, 0.0, "measure"),
        "worst_lpips_bg": _worst([r.lpips_bg for r in pos]),
        "min_face_sim": _worst([r.face_sim for r in pos], worst=min),
    }
    if s.positive["intercept"] is not None and s.positive["measure_at_0"] is not None:
        s.positive["intercept_minus_m0"] = s.positive["intercept"] - s.positive["measure_at_0"]

    s.diagnostic = {
        "measure_at_-1": _at(neg, -1.0, "measure"),
        "worst_lpips_bg": _worst([r.lpips_bg for r in neg]),
        "min_face_sim": _worst([r.face_sim for r in neg], worst=min),
        "lpips_bg_at_-1": _at(neg, -1.0, "lpips_bg"), "lpips_bg_at_+1": _at(pos, 1.0, "lpips_bg"),
        "face_sim_at_-1": _at(neg, -1.0, "face_sim"), "face_sim_at_+1": _at(pos, 1.0, "face_sim"),
    }
    d = s.diagnostic
    if d["lpips_bg_at_-1"] is not None and d["lpips_bg_at_+1"] is not None and d["lpips_bg_at_-1"] > d["lpips_bg_at_+1"] + margin:
        s.flags.append(f"background LPIPS at m=-1 ({d['lpips_bg_at_-1']:.4g}) exceeds m=+1 ({d['lpips_bg_at_+1']:.4g}) by more than {margin}")
    if d["face_sim_at_-1"] is not None and d["face_sim_at_+1"] is not None and d["face_sim_at_-1"] < d["face_sim_at_+1"] - margin:
        s.flags.append(f"face similarity at m=-1 ({d['face_sim_at_-1']:.4g}) is below m=+1 ({d['face_sim_at_+1']:.4g}) by more than {margin}")
    return s


def write_csv(rows: Sequence[SweepRow], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: asdict(r)[k] for k in CSV_FIELDS})
    return path


def read_csv(path: str) -> List[SweepRow]:
    def num(v):
        return None if v in ("", "None", None) else float(v)
    with open(path, newline="", encoding="utf-8") as f:
        return [
            SweepRow(prompt_idx=int(d["prompt_idx"]), prompt=d["prompt"], seed=int(d["seed"]), m=float(d["m"]),
                     path=d["path"], measure=num(d["measure"]), lpips_bg=num(d["lpips_bg"]), face_sim=num(d["face_sim"]))
            for d in csv.DictReader(f)
        ]


def contact_sheet(rows: Sequence[SweepRow], out_path: str, thumb: int = 256) -> str:
    """One prompt: a row per seed, a column per multiplier, labeled with m and the metrics."""
    seeds = sorted({r.seed for r in rows})
    ms = sorted({r.m for r in rows})
    sheet = Image.new("RGB", (thumb * len(ms), (thumb + 28) * len(seeds)), "black")
    draw = ImageDraw.Draw(sheet)
    for r in rows:
        x, y = ms.index(r.m) * thumb, seeds.index(r.seed) * (thumb + 28)
        with Image.open(r.path) as img:
            img = img.convert("RGB")
            img.thumbnail((thumb, thumb))
            sheet.paste(img, (x + (thumb - img.width) // 2, y + 28))
        parts = [f"m={r.m:+g}"]
        if r.measure is not None:
            parts.append(f"v={r.measure:.3g}")
        if r.lpips_bg is not None:
            parts.append(f"bg={r.lpips_bg:.3f}")
        if r.face_sim is not None:
            parts.append(f"face={r.face_sim:.3f}")
        draw.text((x + 4, y + 2), " ".join(parts[:2]), fill="yellow")
        draw.text((x + 4, y + 14), " ".join(parts[2:]) or f"seed {r.seed}", fill="yellow")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sheet.save(out_path)
    return out_path
