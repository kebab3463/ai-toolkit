"""Tuple datasets (F2): turn a tuples.jsonl manifest into an upstream dataset.

Each tuple becomes one dataloader item: its anchor (first) member, captioned
from the manifest, via a generated JSON dataset ({path: {"caption": ...}}, the
format AiToolkitDataset reads from ``dataset_path``). Upstream then buckets,
crops, flips and caches latents / text embeddings for the anchor as usual; the
trainer loads the other members crop-aligned at loss time (members.py).

Process-level options live in a ``slider:`` block:

    slider:
      members_per_step: 2          # int >= 2, or "all"
      force_opposite_signs: false  # always pair one positive with one negative
      zero_sum_weights: false      # weights with sum(w_i * s_i) = 0 per step
      backward_per_member: false   # one forward/backward per member (flat memory)
      debug_dump_steps: 0          # dump crop-aligned members for the first N steps
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List

from .manifest import SliderTuple, load_manifest
from .sampling import MembersPerStep, validate_members_per_step
from .scales import normalize_scales

SLIDER_TRAIN_KEYS = {"members_per_step", "force_opposite_signs", "zero_sum_weights", "backward_per_member", "debug_dump_steps"}


@dataclass
class SliderTrainConfig:
    members_per_step: MembersPerStep = 2
    force_opposite_signs: bool = False
    zero_sum_weights: bool = False
    backward_per_member: bool = False
    debug_dump_steps: int = 0

    @classmethod
    def from_raw(cls, raw: dict) -> "SliderTrainConfig":
        raw = raw or {}
        unknown = set(raw) - SLIDER_TRAIN_KEYS
        if unknown:
            raise ValueError(f"slider: unknown keys {sorted(unknown)} (known: {sorted(SLIDER_TRAIN_KEYS)})")
        debug = raw.get("debug_dump_steps", 0)
        if isinstance(debug, bool) or not isinstance(debug, int) or debug < 0:
            raise ValueError(f"slider.debug_dump_steps must be an integer >= 0, got {debug!r}")
        return cls(
            members_per_step=validate_members_per_step(raw.get("members_per_step", 2)),
            force_opposite_signs=bool(raw.get("force_opposite_signs", False)),
            zero_sum_weights=bool(raw.get("zero_sum_weights", False)),
            backward_per_member=bool(raw.get("backward_per_member", False)),
            debug_dump_steps=debug,
        )


def check_tuple_dataset_config(config, where: str):
    """Reject dataset options that would change one member's pixels without the others."""
    problems = []
    if config.folder_path is not None or config.dataset_path is not None:
        problems.append("set only slider_tuples (no folder_path / dataset_path); images come from the manifest")
    if not config.buckets:
        problems.append("buckets must be enabled")
    if config.augments or config.augmentations:
        problems.append("augments / augmentations are not supported (members would differ)")
    if config.alpha_mask:
        problems.append("alpha_mask is not supported")
    if getattr(config, "num_frames", 1) > 1:
        problems.append("video datasets are not supported")
    if config.is_reg:
        problems.append("regularization (is_reg) datasets cannot be tuple datasets")
    if problems:
        raise ValueError(f"{where}: " + "; ".join(problems))


def normalize_tuple_scales(tuples: List[SliderTuple], mode: str) -> float:
    """Apply slider_scale_normalize across every member of a dataset in place."""
    flat = [m.scale for t in tuples for m in t.members]
    normalized, divisor = normalize_scales(flat, mode)
    it = iter(normalized)
    for t in tuples:
        for m in t.members:
            m.scale = next(it)
    return divisor


def write_anchor_dataset(tuples: List[SliderTuple], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({t.anchor.image: {"caption": t.caption} for t in tuples}, f, indent=1)
    return path


def prepare_tuple_dataset(config, manifest_path: str, normalize: str, out_path: str, where: str) -> Dict[str, SliderTuple]:
    """Load the manifest, point ``config`` at a generated anchor dataset, and
    return {anchor image path: SliderTuple}."""
    check_tuple_dataset_config(config, where)
    tuples = load_manifest(manifest_path)
    if normalize != "none":
        normalize_tuple_scales(tuples, normalize)
    config.dataset_path = write_anchor_dataset(tuples, out_path)
    return {t.anchor.image: t for t in tuples}
