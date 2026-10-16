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
      # anchor preservation (F4, anchors.py); off while anchor_strength is 0
      anchor_strength: 0.0
      anchor_prompts: []           # prompts the slider must leave unchanged
      anchor_prob: 1.0             # chance per step of adding the anchor loss
      anchor_max_multiplier: 1.0   # LoRA multiplier drawn from [-max, max]
      anchor_images: null          # folder of anchor images; default: the batch's latents
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .manifest import SliderTuple, load_manifest
from .sampling import MembersPerStep, validate_members_per_step
from .scales import normalize_scales

SLIDER_TRAIN_KEYS = {
    "members_per_step", "force_opposite_signs", "zero_sum_weights", "backward_per_member", "debug_dump_steps",
    "anchor_strength", "anchor_prompts", "anchor_prob", "anchor_max_multiplier", "anchor_images",
}


@dataclass
class SliderTrainConfig:
    members_per_step: MembersPerStep = 2
    force_opposite_signs: bool = False
    zero_sum_weights: bool = False
    backward_per_member: bool = False
    debug_dump_steps: int = 0
    anchor_strength: float = 0.0
    anchor_prompts: List[str] = field(default_factory=list)
    anchor_prob: float = 1.0
    anchor_max_multiplier: float = 1.0
    anchor_images: Optional[str] = None

    @property
    def anchors_enabled(self) -> bool:
        return self.anchor_strength > 0

    @classmethod
    def from_raw(cls, raw: dict) -> "SliderTrainConfig":
        raw = raw or {}
        unknown = set(raw) - SLIDER_TRAIN_KEYS
        if unknown:
            raise ValueError(f"slider: unknown keys {sorted(unknown)} (known: {sorted(SLIDER_TRAIN_KEYS)})")
        debug = raw.get("debug_dump_steps", 0)
        if isinstance(debug, bool) or not isinstance(debug, int) or debug < 0:
            raise ValueError(f"slider.debug_dump_steps must be an integer >= 0, got {debug!r}")
        def number(key, default):
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"slider.{key} must be a number, got {value!r}")
            return float(value)

        config = cls(
            members_per_step=validate_members_per_step(raw.get("members_per_step", 2)),
            force_opposite_signs=bool(raw.get("force_opposite_signs", False)),
            zero_sum_weights=bool(raw.get("zero_sum_weights", False)),
            backward_per_member=bool(raw.get("backward_per_member", False)),
            debug_dump_steps=debug,
            anchor_strength=number("anchor_strength", 0.0),
            anchor_prompts=list(raw.get("anchor_prompts") or []),
            anchor_prob=number("anchor_prob", 1.0),
            anchor_max_multiplier=number("anchor_max_multiplier", 1.0),
            anchor_images=raw.get("anchor_images"),
        )
        if config.anchor_strength < 0:
            raise ValueError("slider.anchor_strength must be >= 0")
        if not 0.0 <= config.anchor_prob <= 1.0:
            raise ValueError("slider.anchor_prob must be in [0, 1]")
        if config.anchor_max_multiplier < 0:
            raise ValueError("slider.anchor_max_multiplier must be >= 0")
        if not all(isinstance(p, str) and p.strip() for p in config.anchor_prompts):
            raise ValueError("slider.anchor_prompts must be a list of non-empty strings")
        if config.anchors_enabled and not config.anchor_prompts:
            raise ValueError("slider.anchor_strength > 0 needs anchor_prompts")
        return config


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
