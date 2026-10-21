"""Tuple datasets (F2): turn a tuples.jsonl manifest into an upstream dataset.

Each tuple becomes one dataloader item: its anchor (first) member, captioned
from the manifest, via a generated JSON dataset ({path: {"caption": ...}}, the
format AiToolkitDataset reads from ``dataset_path``). Upstream then buckets,
crops, flips and caches latents / text embeddings for the anchor as usual; the
trainer loads the other members crop-aligned at loss time (members.py).

Process-level options live in a ``slider:`` block (centering.py has the math):

    slider:
      members_per_step: all        # or an int >= 2 to save memory (subsets keep span >= min_span)
      level_norm: median_span      # kappa: median_span | max_span | none
      min_span: 0.3                # in kappa units; smaller tuples / subsets are dropped
      weight_cap_pct: 95           # cap each tuple's leverage at this percentile (100 = off)
      backward_per_member: false   # one forward/backward per member (flat memory)
      center_offset: 0.0           # test hook only: breaks the shared-content cancellation
      debug_dump_steps: 0          # dump crop-aligned members for the first N steps
      # anchor preservation (F4, anchors.py); off while anchor_strength is 0
      anchor_strength: 0.0
      anchor_prompts: []           # prompts the slider must leave unchanged
      anchor_prob: 1.0             # chance per step of adding the anchor loss
      anchor_max_multiplier: 1.0   # LoRA multiplier drawn from [-max, max]
      anchor_images: null          # folder of anchor images; default: the batch's latents
      # symmetry loss (symmetry.py); off while symmetry_strength is 0
      symmetry_strength: 0.0       # weight of mean(((v(+s) + v(-s)) / 2 - v(0))^2) on the training batch
      symmetry_prob: 1.0           # chance per step of adding it (4 extra forwards)
      symmetry_max_multiplier: 2.0 # s drawn from [0, max]; cover the strengths you will use
      # Turbo previews (F7, turbo.py): sample with this Turbo DiT checkpoint instead of the Raw model
      turbo_preview: null          # e.g. a ComfyUI int8_convrot .safetensors
      turbo_preview_steps: 8
      turbo_preview_cfg: 1.0       # Krea 2 CFG is 0-normalized: 1 = off
      turbo_preview_mu: 1.15       # fixed schedule shift the distilled Turbo was trained at
      turbo_preview_offload: true  # move the Raw DiT to CPU while sampling
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

from .centering import LEVEL_NORMS, MembersPerStep, validate_members_per_step
from .manifest import SliderTuple, load_manifest

SLIDER_TRAIN_KEYS = {
    "members_per_step", "level_norm", "min_span", "weight_cap_pct", "backward_per_member", "center_offset",
    "debug_dump_steps",
    "anchor_strength", "anchor_prompts", "anchor_prob", "anchor_max_multiplier", "anchor_images",
    "symmetry_strength", "symmetry_prob", "symmetry_max_multiplier",
    "turbo_preview", "turbo_preview_steps", "turbo_preview_cfg", "turbo_preview_mu", "turbo_preview_offload",
}


@dataclass
class SliderTrainConfig:
    members_per_step: MembersPerStep = "all"
    level_norm: str = "median_span"
    min_span: float = 0.3
    weight_cap_pct: float = 95.0
    backward_per_member: bool = False
    center_offset: float = 0.0
    debug_dump_steps: int = 0
    anchor_strength: float = 0.0
    anchor_prompts: List[str] = field(default_factory=list)
    anchor_prob: float = 1.0
    anchor_max_multiplier: float = 1.0
    anchor_images: Optional[str] = None
    symmetry_strength: float = 0.0
    symmetry_prob: float = 1.0
    symmetry_max_multiplier: float = 2.0
    turbo_preview: Optional[str] = None
    turbo_preview_steps: int = 8
    turbo_preview_cfg: float = 1.0
    turbo_preview_mu: float = 1.15
    turbo_preview_offload: bool = True

    @property
    def symmetry_enabled(self) -> bool:
        return self.symmetry_strength > 0

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
            members_per_step=validate_members_per_step(raw.get("members_per_step", "all")),
            level_norm=str(raw.get("level_norm", "median_span")),
            min_span=number("min_span", 0.3),
            weight_cap_pct=number("weight_cap_pct", 95.0),
            backward_per_member=bool(raw.get("backward_per_member", False)),
            center_offset=number("center_offset", 0.0),
            debug_dump_steps=debug,
            anchor_strength=number("anchor_strength", 0.0),
            anchor_prompts=list(raw.get("anchor_prompts") or []),
            anchor_prob=number("anchor_prob", 1.0),
            anchor_max_multiplier=number("anchor_max_multiplier", 1.0),
            anchor_images=raw.get("anchor_images"),
            symmetry_strength=number("symmetry_strength", 0.0),
            symmetry_prob=number("symmetry_prob", 1.0),
            symmetry_max_multiplier=number("symmetry_max_multiplier", 2.0),
            turbo_preview=raw.get("turbo_preview"),
            turbo_preview_steps=int(number("turbo_preview_steps", 8)),
            turbo_preview_cfg=number("turbo_preview_cfg", 1.0),
            turbo_preview_mu=number("turbo_preview_mu", 1.15),
            turbo_preview_offload=bool(raw.get("turbo_preview_offload", True)),
        )
        if config.level_norm not in LEVEL_NORMS:
            raise ValueError(f"slider.level_norm must be one of {LEVEL_NORMS}, got {config.level_norm!r}")
        if config.min_span < 0:
            raise ValueError("slider.min_span must be >= 0")
        if not 0 < config.weight_cap_pct <= 100:
            raise ValueError("slider.weight_cap_pct must be in (0, 100]")
        if config.anchor_strength < 0:
            raise ValueError("slider.anchor_strength must be >= 0")
        if not 0.0 <= config.anchor_prob <= 1.0:
            raise ValueError("slider.anchor_prob must be in [0, 1]")
        if config.symmetry_strength < 0:
            raise ValueError("slider.symmetry_strength must be >= 0")
        if not 0.0 <= config.symmetry_prob <= 1.0:
            raise ValueError("slider.symmetry_prob must be in [0, 1]")
        if config.symmetry_max_multiplier <= 0:
            raise ValueError("slider.symmetry_max_multiplier must be > 0")
        if config.anchor_max_multiplier < 0:
            raise ValueError("slider.anchor_max_multiplier must be >= 0")
        if not all(isinstance(p, str) and p.strip() for p in config.anchor_prompts):
            raise ValueError("slider.anchor_prompts must be a list of non-empty strings")
        if config.turbo_preview is not None and not (
            isinstance(config.turbo_preview, str) and config.turbo_preview.endswith(".safetensors")
        ):
            raise ValueError("slider.turbo_preview must be a .safetensors checkpoint path")
        if config.turbo_preview_steps < 1:
            raise ValueError("slider.turbo_preview_steps must be >= 1")
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


def write_anchor_dataset(tuples: List[SliderTuple], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({t.anchor.image: {"caption": t.caption} for t in tuples}, f, indent=1)
    return path


def load_tuple_dataset(config, manifest_path: str, where: str) -> List[SliderTuple]:
    """Check the dataset options and load its manifest (the trainer filters the
    tuples over the whole training set, then calls write_anchor_dataset)."""
    check_tuple_dataset_config(config, where)
    return load_manifest(manifest_path)
