"""Per-image slider scales (F1), for ordinary (non-tuple) datasets and polarity runs.

Tuple datasets (``slider_tuples``) do not use these: they read measured levels
from the manifest and compute centered scales themselves (centering.py).

A dataset opts in with ``slider_scales: true``. Each image's LoRA multiplier
then comes from, in order:

1. a sidecar next to the image, ``<stem>.scale``, holding one float;
2. ``scales.json`` in the dataset folder, ``{"<stem>": float, ...}``;
3. the dataset's ``network_weight`` (the upstream per-dataset value).

A stem defined by both a sidecar and ``scales.json`` must agree. With
``slider_scale_normalize: max_abs`` every scale in the dataset is divided by
the largest |scale| so the trained range is [-1, 1].
"""

import copy
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

SCALE_SIDECAR_EXT = ".scale"
SCALES_JSON_NAME = "scales.json"
NORMALIZE_MODES = ("none", "max_abs")


@dataclass
class SliderDatasetOptions:
    scales: bool = False
    normalize: str = "none"
    tuples: Optional[str] = None  # tuples.jsonl manifest (guidance_type: slider_tuple)

    @classmethod
    def from_raw(cls, raw: dict) -> "SliderDatasetOptions":
        options = cls(
            scales=bool(raw.get("slider_scales", False)),
            normalize=str(raw.get("slider_scale_normalize", "none")),
            tuples=raw.get("slider_tuples"),
        )
        if options.normalize not in NORMALIZE_MODES:
            raise ValueError(
                f"slider_scale_normalize must be one of {NORMALIZE_MODES}, got {options.normalize!r}"
            )
        if options.normalize != "none" and not options.scales:
            raise ValueError("slider_scale_normalize requires slider_scales: true")
        is_tuple_guidance = raw.get("guidance_type") == "slider_tuple"
        if options.tuples is not None and not is_tuple_guidance:
            raise ValueError("slider_tuples requires guidance_type: slider_tuple")
        if is_tuple_guidance and options.tuples is None:
            raise ValueError("guidance_type: slider_tuple requires slider_tuples (a tuples.jsonl manifest)")
        if options.tuples is not None and options.scales:
            raise ValueError("slider_scales does not apply to tuple datasets; they center the manifest's levels")
        return options


@dataclass
class ScaleSummary:
    count: int
    from_sidecar: int
    from_json: int
    from_default: int
    normalized_by: Optional[float]
    min_scale: float
    max_scale: float


def slider_options_by_config(raw_datasets: Sequence[dict], dataset_configs: Sequence) -> Dict[int, SliderDatasetOptions]:
    """Pair raw dataset dicts with the DatasetConfigs built from them.

    BaseSDTrainProcess builds one DatasetConfig per entry of
    preprocess_dataset_raw_config(raw), in order, and DatasetConfig drops keys
    it does not know; this re-splits the raw config the same way and keys the
    parsed options by DatasetConfig identity (what file items reference).
    """
    from toolkit.config_modules import preprocess_dataset_raw_config

    split = preprocess_dataset_raw_config(copy.deepcopy(list(raw_datasets))) if raw_datasets else []
    if len(split) != len(dataset_configs):
        raise RuntimeError(f"slider_tuple: {len(split)} raw dataset configs but {len(dataset_configs)} DatasetConfigs")
    return {id(config): SliderDatasetOptions.from_raw(raw) for raw, config in zip(split, dataset_configs)}


def _parse_scale(value, where: str) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: scale must be a number, got {value!r}") from None
    if not math.isfinite(scale):
        raise ValueError(f"{where}: scale must be finite, got {scale}")
    return scale


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_scale_sidecar(image_path: str) -> Optional[float]:
    path = os.path.splitext(image_path)[0] + SCALE_SIDECAR_EXT
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return _parse_scale(f.read().strip(), path)


def load_scales_json(folder: str) -> Dict[str, float]:
    path = os.path.join(folder, SCALES_JSON_NAME)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object mapping image stems to scales")
    return {str(stem): _parse_scale(value, f"{path}[{stem!r}]") for stem, value in data.items()}


def resolve_scales(image_paths: Sequence[str], folder: Optional[str], default: float):
    """Scales for ``image_paths`` (sidecar > scales.json > default) plus a ScaleSummary."""
    json_scales = load_scales_json(folder) if folder and os.path.isdir(folder) else {}
    if json_scales:
        stems = Counter(_stem(p) for p in set(image_paths))
        duplicated = sorted(s for s, n in stems.items() if n > 1 and s in json_scales)
        if duplicated:
            raise ValueError(
                f"{os.path.join(folder, SCALES_JSON_NAME)}: stems {duplicated} match more than one image; "
                f"use {SCALE_SIDECAR_EXT} sidecars for those images"
            )

    scales: List[float] = []
    sources = Counter()
    for path in image_paths:
        sidecar = read_scale_sidecar(path)
        from_json = json_scales.get(_stem(path))
        if sidecar is not None and from_json is not None and sidecar != from_json:
            raise ValueError(
                f"{path}: {SCALE_SIDECAR_EXT} sidecar says {sidecar} but {SCALES_JSON_NAME} says {from_json}"
            )
        if sidecar is not None:
            scales.append(sidecar)
            sources["sidecar"] += 1
        elif from_json is not None:
            scales.append(from_json)
            sources["json"] += 1
        else:
            scales.append(float(default))
            sources["default"] += 1
    return scales, sources


def normalize_scales(scales: Sequence[float], mode: str):
    """Returns (normalized scales, divisor or None)."""
    if mode == "none":
        return list(scales), None
    if mode == "max_abs":
        divisor = max((abs(s) for s in scales), default=0.0)
        if divisor == 0.0:
            raise ValueError("slider_scale_normalize: max_abs needs at least one nonzero scale")
        return [s / divisor for s in scales], divisor
    raise ValueError(f"unknown slider_scale_normalize mode {mode!r}")


def apply_slider_scales(file_items: Iterable, options: SliderDatasetOptions, dataset_config) -> Optional[ScaleSummary]:
    """Set ``network_weight`` on every file item of one dataset from its scales."""
    if not options.scales:
        return None
    if dataset_config.folder_path is None or not os.path.isdir(dataset_config.folder_path):
        raise ValueError("slider_scales needs a dataset with a folder_path directory")

    file_items = list(file_items)
    scales, sources = resolve_scales(
        [item.path for item in file_items], dataset_config.folder_path, dataset_config.network_weight
    )
    scales, divisor = normalize_scales(scales, options.normalize)
    for item, scale in zip(file_items, scales):
        item.network_weight = scale
    return ScaleSummary(
        count=len(file_items),
        from_sidecar=sources["sidecar"],
        from_json=sources["json"],
        from_default=sources["default"],
        normalized_by=divisor,
        min_scale=min(scales, default=0.0),
        max_scale=max(scales, default=0.0),
    )
