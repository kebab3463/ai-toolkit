"""Tuple manifests (F2): ``tuples.jsonl``, one tuple per line.

    {"id": "p01", "caption": "photo of a person ...",
     "members": [{"image": "p01/m2.png", "scale": -1.8},
                 {"image": "p01/ref.png", "scale": 0},
                 {"image": "p01/p1.png", "scale": 1.4, "weight": 1.0, "mask": "p01/p1_mask.png"}]}

Image and mask paths are relative to the manifest's folder (or absolute).
Every member of a tuple must have the same pixel size. A tuple needs at least
two members with a nonzero scale; scale-0 members (the unedited reference)
are loaded for metrics and debug dumps only and never trained.
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

TUPLE_KEYS = {"id", "caption", "members"}
MEMBER_KEYS = {"image", "scale", "weight", "mask"}


@dataclass
class TupleMember:
    image: str
    scale: float
    weight: Optional[float] = None
    mask: Optional[str] = None


@dataclass
class SliderTuple:
    id: str
    caption: str
    members: List[TupleMember]
    size: tuple = field(default=(0, 0))

    @property
    def anchor(self) -> TupleMember:
        # the member the dataloader loads as the file item (crop, flip, cached latent)
        return self.members[0]

    @property
    def trainable(self) -> List[int]:
        return [i for i, m in enumerate(self.members) if m.scale != 0]


class ManifestError(ValueError):
    pass


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{where}: expected a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ManifestError(f"{where}: must be finite, got {value}")
    return value


def _path(value, base: str, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: expected a path string, got {value!r}")
    path = value if os.path.isabs(value) else os.path.join(base, value)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ManifestError(f"{where}: file not found: {path}")
    return path


def parse_tuple(data, base: str, where: str) -> SliderTuple:
    if not isinstance(data, dict):
        raise ManifestError(f"{where}: expected an object")
    unknown = set(data) - TUPLE_KEYS
    if unknown:
        raise ManifestError(f"{where}: unknown keys {sorted(unknown)}")
    for key in ("id", "caption", "members"):
        if key not in data:
            raise ManifestError(f"{where}: missing {key!r}")
    tuple_id = data["id"]
    if not isinstance(tuple_id, str) or not tuple_id:
        raise ManifestError(f"{where}: 'id' must be a non-empty string")
    where = f"{where} (id {tuple_id!r})"
    caption = data["caption"]
    if not isinstance(caption, str) or not caption.strip():
        raise ManifestError(f"{where}: 'caption' must be a non-empty string")
    raw_members = data["members"]
    if not isinstance(raw_members, list) or len(raw_members) < 2:
        raise ManifestError(f"{where}: 'members' must list at least two members")

    members = []
    for i, m in enumerate(raw_members):
        mw = f"{where} member {i}"
        if not isinstance(m, dict):
            raise ManifestError(f"{mw}: expected an object")
        unknown = set(m) - MEMBER_KEYS
        if unknown:
            raise ManifestError(f"{mw}: unknown keys {sorted(unknown)}")
        if "image" not in m or "scale" not in m:
            raise ManifestError(f"{mw}: needs 'image' and 'scale'")
        weight = m.get("weight")
        if weight is not None:
            weight = _number(weight, f"{mw} weight")
            if weight < 0:
                raise ManifestError(f"{mw} weight: must be >= 0, got {weight}")
        members.append(TupleMember(
            image=_path(m["image"], base, f"{mw} image"),
            scale=_number(m["scale"], f"{mw} scale"),
            weight=weight,
            mask=_path(m["mask"], base, f"{mw} mask") if m.get("mask") is not None else None,
        ))

    if len({m.image for m in members}) != len(members):
        raise ManifestError(f"{where}: the same image is listed twice")
    if sum(m.scale != 0 for m in members) < 2:
        raise ManifestError(f"{where}: needs at least two members with a nonzero scale")

    sizes = {}
    for m in members:
        with Image.open(m.image) as img:
            sizes[m.image] = img.size
    if len(set(sizes.values())) != 1:
        detail = ", ".join(f"{os.path.basename(p)} {w}x{h}" for p, (w, h) in sizes.items())
        raise ManifestError(f"{where}: members must share one pixel size ({detail})")

    return SliderTuple(id=tuple_id, caption=caption.strip(), members=members, size=next(iter(sizes.values())))


def load_manifest(path: str) -> List[SliderTuple]:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ManifestError(f"tuple manifest not found: {path}")
    base = os.path.dirname(path)
    tuples, ids = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{line_no}"
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ManifestError(f"{where}: invalid JSON ({e.msg})") from None
            t = parse_tuple(data, base, where)
            if t.id in ids:
                raise ManifestError(f"{where}: duplicate id {t.id!r}")
            ids.add(t.id)
            tuples.append(t)
    if not tuples:
        raise ManifestError(f"{path}: no tuples")
    anchors = [t.anchor.image for t in tuples]
    if len(set(anchors)) != len(anchors):
        raise ManifestError(f"{path}: two tuples use the same first (anchor) image")
    return tuples
