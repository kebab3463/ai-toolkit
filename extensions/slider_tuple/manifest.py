"""Tuple manifests (F2): ``tuples.jsonl``, one tuple per line.

    {"id": "p01", "caption": "photo of a person ...", "quality": 1.0,
     "original": "p01/original.png", "mask": "p01/mask.png",
     "members": [{"image": "p01/real.png", "level": 3.1, "role": "real"},
                 {"image": "p01/edit1.png", "level": 2.4, "role": "edit", "quality": 0.8},
                 {"image": "p01/edit2.png", "level": 1.6, "role": "edit", "level_var": 0.04,
                  "mask": "p01/edit2_mask.png"}]}

``level`` is the member's measured attribute level; only within-tuple level
differences matter (training centers them per step, see centering.py).
``role`` is "real" (the re-textured real photo, normally the highest level) or
"edit" (default); at most one real member per tuple. Optional ``quality`` in
[0, 1] and measurement variance ``level_var`` > 0 set member weights; tuple
``quality`` sets the tuple weight. ``original`` is the untouched real photo,
used for metrics and debug dumps only. A tuple ``mask`` is the loss mask for
every member without its own ``mask``; one mask per tuple is recommended,
because centering only cancels shared content when every member is weighted
identically pixel by pixel (see mask_mismatch). Paths are relative to the manifest's
folder (or absolute); every member of a tuple must have the same pixel size.
Pairs are tuples with two members.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

TUPLE_KEYS = {"id", "caption", "members", "quality", "original", "mask"}
MEMBER_KEYS = {"image", "level", "role", "mask", "quality", "level_var"}
ROLES = ("real", "edit")


@dataclass
class TupleMember:
    image: str
    level: float
    role: str = "edit"
    mask: Optional[str] = None
    quality: Optional[float] = None
    level_var: Optional[float] = None


@dataclass
class SliderTuple:
    id: str
    caption: str
    members: List[TupleMember]
    quality: Optional[float] = None
    original: Optional[str] = None
    mask: Optional[str] = None
    size: tuple = (0, 0)

    @property
    def anchor(self) -> TupleMember:
        # the member the dataloader loads as the file item (crop, flip, cached latent)
        return self.members[0]

    @property
    def span(self) -> float:
        levels = [m.level for m in self.members]
        return max(levels) - min(levels)

    @property
    def mask_mismatch(self) -> bool:
        """True if members are masked differently (different files, or some
        unmasked): shared content then no longer cancels where the masks differ."""
        return len({m.mask for m in self.members}) > 1

    @property
    def by_level(self) -> List[int]:
        """Member indices from highest to lowest level; position = rank (0 = top, normally the real photo)."""
        return sorted(range(len(self.members)), key=lambda i: -self.members[i].level)


class ManifestError(ValueError):
    pass


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{where}: expected a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ManifestError(f"{where}: must be finite, got {value}")
    return value


def _quality(value, where: str) -> Optional[float]:
    if value is None:
        return None
    value = _number(value, where)
    if not 0.0 <= value <= 1.0:
        raise ManifestError(f"{where}: must be in [0, 1], got {value}")
    return value


def _path(value, base: str, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: expected a path string, got {value!r}")
    path = os.path.abspath(value if os.path.isabs(value) else os.path.join(base, value))
    if not os.path.isfile(path):
        raise ManifestError(f"{where}: file not found: {path}")
    return path


def _member(m, base: str, where: str) -> TupleMember:
    if not isinstance(m, dict):
        raise ManifestError(f"{where}: expected an object")
    unknown = set(m) - MEMBER_KEYS
    if unknown:
        raise ManifestError(f"{where}: unknown keys {sorted(unknown)}")
    if "image" not in m or "level" not in m:
        raise ManifestError(f"{where}: needs 'image' and 'level'")
    role = m.get("role", "edit")
    if role not in ROLES:
        raise ManifestError(f"{where}: role must be one of {ROLES}, got {role!r}")
    level_var = m.get("level_var")
    if level_var is not None:
        level_var = _number(level_var, f"{where} level_var")
        if level_var <= 0:
            raise ManifestError(f"{where} level_var: must be > 0, got {level_var}")
    return TupleMember(
        image=_path(m["image"], base, f"{where} image"),
        level=_number(m["level"], f"{where} level"),
        role=role,
        mask=_path(m["mask"], base, f"{where} mask") if m.get("mask") is not None else None,
        quality=_quality(m.get("quality"), f"{where} quality"),
        level_var=level_var,
    )


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

    members = [_member(m, base, f"{where} member {i}") for i, m in enumerate(raw_members)]
    tuple_mask = _path(data["mask"], base, f"{where} mask") if data.get("mask") is not None else None
    for m in members:
        if m.mask is None:
            m.mask = tuple_mask
    if len({m.image for m in members}) != len(members):
        raise ManifestError(f"{where}: the same image is listed twice")
    if sum(m.role == "real" for m in members) > 1:
        raise ManifestError(f"{where}: at most one member can have role 'real'")
    if len({m.level for m in members}) == 1:
        raise ManifestError(f"{where}: all members have the same level (zero span)")

    sizes = {}
    for m in members:
        with Image.open(m.image) as img:
            sizes[m.image] = img.size
    if len(set(sizes.values())) != 1:
        detail = ", ".join(f"{os.path.basename(p)} {w}x{h}" for p, (w, h) in sizes.items())
        raise ManifestError(f"{where}: members must share one pixel size ({detail})")

    return SliderTuple(
        id=tuple_id,
        caption=caption.strip(),
        members=members,
        quality=_quality(data.get("quality"), f"{where} quality"),
        original=_path(data["original"], base, f"{where} original") if data.get("original") is not None else None,
        mask=tuple_mask,
        size=next(iter(sizes.values())),
    )


def rebase_entry(entry: dict, src_dir: str, out_dir: str) -> dict:
    """Copy of a raw manifest entry with every relative path (member image /
    mask, tuple original / mask) rewritten from src_dir to be relative to out_dir."""
    def rebase(path):
        absolute = path if os.path.isabs(path) else os.path.join(src_dir, path)
        return os.path.relpath(absolute, out_dir)

    out = dict(entry, members=[dict(m) for m in entry["members"]])
    for m in out["members"]:
        for key in ("image", "mask"):
            if m.get(key):
                m[key] = rebase(m[key])
    for key in ("original", "mask"):
        if out.get(key):
            out[key] = rebase(out[key])
    return out


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
