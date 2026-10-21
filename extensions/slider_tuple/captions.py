"""Caption files next to images: <stem>.txt, or <stem>.json with the caption under a key."""

import json
import os
from typing import Optional

DEFAULT_CAPTION_KEY = "caption_long"


def read_caption(folder: str, stem: str, key: str = DEFAULT_CAPTION_KEY) -> Optional[str]:
    """<folder>/<stem>.txt, else the ``key`` string of <folder>/<stem>.json; None if neither exists
    (or the caption is empty). A .json without a string ``key`` raises ValueError."""
    txt, js = os.path.join(folder, stem + ".txt"), os.path.join(folder, stem + ".json")
    if os.path.exists(txt):
        with open(txt, encoding="utf-8") as f:
            return f.read().strip() or None
    if os.path.exists(js):
        with open(js, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get(key), str):
            fields = sorted(data) if isinstance(data, dict) else type(data).__name__
            raise ValueError(f"{js}: no string field {key!r} (fields: {fields})")
        return data[key].strip() or None
    return None
