"""Convert a polarity dataset (folder_path + unconditional_path) to tuples.jsonl.

Each image in the positive folder with a same-stem partner in the negative
folder becomes a two-member tuple: the positive image first (the anchor) at
level +scale, the partner at level -scale, captioned from the positive image's
caption file. With ``slider.level_norm: none`` the centered multipliers are
+-scale, i.e. polarity at ``network_weight: scale``.
Tuples are written in the order upstream's dataset scan visits the positive
folder (os.walk), so a slider_tuple run iterates the pairs like the polarity run.

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/pairs_to_manifest.py \\
        datasets/slider_smoke/after datasets/slider_smoke/before datasets/slider_smoke/tuples.jsonl
"""

import argparse
import json
import os

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("positive_dir", help="polarity folder_path (trained at +scale)")
    p.add_argument("negative_dir", help="polarity unconditional_path (trained at -scale)")
    p.add_argument("out", help="tuples.jsonl to write")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--caption-ext", default=".txt")
    return p.parse_args()


def find_partner(negative_dir: str, stem: str):
    for ext in IMAGE_EXTS:
        path = os.path.join(negative_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def main():
    args = parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    lines, skipped = [], []
    for root, dirs, files in os.walk(args.positive_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith(".") or not name.lower().endswith(IMAGE_EXTS):
                continue
            positive = os.path.join(root, name)
            stem = os.path.splitext(name)[0]
            negative = find_partner(args.negative_dir, stem)
            caption_path = os.path.splitext(positive)[0] + args.caption_ext
            if negative is None or not os.path.exists(caption_path):
                skipped.append(name)
                continue
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
            lines.append(json.dumps({
                "id": stem,
                "caption": caption,
                "members": [
                    {"image": os.path.relpath(os.path.abspath(positive), out_dir), "level": args.scale},
                    {"image": os.path.relpath(os.path.abspath(negative), out_dir), "level": -args.scale},
                ],
            }))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} tuples to {args.out}" + (f"; skipped (no partner or caption): {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
