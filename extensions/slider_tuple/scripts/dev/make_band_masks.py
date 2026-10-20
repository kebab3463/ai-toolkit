"""Add placeholder loss masks to a tuples.jsonl for smoke tests.

Writes one grayscale mask per member: 0 over the top ``--head-fraction`` of the
frame (roughly where heads are in full-body shots) and ``--body`` below it. This
exercises the masked-loss path only; real training needs real body / head /
background masks (see NOTES.md, M5).

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/dev/make_band_masks.py \\
        datasets/slider_smoke/tuples.jsonl datasets/slider_smoke/tuples_masked.jsonl
"""

import argparse
import json
import os

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--head-fraction", type=float, default=0.25)
    p.add_argument("--body", type=float, default=1.0)
    p.add_argument("--mask-dir", default="masks", help="folder for masks, relative to the output manifest")
    return p.parse_args()


def main():
    args = parse_args()
    src_dir = os.path.dirname(os.path.abspath(args.manifest))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    mask_dir = os.path.join(out_dir, args.mask_dir)
    os.makedirs(mask_dir, exist_ok=True)
    lines = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            for i, member in enumerate(t["members"]):
                image = member["image"] if os.path.isabs(member["image"]) else os.path.join(src_dir, member["image"])
                with Image.open(image) as img:
                    w, h = img.size
                band = round(h * args.head_fraction)
                mask = Image.new("L", (w, h), round(255 * args.body))
                mask.paste(0, (0, 0, w, band))
                mask_path = os.path.join(mask_dir, f"{t['id']}_{i}.png")
                mask.save(mask_path)
                member["image"] = os.path.relpath(image, out_dir)
                member["mask"] = os.path.relpath(mask_path, out_dir)
            lines.append(json.dumps(t))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} tuples with band masks to {args.out}")


if __name__ == "__main__":
    main()
