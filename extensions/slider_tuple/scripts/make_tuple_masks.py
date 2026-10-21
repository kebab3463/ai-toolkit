"""Generate one loss mask per tuple and write a manifest that uses them.

Per tuple: body = union of every member's Sapiens2 person matte (soft edges
kept); head = union of keypoint-estimated head ellipses (torchvision Keypoint
R-CNN, COCO keypoints) or, with --head-masks, your own head mask
<head-masks>/<tuple id>.png (white = head), e.g. the paste-back region from the
editing pipeline. mask = background + (body - background) * body_union, head 0.
The output manifest gets a tuple-level "mask" and drops per-member masks
(see NOTES.md, "One mask per tuple"). Tuples where no head is found are
reported and left unmasked. --preview writes overlays to check by eye.

First use downloads facebook/sapiens2-matting-1b (into models/sapiens2) and the
torchvision Keypoint R-CNN weights (~230 MB, torch hub cache).

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/make_tuple_masks.py \\
        data/tuples.jsonl data/tuples_masked.jsonl --preview
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from extensions.slider_tuple.manifest import load_manifest, rebase_entry  # noqa: E402
from extensions.slider_tuple.tuple_masks import (  # noqa: E402
    compose, head_ellipse, load_mask_models, overlay, to_png_array, union,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("out")
    p.add_argument("--mask-dir", default="masks", help="folder for the masks, relative to the output manifest")
    p.add_argument("--background", type=float, default=0.2)
    p.add_argument("--body", type=float, default=1.0)
    p.add_argument("--head-weight", type=float, default=0.0,
                   help="loss weight on the head (0 = face left out; keep below --background)")
    p.add_argument("--head-margin", type=float, default=0.15, help="enlarge the estimated head ellipse by this fraction")
    p.add_argument("--kp-score", type=float, default=1.0, help="minimum keypoint score counted as visible")
    p.add_argument("--head-masks", help="folder of <tuple id>.png head masks to use instead of keypoints")
    p.add_argument("--preview", action="store_true", help="write <mask-dir>/preview/<id>.jpg overlays")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args(argv)


def tuple_mask(images, matte_fn, keypoints_fn, head_mask=None, background=0.2, body=1.0, head_margin=0.15, kp_score=1.0,
               head_weight=0.0):
    """(mask H x W, head found) for one tuple's member images (PIL RGB, same size)."""
    size = images[0].size
    body_union = union([matte_fn(img) for img in images])
    if head_mask is not None:
        head = (np.asarray(head_mask.convert("L").resize(size), dtype=np.float32) >= 128).astype(np.float32)
    else:
        heads = []
        for img in images:
            kps = keypoints_fn(img)
            if kps is not None:
                ellipse = head_ellipse(kps, size, min_score=kp_score, margin=head_margin)
                if ellipse is not None:
                    heads.append(ellipse)
        if not heads:
            return None, False
        head = union(heads)
    return compose(body_union, head, background=background, body_weight=body, head_weight=head_weight), True


def run(args, matte_fn, keypoints_fn):
    tuples = load_manifest(args.manifest)  # validates every path and size first
    src_dir = os.path.dirname(os.path.abspath(args.manifest))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    mask_dir = os.path.join(out_dir, args.mask_dir)
    os.makedirs(mask_dir, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f:
        raw = {e["id"]: e for e in (json.loads(line) for line in f if line.strip())}

    lines, missing = [], []
    for n, t in enumerate(tuples, 1):
        images = [Image.open(m.image).convert("RGB") for m in t.members]
        head_mask = None
        if args.head_masks:
            path = os.path.join(args.head_masks, f"{t.id}.png")
            if os.path.exists(path):
                head_mask = Image.open(path)
        mask, found = tuple_mask(images, matte_fn, keypoints_fn, head_mask,
                                 args.background, args.body, args.head_margin, args.kp_score, args.head_weight)
        entry = rebase_entry(raw[t.id], src_dir, out_dir)
        for m in entry["members"]:
            m.pop("mask", None)
        entry.pop("mask", None)
        if found:
            mask_path = os.path.join(mask_dir, f"{t.id}.png")
            Image.fromarray(to_png_array(mask)).save(mask_path)
            entry["mask"] = os.path.relpath(mask_path, out_dir)
            if args.preview:
                top = images[t.by_level[0]]
                preview = np.concatenate([np.asarray(top), overlay(np.asarray(top), mask)], axis=1)
                os.makedirs(os.path.join(mask_dir, "preview"), exist_ok=True)
                Image.fromarray(preview).save(os.path.join(mask_dir, "preview", f"{t.id}.jpg"), quality=90)
        else:
            missing.append(t.id)
        lines.append(json.dumps(entry))
        print(f"[{n}/{len(tuples)}] {t.id}: {'mask written' if found else 'NO HEAD FOUND, left unmasked'}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(tuples) - len(missing)} tuples masked"
          + (f", {len(missing)} without a head detection (unmasked): {missing}" if missing else ""))
    return missing


def main():
    args = parse_args()
    if args.head_masks and not os.path.isdir(args.head_masks):
        raise SystemExit(f"--head-masks: not a folder: {args.head_masks}")
    matte_fn, keypoints_fn = load_mask_models(args.device)
    run(args, matte_fn, keypoints_fn)


if __name__ == "__main__":
    main()
