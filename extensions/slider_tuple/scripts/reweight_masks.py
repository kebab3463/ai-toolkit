"""Change the background / head / body weights of a tuple dataset's existing loss masks.

paste_back.py and make_tuple_masks.py write masks as background + (body -
background) * body_matte with the head set to its own weight (0 by default).
This recovers the body matte and head region from each mask and recomposes it
with new weights, so no model has to run again. Writes new mask files and a
manifest pointing at them; the original masks stay untouched.

Usage (from the repo root):
    # heads trained ("keep the face") and a stronger background, for the measured weight data
    .venv/bin/python extensions/slider_tuple/scripts/reweight_masks.py \\
        datasets/weight_slider_tuples/measured/train.jsonl \\
        datasets/weight_slider_tuples/measured_bg05_head03/train.jsonl \\
        --background 0.5 --head-weight 0.3

The old weights default to what the scripts write (background 0.2, body 1,
head 0); pass --old-* if the masks were made with others. The head is
recognized by its exact value, so the old head weight must be below the old
background. Run it on held_out.jsonl too if the sweep should use the same manifest.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from extensions.slider_tuple.manifest import rebase_entry  # noqa: E402
from extensions.slider_tuple.tuple_masks import compose, decompose, to_png_array  # noqa: E402


def reweight(mask: np.ndarray, old: dict, new: dict) -> np.ndarray:
    """mask in [0, 1] made with ``old`` weights -> the same body / head with ``new`` weights."""
    body, head = decompose(mask, old["background"], old["body"], old["head"])
    return compose(body, head, background=new["background"], body_weight=new["body"], head_weight=new["head"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("out", help="manifest to write; new masks go to <its folder>/masks/")
    p.add_argument("--background", type=float, required=True)
    p.add_argument("--head-weight", type=float, required=True)
    p.add_argument("--body", type=float, default=1.0)
    p.add_argument("--old-background", type=float, default=0.2)
    p.add_argument("--old-head-weight", type=float, default=0.0)
    p.add_argument("--old-body", type=float, default=1.0)
    args = p.parse_args(argv)
    old = {"background": args.old_background, "head": args.old_head_weight, "body": args.old_body}
    new = {"background": args.background, "head": args.head_weight, "body": args.body}

    src_dir = os.path.dirname(os.path.abspath(args.manifest))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if os.path.abspath(args.out) == os.path.abspath(args.manifest):
        raise SystemExit("write to a new manifest; the original masks and manifest stay as they are")
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    done = {}  # source mask -> new mask (tuple- and member-level masks may be shared)

    def convert(path):
        source = path if os.path.isabs(path) else os.path.normpath(os.path.join(src_dir, path))
        if source not in done:
            mask = np.asarray(Image.open(source).convert("L"), dtype=np.float32) / 255.0
            target = os.path.join(mask_dir, os.path.basename(source))
            if target in done.values():
                raise SystemExit(f"two masks named {os.path.basename(source)}; cannot share one folder")
            Image.fromarray(to_png_array(reweight(mask, old, new))).save(target)
            done[source] = target
        return os.path.relpath(done[source], src_dir)  # relative to the source; rebased below

    with open(args.manifest, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    out = []
    for entry in entries:
        entry = dict(entry, members=[dict(m) for m in entry["members"]])
        if entry.get("mask"):
            entry["mask"] = convert(entry["mask"])
        for m in entry["members"]:
            if m.get("mask"):
                m["mask"] = convert(m["mask"])
        out.append(rebase_entry(entry, src_dir, out_dir))
    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(e) + "\n" for e in out)
    print(f"{len(out)} tuples, {len(done)} masks reweighted (background {args.background}, head {args.head_weight}, "
          f"body {args.body}) -> {args.out}")


if __name__ == "__main__":
    main()
