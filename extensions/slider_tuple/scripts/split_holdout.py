"""Split a tuples.jsonl into a training manifest and a held-out manifest.

The held-out tuples are never trained on; their captions are the prompts for
the linearity sweep (slider_sweep.py --prompts held_out.jsonl). The split is
deterministic for a given --seed and set of tuple ids, so re-running after a
caption change (paste_back.py --captions) holds out the same tuples.

Writes <out-dir>/train.jsonl and <out-dir>/held_out.jsonl (default: next to
the input), with paths rewritten relative to the output folder.

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/split_holdout.py datasets/weight_slider_tuples/tuples.jsonl
    # hold out specific tuples instead of a random 10%
    ... --ids v3jf2w2lrvih1,abc123
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from extensions.slider_tuple.manifest import rebase_entry  # noqa: E402


def choose_held_out(ids, fraction=0.1, seed=0, forced=()):
    """Tuple ids to hold out: ``forced`` if given, else round(fraction * n) (at least 1
    when there are 2+ tuples) picked by a seeded shuffle of the sorted ids."""
    if forced:
        missing = set(forced) - set(ids)
        if missing:
            raise ValueError(f"unknown tuple ids: {sorted(missing)}")
        return set(forced)
    n = len(ids)
    k = 0 if n < 2 else min(n - 1, max(1, round(fraction * n)))
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return set(order[:k])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("--out-dir", help="folder for train.jsonl / held_out.jsonl (default: the manifest's folder)")
    p.add_argument("--fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ids", default="", help="comma-separated tuple ids to hold out (overrides --fraction)")
    args = p.parse_args(argv)
    if not 0 <= args.fraction < 1:
        p.error("--fraction must be in [0, 1)")

    src_dir = os.path.dirname(os.path.abspath(args.manifest))
    out_dir = os.path.abspath(args.out_dir or src_dir)
    with open(args.manifest, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate tuple ids in the manifest")
    forced = [x.strip() for x in args.ids.split(",") if x.strip()]
    held = choose_held_out(ids, args.fraction, args.seed, forced)

    os.makedirs(out_dir, exist_ok=True)
    outputs = {"train.jsonl": [e for e in entries if e["id"] not in held],
               "held_out.jsonl": [e for e in entries if e["id"] in held]}
    for name, rows in outputs.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(rebase_entry(e, src_dir, out_dir)) + "\n" for e in rows)
    print(f"{len(outputs['train.jsonl'])} train, {len(held)} held out "
          f"({', '.join(sorted(held)) or 'none'}) -> {out_dir}")


if __name__ == "__main__":
    main()
