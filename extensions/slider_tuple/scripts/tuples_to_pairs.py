"""Reduce a tuples.jsonl to pairs drawn from the same data (M5: tuples vs pairs).

Each tuple keeps only its highest- and lowest-level members (usually the real
photo and the strongest edit), with every other field unchanged, so a pairs run
and a tuples run differ only in the intermediate levels. Paths are rewritten
relative to the output manifest.

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/tuples_to_pairs.py data/tuples.jsonl data/pairs.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from extensions.slider_tuple.manifest import rebase_entry  # noqa: E402


def to_pair(entry: dict, src_dir: str, out_dir: str) -> dict:
    members = sorted(entry["members"], key=lambda m: -m["level"])
    return rebase_entry(dict(entry, members=[members[0], members[-1]]), src_dir, out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("out")
    args = p.parse_args()
    src_dir = os.path.dirname(os.path.abspath(args.manifest))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    with open(args.manifest, encoding="utf-8") as f:
        pairs = [to_pair(json.loads(line), src_dir, out_dir) for line in f if line.strip()]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x) for x in pairs) + "\n")
    print(f"wrote {len(pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
