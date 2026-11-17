"""Set the captions of an existing manifest from caption files, keeping everything else.

For captions written after the manifest (paste_back.py --captions would
rewrite it, levels included). Each tuple's caption comes from
<captions>/<id>.txt, or <captions>/<id>.json (the string under --caption-key,
default caption_long). Tuples without a caption file keep their caption and
are listed.

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/set_captions.py \\
        datasets/slider_tuples/tuples.jsonl datasets/slider_tuples/captions
    # writes in place; --out elsewhere.jsonl to keep the original
Run it before split_holdout.py, or on the train / held-out manifests too.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from extensions.slider_tuple import captions  # noqa: E402
from extensions.slider_tuple.manifest import rebase_entry  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest")
    p.add_argument("captions", help="folder of <id>.txt or <id>.json captions")
    p.add_argument("--caption-key", default=captions.DEFAULT_CAPTION_KEY, help="field to read from .json captions")
    p.add_argument("--out", help="manifest to write (default: in place)")
    args = p.parse_args(argv)

    with open(args.manifest, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    missing = []
    for entry in entries:
        try:
            caption = captions.read_caption(args.captions, str(entry["id"]), args.caption_key)
        except ValueError as e:
            raise SystemExit(f"{e}; see --caption-key")
        if caption is None:
            missing.append(entry["id"])
        else:
            entry["caption"] = caption

    out = args.out or args.manifest
    src_dir, out_dir = (os.path.dirname(os.path.abspath(p)) for p in (args.manifest, out))
    text = "".join(json.dumps(rebase_entry(e, src_dir, out_dir)) + "\n" for e in entries)
    with open(out, "w", encoding="utf-8") as f:  # after reading everything: a failed run must not truncate it
        f.write(text)
    print(f"{len(entries) - len(missing)}/{len(entries)} captions set -> {out}")
    if missing:
        print(f"{len(missing)} without a caption file kept theirs: {missing}")
    return out


if __name__ == "__main__":
    main()
