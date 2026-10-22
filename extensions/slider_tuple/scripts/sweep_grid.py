"""Make a training config's sample previews a sweep grid (in-training sweeps).

Replaces the config's sample prompts with every held-out prompt x seed x
multiplier, each with an explicit seed and network_multiplier, and sets PNG
output. Every sample step then renders a full sweep of that checkpoint on
Turbo (with slider.turbo_preview), with no separate sweep job; the web UI
shows it as the run trains. Afterwards sweep_from_samples.py turns the samples
into sweep tables for compare_sweeps.py.

The config is edited as text, so its comments stay: the list under the
sample block's ``prompts:`` / ``samples:`` key is replaced, ``format`` is set to
png and ``walk_seed`` to false.

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/sweep_grid.py CONFIG.yaml \\
        --prompts datasets/weight_slider_tuples/held_out.jsonl
    # defaults: --seeds 1000,1001,1002 --multipliers 0,1,2,3,-1,-2,-3 (the sweeps so far)

Cost on Turbo: about 4 s per image, so 42 images (2 prompts x 3 seeds x 7) ~3 min
per sample step.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from extensions.slider_tuple.sweep import grid_samples  # noqa: E402


def read_prompts(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line)["caption"] for line in f if line.strip()]
        return [line.strip() for line in f if line.strip()]


def set_grid(text: str, samples) -> str:
    """The config text with the sample block's prompt list replaced by ``samples`` (png, fixed seeds)."""
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines) if re.match(r"^\s*sample:\s*$", l)), None)
    if start is None:
        raise ValueError("no 'sample:' block in the config")
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= indent), len(lines))
    block = lines[start + 1:end]
    key_indent = " " * (indent + 2)

    # drop the existing prompts / samples list
    out, skipping = [], False
    for line in block:
        stripped = line.strip()
        depth = len(line) - len(line.lstrip())
        if re.match(rf"^{key_indent}(prompts|samples):", line):
            skipping = True
            continue
        if skipping and (not stripped or depth > len(key_indent) or stripped.startswith("- ")):
            continue
        skipping = False
        out.append(line)

    def set_key(key, value):
        for i, line in enumerate(out):
            if re.match(rf"^{key_indent}{key}:", line):
                comment = line[line.index("#"):] if "#" in line else ""
                out[i] = f"{key_indent}{key}: {value}" + (f" {comment}" if comment else "")
                return
        out.insert(0, f"{key_indent}{key}: {value}")

    set_key("format", '"png"')
    set_key("walk_seed", "false")
    grid = [f"{key_indent}# sweep grid (sweep_grid.py): prompt x seed x multiplier",
            f"{key_indent}samples:"]
    grid += [f"{key_indent}  - {{prompt: {json.dumps(s['prompt'])}, seed: {s['seed']}, "
             f"network_multiplier: {s['network_multiplier']}}}" for s in samples]
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(lines[:start + 1] + out + grid + lines[end:])


def floats(text):
    return [float(x) for x in text.split(",") if x.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--prompts", required=True, help="held_out.jsonl (captions) or a text file, one prompt per line")
    p.add_argument("--seeds", default="1000,1001,1002")
    p.add_argument("--multipliers", default="0,1,2,3,-1,-2,-3")
    p.add_argument("--out", help="write here instead of editing the config in place")
    args = p.parse_args(argv)
    samples = grid_samples(read_prompts(args.prompts), [int(s) for s in floats(args.seeds)], floats(args.multipliers))
    with open(args.config, encoding="utf-8") as f:
        text = f.read()
    with open(args.out or args.config, "w", encoding="utf-8") as f:
        f.write(set_grid(text, samples))
    print(f"{len(samples)} samples per sample step -> {args.out or args.config}")
    return samples


if __name__ == "__main__":
    main()
