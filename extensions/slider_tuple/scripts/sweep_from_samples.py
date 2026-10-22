"""Sweep tables from a run's own sample images (in-training sweeps, see sweep_grid.py).

When a run's sample list is a sweep grid (each sample with a seed and a
multiplier, including 0 for every prompt/seed), every sample step is already a
sweep of that checkpoint. This measures those images like slider_sweep.py
(the same measure, LPIPS and summaries) and writes one sweep folder per step:

    <out>/<run>_<step>/  sweep.csv, summary.txt / .json, sheets/

so compare_sweeps.py works on them as on checkpoint sweeps. Steps already
measured are skipped (--force redoes them); run it again while training goes on.

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/sweep_from_samples.py output/weight_r4c_shared0 --measure body
    .venv/bin/python extensions/slider_tuple/scripts/compare_sweeps.py --parity output/sweeps/weight_r4c_shared0/*
"""

import argparse
import importlib.util
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from extensions.slider_tuple.sweep import rows_from_samples, write_csv  # noqa: E402

SAMPLE_NAME = re.compile(r"^(\d+)__(\d+)_(\d+)\.\w+$")  # <time>__<step>_<index>.<ext>


def load_sweep_script():
    spec = importlib.util.spec_from_file_location("slider_sweep", os.path.join(os.path.dirname(__file__), "slider_sweep.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_samples(config_path):
    import yaml

    from toolkit.config import fixed_loader

    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=fixed_loader) if config_path.endswith((".yaml", ".yml")) else __import__("json").load(f)
    sample = config["config"]["process"][0].get("sample") or {}
    samples = sample.get("samples")
    if not samples:
        raise SystemExit(f"{config_path}: no sample.samples list (make one with sweep_grid.py)")
    return config["config"]["name"], samples


def images_by_step(sample_dir):
    """{step: {index: path}}; when a step was sampled twice (restart), the later images win."""
    found = {}
    for name in sorted(os.listdir(sample_dir)):
        match = SAMPLE_NAME.match(name)
        if match:
            stamp, step, index = (int(g) for g in match.groups())
            slot = found.setdefault(step, {})
            if index not in slot or stamp > slot[index][0]:
                slot[index] = (stamp, os.path.join(sample_dir, name))
    return {step: {i: p for i, (_, p) in slots.items()} for step, slots in found.items()}


def main(argv=None, models=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="the run's folder, e.g. output/<name>")
    p.add_argument("--config", help="its config (default: <run_dir>/config.yaml)")
    p.add_argument("--out", help="default: output/sweeps/<run name>")
    p.add_argument("--steps", help="comma-separated steps to measure (default: all)")
    p.add_argument("--force", action="store_true", help="re-measure steps already done")
    p.add_argument("--measure", help="'body' or module:function, as for slider_sweep.py")
    p.add_argument("--no-lpips", action="store_true")
    p.add_argument("--lpips-res", type=int, default=512)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)

    name, samples = run_samples(args.config or os.path.join(args.run_dir, "config.yaml"))
    out = args.out or os.path.join(REPO_ROOT, "output", "sweeps", name)
    by_step = images_by_step(os.path.join(args.run_dir, "samples"))
    wanted = {int(s) for s in args.steps.split(",")} if args.steps else set(by_step)
    todo = []
    for step in sorted(wanted & set(by_step)):
        folder = os.path.join(out, f"{name}_{step:09d}")
        if len(by_step[step]) < len(samples):
            print(f"step {step}: {len(by_step[step])}/{len(samples)} images, skipped (still sampling?)")
            continue
        if os.path.exists(os.path.join(folder, "sweep.csv")) and not args.force:
            continue
        todo.append((step, folder, rows_from_samples(samples, by_step[step])))
    print(f"{name}: {len(by_step)} sample steps, {len(todo)} to measure -> {out}")
    if not todo:
        return []

    sweep = load_sweep_script()
    sweep_argv = ["--prompts", "unused", "--out", out, "--lpips-res", str(args.lpips_res), "--device", args.device]
    if args.measure:
        sweep_argv += ["--measure", args.measure]
    if args.no_lpips:
        sweep_argv.append("--no-lpips")
    sweep_args = sweep.parse_args(sweep_argv)
    models = models if models is not None else sweep.load_measure_models(sweep_args)
    done = []
    for step, folder, rows in todo:
        os.makedirs(folder, exist_ok=True)
        sweep.measure(sweep_args, rows, models=models)
        write_csv(rows, os.path.join(folder, "sweep.csv"))
        sweep_args.out = folder
        sweep.write_outputs(sweep_args, folder, rows, len({r.prompt_idx for r in rows}))
        print(f"step {step}: measured {len(rows)} images -> {folder}")
        done.append(folder)
    return done


if __name__ == "__main__":
    main()
