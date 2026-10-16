"""Run two training configs briefly with the same seed and compare their loss curves.

Each config is copied with a short step count, sampling disabled and per-step
loss logging enabled, then run through run.py one after the other with SEED set.
While each run trains, GPU memory is polled with nvidia-smi. At the end the
per-step `loss/loss` values from both runs' loss_log.db are compared.

Used for M2 (sd_trainer vs slider_tuple_trainer on the same polarity data) and
reusable for later equivalence checks (e.g. polarity vs slider_tuple in M4).

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/compare_runs.py \\
        extensions/slider_tuple/configs/krea2_polarity_smoke.yaml \\
        extensions/slider_tuple/configs/krea2_polarity_smoke_ext.yaml

Same config with one setting changed (an A/B):
    .venv/bin/python extensions/slider_tuple/scripts/compare_runs.py \\
        extensions/slider_tuple/configs/krea2_polarity_smoke.yaml \\
        extensions/slider_tuple/configs/krea2_polarity_smoke.yaml \\
        --set-b model.qtype=convrot8 --suffix-b _convrot8

Run folders are output/<config name><suffix>; pass --overwrite to replace them.
"""

import argparse
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

NVIDIA_SMI_CANDIDATES = ["/usr/lib/wsl/lib/nvidia-smi", "nvidia-smi"]

# Runs run.py unchanged and, at interpreter exit, writes torch's own peak CUDA
# memory counters to the JSON path in AITK_CMP_MEM_OUT. nvidia-smi under WSL
# cannot attribute memory to a process, so this is the per-run number.
RUN_WRAPPER = """
import atexit, json, os, runpy, sys
def _dump():
    import torch
    if torch.cuda.is_available():
        with open(os.environ["AITK_CMP_MEM_OUT"], "w") as f:
            json.dump({"max_allocated": torch.cuda.max_memory_allocated(),
                       "max_reserved": torch.cuda.max_memory_reserved()}, f)
atexit.register(_dump)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config_a")
    p.add_argument("config_b")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--suffix-a", default="_cmp", help="appended to config A's name for its run folder")
    p.add_argument("--suffix-b", default="_cmp", help="appended to config B's name for its run folder")
    p.add_argument("--set-a", action="append", default=[], metavar="KEY=VALUE",
                   help="override in config A's process, dotted path (e.g. model.qtype=convrot8); repeatable")
    p.add_argument("--set-b", action="append", default=[], metavar="KEY=VALUE",
                   help="override in config B's process, dotted path; repeatable")
    p.add_argument("--rtol", type=float, default=0.02, help="max per-step relative loss difference to call a match")
    p.add_argument("--overwrite", action="store_true", help="delete existing run folders first")
    p.add_argument("--reuse-existing", action="store_true",
                   help="reuse a run folder whose loss_log.db already covers --steps instead of training again "
                        "(its peak GPU memory is not known then)")
    return p.parse_args()


class VramPoller:
    """Polls used GPU memory (MiB) in a background thread and keeps the peak."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mib = None
        self._stop = threading.Event()
        self._cmd = next((c for c in NVIDIA_SMI_CANDIDATES if shutil.which(c) or os.path.exists(c)), None)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def read(self):
        if self._cmd is None:
            return None
        try:
            out = subprocess.run(
                [self._cmd, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            return int(out.splitlines()[0])
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            used = self.read()
            if used is not None:
                self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def apply_override(process: dict, assignment: str):
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise SystemExit(f"override must be KEY=VALUE, got {assignment!r}")
    *parents, leaf = key.split(".")
    node = process
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = yaml.safe_load(raw)  # so 1e-4, true, 512 get their YAML types


def make_short_config(path: str, steps: int, suffix: str, overrides=()) -> dict:
    from toolkit.config import get_config

    # round-trip through JSON to turn OrderedDicts into plain dicts for yaml.safe_dump
    config = json.loads(json.dumps(get_config(path)))
    config["config"]["name"] = config["config"]["name"] + suffix
    for process in config["config"]["process"]:
        process.setdefault("train", {})
        process["train"]["steps"] = steps
        process["train"]["disable_sampling"] = True
        process.setdefault("save", {})
        process["save"]["save_every"] = steps
        process["logging"] = {**process.get("logging", {}), "log_every": 1, "use_ui_logger": True}
        for assignment in overrides:
            apply_override(process, assignment)
    return config


def run_folder(config: dict) -> str:
    process = config["config"]["process"][0]
    return os.path.join(REPO_ROOT, process.get("training_folder", "output"), config["config"]["name"])


def read_losses(folder: str):
    db = os.path.join(folder, "loss_log.db")
    if not os.path.exists(db):
        raise SystemExit(f"no loss_log.db in {folder}")
    con = sqlite3.connect(db)
    losses = dict(con.execute("SELECT step, value_real FROM metrics WHERE key = 'loss/loss' ORDER BY step"))
    times = [t for (t,) in con.execute("SELECT wall_time FROM steps ORDER BY step")]
    con.close()
    # skip the first interval (includes warm-up / compilation)
    step_times = [b - a for a, b in zip(times[1:], times[2:])]
    return losses, (statistics.median(step_times) if step_times else None)


def run_one(config_path: str, suffix: str, overrides, args) -> dict:
    config = make_short_config(config_path, args.steps, suffix, overrides)
    folder = run_folder(config)
    if args.reuse_existing and os.path.exists(os.path.join(folder, "loss_log.db")):
        losses, s_per_step = read_losses(folder)
        if len(losses) >= args.steps - 1:
            print(f"\n=== {config['config']['name']}: reusing existing run ({len(losses)} logged steps) ===", flush=True)
            return {"name": config["config"]["name"], "losses": losses, "s_per_step": s_per_step,
                    "peak_mib": None, "idle_mib": None, "torch_mem": None, "wall_s": 0.0}
    if os.path.exists(folder):
        if not args.overwrite:
            raise SystemExit(f"{folder} exists; pass --overwrite to replace it")
        shutil.rmtree(folder)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config, f, sort_keys=False)
        tmp_path = f.name

    extra = f", overrides {list(overrides)}" if overrides else ""
    print(f"\n=== {config['config']['name']}: {args.steps} steps, SEED={args.seed}{extra} ===", flush=True)
    mem_out = tmp_path + ".mem.json"
    env = {**os.environ, "SEED": str(args.seed), "AITK_CMP_MEM_OUT": mem_out}
    start = time.time()
    try:
        with VramPoller() as vram:
            idle = vram.read()
            result = subprocess.run(
                [sys.executable, "-c", RUN_WRAPPER, os.path.join(REPO_ROOT, "run.py"), tmp_path],
                cwd=REPO_ROOT, env=env,
            )
        torch_mem = None
        if os.path.exists(mem_out):
            with open(mem_out) as f:
                torch_mem = json.load(f)
    finally:
        os.unlink(tmp_path)
        if os.path.exists(mem_out):
            os.unlink(mem_out)
    if result.returncode != 0:
        raise SystemExit(f"run failed for {config_path} (exit {result.returncode})")

    losses, s_per_step = read_losses(folder)
    return {
        "name": config["config"]["name"],
        "losses": losses,
        "s_per_step": s_per_step,
        "peak_mib": vram.peak_mib,
        "idle_mib": idle,
        "torch_mem": torch_mem,
        "wall_s": time.time() - start,
    }


def main():
    args = parse_args()
    names = [make_short_config(c, args.steps, sfx)["config"]["name"]
             for c, sfx in ((args.config_a, args.suffix_a), (args.config_b, args.suffix_b))]
    if names[0] == names[1]:
        raise SystemExit(f"both runs would use output/{names[0]}; give them different --suffix-a/--suffix-b")
    a = run_one(args.config_a, args.suffix_a, args.set_a, args)
    b = run_one(args.config_b, args.suffix_b, args.set_b, args)

    print("\n=== summary ===")
    for r in (a, b):
        sps = f"{r['s_per_step']:.2f}" if r["s_per_step"] else "n/a"
        if r["peak_mib"] is None:
            mem = "peak GPU memory n/a (reused run)"
        else:
            mem = f"whole-GPU peak {r['peak_mib']} MiB (idle before start {r['idle_mib']} MiB)"
            if r["torch_mem"]:
                gib = 1024 ** 3
                mem = (f"torch peak allocated {r['torch_mem']['max_allocated'] / gib:.2f} GiB, "
                       f"reserved {r['torch_mem']['max_reserved'] / gib:.2f} GiB; " + mem)
        print(f"{r['name']}: {sps} s/step (median), {mem}, wall {r['wall_s'] / 60:.1f} min")

    steps = sorted(set(a["losses"]) & set(b["losses"]))
    if not steps:
        raise SystemExit("no common logged steps to compare")
    print(f"\n{'step':>5} {a['name']:>28} {b['name']:>28} {'rel diff':>9}")
    rel = []
    for s in steps:
        la, lb = a["losses"][s], b["losses"][s]
        r = abs(la - lb) / max(abs(la), abs(lb), 1e-12)
        rel.append(r)
        print(f"{s:>5} {la:>28.6f} {lb:>28.6f} {r:>9.2%}")
    for r in (a, b):
        vals = [r["losses"][s] for s in steps]
        print(f"{r['name']}: mean loss {statistics.mean(vals):.6f}, median {statistics.median(vals):.6f}")
    worst = max(rel)
    verdict = "MATCH" if worst <= args.rtol else "DIFFER"
    print(f"\n{verdict}: max per-step relative difference {worst:.2%} over {len(steps)} steps (rtol {args.rtol:.0%})")
    sys.exit(0 if verdict == "MATCH" else 1)


if __name__ == "__main__":
    main()
