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

Run folders are output/<config name>_cmp; pass --overwrite to replace them.
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config_a")
    p.add_argument("config_b")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--suffix", default="_cmp", help="appended to each config's name for its run folder")
    p.add_argument("--rtol", type=float, default=0.02, help="max per-step relative loss difference to call a match")
    p.add_argument("--overwrite", action="store_true", help="delete existing run folders first")
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


def make_short_config(path: str, steps: int, suffix: str) -> dict:
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


def run_one(config_path: str, args) -> dict:
    config = make_short_config(config_path, args.steps, args.suffix)
    folder = run_folder(config)
    if os.path.exists(folder):
        if not args.overwrite:
            raise SystemExit(f"{folder} exists; pass --overwrite to replace it")
        shutil.rmtree(folder)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config, f, sort_keys=False)
        tmp_path = f.name

    print(f"\n=== {config['config']['name']}: {args.steps} steps, SEED={args.seed} ===", flush=True)
    env = {**os.environ, "SEED": str(args.seed)}
    start = time.time()
    try:
        with VramPoller() as vram:
            idle = vram.read()
            result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "run.py"), tmp_path], cwd=REPO_ROOT, env=env)
    finally:
        os.unlink(tmp_path)
    if result.returncode != 0:
        raise SystemExit(f"run failed for {config_path} (exit {result.returncode})")

    losses, s_per_step = read_losses(folder)
    return {
        "name": config["config"]["name"],
        "losses": losses,
        "s_per_step": s_per_step,
        "peak_mib": vram.peak_mib,
        "idle_mib": idle,
        "wall_s": time.time() - start,
    }


def main():
    args = parse_args()
    a = run_one(args.config_a, args)
    b = run_one(args.config_b, args)

    print("\n=== summary ===")
    for r in (a, b):
        sps = f"{r['s_per_step']:.2f}" if r["s_per_step"] else "n/a"
        print(f"{r['name']}: {sps} s/step (median), peak GPU memory {r['peak_mib']} MiB "
              f"(idle before start {r['idle_mib']} MiB), wall {r['wall_s'] / 60:.1f} min")

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
    worst = max(rel)
    verdict = "MATCH" if worst <= args.rtol else "DIFFER"
    print(f"\n{verdict}: max per-step relative difference {worst:.2%} over {len(steps)} steps (rtol {args.rtol:.0%})")
    sys.exit(0 if verdict == "MATCH" else 1)


if __name__ == "__main__":
    main()
