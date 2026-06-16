#!/usr/bin/env python3
"""
Rewind diffusion training outputs on disk after a given global training step.

Use case: multi-stage (or single-stage) run advanced past step N; you change stage
settings in the job config and want to resume from a clean history. This script:

  - Deletes checkpoints whose 9-digit step suffix is **strictly greater** than --keep-step
    (matches {name}_000000123.safetensors, diffusers folders, embeddings, adapters, refiner, critic, …)
  - Trims loss_log.db (steps + metrics) to step <= keep-step
  - Optionally removes samples, optimizer.pt, tensorboard run dir, and patches config.yaml

Omit **--apply** to dry-run (print actions only). Pass **--apply** to delete or modify files.

Resume checklist (after --apply):
  1. Fix your YAML / UI job config (stages, LR, etc.).
  2. If you removed optimizer.pt, the next run rebuilds optimizer state from scratch (same as missing file).
  3. Set train.start_step in the job config to the step you want the loop to start from (often keep-step + 1,
     or keep-step if you want to repeat that step). Use --set-start-step to patch save_root/config.yaml only.

Examples:
  python scripts/rewind_training_output.py \\
    --run-dir output/my_lora_run --keep-step 2500 --dry-run

  python scripts/rewind_training_output.py \\
    --training-folder output --name my_lora_run --keep-step 2500 --apply \\
    --set-start-step 2501 --remove-samples --remove-optimizer-state
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import yaml

# Matches trailing _000000123 before extension or end of directory name.
_STEP_SUFFIX_RE = re.compile(r"_(\d{9})(?:\.[^./\\]+)?$")
# Sample images: [time]_000000100_0.png
_SAMPLE_STEP_RE = re.compile(r"_(\d{9})_")


def _parse_step_from_basename(name: str) -> Optional[int]:
    """Return training step encoded in a checkpoint-style basename, or None."""
    m = _STEP_SUFFIX_RE.search(name)
    if not m:
        return None
    return int(m.group(1))


def _parse_sample_step(filename: str) -> Optional[int]:
    m = _SAMPLE_STEP_RE.search(filename)
    if not m:
        return None
    return int(m.group(1))


def _iter_checkpoint_paths(save_root: Path) -> Iterable[Path]:
    """Paths that may encode a training step in the basename (files + dirs)."""
    if not save_root.is_dir():
        return
    for entry in save_root.iterdir():
        if entry.name in ("samples", "config.yaml", "README.md", ".job_config.json"):
            continue
        if entry.name == "loss_log.db":
            continue
        if entry.name == "optimizer.pt":
            continue
        if entry.name == "learnable_snr.json":
            continue
        if entry.name.startswith("."):
            continue
        # Only consider artifacts with a step suffix (skip bare name.safetensors without _NNNNNNNNN).
        if _parse_step_from_basename(entry.name) is None:
            continue
        yield entry


def _iter_sample_paths(samples_dir: Path) -> Iterable[Path]:
    if not samples_dir.is_dir():
        return
    for entry in samples_dir.iterdir():
        if entry.is_file() and _parse_sample_step(entry.name) is not None:
            yield entry


def _trim_loss_log_db(db_path: Path, keep_step: int, dry_run: bool) -> List[str]:
    actions: List[str] = []
    if not db_path.is_file():
        actions.append(f"skip loss_log.db (missing): {db_path}")
        return actions

    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('steps','metrics','metric_keys')"
        )
        tables = {r[0] for r in cur.fetchall()}
        if "steps" not in tables or "metrics" not in tables:
            actions.append(f"skip loss_log.db (unexpected schema): {db_path}")
            return actions

        cur.execute("SELECT COUNT(*) FROM steps WHERE step > ?", (keep_step,))
        n_steps = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM metrics WHERE step > ?",
            (keep_step,),
        )
        n_metrics = cur.fetchone()[0]
        actions.append(
            f"loss_log.db: delete {n_steps} step rows and {n_metrics} metric rows with step > {keep_step}"
        )
        if dry_run:
            return actions

        con.execute("PRAGMA foreign_keys = ON;")
        # metrics references steps; CASCADE removes metrics when steps removed (see logging_aitk schema).
        cur.execute("DELETE FROM steps WHERE step > ?", (keep_step,))
        cur.execute(
            "DELETE FROM metric_keys WHERE key NOT IN (SELECT DISTINCT key FROM metrics)"
        )
        con.commit()
        actions.append("loss_log.db: committed deletes")
    finally:
        con.close()
    return actions


def _patch_config_start_step(
    config_path: Path, start_step: int, dry_run: bool
) -> List[str]:
    actions: List[str] = []
    if not config_path.is_file():
        actions.append(f"skip config patch (missing): {config_path}")
        return actions
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        actions.append("skip config patch (no top-level config dict)")
        return actions
    proc = cfg.get("process")
    if not isinstance(proc, list) or not proc:
        actions.append("skip config patch (no config.process list)")
        return actions
    train = proc[0].get("train")
    if not isinstance(train, dict):
        actions.append("skip config patch (no train dict)")
        return actions
    old = train.get("start_step")
    actions.append(f"config.yaml: set train.start_step {old!r} -> {start_step}")
    if dry_run:
        return actions
    train["start_step"] = int(start_step)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    actions.append(f"wrote {config_path}")
    return actions


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--run-dir",
        type=Path,
        help="Training output folder (same as save_root: training_folder / config.name)",
    )
    g.add_argument(
        "--training-folder",
        type=Path,
        help="With --name, save_root = training_folder / name",
    )
    p.add_argument(
        "--name",
        type=str,
        help="Job / run name when using --training-folder",
    )
    p.add_argument(
        "--keep-step",
        type=int,
        required=True,
        help="Keep all artifacts at training step <= this value; remove strictly later steps.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete / rewrite files (default is dry-run).",
    )
    p.add_argument(
        "--remove-samples",
        action="store_true",
        help="Delete sample images under samples/ whose step in the filename is > keep-step.",
    )
    p.add_argument(
        "--remove-optimizer-state",
        action="store_true",
        help="Delete optimizer.pt (recommended after removing newer checkpoints so resume does not load wrong state).",
    )
    p.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="If set, delete this entire directory (e.g. one TensorBoard run folder).",
    )
    p.add_argument(
        "--set-start-step",
        type=int,
        default=None,
        help="If set with --apply, write train.start_step into save_root/config.yaml (nested job format).",
    )
    args = p.parse_args(argv)

    dry_run = not args.apply
    if args.training_folder:
        if not args.name:
            p.error("--name is required with --training-folder")
        save_root = (args.training_folder / args.name).resolve()
    else:
        save_root = args.run_dir.resolve()

    if not save_root.is_dir():
        print(f"Error: save root is not a directory: {save_root}", file=sys.stderr)
        return 2

    keep = int(args.keep_step)
    actions: List[str] = []

    to_delete: List[Path] = []
    for path in sorted(list(_iter_checkpoint_paths(save_root)), key=lambda x: x.name):
        step = _parse_step_from_basename(path.name)
        if step is None:
            continue
        if step > keep:
            to_delete.append(path)

    for path in to_delete:
        actions.append(f"delete checkpoint artifact: {path}")
    if not dry_run:
        for path in to_delete:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=False)
            else:
                path.unlink(missing_ok=True)
                yml = path.with_suffix(".yaml")
                if yml.is_file():
                    yml.unlink(missing_ok=True)

    actions.extend(_trim_loss_log_db(save_root / "loss_log.db", keep, dry_run))

    if args.remove_samples:
        samples_dir = save_root / "samples"
        for path in sorted(_iter_sample_paths(samples_dir)):
            st = _parse_sample_step(path.name)
            if st is not None and st > keep:
                actions.append(f"delete sample: {path}")
                if not dry_run:
                    path.unlink(missing_ok=True)

    opt_pt = save_root / "optimizer.pt"
    if args.remove_optimizer_state and opt_pt.is_file():
        actions.append(f"delete optimizer state: {opt_pt}")
        if not dry_run:
            opt_pt.unlink(missing_ok=True)

    if args.tensorboard_dir is not None:
        tb = args.tensorboard_dir.resolve()
        actions.append(f"delete tensorboard dir: {tb}")
        if not dry_run and tb.exists():
            shutil.rmtree(tb, ignore_errors=False)

    if args.set_start_step is not None:
        actions.extend(
            _patch_config_start_step(
                save_root / "config.yaml", int(args.set_start_step), dry_run
            )
        )

    print(f"save_root: {save_root}")
    print(f"keep_step: {keep}  (mode: {'DRY-RUN' if dry_run else 'APPLY'})")
    for line in actions:
        print(f"  - {line}")

    if args.apply and to_delete and not args.remove_optimizer_state:
        print(
            "\nWarning: checkpoints were removed but optimizer.pt was kept. "
            "If it was saved after --keep-step, resume can load a mismatched optimizer. "
            "Consider --remove-optimizer-state.",
            file=sys.stderr,
        )

    if dry_run:
        print("\nRe-run with --apply to perform these actions.")
    else:
        print(
            "\nDone. Update your job definition if it is not loaded from this config.yaml, then resume."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
