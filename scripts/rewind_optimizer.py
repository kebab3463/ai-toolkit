#!/usr/bin/env python3
"""
Manage AI Toolkit optimizer.pt checkpoints when rewinding training.

Important limitation
--------------------
The trainer saves a single optimizer.pt in each run folder (see
jobs/process/BaseSDTrainProcess.py). It is overwritten on every checkpoint
save and does NOT store per-step history. You cannot reconstruct the exact
optimizer state at an earlier step from optimizer.pt alone.

What this script can do instead:
  inspect  - show a summary of the saved optimizer state
  reset    - zero accumulated optimizer state (momentum, Adafactor factors,
             Automagic sign history, etc.) while keeping param_groups intact.
             Use this after loading an earlier LoRA safetensors checkpoint and
             setting start_step in your config.
  delete   - remove optimizer.pt so training starts with a fresh optimizer
  restore  - copy a backup file over optimizer.pt

Typical rewind workflow
-----------------------
1. Stop training.
2. Copy or select the LoRA checkpoint at the target step
   (e.g. my_job_000001500.safetensors).
3. Run:  python scripts/rewind_optimizer.py reset output/my_job/optimizer.pt
   or:    python scripts/rewind_optimizer.py delete output/my_job/optimizer.pt
4. Run:  python scripts/prune_loss_log.py output/my_job/loss_log.db --after-step 1500
5. Set train.start_step: 1500 in your job config and resume.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect, reset, delete, or restore AI Toolkit optimizer.pt files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print optimizer state summary")
    inspect_parser.add_argument("optimizer", type=str, help="Path to optimizer.pt")

    reset_parser = subparsers.add_parser(
        "reset",
        help="Zero accumulated optimizer state (approximate fresh optimizer)",
    )
    reset_parser.add_argument("optimizer", type=str, help="Path to optimizer.pt")
    reset_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be reset without writing changes.",
    )
    reset_parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy optimizer.pt to optimizer.pt.bak.<timestamp> before resetting.",
    )

    delete_parser = subparsers.add_parser("delete", help="Delete optimizer.pt")
    delete_parser.add_argument("optimizer", type=str, help="Path to optimizer.pt")
    delete_parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy optimizer.pt to optimizer.pt.bak.<timestamp> before deleting.",
    )

    restore_parser = subparsers.add_parser(
        "restore",
        help="Restore optimizer.pt from a backup copy",
    )
    restore_parser.add_argument("optimizer", type=str, help="Path to optimizer.pt")
    restore_parser.add_argument(
        "--from",
        dest="source",
        type=str,
        required=True,
        metavar="BACKUP",
        help="Backup file to copy over optimizer.pt",
    )
    restore_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite optimizer.pt if it already exists.",
    )

    return parser.parse_args()


def load_state_dict(path: str) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def save_state_dict(path: str, state_dict: dict[str, Any]) -> None:
    import torch

    torch.save(state_dict, path)


def backup_file(path: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak.{stamp}"
    shutil.copy2(path, backup_path)
    return backup_path


INT_RESET_KEYS = {"hist_idx", "hist_fill", "step"}


def reset_tensor(value):
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    if value.numel() == 1 and value.dtype in (torch.int64, torch.int32, torch.long):
        return torch.zeros((), dtype=value.dtype)
    return torch.zeros_like(value)


def reset_param_state(param_state: dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for key, value in list(param_state.items()):
        if isinstance(value, dict):
            nested = reset_param_state(value)
            if nested:
                changed.extend(f"{key}.{item}" for item in nested)
            continue

        if key in INT_RESET_KEYS and isinstance(value, int):
            if value != 0:
                param_state[key] = 0
                changed.append(key)
            continue

        import torch

        if isinstance(value, torch.Tensor):
            new_value = reset_tensor(value)
            if not torch.equal(value, new_value):
                param_state[key] = new_value
                changed.append(key)

    return changed


def reset_optimizer_state_dict(state_dict: dict[str, Any]) -> tuple[dict[str, Any], int]:
    changed_keys = 0
    for param_id, param_state in state_dict.get("state", {}).items():
        touched = reset_param_state(param_state)
        if touched:
            changed_keys += 1
    return state_dict, changed_keys


def tensor_summary(value) -> str:
    import torch

    if not isinstance(value, torch.Tensor):
        return repr(value)
    return f"Tensor shape={tuple(value.shape)} dtype={value.dtype}"


def inspect_state_dict(state_dict: dict[str, Any]) -> None:
    param_groups = state_dict.get("param_groups", [])
    state = state_dict.get("state", {})

    print(f"Param groups: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        params = group.get("params", [])
        lr = group.get("lr", "?")
        print(f"  group[{i}]: {len(params)} param ref(s), lr={lr}")
        for key, value in sorted(group.items()):
            if key == "params":
                continue
            print(f"    {key}: {value}")

    print(f"Tracked parameter states: {len(state)}")
    sample_ids = list(state.keys())[:3]
    for param_id in sample_ids:
        param_state = state[param_id]
        print(f"  state[{param_id}]:")
        for key, value in sorted(param_state.items()):
            print(f"    {key}: {tensor_summary(value)}")

    if len(state) > len(sample_ids):
        print(f"  ... and {len(state) - len(sample_ids)} more parameter state entries")


def cmd_inspect(path: str) -> int:
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    state_dict = load_state_dict(path)
    print(f"Optimizer: {os.path.abspath(path)}")
    inspect_state_dict(state_dict)
    return 0


def cmd_reset(path: str, dry_run: bool, do_backup: bool) -> int:
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    state_dict = load_state_dict(path)
    _, num_changed = reset_optimizer_state_dict(state_dict)

    print(f"Optimizer: {os.path.abspath(path)}")
    print(f"Parameter states with reset tensors/scalars: {num_changed}")

    if num_changed == 0:
        print("Nothing to reset.")
        return 0

    if dry_run:
        print("Dry run: no changes written.")
        return 0

    if do_backup:
        backup_path = backup_file(path)
        print(f"Backup written to {backup_path}")

    save_state_dict(path, state_dict)
    print("Reset complete.")
    print(
        "Note: this clears accumulated optimizer state but cannot restore the "
        "exact historical state at an earlier step."
    )
    return 0


def cmd_delete(path: str, do_backup: bool) -> int:
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    if do_backup:
        backup_path = backup_file(path)
        print(f"Backup written to {backup_path}")

    os.remove(path)
    print(f"Deleted {os.path.abspath(path)}")
    print("Training will create a fresh optimizer on next resume.")
    return 0


def cmd_restore(path: str, source: str, force: bool) -> int:
    source = os.path.abspath(source)
    path = os.path.abspath(path)

    if not os.path.isfile(source):
        print(f"Error: backup not found: {source}", file=sys.stderr)
        return 1

    if os.path.exists(path) and not force:
        print(
            f"Error: {path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    shutil.copy2(source, path)
    print(f"Restored {source} -> {path}")
    return 0


def main() -> int:
    args = parse_args()

    if args.command == "inspect":
        return cmd_inspect(args.optimizer)
    if args.command == "reset":
        return cmd_reset(args.optimizer, args.dry_run, args.backup)
    if args.command == "delete":
        return cmd_delete(args.optimizer, args.backup)
    if args.command == "restore":
        return cmd_restore(args.optimizer, args.source, args.force)

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
