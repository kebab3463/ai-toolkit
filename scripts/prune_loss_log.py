#!/usr/bin/env python3
"""
Prune entries from an AI Toolkit loss_log.db after a given training step.

The database schema matches toolkit.logging_aitk.UILogger:
  - steps(step, wall_time)
  - metrics(step, key, value_real, value_text)  [FK -> steps, ON DELETE CASCADE]
  - metric_keys(key, first_seen_step, last_seen_step)

By default, rows with step > AFTER_STEP are removed (step AFTER_STEP is kept).
This mirrors UILogger._prune_future_steps(), which runs automatically when
resuming training at a lower step.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove loss-log rows after a given training step."
    )
    parser.add_argument(
        "loss_log",
        type=str,
        help="Path to loss_log.db (e.g. output/my_job/loss_log.db)",
    )
    parser.add_argument(
        "--after-step",
        type=int,
        required=True,
        metavar="STEP",
        help="Keep this step and earlier; delete all rows with step > STEP.",
    )
    parser.add_argument(
        "--include-step",
        action="store_true",
        help="Also delete rows at STEP itself (delete step >= STEP).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without modifying the database.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy the database to loss_log.db.bak.<timestamp> before pruning.",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM after pruning to reclaim disk space.",
    )
    return parser.parse_args()


def count_rows(con: sqlite3.Connection, table: str, where: str, params: tuple) -> int:
    row = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {where};", params).fetchone()
    return int(row[0])


def summarize(con: sqlite3.Connection) -> dict[str, int | None]:
    max_step = con.execute("SELECT MAX(step) FROM steps;").fetchone()[0]
    return {
        "steps": count_rows(con, "steps", "1=1", ()),
        "metrics": count_rows(con, "metrics", "1=1", ()),
        "metric_keys": count_rows(con, "metric_keys", "1=1", ()),
        "max_step": max_step,
    }


def prune_loss_log(
    con: sqlite3.Connection,
    after_step: int,
    include_step: bool,
) -> None:
    threshold = after_step if include_step else after_step
    comparator = ">=" if include_step else ">"

    con.execute("BEGIN;")
    # metrics rows cascade via FK ON DELETE CASCADE
    con.execute(f"DELETE FROM steps WHERE step {comparator} ?;", (threshold,))
    con.execute(
        "DELETE FROM metric_keys "
        "WHERE NOT EXISTS (SELECT 1 FROM metrics WHERE metrics.key = metric_keys.key);"
    )
    con.execute(
        "UPDATE metric_keys "
        "SET last_seen_step = (SELECT MAX(step) FROM metrics WHERE metrics.key = metric_keys.key) "
        f"WHERE last_seen_step {comparator} ?;",
        (threshold,),
    )
    con.execute("COMMIT;")


def main() -> int:
    args = parse_args()
    log_path = os.path.abspath(args.loss_log)

    if not os.path.isfile(log_path):
        print(f"Error: file not found: {log_path}", file=sys.stderr)
        return 1

    if args.after_step < 0:
        print("Error: --after-step must be >= 0", file=sys.stderr)
        return 1

    con = sqlite3.connect(log_path, timeout=30.0)
    try:
        con.execute("PRAGMA foreign_keys=ON;")

        before = summarize(con)
        op = ">=" if args.include_step else ">"
        steps_to_delete = count_rows(con, "steps", f"step {op} ?", (args.after_step,))
        metrics_to_delete = count_rows(con, "metrics", f"step {op} ?", (args.after_step,))

        print(f"Database: {log_path}")
        print(f"Current max step: {before['max_step']}")
        print(f"Prune mode: delete steps {op} {args.after_step}")
        print(
            f"Would remove {steps_to_delete} step row(s) "
            f"and {metrics_to_delete} metric row(s)"
        )

        if steps_to_delete == 0 and metrics_to_delete == 0:
            print("Nothing to do.")
            return 0

        if args.dry_run:
            print("Dry run: no changes made.")
            return 0

        if args.backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{log_path}.bak.{stamp}"
            shutil.copy2(log_path, backup_path)
            print(f"Backup written to {backup_path}")

        prune_loss_log(con, args.after_step, args.include_step)

        if args.vacuum:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            con.execute("VACUUM;")

        after = summarize(con)
        print("Done.")
        print(
            f"Steps: {before['steps']} -> {after['steps']}, "
            f"metrics: {before['metrics']} -> {after['metrics']}, "
            f"max step: {before['max_step']} -> {after['max_step']}"
        )
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
