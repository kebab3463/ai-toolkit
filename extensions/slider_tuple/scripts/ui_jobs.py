"""Register slider_tuple experiment configs as jobs in the ai-toolkit web UI.

The web UI starts, stops and tracks jobs stored in its database (aitk_db.db,
table Job). This adds a job per config so it can be started from the browser
(Jobs -> the job -> start), shows the run's samples as a grid (one row per
sample step, one column per prompt), its loss graphs, and its status.

    add CONFIG...       one job per config, named like config.name. If the run
                        already finished in the UI's training folder
                        (<name>/<name>.safetensors), the job is added as
                        completed at its last step, so past runs and their
                        samples show up too; otherwise it is added stopped,
                        ready to start.
    list                the UI's jobs (name, status, step)

The job config is the YAML config with two UI conventions applied:
training_folder set to the UI's training folder, and sample.prompts written as
sample.samples ([{prompt: ...}]), the form the UI reads to lay out the grid.
Python reads both forms. An existing job of the same name is left alone unless
--update (refused while it is running or queued).

Usage (from the repo root; the UI does not need to be running):
    .venv/bin/python extensions/slider_tuple/scripts/ui_jobs.py add \\
        extensions/slider_tuple/configs/experiments/weight_r4_symmetry.yaml
    .venv/bin/python extensions/slider_tuple/scripts/ui_jobs.py add extensions/slider_tuple/configs/experiments/*.yaml
    .venv/bin/python extensions/slider_tuple/scripts/ui_jobs.py list
"""

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
import time
import uuid

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_config(path):
    import yaml

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) if path.endswith((".yaml", ".yml")) else json.load(f)
    if not isinstance(config, dict) or "config" not in config or not config["config"].get("process"):
        raise SystemExit(f"{path}: not a job config (needs config.process)")
    if not config["config"].get("name"):
        raise SystemExit(f"{path}: config.name is required (it is the UI job name)")
    return config


def ui_job_config(config, training_folder):
    """The config as the UI stores it: training_folder from the UI settings, samples instead of prompts."""
    config = json.loads(json.dumps(config))  # deep copy
    process = config["config"]["process"][0]
    process["training_folder"] = training_folder
    sample = process.get("sample")
    if sample and "samples" not in sample:
        sample["samples"] = [{"prompt": p} for p in sample.pop("prompts", [])]
    return config


def last_step(run_dir, name):
    """Highest step among <name>_<step>.safetensors and samples/*__<step>_<i>.*, else None."""
    steps = []
    for path in glob.glob(os.path.join(run_dir, f"{glob.escape(name)}_*.safetensors")):
        match = re.search(r"_(\d+)\.safetensors$", path)
        if match:
            steps.append(int(match.group(1)))
    for path in glob.glob(os.path.join(run_dir, "samples", "*")):
        match = re.search(r"__(\d+)_\d+\.\w+$", os.path.basename(path))
        if match:
            steps.append(int(match.group(1)))
    return max(steps) if steps else None


def connect(db_path):
    if not os.path.exists(db_path):
        raise SystemExit(f"no UI database at {db_path} (start the UI once, or pass --db)")
    return sqlite3.connect(db_path)


def setting(db, key, default=None):
    row = db.execute("SELECT value FROM Settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row and row[0] else default


def add(db, config_path, gpu_ids="0", update=False, now_ms=None):
    """Insert (or with update, refresh) the job for one config; returns a status line."""
    now_ms = now_ms or int(time.time() * 1000)
    training_folder = setting(db, "TRAINING_FOLDER", os.path.join(REPO_ROOT, "output"))
    config = load_config(config_path)
    name = config["config"]["name"]
    job_config = ui_job_config(config, training_folder)
    total = job_config["config"]["process"][0].get("train", {}).get("steps")
    run_dir = os.path.join(training_folder, name)
    finished = os.path.exists(os.path.join(run_dir, f"{name}.safetensors"))
    step = (last_step(run_dir, name) or total or 0) if finished else 0
    status = "completed" if finished else "stopped"
    info = "registered from " + os.path.relpath(config_path, REPO_ROOT)

    existing = db.execute("SELECT id, status FROM Job WHERE name = ?", (name,)).fetchone()
    if existing:
        if not update:
            return f"{name}: already a UI job ({existing[1]}); --update to replace its config"
        if existing[1] in ("running", "queued"):
            return f"{name}: is {existing[1]} in the UI; not changed"
        db.execute("UPDATE Job SET job_config = ?, total_steps = ?, status = ?, step = ?, info = ?, updated_at = ? "
                   "WHERE id = ?", (json.dumps(job_config), total, status, step, info, now_ms, existing[0]))
        return f"{name}: updated ({status}, step {step})"
    db.execute(
        "INSERT INTO Job (id, name, gpu_ids, job_config, created_at, updated_at, status, stop, return_to_queue, "
        "step, total_steps, info, speed_string, queue_position, job_type, save_now, sample_now) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, '', 0, 'train', 0, 0)",
        (str(uuid.uuid4()), name, gpu_ids, json.dumps(job_config), now_ms, now_ms, status, step, total, info),
    )
    return f"{name}: added ({status}" + (f", step {step}" if finished else ", start it from the UI") + ")"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=os.path.join(REPO_ROOT, "aitk_db.db"))
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add")
    a.add_argument("configs", nargs="+")
    a.add_argument("--gpu", default="0", help="the UI's gpu_ids for the job")
    a.add_argument("--update", action="store_true", help="replace the config of an existing (not running) job")
    sub.add_parser("list")
    args = p.parse_args(argv)

    db = connect(args.db)
    try:
        if args.command == "add":
            lines = [add(db, path, args.gpu, args.update) for path in args.configs]
            db.commit()
            print("\n".join(lines))
            return lines
        rows = db.execute("SELECT name, status, step, total_steps FROM Job ORDER BY created_at").fetchall()
        for name, status, step, total in rows:
            print(f"{name:45s} {status:10s} {step}/{total or '?'}")
        return rows
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
