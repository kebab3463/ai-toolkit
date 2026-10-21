"""scripts/ui_jobs.py against a database with the web UI's Job / Settings tables."""

import importlib.util
import json
import os
import sqlite3

import yaml

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "ui_jobs.py")


def load_script():
    spec = importlib.util.spec_from_file_location("ui_jobs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_db(path, training_folder):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE Settings (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL UNIQUE, value TEXT NOT NULL);
        CREATE TABLE Job (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, gpu_ids TEXT NOT NULL, job_config TEXT NOT NULL,
            created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, status TEXT NOT NULL DEFAULT 'stopped',
            stop BOOLEAN NOT NULL DEFAULT false, return_to_queue BOOLEAN NOT NULL DEFAULT false,
            step INTEGER NOT NULL DEFAULT 0, total_steps INTEGER, info TEXT NOT NULL DEFAULT '',
            speed_string TEXT NOT NULL DEFAULT '', queue_position INTEGER NOT NULL DEFAULT 0, pid INTEGER,
            job_type TEXT NOT NULL DEFAULT 'train', job_ref TEXT, save_now BOOLEAN NOT NULL DEFAULT false,
            sample_now BOOLEAN NOT NULL DEFAULT false);
    """)
    db.execute("INSERT INTO Settings (key, value) VALUES ('TRAINING_FOLDER', ?)", (training_folder,))
    db.commit()
    db.close()


def write_config(path, name):
    config = {"job": "extension", "config": {"name": name, "process": [{
        "type": "slider_tuple_trainer", "training_folder": "output", "train": {"steps": 1000},
        "sample": {"sample_every": 125, "prompts": ["a --m -1", "a --m 0", "a --m 1"]}}]}}
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_add_new_and_finished_runs_and_update(tmp_path):
    script = load_script()
    out, db_path = tmp_path / "output", str(tmp_path / "aitk_db.db")
    make_db(db_path, str(out))
    fresh = write_config(tmp_path / "fresh.yaml", "fresh_run")
    done = write_config(tmp_path / "done.yaml", "done_run")
    (out / "done_run" / "samples").mkdir(parents=True)
    (out / "done_run" / "done_run.safetensors").write_bytes(b"")
    (out / "done_run" / "done_run_000000750.safetensors").write_bytes(b"")
    (out / "done_run" / "samples" / "1792__000001000_2.png").write_bytes(b"")

    lines = script.main(["--db", db_path, "add", fresh, done])
    assert "fresh_run: added (stopped" in lines[0] and "done_run: added (completed, step 1000)" in lines[1]
    db = sqlite3.connect(db_path)
    rows = {r[0]: r[1:] for r in db.execute("SELECT name, status, step, total_steps, job_config, gpu_ids FROM Job")}
    assert rows["fresh_run"][:3] == ("stopped", 0, 1000) and rows["done_run"][:2] == ("completed", 1000)
    stored = json.loads(rows["fresh_run"][3])["config"]["process"][0]
    assert stored["training_folder"] == str(out)
    assert stored["sample"]["samples"] == [{"prompt": "a --m -1"}, {"prompt": "a --m 0"}, {"prompt": "a --m 1"}]
    assert "prompts" not in stored["sample"]

    assert "already a UI job" in script.main(["--db", db_path, "add", fresh])[0]
    db.execute("UPDATE Job SET status = 'running' WHERE name = 'fresh_run'")
    db.commit()
    assert "not changed" in script.main(["--db", db_path, "add", "--update", fresh])[0]
    assert "updated (completed" in script.main(["--db", db_path, "add", "--update", done])[0]
    assert len(script.main(["--db", db_path, "list"])) == 2
