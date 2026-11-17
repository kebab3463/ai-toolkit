"""scripts/levels_from_csv.py: measured levels from a CSV of per-member values."""

import importlib.util
import json
import math
import os

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "levels_from_csv.py")


def load_script():
    spec = importlib.util.spec_from_file_location("levels_from_csv", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_dataset(tmp_path):
    data = tmp_path / "data"
    entries = []
    for tuple_id in ("a", "b", "c"):
        for folder in ("heavy", "thin"):
            (data / folder).mkdir(parents=True, exist_ok=True)
            (data / folder / f"{tuple_id}.png").write_bytes(b"")
        entries.append({"id": tuple_id, "caption": "a photo", "members": [
            {"image": f"heavy/{tuple_id}.png", "level": 1.0, "role": "edit"},
            {"image": f"thin/{tuple_id}.png", "level": -1.0, "role": "edit"}]})
    (data / "tuples.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    # spaces after commas, as in hand-written CSVs; 'c' has no row
    (data / "m.csv").write_text("stem, neg, pos\na, 100, 200\nb, 100, 101\n")
    return data


def read(path):
    return [json.loads(line) for line in open(path)]


def test_log_levels_and_rebased_paths(tmp_path, capsys):
    data = fake_dataset(tmp_path)
    out = tmp_path / "elsewhere" / "tuples.jsonl"
    out.parent.mkdir()
    load_script().main([str(data / "tuples.jsonl"), str(data / "m.csv"), "--column", "neg=thin",
                        "--column", "pos=heavy", "--out", str(out)])
    entries = read(out)
    assert [e["id"] for e in entries] == ["a", "b"]
    heavy, thin = entries[0]["members"]
    assert heavy["level"] - thin["level"] == pytest.approx(math.log(2), abs=1e-5)
    assert heavy["image"] == os.path.join("..", "data", "heavy", "a.png")
    assert "skipped c: no CSV row" in capsys.readouterr().out


def test_min_ratio_and_linear(tmp_path, capsys):
    data = fake_dataset(tmp_path)
    out = tmp_path / "t.jsonl"
    load_script().main([str(data / "tuples.jsonl"), str(data / "m.csv"), "--column", "neg=thin",
                        "--column", "pos=heavy", "--out", str(out), "--transform", "linear", "--min-ratio", "1.05"])
    entries = read(out)
    assert [e["id"] for e in entries] == ["a"]
    assert [m["level"] for m in entries[0]["members"]] == [200.0, 100.0]
    assert "skipped b: ratio 1.010" in capsys.readouterr().out


def test_unmapped_folder(tmp_path):
    data = fake_dataset(tmp_path)
    with pytest.raises(SystemExit, match="no tuples measured"):
        load_script().main([str(data / "tuples.jsonl"), str(data / "m.csv"), "--out", str(tmp_path / "t.jsonl")])
