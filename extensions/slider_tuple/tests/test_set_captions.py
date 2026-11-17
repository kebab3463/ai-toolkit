"""scripts/set_captions.py: captions into an existing manifest."""

import importlib.util
import json
import os

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "set_captions.py")


def load_script():
    spec = importlib.util.spec_from_file_location("set_captions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_dataset(tmp_path):
    data = tmp_path / "data"
    (data / "captions").mkdir(parents=True)
    entries = [{"id": i, "caption": "a photo", "members": [
        {"image": f"heavy/{i}.png", "level": 0.3, "role": "edit"},
        {"image": f"thin/{i}.png", "level": 0.1, "role": "edit"}]} for i in ("a", "b", "c")]
    (data / "tuples.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (data / "captions" / "a.json").write_text(json.dumps({"caption_long": " long a ", "caption_short": "short a"}))
    (data / "captions" / "b.txt").write_text("text b\n")
    return data


def read(path):
    return [json.loads(line) for line in open(path)]


def test_in_place_keeps_levels(tmp_path, capsys):
    data = fake_dataset(tmp_path)
    load_script().main([str(data / "tuples.jsonl"), str(data / "captions")])
    entries = read(data / "tuples.jsonl")
    assert [e["caption"] for e in entries] == ["long a", "text b", "a photo"]
    assert entries[0]["members"][0] == {"image": "heavy/a.png", "level": 0.3, "role": "edit"}
    assert "kept theirs: ['c']" in capsys.readouterr().out


def test_key_and_out(tmp_path):
    data = fake_dataset(tmp_path)
    out = tmp_path / "other" / "t.jsonl"
    out.parent.mkdir()
    load_script().main([str(data / "tuples.jsonl"), str(data / "captions"), "--caption-key", "caption_short",
                        "--out", str(out)])
    entries = read(out)
    assert entries[0]["caption"] == "short a"
    assert entries[0]["members"][0]["image"] == os.path.join("..", "data", "heavy", "a.png")
    assert read(data / "tuples.jsonl")[0]["caption"] == "a photo"


def test_bad_key(tmp_path):
    data = fake_dataset(tmp_path)
    before = (data / "tuples.jsonl").read_text()
    with pytest.raises(SystemExit, match="caption-key"):
        load_script().main([str(data / "tuples.jsonl"), str(data / "captions"), "--caption-key", "nope"])
    assert (data / "tuples.jsonl").read_text() == before
