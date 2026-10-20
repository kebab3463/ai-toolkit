"""scripts/split_holdout.py: deterministic held-out split with rebased paths."""

import importlib.util
import json
import os

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "split_holdout.py")


def load_script():
    spec = importlib.util.spec_from_file_location("split_holdout", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_choose_held_out_is_deterministic_and_sized():
    script = load_script()
    ids = [f"t{i:02d}" for i in range(20)]
    held = script.choose_held_out(ids, 0.1, seed=3)
    assert len(held) == 2 and held == script.choose_held_out(list(reversed(ids)), 0.1, seed=3)
    assert len(script.choose_held_out(ids[:3], 0.1)) == 1   # at least one ...
    assert script.choose_held_out(ids[:1], 0.1) == set()    # ... but never every tuple
    assert script.choose_held_out(ids, 0.1, forced=["t05"]) == {"t05"}
    with pytest.raises(ValueError, match="unknown"):
        script.choose_held_out(ids, 0.1, forced=["nope"])


def test_split_writes_both_manifests_with_rebased_paths(tmp_path):
    script = load_script()
    src = tmp_path / "data"
    src.mkdir()
    entries = [{"id": f"t{i}", "caption": "a photo of a woman", "mask": f"masks/t{i}.png",
                "members": [{"image": f"heavy/t{i}.png", "level": 1}, {"image": f"thin/t{i}.png", "level": -1}]}
               for i in range(10)]
    (src / "tuples.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    out = tmp_path / "split"
    script.main([str(src / "tuples.jsonl"), "--out-dir", str(out), "--ids", "t3"])
    train = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    held = [json.loads(line) for line in (out / "held_out.jsonl").read_text().splitlines()]
    assert [e["id"] for e in held] == ["t3"] and len(train) == 9
    assert held[0]["members"][0]["image"] == os.path.join("..", "data", "heavy", "t3.png")
    assert held[0]["mask"] == os.path.join("..", "data", "masks", "t3.png")
