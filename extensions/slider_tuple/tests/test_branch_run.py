"""scripts/branch_run.py: a branch config from a run's checkpoint."""

import importlib.util
import os

import pytest
import yaml

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "branch_run.py")


def load_script():
    spec = importlib.util.spec_from_file_location("branch_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_run(tmp_path):
    run = tmp_path / "output" / "myrun"
    run.mkdir(parents=True)
    config = {"job": "extension", "config": {"name": "myrun", "process": [{
        "type": "slider_tuple_trainer", "sqlite_db_path": "./aitk_db.db", "network": {"type": "lora", "linear": 4},
        "train": {"steps": 500, "lr": 3e-4, "skip_first_sample": True,
                  "stages": [{"steps": 250, "lr": 3e-4}, {"steps": 250, "lr": 1e-4}]},
        "sample": {"samples": [{"prompt": "p", "seed": 1, "network_multiplier": 0.0}]}}]}}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    for step in (250, 312):
        (run / f"myrun_{step:09d}.safetensors").write_bytes(b"")
    return run


@pytest.mark.parametrize("scheduler", ["constant", "cosine", "linear"])
def test_branch_from_step(tmp_path, scheduler):
    script = load_script()
    run = fake_run(tmp_path)
    out = script.main([str(run), "312", "--lr", "1e-4", "--steps", "200", "--scheduler", scheduler,
                       "--out", str(tmp_path / "b.yaml")])
    text = open(out).read()
    assert text.startswith("# Branch of myrun from its step-312 checkpoint")
    config = yaml.safe_load(text)
    process = config["config"]["process"][0]
    assert config["config"]["name"] == "myrun_from312"
    assert process["network"]["pretrained_lora_path"].endswith("myrun_000000312.safetensors")
    train = process["train"]
    assert (train["steps"], train["lr"], train["lr_scheduler"], train["skip_first_sample"]) == (200, 1e-4, scheduler, False)
    assert "stages" not in train and "sqlite_db_path" not in process
    assert (train.get("lr_scheduler_params") == {"start_factor": 1.0, "end_factor": 0.0}) == (scheduler == "linear")
    assert process["sample"]["samples"]  # the parent's previews (the sweep grid) carry over


def test_missing_checkpoint_lists_the_steps(tmp_path):
    script = load_script()
    with pytest.raises(SystemExit, match=r"have steps \[250, 312\]"):
        script.main([str(fake_run(tmp_path)), "375", "--lr", "1e-4", "--out", str(tmp_path / "b.yaml")])
