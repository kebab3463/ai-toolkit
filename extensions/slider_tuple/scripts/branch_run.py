"""Branch a training run from one of its checkpoints with a new learning rate / schedule.

For acting on what a run's previews show: e.g. a slider that is good at step
312 and drifting by 375 can be continued from the 312 checkpoint with a lower
or decaying learning rate. The branch is the parent's config (from
<run>/config.yaml) with:

    config.name                     <run>_from<step> (or --name)
    network.pretrained_lora_path    the checkpoint (weights only)
    train.steps / lr / lr_scheduler --steps / --lr / --scheduler
    train.skip_first_sample         false: the branch's step-0 previews show the
                                    checkpoint it starts from, as a reference

Branch steps count from 0. The optimizer starts fresh (ai-toolkit saves only the
latest optimizer state), which usually costs a few unsettled steps. A
``cosine`` / ``linear`` schedule decays from --lr to 0 over the branch.

Usage (from the repo root):
    .venv/bin/python extensions/slider_tuple/scripts/branch_run.py output/weight_r3b_rank4_512px 312 \\
        --lr 1e-4 --scheduler cosine --steps 250 --register
Writes extensions/slider_tuple/configs/experiments/branches/<name>.yaml; --register adds it to the web UI.
"""

import argparse
import glob
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
BRANCH_DIR = os.path.join(REPO_ROOT, "extensions", "slider_tuple", "configs", "experiments", "branches")
SCHEDULERS = ("constant", "cosine", "linear")


def find_checkpoint(run_dir, checkpoint):
    """(path, step) for a checkpoint given as a path or a step number."""
    if os.path.isfile(checkpoint):
        path = checkpoint
    else:
        step = int(checkpoint)
        matches = [p for p in glob.glob(os.path.join(run_dir, "*.safetensors"))
                   if re.search(rf"_0*{step}\.safetensors$", p)]
        if not matches:
            have = sorted(int(m.group(1)) for p in glob.glob(os.path.join(run_dir, "*.safetensors"))
                          if (m := re.search(r"_(\d+)\.safetensors$", p)))
            raise SystemExit(f"no checkpoint for step {step} in {run_dir} (have steps {have})")
        path = matches[0]
    match = re.search(r"_(\d+)\.safetensors$", path)
    return os.path.abspath(path), int(match.group(1)) if match else None


def branch_config(parent: dict, checkpoint: str, name: str, lr: float, steps: int, scheduler: str) -> dict:
    import copy

    config = copy.deepcopy(parent)
    config["config"]["name"] = name
    process = config["config"]["process"][0]
    process.pop("sqlite_db_path", None)  # set by the UI per launch
    process.setdefault("network", {})["pretrained_lora_path"] = checkpoint
    train = process.setdefault("train", {})
    train.pop("stages", None)
    train.update(steps=steps, lr=lr, lr_scheduler=scheduler, skip_first_sample=False)
    train.pop("lr_scheduler_params", None)
    if scheduler == "linear":  # LinearLR: from lr down to 0 over the branch
        train["lr_scheduler_params"] = {"start_factor": 1.0, "end_factor": 0.0}
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="the parent run's folder, e.g. output/<run>")
    p.add_argument("checkpoint", help="a step number or a checkpoint path")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--scheduler", choices=SCHEDULERS, default="constant")
    p.add_argument("--name")
    p.add_argument("--out", help=f"config to write (default: {os.path.relpath(BRANCH_DIR, REPO_ROOT)}/<name>.yaml)")
    p.add_argument("--register", action="store_true", help="also add it to the web UI (scripts/ui_jobs.py)")
    args = p.parse_args(argv)

    import yaml

    from toolkit.config import fixed_loader

    parent_path = os.path.join(args.run_dir, "config.yaml")
    with open(parent_path, encoding="utf-8") as f:
        parent = yaml.load(f, Loader=fixed_loader)
    checkpoint, step = find_checkpoint(args.run_dir, args.checkpoint)
    parent_name = parent["config"]["name"]
    name = args.name or f"{parent_name}_from{step if step is not None else 'ckpt'}"
    config = branch_config(parent, checkpoint, name, args.lr, args.steps, args.scheduler)

    out = args.out or os.path.join(BRANCH_DIR, f"{name}.yaml")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    header = (f"# Branch of {parent_name} from its step-{step} checkpoint (scripts/branch_run.py):\n"
              f"# {args.steps} steps at lr {args.lr:g}, {args.scheduler} schedule; steps count from 0, fresh optimizer.\n"
              f"# Step-0 previews show the starting checkpoint.\n"
              f"#   .venv/bin/python run.py {os.path.relpath(out, REPO_ROOT)}\n")
    with open(out, "w", encoding="utf-8") as f:
        f.write(header + "---\n" + yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=100))
    print(f"branch {name}: {checkpoint} -> {out}")
    if args.register:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "ui_jobs.py"), "add", out], check=True)
    return out


if __name__ == "__main__":
    main()
