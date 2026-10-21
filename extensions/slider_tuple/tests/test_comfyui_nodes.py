"""comfyui/nodes.py with stub models (no ComfyUI import needed)."""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from .test_body_measure import body, keypoints

NODES = os.path.join(os.path.dirname(__file__), "..", "comfyui", "nodes.py")


def load_nodes():
    spec = importlib.util.spec_from_file_location("slider_comfy_nodes", NODES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STUB = (lambda img: (np.asarray(img.convert("L"), np.float32) > 127).astype(np.float32), lambda img: keypoints())


def batch(*widths):
    """IMAGE [B, H, W, 3] with one synthetic body per width."""
    return torch.from_numpy(np.stack([np.repeat(body(w)[..., None], 3, axis=2) for w in widths]))


def test_edit_strength_verdicts():
    nodes = load_nodes()
    node = nodes.SliderEditStrength()
    # torso 80 px: reference width 64 -> 0.8; edits 48 (0.6), 60 (0.75), 70 (0.875)
    change, rel, ok, verdict, preview = node.run(batch(64), batch(48, 60, 70), "thinner", 0.08, 1.0, True, models=STUB)
    assert change == pytest.approx(-0.2) and rel == pytest.approx(-0.25) and ok
    lines = verdict.splitlines()
    assert lines[0].startswith("[0] OK") and "TOO WEAK" in lines[1] and "WRONG DIRECTION" in lines[2]
    assert preview.shape[0] == 3 and preview.shape[-1] == 3 and float(preview.max()) <= 1.0
    with pytest.raises(ValueError, match="reference batch"):
        node.run(batch(64, 64), batch(48, 60, 70), "thinner", 0.08, 1.0, True, models=STUB)


def test_unmeasurable_and_body_width():
    nodes = load_nodes()
    blank = torch.zeros(1, 240, 200, 3)
    no_person = (STUB[0], lambda img: None)
    _, _, ok, verdict, _ = nodes.SliderEditStrength().run(blank, blank, "thinner", 0.08, 1.0, True, models=no_person)
    assert not ok and verdict.startswith("[0] UNMEASURABLE")
    score, report, preview = nodes.SliderBodyWidth().run(batch(48, 64), 1.0, True, models=STUB)
    assert score == pytest.approx(0.6) and report.splitlines()[1] == "[1] 0.8000"


def test_toolkit_stubs_load_sapiens2_and_clean_up():
    nodes = load_nodes()
    before = {k for k in sys.modules if k == "toolkit" or k.startswith("toolkit.")}
    with nodes._toolkit_stubs():
        from toolkit.models.sapiens2 import Sapiens2Matting  # noqa: F401  (resolved through the stubs or the real package)
        from toolkit.paths import MODELS_PATH
        assert MODELS_PATH.endswith("models")
    after = {k for k in sys.modules if k == "toolkit" or k.startswith("toolkit.")}
    assert after == before
