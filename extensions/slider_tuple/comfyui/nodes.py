"""ComfyUI nodes for the slider_tuple body-width measure (body_measure.py).

The nodes run the same code as measure_tuples.py / slider_sweep.py --measure
body, loaded by file path from this ai-toolkit checkout (no ai-toolkit
package import, nothing installed into ComfyUI's environment), and reuse its
Sapiens2 weights in <ai-toolkit>/models/sapiens2. Needs torch, torchvision,
numpy, opencv-python, safetensors and huggingface_hub, which ComfyUI already has.
"""

import contextlib
import importlib.util
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.realpath(__file__))
EXTENSION = os.path.dirname(HERE)  # extensions/slider_tuple
REPO_ROOT = os.path.dirname(os.path.dirname(EXTENSION))

_MODULES = {}
_MODELS = {}  # device -> (matte_fn, keypoints_fn)


def _load(name, path):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[name] = module
    return _MODULES[name]


@contextlib.contextmanager
def _toolkit_stubs():
    """Just enough of ai-toolkit's ``toolkit`` package for sapiens2.py and
    tuple_masks.load_mask_models, present only while they import."""
    added = []

    def put(name, module):
        if name not in sys.modules:
            sys.modules[name] = module
            added.append(name)

    try:
        for name in ("toolkit", "toolkit.models"):
            stub = types.ModuleType(name)
            stub.__path__ = []
            put(name, stub)
        paths = types.ModuleType("toolkit.paths")
        paths.MODELS_PATH = os.path.join(REPO_ROOT, "models")
        put("toolkit.paths", paths)
        if "toolkit.models.sapiens2" not in sys.modules:
            put("toolkit.models.sapiens2",
                _load("slider_tuple_sapiens2", os.path.join(REPO_ROOT, "toolkit", "models", "sapiens2.py")))
        yield
    finally:
        for name in added:
            del sys.modules[name]


def body_measure():
    return _load("slider_tuple_body_measure", os.path.join(EXTENSION, "body_measure.py"))


def _device():
    try:
        import comfy.model_management
        return str(comfy.model_management.get_torch_device())
    except ImportError:
        return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_models(device=None):
    device = device or _device()
    if device not in _MODELS:
        tuple_masks = _load("slider_tuple_tuple_masks", os.path.join(EXTENSION, "tuple_masks.py"))
        with _toolkit_stubs():
            _MODELS[device] = tuple_masks.load_mask_models(device)
    return _MODELS[device]


def unload_models():
    _MODELS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_pil_list(images):
    """ComfyUI IMAGE [B, H, W, C] float 0-1 -> list of PIL RGB images."""
    from PIL import Image

    arrays = (images.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    return [Image.fromarray(a[..., :3]) for a in arrays]


def to_image_batch(arrays):
    """List of HxWx3 uint8 arrays (padded to one size) -> IMAGE [B, H, W, C]."""
    h = max(a.shape[0] for a in arrays)
    w = max(a.shape[1] for a in arrays)
    out = np.zeros((len(arrays), h, w, 3), np.float32)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0], :a.shape[1]] = a / 255.0
    return torch.from_numpy(out)


def measure(img, models, min_score):
    matte_fn, keypoints_fn = models
    return body_measure().measure_body(matte_fn(img), keypoints_fn(img), min_score=min_score)


def side_by_side(arrays, height=768):
    import cv2

    scaled = [cv2.resize(a, (round(a.shape[1] * height / a.shape[0]), height), interpolation=cv2.INTER_AREA)
              for a in arrays]
    return np.concatenate(scaled, axis=1)


def judge(ref, edit, direction, min_change):
    """(change, relative, verdict) for one reference / edit pair of BodyMeasurements."""
    if ref.score is None or edit.score is None:
        which = "reference" if ref.score is None else "edit"
        return 0.0, 0.0, f"UNMEASURABLE ({which}: {(ref if ref.score is None else edit).reason})"
    change = edit.score - ref.score
    relative = change / ref.score
    wanted = -1.0 if direction == "thinner" else 1.0
    signed = change * wanted
    text = f"{change:+.3f} ({relative * 100:+.1f}%), reference {ref.score:.3f} -> edit {edit.score:.3f}"
    if signed <= 0:
        return change, relative, f"WRONG DIRECTION: {text}"
    if signed < min_change:
        return change, relative, f"TOO WEAK (< {min_change:g}): {text}"
    return change, relative, f"OK: {text}"


class SliderBodyWidth:
    """Body width in torso lengths for each image (the slider_tuple weight measure)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "keypoint_score": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1,
                                             "tooltip": "minimum Keypoint R-CNN score for shoulders and hips"}),
                "keep_models_loaded": ("BOOLEAN", {"default": True,
                                                   "tooltip": "keep Sapiens2 + Keypoint R-CNN (~3 GB) in memory"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("score", "report", "preview")
    FUNCTION = "run"
    CATEGORY = "slider_tuple"
    DESCRIPTION = ("Body width over the torso-to-thigh band divided by torso length (Sapiens2 matte + "
                   "Keypoint R-CNN). Comparable between versions of one photo, not across different photos. "
                   "score is the first image's (0 if unmeasurable); report lists all.")

    def run(self, image, keypoint_score, keep_models_loaded, models=None):
        loaded = models or get_models()
        lines, previews, first = [], [], None
        try:
            for i, img in enumerate(to_pil_list(image)):
                m = measure(img, loaded, keypoint_score)
                if i == 0:
                    first = m.score
                lines.append(f"[{i}] " + (f"{m.score:.4f}" if m.score is not None else f"unmeasurable: {m.reason}"))
                previews.append(body_measure().draw_measurement(np.asarray(img), m))
        finally:
            if not keep_models_loaded and models is None:
                unload_models()
        return (float(first or 0.0), "\n".join(lines), to_image_batch(previews))


class SliderEditStrength:
    """How much an edit changed body width against its reference, with a pass / redo verdict."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE", {"tooltip": "the pass-through edit (or the original); one image, "
                                                   "or one per edited image"}),
                "edited": ("IMAGE",),
                "direction": (["thinner", "heavier"], {"default": "thinner"}),
                "min_change": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.005,
                                         "tooltip": "smallest useful change in body width (torso lengths). "
                                                    "The measure's per-image noise is ~0.03; the first weight "
                                                    "data's median edit was 0.11"}),
                "keypoint_score": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "keep_models_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "BOOLEAN", "STRING", "IMAGE")
    RETURN_NAMES = ("change", "relative_change", "ok", "verdict", "preview")
    FUNCTION = "run"
    CATEGORY = "slider_tuple"
    DESCRIPTION = ("Measures reference and edit with the slider_tuple body-width measure. change = edit - "
                   "reference (negative = thinner), relative_change as a fraction of the reference. "
                   "ok is True when the change goes in the chosen direction by at least min_change. "
                   "With batches the outputs are for the first pair; verdict lists every pair.")

    def run(self, reference, edited, direction, min_change, keypoint_score, keep_models_loaded, models=None):
        refs, edits = to_pil_list(reference), to_pil_list(edited)
        if len(refs) not in (1, len(edits)):
            raise ValueError(f"reference batch ({len(refs)}) must be 1 or match edited ({len(edits)})")
        loaded = models or get_models()
        bm = body_measure()
        results, lines, previews = [], [], []
        try:
            ref_cache = {}
            for i, edit in enumerate(edits):
                r = 0 if len(refs) == 1 else i
                if r not in ref_cache:
                    ref_cache[r] = measure(refs[r], loaded, keypoint_score)
                ref_m, edit_m = ref_cache[r], measure(edit, loaded, keypoint_score)
                change, relative, verdict = judge(ref_m, edit_m, direction, min_change)
                results.append((change, relative, verdict.startswith("OK")))
                lines.append(f"[{i}] {verdict}")
                previews.append(side_by_side([bm.draw_measurement(np.asarray(refs[r]), ref_m),
                                              bm.draw_measurement(np.asarray(edit), edit_m)]))
        finally:
            if not keep_models_loaded and models is None:
                unload_models()
        change, relative, ok = results[0]
        return (float(change), float(relative), bool(ok), "\n".join(lines), to_image_batch(previews))


NODE_CLASS_MAPPINGS = {
    "SliderBodyWidth": SliderBodyWidth,
    "SliderEditStrength": SliderEditStrength,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SliderBodyWidth": "Body Width Score (slider_tuple)",
    "SliderEditStrength": "Slider Edit Strength (body width)",
}
