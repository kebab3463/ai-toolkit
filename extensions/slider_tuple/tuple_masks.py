"""Build one loss mask per tuple (see NOTES.md, "One mask per tuple").

    mask = background + (body_weight - background) * union_of_member_bodies,  head -> 0

Bodies come from person mattes (H x W in [0, 1]); the head region from COCO
keypoints (nose, eyes, ears, shoulders) as an ellipse, or from the user's own
head masks (e.g. the paste-back region of their editing pipeline). Pure
functions here; the models are wired up in scripts/make_tuple_masks.py.
"""

from typing import Optional, Sequence

import numpy as np

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 1, 2, 3, 4, 5, 6


def union(maps: Sequence[np.ndarray]) -> np.ndarray:
    return np.maximum.reduce([np.asarray(m, dtype=np.float32) for m in maps])


def head_ellipse(keypoints: np.ndarray, size, min_score: float = 1.0, margin: float = 0.15) -> Optional[np.ndarray]:
    """Head region (H x W, 0/1) from one person's COCO keypoints, rows (x, y, score).

    The head width is the largest of: 1.4 x ear distance (hair), 3 x eye
    distance, 0.45 x shoulder width, and the face-to-shoulder-midpoint distance
    (about a head height) / 1.35. Taking the largest matters: in profile and
    back views both ears (and eyes) land on nearly the same point, which alone
    would shrink the head to a few pixels. Height is 1.35 x width, centered a
    little above the face keypoints; ``margin`` enlarges both axes. Returns None
    if no face keypoint is visible or nothing sizes the head."""
    width, height = size
    kp = np.asarray(keypoints, dtype=np.float64)
    visible = kp[:, 2] >= min_score

    def dist(a, b):
        return float(np.hypot(*(kp[a, :2] - kp[b, :2]))) if visible[a] and visible[b] else None

    face = [i for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if visible[i]]
    if not face:
        return None
    cx, cy = kp[face, 0].mean(), kp[face, 1].mean()
    estimates = []
    for value, factor in ((dist(L_EAR, R_EAR), 1.4), (dist(L_EYE, R_EYE), 3.0), (dist(L_SHOULDER, R_SHOULDER), 0.45)):
        if value:
            estimates.append(value * factor)
    shoulders = [i for i in (L_SHOULDER, R_SHOULDER) if visible[i]]
    if shoulders:
        neck = float(np.hypot(cx - kp[shoulders, 0].mean(), cy - kp[shoulders, 1].mean()))
        estimates.append(neck / 1.35)
    if not estimates:
        return None
    head_w = min(max(estimates), 0.6 * min(width, height))
    head_h = 1.35 * head_w
    cy -= 0.05 * head_h
    rx, ry = head_w / 2 * (1 + margin), head_h / 2 * (1 + margin)
    ys, xs = np.mgrid[0:height, 0:width]
    return ((((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2) <= 1.0).astype(np.float32)


def compose(body: np.ndarray, head: np.ndarray, background: float = 0.2, body_weight: float = 1.0,
            head_weight: float = 0.0) -> np.ndarray:
    """Loss mask in [0, 1] (H x W): background + (body_weight - background) * body, head -> head_weight.

    head_weight 0 (default) leaves the face out of the loss. Above 0 it trains the pasted-back
    head, which is the same in every member, so any change the LoRA makes to a face costs loss
    ("keep the face"). Keep it below background if the masks may be reweighted later:
    decompose() (reweight_masks.py) tells the head from the rest by that."""
    if not 0 <= background <= body_weight <= 1:
        raise ValueError("need 0 <= background <= body_weight <= 1")
    if not 0 <= head_weight <= 1:
        raise ValueError("need 0 <= head_weight <= 1")
    mask = background + (body_weight - background) * np.clip(body, 0.0, 1.0)
    mask[np.asarray(head) >= 0.5] = head_weight
    return mask.astype(np.float32)


def decompose(mask: np.ndarray, background: float = 0.2, body_weight: float = 1.0,
              head_weight: float = 0.0) -> tuple:
    """(body, head) back from a compose() mask in [0, 1] made with these weights; head_weight must
    be below background. Inverse of compose up to 8-bit rounding."""
    if not head_weight < background < body_weight:
        raise ValueError("decompose needs head_weight < background < body_weight")
    tol = 0.5 / 255
    head = (np.abs(mask - head_weight) <= tol).astype(np.float32)
    body = np.clip((mask - background) / (body_weight - background), 0.0, 1.0)
    body[head > 0] = 0.0
    return body.astype(np.float32), head


def to_png_array(mask: np.ndarray) -> np.ndarray:
    return np.clip(np.round(mask * 255.0), 0, 255).astype(np.uint8)


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Preview: the image darkened where the mask is low and tinted red on the head (mask 0)."""
    img = np.asarray(image, dtype=np.float32)
    shown = img * (0.25 + 0.75 * mask[..., None])
    head = mask <= 0.0
    shown[head] = 0.5 * shown[head] + 0.5 * np.array([255.0, 0.0, 0.0])
    return np.clip(shown, 0, 255).astype(np.uint8)


def load_mask_models(device: str = "cuda:0"):
    """(matte_fn, keypoints_fn) for PIL RGB images.

    matte_fn: Sapiens2 person matte, H x W float in [0, 1]
    (facebook/sapiens2-matting-1b, downloaded into models/sapiens2 on first use).
    keypoints_fn: the most confident person's 17 COCO keypoints as (x, y, score)
    rows, or None (torchvision Keypoint R-CNN, weights downloaded on first use)."""
    import torch
    import torchvision
    from torchvision.transforms.functional import to_tensor

    from toolkit.models.sapiens2 import Sapiens2Matting

    matting = Sapiens2Matting.from_pretrained(device=device, dtype=torch.float16)
    weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    detector = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights).to(device).eval()

    @torch.no_grad()
    def matte_fn(img):
        return np.asarray(matting(img).convert("L"), dtype=np.float32) / 255.0

    @torch.no_grad()
    def keypoints_fn(img):
        out = detector([to_tensor(img).to(device)])[0]
        if len(out["scores"]) == 0:
            return None
        best = int(out["scores"].argmax())  # the tuple's one person: the most confident detection
        kps = out["keypoints"][best, :, :2].cpu().numpy()
        scores = out["keypoints_scores"][best].cpu().numpy()
        return np.concatenate([kps, scores[:, None]], axis=1)

    return matte_fn, keypoints_fn
