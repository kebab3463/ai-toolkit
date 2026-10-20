"""Paste the original's head and background back into every tuple member.

All outputs are in the **original's frame**, so every member's background is
the original's own pixels (identical across members, which is what lets
centering cancel it) and the members are consistent with each other.

Per member (images uint8 RGB, masks float H x W in [0, 1]):

1. Registration: the member is aligned to the original over the background
   (outside the people): ORB features + RANSAC similarity, refined with ECC
   (sub-pixel, affine). Under ``min_shift`` px of motion the member is used
   as is, without resampling.
2. Head registration: the original's head is aligned onto the registered
   member's head: a normalized template match of the head core finds the
   integer-pixel optimum within the allowed shift, a step-halving search refines
   the translation to 1/16 px by maximizing masked normalized cross-correlation
   on the head's core, and an ECC affine refinement is kept only if it raises
   that correlation (ECC's own score proved unreliable: it can settle about a
   pixel off while reporting a high value). All of it uses the head's core only
   (an eroded head mask: the background ring around a head follows the
   global alignment, not the head, and would bias the fit and inflate the
   score), falling back to ORB inside the core. It must pass sanity checks
   (scale, rotation, shear, shift, correlation); otherwise the head is not
   pasted and the member keeps its own. A misaligned face is worse than an
   unpasted one.
3. Composition by multi-band (Laplacian pyramid) blending: low frequencies
   cross a seam over a wide band (tone differences fade out), high
   frequencies over a narrow one (detail stays sharp). The member keeps its
   pixels inside the dilated union of all bodies (members and original),
   minus the head; the original fills head and background.

Matrices are sampling maps (output coordinates -> source coordinates), used
with cv2.WARP_INVERSE_MAP, as cv2.findTransformECC returns them. Pure
functions; the models are wired up in scripts/paste_back.py.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)


@dataclass
class Alignment:
    matrix: np.ndarray  # 2x3 sampling map: output (x, y) -> source (x, y)
    method: str  # "identity", "orb", "ecc"
    score: float = 0.0  # ECC correlation, or ORB inlier ratio
    inliers: int = 0

    def shift_at(self, x: float, y: float) -> Tuple[float, float]:
        """How far the source content at (x, y) moves in the output (output - source)."""
        sx, sy = self.matrix @ np.array([x, y, 1.0])
        return float(x - sx), float(y - sy)

    @property
    def scale_rotation(self) -> Tuple[float, float]:
        a = self.matrix[:, :2]
        return float(np.sqrt(abs(np.linalg.det(a)))), float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def max_shift(matrix: np.ndarray, size) -> float:
    w, h = size
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=np.float64)
    return float(np.abs(corners @ matrix.T - corners[:, :2]).max())


def resample(image: np.ndarray, matrix: np.ndarray, size) -> np.ndarray:
    if np.allclose(matrix, IDENTITY):
        return image.copy()
    return cv2.warpAffine(image, matrix, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def _u8(mask):
    return None if mask is None else (np.asarray(mask) > 0.5).astype(np.uint8) * 255


def orb_map(output_img, source_img, output_mask=None, source_mask=None, n_features=8000, threshold=1.5):
    """Similarity sampling map output -> source from ORB matches inside the masks."""
    orb = cv2.ORB_create(n_features)
    ko, do = orb.detectAndCompute(gray(output_img), _u8(output_mask))
    ks, ds = orb.detectAndCompute(gray(source_img), _u8(source_mask))
    if do is None or ds is None or len(ko) < 8 or len(ks) < 8:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(do, ds)
    if len(matches) < 8:
        return None
    p_out = np.float32([ko[m.queryIdx].pt for m in matches])
    p_src = np.float32([ks[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(p_out, p_src, method=cv2.RANSAC,
                                                  ransacReprojThreshold=threshold, maxIters=5000)
    if matrix is None:
        return None
    n = int(inliers.sum())
    return Alignment(matrix.astype(np.float64), "orb", score=n / len(matches), inliers=n)


def ecc_map(output_img, source_img, mask, init=None, motions=(cv2.MOTION_AFFINE,)):
    """ECC sampling map output -> source over ``mask`` (output frame); (matrix, correlation) or None."""
    tpl = gray(output_img).astype(np.float32)
    src = gray(source_img).astype(np.float32)
    w = (IDENTITY if init is None else init).astype(np.float32).copy()
    score = None
    try:
        for motion in motions:
            score, w = cv2.findTransformECC(tpl, src, w, motion, ECC_CRITERIA, _u8(mask), 5)
    except cv2.error:
        return None
    return w.astype(np.float64), float(score)


def register_member(original, member, original_outside, member_outside, min_shift: float = 0.25,
                    min_inliers: int = 30) -> Alignment:
    """Sampling map original frame -> member, fitted on the background."""
    h, w = original.shape[:2]
    orb = orb_map(original, member, original_outside, member_outside)
    init = orb.matrix if orb is not None and orb.inliers >= min_inliers else IDENTITY
    refined = ecc_map(original, member, original_outside, init=init)
    if refined is not None:
        fit = Alignment(refined[0], "ecc", score=refined[1], inliers=0 if orb is None else orb.inliers)
    elif orb is not None and orb.inliers >= min_inliers:
        fit = orb
    else:
        return Alignment(IDENTITY.copy(), "identity")
    if max_shift(fit.matrix, (w, h)) < min_shift:
        return Alignment(IDENTITY.copy(), "identity", score=fit.score, inliers=fit.inliers)
    return fit


def bbox(mask: np.ndarray, pad: float = 0.0):
    ys, xs = np.nonzero(np.asarray(mask) > 0.5)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    h, w = mask.shape[:2]
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def head_core(head_mask: np.ndarray, shrink: float = 0.2) -> np.ndarray:
    """The head mask eroded by ``shrink`` of its size: face and hair, no background ring."""
    box = bbox(head_mask)
    if box is None:
        return np.zeros_like(head_mask, dtype=np.float32)
    size = max(box[2] - box[0], box[3] - box[1])
    r = max(1, int(size * shrink / 2))
    core = cv2.erode((np.asarray(head_mask) > 0.5).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return core.astype(np.float32)


def _sane(local: np.ndarray, max_scale_dev, max_rot, max_shear, max_shift_px) -> bool:
    a = local[:, :2]
    sx, sy = np.linalg.norm(a[:, 0]), np.linalg.norm(a[:, 1])
    shear = abs(float(a[:, 0] @ a[:, 1]) / (sx * sy))
    rot = abs(np.degrees(np.arctan2(a[1, 0], a[0, 0])))
    return (abs(sx - 1) <= max_scale_dev and abs(sy - 1) <= max_scale_dev and rot <= max_rot
            and shear <= max_shear and np.abs(local[:, 2]).max() <= max_shift_px)


def template_shift(member, original, core, search_px: int):
    """Integer translation (sampling map member -> original) of the head core,
    by normalized cross-correlation within +-search_px; (tx, ty, peak) or None."""
    box = bbox(core)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    h, w = original.shape[:2]
    sx0, sy0 = max(0, x0 - search_px), max(0, y0 - search_px)
    sx1, sy1 = min(w, x1 + search_px), min(h, y1 + search_px)
    tpl = gray(member)[y0:y1, x0:x1].astype(np.float32)
    region = gray(original)[sy0:sy1, sx0:sx1].astype(np.float32)
    if region.shape[0] < tpl.shape[0] or region.shape[1] < tpl.shape[1]:
        return None
    scores = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (px, py) = cv2.minMaxLoc(scores)
    return float(sx0 + px - x0), float(sy0 + py - y0), float(peak)


def masked_ncc(member, original, mask, matrix) -> float:
    """Pearson correlation over ``mask`` between the member and the original sampled through ``matrix``."""
    h, w = member.shape[:2]
    a = gray(member).astype(np.float64)[mask > 0.5]
    b = gray(resample(original, matrix, (w, h))).astype(np.float64)[mask > 0.5]
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else 0.0


def refine_translation(member, original, mask, tx: float, ty: float, start: float = 0.5, stop: float = 1 / 16):
    """Step-halving pattern search on the translation maximizing masked_ncc; (tx, ty, ncc)."""
    def score(x, y):
        return masked_ncc(member, original, mask, np.array([[1.0, 0.0, x], [0.0, 1.0, y]]))
    best = score(tx, ty)
    step = start
    while step >= stop:
        improved = True
        while improved:
            improved = False
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                s = score(tx + dx, ty + dy)
                if s > best:
                    best, tx, ty, improved = s, tx + dx, ty + dy, True
        step /= 2
    return tx, ty, best


def head_alignment(original, member, head_mask, shrink: float = 0.2, pad: float = 0.1, min_ncc: float = 0.85,
                   max_scale_dev: float = 0.08, max_rot: float = 8.0, max_shear: float = 0.05,
                   max_shift_px: float = 40.0) -> Optional[Alignment]:
    """Sampling map member frame -> original that puts the original's head on the member's.

    ``member`` is already registered into the original's frame. None when no
    fit passes the sanity checks (then the head must not be pasted)."""
    core = head_core(head_mask, shrink)
    box = bbox(head_mask, pad)
    if box is None or core.sum() < 64:
        return None
    x0, y0, x1, y1 = box
    crop = lambda a: a[y0:y1, x0:x1]
    to_full = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
    to_crop = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    full = lambda local: (to_full @ np.vstack([local, [0, 0, 1]]) @ to_crop)[:2]

    coarse = template_shift(member, original, core, int(max_shift_px))
    if coarse is not None:
        mc, oc, kc = crop(member), crop(original), crop(core)
        tx, ty, ncc = refine_translation(mc, oc, kc, coarse[0], coarse[1])
        local = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])
        if ncc >= min_ncc and _sane(local, max_scale_dev, max_rot, max_shear, max_shift_px):
            affine = ecc_map(mc, oc, kc, init=local, motions=(cv2.MOTION_AFFINE,))
            if affine is not None and _sane(affine[0], max_scale_dev, max_rot, max_shear, max_shift_px):
                affine_ncc = masked_ncc(mc, oc, kc, affine[0])
                if affine_ncc > ncc + 0.002:
                    return Alignment(full(affine[0]), "affine", score=affine_ncc)
            return Alignment(full(local), "translation", score=ncc)
    orb = orb_map(crop(member), crop(original), crop(core), crop(core), n_features=2000)
    if orb is not None and orb.inliers >= 15 and _sane(orb.matrix, max_scale_dev, max_rot, max_shear, max_shift_px):
        return Alignment(full(orb.matrix), "orb", score=orb.score, inliers=orb.inliers)
    return None


def feather(mask: np.ndarray, dilate_px: int = 0, blur_px: float = 0) -> np.ndarray:
    m = (np.asarray(mask) > 0.5).astype(np.uint8)
    if dilate_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)))
    m = m.astype(np.float32)
    if blur_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur_px)
    return np.clip(m, 0.0, 1.0)


def pyramid_blend(source: np.ndarray, target: np.ndarray, alpha: np.ndarray, levels: int = 4) -> np.ndarray:
    """Multi-band blend: alpha = 1 takes ``source``, 0 keeps ``target``. The
    coarsest band spans about 2**levels px; more levels hide tone seams over a
    wider band but leak more low frequencies into small kept regions."""
    h, w = target.shape[:2]
    gs, gt, ga = [source.astype(np.float32)], [target.astype(np.float32)], [alpha.astype(np.float32)]
    for _ in range(levels):
        if min(gs[-1].shape[:2]) < 16:
            break
        gs.append(cv2.pyrDown(gs[-1])); gt.append(cv2.pyrDown(gt[-1])); ga.append(cv2.pyrDown(ga[-1]))

    def laplacian(g):
        return [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0])) for i in range(len(g) - 1)] + [g[-1]]

    blended = []
    for ls, lt, al in zip(laplacian(gs), laplacian(gt), ga):
        al = al[..., None] if ls.ndim == 3 else al
        blended.append(al * ls + (1 - al) * lt)
    out = blended[-1]
    for lap in reversed(blended[:-1]):
        out = cv2.pyrUp(out, dstsize=(lap.shape[1], lap.shape[0])) + lap
    return np.clip(np.round(out), 0, 255).astype(np.uint8)[:h, :w]


@dataclass
class PasteResult:
    image: np.ndarray
    head_alignment: Optional[Alignment]  # None: head not pasted


def paste_member(original, member, body_union, head_mask, dilate_px: int = 12, seam_px: float = 4,
                 align_head: bool = True) -> PasteResult:
    """Original's head (aligned) and background into a member, all in the original's frame.

    ``member`` must already be registered to the original (register_member +
    resample); body_union covers every member's and the original's body."""
    h, w = original.shape[:2]
    head_fit = head_alignment(original, member, head_mask) if align_head and np.any(head_mask > 0.5) else None

    source = original
    head_alpha = np.zeros((h, w), np.float32)
    if head_fit is not None:
        # original moved onto the member's head inside the (dilated) head, unmoved
        # outside, joined by a multi-band blend so a local head shift leaves no seam
        moved = resample(original, head_fit.matrix, (w, h))
        source = pyramid_blend(moved, original, feather(head_mask, dilate_px=dilate_px, blur_px=seam_px))
        head_alpha = feather(head_mask, blur_px=seam_px)

    take_original = np.maximum(1.0 - feather(body_union, dilate_px=dilate_px), head_alpha)
    if head_fit is None:
        take_original = take_original * (1.0 - feather(head_mask, dilate_px=dilate_px))  # keep the member's head
    take_original = cv2.GaussianBlur(take_original, (0, 0), max(1.0, seam_px / 2))
    return PasteResult(image=pyramid_blend(source, member, take_original), head_alignment=head_fit)
