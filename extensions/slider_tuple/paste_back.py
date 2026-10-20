"""Paste the original's head and background back into every tuple member.

All outputs are in the **original's frame**, so every member's background is
the original's own pixels (identical across members, which is what lets
centering cancel it) and the members are consistent with each other.

Per member (images uint8 RGB, masks float H x W in [0, 1]):

1. Registration: the member is aligned to the original over the background
   (outside the people). Candidates: identity, an ORB + RANSAC similarity,
   and coarse-to-fine ECC (affine) started from each. Every candidate must be
   plausible (scale, rotation, displacement bounds) and is scored by masked
   NCC over the background; the best wins, and identity unless something
   beats it. (ECC alone can diverge to wild transforms on low-texture or
   person-filled frames.) Under ``min_shift`` px of motion the member is used
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
3. Tone matching: the editor's large-scale tone shift is measured on pure
   background (original - member where both show background), extended under
   the person by normalized convolution, and removed from the whole member.
   Otherwise background the editor regenerated where a body used to be shows
   as a lighter/darker ring next to the pasted original, and the shift on the
   body itself would differ between members.
4. Composition by multi-band (Laplacian pyramid) blending: low frequencies
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
    """``image`` sampled into an output of ``size`` (w, h) through the sampling map."""
    if np.allclose(matrix, IDENTITY) and image.shape[1] == size[0] and image.shape[0] == size[1]:
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


def masked_ncc(member, original, mask, matrix) -> float:
    """Pearson correlation over ``mask`` between the member and the original sampled through ``matrix``."""
    h, w = member.shape[:2]
    a = gray(member).astype(np.float64)[mask > 0.5]
    b = gray(resample(original, matrix, (w, h))).astype(np.float64)[mask > 0.5]
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else 0.0


def _scaled(matrix: np.ndarray, factor: float) -> np.ndarray:
    """The same sampling map expressed in coordinates scaled by ``factor``."""
    out = matrix.copy()
    out[:, 2] *= factor
    return out


def ecc_coarse_to_fine(output_img, source_img, mask, init, scale: float = 0.25):
    """ECC affine at ``scale`` (wider capture range), then refined at full resolution."""
    h, w = output_img.shape[:2]
    small = (max(16, round(w * scale)), max(16, round(h * scale)))
    shrink = lambda a, interp=cv2.INTER_AREA: cv2.resize(a, small, interpolation=interp)
    coarse = ecc_map(shrink(output_img), shrink(source_img), shrink(np.asarray(mask, np.float32), cv2.INTER_NEAREST),
                     init=_scaled(init, scale))
    start = init if coarse is None else _scaled(coarse[0], 1 / scale)
    fine = ecc_map(output_img, source_img, mask, init=start)
    return None if fine is None else fine[0]


def plausible(matrix: np.ndarray, size, max_shift_px: float, max_scale_dev: float = 0.05, max_rot: float = 2.0) -> bool:
    a = matrix[:, :2]
    sx, sy = np.linalg.norm(a[:, 0]), np.linalg.norm(a[:, 1])
    rot = abs(np.degrees(np.arctan2(a[1, 0], a[0, 0])))
    return (np.isfinite(matrix).all() and abs(sx - 1) <= max_scale_dev and abs(sy - 1) <= max_scale_dev
            and rot <= max_rot and max_shift(matrix, size) <= max_shift_px)


def register_member(original, member, original_outside, member_outside, min_shift: float = 0.25,
                    max_shift_frac: float = 0.03, min_gain: float = 0.001, min_inliers: int = 12) -> Alignment:
    """Sampling map original frame -> member, fitted and validated on the background."""
    h, w = original.shape[:2]
    limit = max_shift_frac * min(h, w)
    candidates = [("identity", IDENTITY.copy(), 0)]
    orb = orb_map(original, member, original_outside, member_outside)
    if orb is not None and orb.inliers >= min_inliers:
        candidates.append(("orb", orb.matrix, orb.inliers))
    for _, init, inliers in list(candidates):
        refined = ecc_coarse_to_fine(original, member, original_outside, init)
        if refined is not None:
            candidates.append(("ecc", refined, inliers))

    scored = []
    for method, matrix, inliers in candidates:
        if method == "identity" or plausible(matrix, (w, h), limit):
            scored.append((masked_ncc(original, member, original_outside, matrix), method, matrix, inliers))
    identity_ncc = scored[0][0]
    ncc, method, matrix, inliers = max(scored, key=lambda c: c[0])
    if method == "identity" or ncc < identity_ncc + min_gain or max_shift(matrix, (w, h)) < min_shift:
        return Alignment(IDENTITY.copy(), "identity", score=identity_ncc)
    return Alignment(matrix, method, score=ncc, inliers=inliers)


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


def tone_correction(original, member, background, sigmas=(0.03, 0.12)) -> np.ndarray:
    """Smooth per-channel field (H x W x 3, float) that maps the member's
    large-scale tone onto the original's.

    Measured as original - member over ``background`` (both show background),
    then extended everywhere by normalized convolution: a Gaussian at each of
    ``sigmas`` (fractions of the short side), finer scales winning where they
    have enough support, so the field reaches deep inside large bodies."""
    h, w = original.shape[:2]
    weight = (np.asarray(background) > 0.5).astype(np.float32)
    diff = (original.astype(np.float32) - member.astype(np.float32)) * weight[..., None]
    field = np.zeros((h, w, 3), np.float32)
    filled = np.zeros((h, w), np.float32)
    for frac in sigmas:
        sigma = max(2.0, frac * min(h, w))
        support = cv2.GaussianBlur(weight, (0, 0), sigma)
        estimate = cv2.GaussianBlur(diff, (0, 0), sigma) / np.maximum(support, 1e-6)[..., None]
        trust = np.clip(support / 0.2, 0.0, 1.0) * (1.0 - filled)  # 20% background nearby = fully trusted
        field += estimate * trust[..., None]
        filled += trust
    if filled.max() > 0:  # anything still unsupported: the mean background difference
        mean = diff.sum(axis=(0, 1)) / max(weight.sum(), 1.0)
        field += mean * (1.0 - np.clip(filled, 0, 1))[..., None]
    return field


def tone_match(original, member, background, sigmas=(0.03, 0.12)) -> np.ndarray:
    corrected = member.astype(np.float32) + tone_correction(original, member, background, sigmas)
    return np.clip(np.round(corrected), 0, 255).astype(np.uint8)


@dataclass
class PasteResult:
    image: np.ndarray
    head_alignment: Optional[Alignment]  # None: head not pasted


def paste_member(original, member, keep_region, head_mask, dilate_px: int = 12, seam_px: float = 4,
                 align_head: bool = True, background_levels: int = 4, head_levels: int = 4) -> PasteResult:
    """Original's head (aligned) and background into a member, all in the original's frame.

    ``member`` must already be registered to the original (register_member +
    resample). ``keep_region`` is where the member keeps its own pixels: its
    own body plus the original's body (where the original shows a person, its
    pixels cannot serve as background). Using every member's body here instead
    would leave each member's regenerated background inside the other members'
    outlines, a ghost of the heavier silhouette in the thinner image.

    Two multi-band passes: body/background, then the aligned head. Tone
    differences are removed by tone_match first, so both blends stay narrow
    (2**levels px): a wide low-frequency band reached past the dilated seam and
    smeared nearby body content (the original's arm) into the background as a
    soft ghost."""
    h, w = original.shape[:2]
    head_fit = head_alignment(original, member, head_mask) if align_head and np.any(head_mask > 0.5) else None

    # remove the editor's large-scale tone shift, measured where both show background
    background = 1.0 - feather(np.maximum(keep_region, head_mask), dilate_px=3 * dilate_px)
    member = tone_match(original, member, background)

    keep = feather(keep_region, dilate_px=dilate_px)
    if head_fit is None:  # keep the member's own head too
        keep = np.maximum(keep, feather(head_mask, dilate_px=dilate_px))
    take_original = cv2.GaussianBlur(1.0 - keep, (0, 0), max(1.0, seam_px / 2))
    image = pyramid_blend(original, member, take_original, levels=background_levels)

    # head pass: the original moved onto the member's head, or (if it could not be
    # aligned) the member's own head restored without the wide pass's tone mixing
    head_source = member if head_fit is None else resample(original, head_fit.matrix, (w, h))
    if np.any(head_mask > 0.5):
        image = pyramid_blend(head_source, image, feather(head_mask, blur_px=seam_px), levels=head_levels)
    return PasteResult(image=image, head_alignment=head_fit)
