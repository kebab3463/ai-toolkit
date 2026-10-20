"""Paste-back of the original's head and background, on synthetic scenes with known shifts."""

import cv2
import numpy as np
import pytest

from extensions.slider_tuple.paste_back import (
    feather,
    head_alignment,
    paste_member,
    pyramid_blend,
    register_member,
    resample,
)

SIZE = 320
HEAD_CENTER, HEAD_R = (160.0, 70.0), 26
BODY_CENTER = (160.0, 210.0)


def moved(image, dx, dy):
    """``image`` with its content moved by (dx, dy)."""
    return resample(image, np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]]), (SIZE, SIZE))


def textured(seed, size=SIZE):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (size, size, 3)).astype(np.float32), (0, 0), 2)
    for _ in range(60):  # corners for ORB
        x, y = rng.integers(0, size, 2)
        w, h = rng.integers(6, 30, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), rng.uniform(0, 255, 3).tolist(), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def ellipse(center, axes):
    m = np.zeros((SIZE, SIZE), np.uint8)
    cv2.ellipse(m, (round(center[0]), round(center[1])), (round(axes[0]), round(axes[1])), 0, 0, 360, 1, -1)
    return m.astype(np.float32)


def scene(global_shift=(3.0, -2.0), head_shift=(2.5, 1.5), body_w=45, noise_seed=None, bad_head=False):
    """An original and one edited member: the member is the original moved by
    ``global_shift``, with its own body, its head moved by a further
    ``head_shift``, and optional regenerated background noise. Masks are in
    the original's frame (where everything is composed)."""
    background, head_tex = textured(1), textured(2)
    orig_head = ellipse(HEAD_CENTER, (HEAD_R, HEAD_R))
    orig_body = ellipse(BODY_CENTER, (55, 90))
    original = background.copy()
    original[orig_body > 0] = textured(99)[orig_body > 0]
    original[orig_head > 0] = head_tex[orig_head > 0]

    gx, gy = global_shift
    hx, hy = head_shift
    member = moved(background, gx, gy).astype(np.float32)
    if noise_seed is not None:
        member += cv2.GaussianBlur(np.random.default_rng(noise_seed).normal(0, 60, member.shape).astype(np.float32), (0, 0), 2)
    member = np.clip(member, 0, 255).astype(np.uint8)
    body = ellipse((BODY_CENTER[0] + gx, BODY_CENTER[1] + gy), (body_w, 90))
    member[body > 0] = textured(3 + body_w)[body > 0]
    head_layer = moved(textured(77) if bad_head else head_tex, gx + hx, gy + hy)
    head_in_member = moved(orig_head, gx + hx, gy + hy) > 0.5
    member[head_in_member] = head_layer[head_in_member]

    # in the original's frame
    body_union = np.maximum(ellipse(BODY_CENTER, (body_w, 90)), orig_body)
    head_mask = ellipse((HEAD_CENTER[0] + hx, HEAD_CENTER[1] + hy), (HEAD_R * 1.15, HEAD_R * 1.15))
    true_head = moved(orig_head, hx, hy)
    outside = 1 - feather(np.maximum(np.maximum(body_union, head_mask), orig_head), dilate_px=14)
    return original, member, body_union, head_mask, true_head, outside


def register(original, member, outside):
    alignment = register_member(original, member, outside, moved(outside, 3.0, -2.0) > 0.5)
    return alignment, resample(member, alignment.matrix, (SIZE, SIZE))


def test_registration_and_head_shift_are_sub_pixel():
    original, member, _, head_mask, _, outside = scene()
    alignment, registered = register(original, member, outside)
    assert alignment.method == "ecc"
    assert alignment.matrix[:, 2] == pytest.approx([3.0, -2.0], abs=0.1)
    assert alignment.matrix[:, :2] == pytest.approx(np.eye(2), abs=0.002)

    head = head_alignment(original, registered, head_mask)
    assert head is not None and head.method in ("translation", "affine") and head.score > 0.99
    # the original's head content moves onto the member's head, 2.5 / 1.5 px away
    assert head.shift_at(HEAD_CENTER[0] + 2.5, HEAD_CENTER[1] + 1.5) == pytest.approx((2.5, 1.5), abs=0.15)


def test_pasted_head_lands_on_the_members_head():
    original, member, body, head_mask, true_head, outside = scene()
    _, registered = register(original, member, outside)
    result = paste_member(original, registered, body, head_mask, dilate_px=6, seam_px=3)
    core = cv2.erode(true_head.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    err = lambda img: np.abs(img.astype(np.float32) - registered.astype(np.float32))[core].mean()
    assert err(result.image) < 4.0
    assert err(original) > 4 * err(result.image)  # pasting the head unaligned would miss


def test_members_share_the_originals_background_exactly():
    o, heavy, body_h, head_h, _, outside_h = scene(body_w=60, noise_seed=10)
    _, thin, body_t, head_t, _, outside_t = scene(body_w=35, noise_seed=11)
    union = np.maximum(body_h, body_t)
    _, heavy_r = register(o, heavy, np.minimum(outside_h, outside_t))
    _, thin_r = register(o, thin, np.minimum(outside_h, outside_t))
    away = feather(np.maximum(union, np.maximum(head_h, head_t)), dilate_px=24) < 0.5
    raw = np.abs(heavy_r.astype(np.float32) - thin_r.astype(np.float32))[away].mean()
    a = paste_member(o, heavy_r, union, head_h, dilate_px=6, seam_px=3).image
    b = paste_member(o, thin_r, union, head_t, dilate_px=6, seam_px=3).image
    pasted = np.abs(a.astype(np.float32) - b.astype(np.float32))[away].mean()
    assert raw > 5 and pasted < 0.5
    assert np.abs(a.astype(np.float32) - o.astype(np.float32))[away].mean() < 0.5
    inside = cv2.erode(np.minimum(body_h, body_t).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    assert np.abs(a.astype(np.float32) - heavy_r.astype(np.float32))[inside].mean() < 1.0  # own body kept


def test_unrelated_head_is_not_pasted():
    original, member, body, head_mask, true_head, outside = scene(bad_head=True)
    _, registered = register(original, member, outside)
    result = paste_member(original, registered, body, head_mask, dilate_px=6, seam_px=3)
    assert result.head_alignment is None
    core = cv2.erode(true_head.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert np.abs(result.image.astype(np.float32) - registered.astype(np.float32))[core].mean() < 1.0


def test_small_motion_is_not_resampled():
    original, member, _, _, _, outside = scene(global_shift=(0.0, 0.0))
    alignment, registered = register(original, member, outside)
    assert alignment.method == "identity" and np.array_equal(registered, member)


def test_pyramid_blend_endpoints():
    a, b = textured(5), textured(6)
    assert np.abs(pyramid_blend(a, b, np.ones((SIZE, SIZE), np.float32)).astype(int) - a).max() <= 1
    assert np.abs(pyramid_blend(a, b, np.zeros((SIZE, SIZE), np.float32)).astype(int) - b).max() <= 1


@pytest.mark.parametrize("global_shift, head_shift", [
    ((0.0, 0.0), (0.0, 0.0)), ((1.3, 0.7), (-1.8, 2.2)), ((-4.2, 3.1), (0.4, -0.6)),
    ((6.0, -5.5), (3.3, 3.3)), ((0.2, 0.1), (-4.75, -2.25)),
])
def test_head_shift_accuracy_across_shifts(global_shift, head_shift):
    original, member, _, head_mask, _, outside = scene(global_shift=global_shift, head_shift=head_shift)
    gx, gy = global_shift
    alignment = register_member(original, member, outside, moved(outside, gx, gy) > 0.5)
    registered = resample(member, alignment.matrix, (SIZE, SIZE))
    head = head_alignment(original, registered, head_mask)
    assert head is not None
    hx, hy = head_shift
    if alignment.method == "identity":  # sub-threshold global motion is left in place, so the head carries it
        hx, hy = hx + gx, hy + gy
    assert head.shift_at(HEAD_CENTER[0] + hx, HEAD_CENTER[1] + hy) == pytest.approx((hx, hy), abs=0.15)
