"""M4 / F2: the tuple slider loss on the stub model."""

import pytest
import torch

from extensions.slider_tuple.loss import slider_tuple_loss
from extensions.slider_tuple.centering import centered_scales, weighted_scale_sum
from toolkit.guidance import get_guided_loss_polarity

from .stubs import StubModel, make_batch, make_embeds, make_pair, make_train_config


def run_slider(sd, slot_latents, slot_scales, slot_weights, noise, timesteps, embeds, backward_per_member=False):
    return slider_tuple_loss(
        sd,
        slot_latents=slot_latents,
        slot_scales=slot_scales,
        slot_weights=slot_weights,
        noise=noise,
        timesteps=timesteps,
        conditional_embeds=embeds,
        pred_kwargs={},
        batch=make_batch(slot_latents[0]),
        train_config=make_train_config(),
        backward_per_member=backward_per_member,
    )


@pytest.mark.parametrize("w", [1.0, 0.7])
def test_two_member_tuple_equals_polarity(w):
    after, before, noise, timesteps = make_pair(batch_size=2)
    embeds = make_embeds(2)

    sd_ref = StubModel()
    ref = get_guided_loss_polarity(
        noisy_latents=None, conditional_embeds=embeds, match_adapter_assist=False,
        network_weight_list=[w, w], timesteps=timesteps, pred_kwargs={},
        batch=make_batch(after, before), noise=noise, sd=sd_ref, train_config=make_train_config(),
    )

    sd = StubModel()
    loss = run_slider(sd, [after, before], [[w, w], [-w, -w]], [[1.0, 1.0], [1.0, 1.0]], noise, timesteps, embeds)

    assert loss.item() == pytest.approx(ref.item(), rel=1e-6)
    assert torch.allclose(sd.lora.weight.grad, sd_ref.lora.weight.grad, rtol=1e-5, atol=1e-7)
    assert sd.network.multiplier == sd_ref.network.multiplier == [w, w, -w, -w]
    assert sd.noise_scheduler.set_train_timesteps_calls == sd_ref.noise_scheduler.set_train_timesteps_calls == 1
    assert loss.requires_grad and loss.grad_fn is None


def three_slots(seed=3):
    gen = torch.Generator().manual_seed(seed)
    shape = (2, 4, 8, 8)
    slots = [torch.randn(shape, generator=gen) for _ in range(3)]
    noise = torch.randn(shape, generator=gen)
    timesteps = torch.tensor([150.0, 820.0])
    scales = [[1.4, 2.0], [0.3, -0.5], [-0.6, -1.0]]
    weights = [[0.6, 0.4], [1.0, 1.3], [1.4, 1.3]]
    return slots, scales, weights, noise, timesteps


def test_backward_per_member_matches_single_forward():
    slots, scales, weights, noise, timesteps = three_slots()
    embeds = make_embeds(2)
    sd_one, sd_per = StubModel(), StubModel()
    loss_one = run_slider(sd_one, slots, scales, weights, noise, timesteps, embeds)
    loss_per = run_slider(sd_per, slots, scales, weights, noise, timesteps, embeds, backward_per_member=True)
    assert loss_per.item() == pytest.approx(loss_one.item(), rel=1e-6)
    assert torch.allclose(sd_per.lora.weight.grad, sd_one.lora.weight.grad, rtol=1e-5, atol=1e-7)


def test_loss_is_weighted_sum_of_member_mses():
    slots, scales, weights, noise, timesteps = three_slots()
    sd = StubModel()
    loss = run_slider(sd, slots, scales, weights, noise, timesteps, make_embeds(2))
    expected = 0.0
    with torch.no_grad():
        for x, s, w in zip(slots, scales, weights):
            sd.network.is_active, sd.network.multiplier = True, s
            pred = sd.predict_noise(sd.add_noise(x, noise, timesteps))
            per_item = ((pred - (noise - x)) ** 2).mean(dim=(1, 2, 3))
            expected += (torch.tensor(w) * per_item).mean().item()
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def tuple_grad(images, levels, weights, kappa=1.0, offset=0.0, seed=7, zero_lora=True):
    """LoRA gradient of one tuple with centered multipliers (+ optional offset)."""
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(1, 4, 8, 8, generator=gen)
    scales = centered_scales(levels, weights, kappa, offset)
    sd = StubModel()
    if zero_lora:
        with torch.no_grad():
            sd.lora.weight.zero_()  # a fresh LoRA contributes nothing (its up-projection starts at zero)
    run_slider(sd, [x.clone() for x in images], [[s] for s in scales], [[w] for w in weights],
               noise, torch.tensor([400.0]), make_embeds(1))
    return sd.lora.weight.grad, scales


def same_image(n, seed=11):
    image = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(seed))
    return [image] * n


# acceptance (2): identical images -> zero LoRA gradient for any weights
@pytest.mark.parametrize(
    "levels, weights",
    [([1.0, -1.0], [1.0, 1.0]), ([1.4, 0.9, -0.2], [1.0, 0.5, 1.0]), ([3.1, 2.4, 1.6, 0.2], [0.3, 2.0, 1.0, 0.7])],
)
def test_identical_members_give_zero_lora_gradient_for_any_weights(levels, weights):
    grad, scales = tuple_grad(same_image(len(levels)), levels, weights, kappa=0.9)
    assert weighted_scale_sum(scales, weights) == pytest.approx(0.0, abs=1e-12)
    assert grad.abs().max().item() < 1e-6


# acceptance (4): with an offset the shared content leaks into the gradient, and for a pair
# offset by delta / 2 kappa the lower member sits at multiplier 0 and contributes nothing
def test_center_offset_breaks_cancellation():
    grad, _ = tuple_grad(same_image(3), [1.4, 0.9, -0.2], [1.0, 0.5, 1.0], offset=0.3)
    assert grad.abs().max().item() > 1e-2


def test_offset_half_span_zeroes_the_lower_members_gradient():
    gen = torch.Generator().manual_seed(4)
    upper, lower = torch.randn(1, 4, 8, 8, generator=gen), torch.randn(1, 4, 8, 8, generator=gen)
    levels, kappa = [2.5, 1.0], 1.5
    offset = (levels[0] - levels[1]) / (2 * kappa)
    grad_pair, scales = tuple_grad([upper, lower], levels, [1.0, 1.0], kappa=kappa, offset=offset, zero_lora=False)
    assert scales == pytest.approx([1.0, 0.0])
    grad_upper_only, _ = tuple_grad([upper], [levels[0]], [1.0], kappa=kappa, offset=1.0, zero_lora=False)
    assert torch.allclose(grad_pair, grad_upper_only, rtol=1e-5, atol=1e-7)
    assert grad_pair.abs().max() > 1e-3
