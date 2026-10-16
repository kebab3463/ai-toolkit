"""M4 / F2: the tuple slider loss on the stub model."""

import pytest
import torch

from extensions.slider_tuple.loss import slider_tuple_loss
from extensions.slider_tuple.sampling import member_weights, weighted_scale_sum
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


def identical_tuple_grad(scales, zero_sum):
    gen = torch.Generator().manual_seed(7)
    image = torch.randn(1, 4, 8, 8, generator=gen)
    noise = torch.randn(1, 4, 8, 8, generator=gen)
    weights = member_weights(scales, [None] * len(scales), zero_sum=zero_sum)
    sd = StubModel()
    with torch.no_grad():
        sd.lora.weight.zero_()  # a fresh LoRA contributes nothing (its up-projection starts at zero)
    run_slider(sd, [image.clone() for _ in scales], [[s] for s in scales], [[w] for w in weights],
               noise, torch.tensor([400.0]), make_embeds(1))
    return sd.lora.weight.grad, weighted_scale_sum(scales, weights)


@pytest.mark.parametrize("scales", [(1.4, -0.6), (2.0, 0.5, -1.0)])
def test_identical_members_give_zero_lora_gradient_with_zero_sum_weights(scales):
    grad, wsum = identical_tuple_grad(scales, zero_sum=True)
    assert wsum == pytest.approx(0.0, abs=1e-12)
    assert grad.abs().max().item() < 1e-6
    # control: unit weights on the same asymmetric tuple do push the LoRA
    grad_unit, wsum_unit = identical_tuple_grad(scales, zero_sum=False)
    assert abs(wsum_unit) > 0.1
    assert grad_unit.abs().max().item() > 1e-2
