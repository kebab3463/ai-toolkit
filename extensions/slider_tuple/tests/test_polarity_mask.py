"""M5 / F3: upstream polarity honors mask_multiplier (patch in toolkit/guidance.py)."""

import torch

from toolkit.guidance import get_guidance_loss

from .stubs import StubModel, make_batch, make_embeds, make_pair, make_train_config


def polarity(after, before, noise, timesteps, mask_multiplier):
    sd = StubModel()
    loss = get_guidance_loss(
        noisy_latents=None, conditional_embeds=make_embeds(after.shape[0]), match_adapter_assist=False,
        network_weight_list=[1.0] * after.shape[0], timesteps=timesteps, pred_kwargs={},
        batch=make_batch(after, before), noise=noise, sd=sd, mask_multiplier=mask_multiplier,
        train_config=make_train_config(),
    )
    return loss, sd.lora.weight.grad.clone()


def sdtrainer_mask(mask_pixels, latents):
    # how SDTrainer builds mask_multiplier before calling get_guided_loss
    m = torch.nn.functional.interpolate(mask_pixels, size=latents.shape[-2:])
    m = m.expand(-1, latents.shape[1], -1, -1)
    return m / m.mean()


def test_default_all_ones_multiplier_changes_nothing():
    after, before, noise, timesteps = make_pair()
    ref = polarity(after, before, noise, timesteps, None)
    ones = polarity(after, before, noise, timesteps, torch.ones(after.shape[0], 1, 1, 1))
    assert torch.equal(ones[0], ref[0]) and torch.equal(ones[1], ref[1])


def test_masked_polarity_ignores_head_region():
    after, before, noise, timesteps = make_pair()
    mask_pixels = torch.ones(after.shape[0], 1, 64, 64)
    mask_pixels[:, :, :16] = 0.0
    mask = sdtrainer_mask(mask_pixels, after)
    after2, before2 = after.clone(), before.clone()
    after2[:, :, :2] += 4.0
    before2[:, :, :2] -= 2.0

    loss, grad = polarity(after, before, noise, timesteps, mask)
    _, grad_changed = polarity(after2, before2, noise, timesteps, mask)
    assert torch.allclose(grad, grad_changed, rtol=1e-6, atol=1e-8)

    unmasked_loss, grad_plain = polarity(after, before, noise, timesteps, None)
    _, grad_plain_changed = polarity(after2, before2, noise, timesteps, None)
    assert (grad_plain - grad_plain_changed).abs().max() > 1e-3
    assert not torch.isclose(loss, unmasked_loss)
