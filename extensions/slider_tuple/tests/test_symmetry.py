"""symmetry.py: the loss sees only the part of the LoRA's effect that is even in the multiplier."""

import pytest
import torch

from extensions.slider_tuple.symmetry import symmetry_loss
from extensions.slider_tuple.tuples import SliderTrainConfig

from .stubs import StubModel, make_batch, make_embeds
from .test_anchors import anchor_trainer, inputs
from .test_tuple_trainer import tuple_loss


class EvenStub(StubModel):
    """v(m) = base(x) + m lora(x) + m^2 even(x): an odd (slider) and an even (drift) part."""

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        gen = torch.Generator().manual_seed(seed + 1)
        self.even = torch.nn.Conv2d(4, 4, 1, bias=False)
        with torch.no_grad():
            self.even.weight.copy_(0.1 * torch.randn(self.even.weight.shape, generator=gen))

    def lora_parameters(self):
        return list(self.lora.parameters()) + list(self.even.parameters())

    def predict_noise(self, latents, conditional_embeddings=None, timestep=None, guidance_scale=1.0, **kwargs):
        m = self._multiplier_tensor(latents.shape[0])
        return self.base(latents) + m * self.lora(latents) + m * m * self.even(latents)


def run(sd, multipliers=(1.5, 0.7), strength=1.0, seed=0):
    latents, noise, timesteps = inputs(seed, len(multipliers))
    return symmetry_loss(sd, latents, noise, timesteps, make_embeds(len(multipliers)), list(multipliers),
                         strength, {}, make_batch(latents))


def test_odd_slider_is_free_even_drift_is_penalized():
    linear = StubModel()  # a purely linear slider: no even part
    _, mse = run(linear)
    assert mse == pytest.approx(0.0, abs=1e-12) and linear.lora.weight.grad.abs().max() == 0

    sd = EvenStub()
    loss, mse = run(sd, strength=2.0)
    latents, noise, timesteps = inputs()
    x_t = sd.add_noise(latents, noise, timesteps)
    with torch.no_grad():
        expected = (torch.tensor([1.5, 0.7]).view(-1, 1, 1, 1) ** 2 * sd.even(x_t)).pow(2).mean()
    assert mse == pytest.approx(expected.item(), rel=1e-5) and loss.item() == pytest.approx(2 * mse, rel=1e-6)
    assert sd.lora.weight.grad.abs().max() < 1e-7       # the slider direction gets no gradient
    assert sd.even.weight.grad.abs().max() > 1e-4


def test_gradient_matches_autograd_and_shrinks_the_even_part():
    sd = EvenStub()
    run(sd, strength=1.0)
    ours = sd.even.weight.grad.clone()
    sd.zero_lora_grad(); sd.even.weight.grad = None
    latents, noise, timesteps = inputs()
    x_t = sd.add_noise(latents, noise, timesteps)
    m = torch.tensor([1.5, 0.7]).view(-1, 1, 1, 1)
    full = lambda s: sd.base(x_t) + s * sd.lora(x_t) + s * s * sd.even(x_t)
    ((full(m) + full(-m)) / 2 - sd.base(x_t)).pow(2).mean().backward()
    assert torch.allclose(ours, sd.even.weight.grad, atol=1e-6)

    sd = EvenStub()
    optimizer = torch.optim.SGD(sd.lora_parameters(), lr=2.0)
    history, lora_before = [], sd.lora.weight.detach().clone()
    for step in range(30):
        optimizer.zero_grad()
        history.append(run(sd, seed=step)[1])
        optimizer.step()
    assert history[-1] < 0.1 * history[0] and torch.allclose(sd.lora.weight, lora_before)


def test_trainer_adds_and_logs_symmetry_loss(tmp_path):
    plain, batch_plain = anchor_trainer(tmp_path / "plain", None)
    slider_only = tuple_loss(plain, batch_plain)[0].item()
    trainer, batch = anchor_trainer(tmp_path / "sym", {"symmetry_strength": 0.5})
    trainer.sd = EvenStub()  # the stub trainer's model has no even part
    plain.sd = EvenStub()
    slider_only = tuple_loss(plain, batch_plain)[0].item()
    torch.manual_seed(0)
    total = tuple_loss(trainer, batch)[0]
    mse = trainer.additional_logs["slider/symmetry_loss"]
    assert mse > 0 and total.item() == pytest.approx(slider_only + 0.5 * mse, rel=1e-4)
    never, batch_never = anchor_trainer(tmp_path / "never", {"symmetry_strength": 0.5, "symmetry_prob": 0.0})
    tuple_loss(never, batch_never)
    assert "slider/symmetry_loss" not in never.additional_logs


@pytest.mark.parametrize("slider, message", [
    ({"symmetry_strength": -1.0}, ">= 0"),
    ({"symmetry_prob": 2.0}, r"in \[0, 1\]"),
    ({"symmetry_max_multiplier": 0.0}, "> 0"),
])
def test_symmetry_option_validation(slider, message):
    with pytest.raises(ValueError, match=message):
        SliderTrainConfig.from_raw(slider)
    assert SliderTrainConfig.from_raw({}).symmetry_enabled is False
