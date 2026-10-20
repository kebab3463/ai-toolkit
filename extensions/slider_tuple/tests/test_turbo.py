"""F7: Turbo preview sampling (stub model holder; the real check is a GPU run)."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from extensions.slider_tuple.tuples import SliderTrainConfig
from extensions.slider_tuple.turbo import TurboPreview, is_comfy_quantized, strip_comfy_prefix


def test_strip_comfy_prefix():
    sd = {"model.diffusion_model.blocks.0.attn.wq.weight": 1, "diffusion_model.first.weight": 2, "last.bias": 3}
    assert strip_comfy_prefix(sd) == {"blocks.0.attn.wq.weight": 1, "first.weight": 2, "last.bias": 3}


def test_is_comfy_quantized(tmp_path):
    plain, quant = tmp_path / "plain.safetensors", tmp_path / "quant.safetensors"
    save_file({"blocks.0.w.weight": torch.zeros(2)}, str(plain))
    save_file({"model.diffusion_model.blocks.0.w.weight": torch.zeros(2, dtype=torch.int8),
               "model.diffusion_model.blocks.0.w.comfy_quant": torch.zeros(3, dtype=torch.uint8)}, str(quant))
    assert not is_comfy_quantized(str(plain)) and is_comfy_quantized(str(quant))
    assert not is_comfy_quantized(str(tmp_path / "missing.safetensors"))


class FakeModule(torch.nn.Linear):
    def __init__(self, name):
        super().__init__(2, 2)
        self.name, self.moves = name, []

    def to(self, *args, **kwargs):
        self.moves.append(str(args[0]) if args else str(kwargs.get("device")))
        return self


class FakeLoRA:
    def __init__(self):
        self.strength, self.attached, self.detached = 1.0, False, False

    def attach(self):
        self.attached = True

    def detach(self):
        self.detached = True


class FakeHolder:
    """The bits of a Krea 2 model holder TurboPreview touches."""

    def __init__(self, lora, fail_on=None):
        self.lora = lora  # to record the hook strength each image renders with
        self.device_torch = torch.device("cpu")
        self.model = FakeModule("raw")
        self.network = object()
        self.model_config = SimpleNamespace(model_kwargs={})
        self.calls, self.fail_on = [], fail_on

    def generate_single_image(self, pipeline, gen_config, *args):
        if self.fail_on is not None and gen_config.network_multiplier == self.fail_on:
            raise RuntimeError("boom")
        self.calls.append((self.model.name, self.network, self.model_config.model_kwargs.get("schedule_mu"),
                           gen_config.num_inference_steps, gen_config.guidance_scale, self.lora.strength))

    def generate_images(self, configs, sampler=None):
        for config in configs:
            self.generate_single_image(None, config)


def run(holder, multipliers=(-1.0, 0.0, 1.0)):
    turbo = TurboPreview("turbo.safetensors", steps=8, cfg=1.0, mu=1.15,
                         lora_factory=lambda path, sd: holder.lora, loader=lambda path, dtype: FakeModule("turbo"))
    configs = [SimpleNamespace(network_multiplier=m, num_inference_steps=25, guidance_scale=4.0) for m in multipliers]
    saved = []
    turbo.generate(holder, configs, "flowmatch", save_lora=saved.append, lora_path="/tmp/x/lora.safetensors")
    return turbo, saved


def test_turbo_generate_swaps_and_restores():
    lora = FakeLoRA()
    holder = FakeHolder(lora)
    raw, network = holder.model, holder.network
    turbo, saved = run(holder)
    assert saved == ["/tmp/x/lora.safetensors"]
    # Turbo DiT, no training network, Turbo schedule, hook strength = each image's --m
    assert holder.calls == [("turbo", None, 1.15, 8, 1.0, m) for m in (-1.0, 0.0, 1.0)]
    assert holder.model is raw and holder.network is network
    assert "schedule_mu" not in holder.model_config.model_kwargs
    assert "generate_single_image" not in vars(holder)
    assert lora.attached and lora.detached
    assert raw.moves == ["cpu", "cpu"] and turbo.dit.moves[-1] == "cpu"  # raw offloaded, then back on its device


def test_turbo_generate_restores_after_errors():
    lora = FakeLoRA()
    holder = FakeHolder(lora, fail_on=0.0)
    holder.model_config.model_kwargs["schedule_mu"] = 0.9
    raw, network = holder.model, holder.network
    with pytest.raises(RuntimeError, match="boom"):
        run(holder)
    assert holder.model is raw and holder.network is network
    assert holder.model_config.model_kwargs["schedule_mu"] == 0.9 and lora.detached


@pytest.mark.parametrize("slider, message", [
    ({"turbo_preview": "model.ckpt"}, ".safetensors"),
    ({"turbo_preview": "m.safetensors", "turbo_preview_steps": 0}, "turbo_preview_steps"),
])
def test_turbo_option_validation(slider, message):
    with pytest.raises(ValueError, match=message):
        SliderTrainConfig.from_raw(slider)
    assert SliderTrainConfig.from_raw({"turbo_preview": "a.safetensors"}).turbo_preview_mu == 1.15
