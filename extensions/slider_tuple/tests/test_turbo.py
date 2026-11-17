"""F7: Turbo preview sampling (stub model holder; the real check is a GPU run)."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from extensions.slider_tuple.tuples import SliderTrainConfig
from extensions.slider_tuple.turbo import (
    TurboPreview, TurboSampleMixin, is_comfy_quantized, strip_comfy_prefix, turbo_preview_from_raw)


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


class ImageHolder(FakeHolder):
    """Returns one new image token per real render, so reuse is visible."""

    def generate_single_image(self, pipeline, gen_config, *args):
        super().generate_single_image(pipeline, gen_config, *args)
        return object()

    def generate_images(self, configs, sampler=None):
        self.images = [self.generate_single_image(None, config) for config in configs]


def preview_config(m, seed=1, prompt="a woman"):
    return SimpleNamespace(prompt=prompt, negative_prompt="", seed=seed, width=1024, height=1024,
                           network_multiplier=m, num_inference_steps=25, guidance_scale=4.0)


def test_weight_zero_rendered_once():
    holder = ImageHolder(FakeLoRA())
    turbo = TurboPreview("turbo.safetensors", reuse_weight_zero=True, lora_factory=lambda path, sd: holder.lora,
                         loader=lambda path, dtype: FakeModule("turbo"))
    grid = lambda: [preview_config(0.0), preview_config(1.0), preview_config(0.0, seed=2),
                    preview_config(0.0, prompt="a man")]
    turbo.generate(holder, grid(), None, save_lora=lambda p: None, lora_path="/tmp/x/lora.safetensors")
    first = holder.images
    assert len(holder.calls) == 4
    turbo.generate(holder, grid(), None, save_lora=lambda p: None, lora_path="/tmp/x/lora.safetensors")
    assert len(holder.calls) == 5  # only weight 1 rendered again
    assert holder.images[0] is first[0] and holder.images[2] is first[2] and holder.images[3] is first[3]
    assert len({id(i) for i in first}) == 4  # other seeds and prompts were not taken for each other


def test_turbo_preview_from_raw(tmp_path):
    ckpt = tmp_path / "turbo.safetensors"
    ckpt.write_bytes(b"")
    assert turbo_preview_from_raw(None) is None
    turbo = turbo_preview_from_raw({"path": str(ckpt), "steps": 6})
    assert (turbo.steps, turbo.cfg, turbo.mu, turbo.offload_raw, turbo.reuse_weight_zero) == (6, 1.0, 1.15, True, True)
    assert turbo_preview_from_raw({"path": str(ckpt), "mu": None}).mu is None
    for raw, message in [("x.safetensors", "mapping"), ({"path": str(ckpt), "stepz": 8}, "unknown keys"),
                         ({"path": "model.ckpt"}, ".safetensors"), ({"path": str(tmp_path / "no.safetensors")}, "not found"),
                         ({"path": str(ckpt), "steps": 0}, "steps")]:
        with pytest.raises(ValueError, match=message):
            turbo_preview_from_raw(raw)


class BaseTrainer:
    """The bits of BaseSDTrainProcess.sample the mixin wraps."""

    def __init__(self, holder):
        self.sd, self.save_root, self.network = holder, "/tmp/run", SimpleNamespace(save_weights=lambda *a, **k: None)

    def sample(self, step=None, is_first=False):
        self.sd.generate_images([preview_config(1.0)])
        return step


class MixedTrainer(TurboSampleMixin, BaseTrainer):
    pass


def test_sample_mixin():
    holder = ImageHolder(FakeLoRA())
    trainer = MixedTrainer(holder)
    assert trainer.sample(5) == 5 and holder.calls[-1][0] == "raw"  # no turbo_preview: the trainer's own
    trainer.turbo_preview = TurboPreview("turbo.safetensors", lora_factory=lambda path, sd: holder.lora,
                                         loader=lambda path, dtype: FakeModule("turbo"))
    assert trainer.sample(6) == 6 and holder.calls[-1][0] == "turbo"
    assert "generate_images" not in vars(holder)  # unwrapped afterwards


def test_preview_defaults_per_arch(tmp_path):
    from extensions.slider_tuple.turbo import load_krea2_dit, load_moss_image_dit

    ckpt = tmp_path / "dit.safetensors"
    ckpt.write_bytes(b"")
    krea = turbo_preview_from_raw({"path": str(ckpt)})
    assert (krea.steps, krea.cfg, krea.mu, krea.loader) == (8, 1.0, 1.15, load_krea2_dit)
    moss = turbo_preview_from_raw({"path": str(ckpt)}, arch="moss_image:base")  # variant suffix ignored
    assert (moss.steps, moss.cfg, moss.mu, moss.loader) == (30, 3.0, None, load_moss_image_dit)
    assert turbo_preview_from_raw({"path": str(ckpt), "steps": 20}, arch="moss_image").steps == 20
    with pytest.raises(ValueError, match="no preview DiT loader"):
        turbo_preview_from_raw({"path": str(ckpt)}, arch="flux2")


class PostLoadDiT(FakeModule):
    def __init__(self):
        super().__init__("turbo")
        self.post_load = None

    def aitk_post_load(self, **kwargs):
        self.post_load = kwargs
        return self


def test_preview_dit_keeps_holder_exclusions():
    holder = ImageHolder(FakeLoRA())
    holder.get_quantization_exclude_modules = lambda: ["first", "tmlp*"]
    dit = PostLoadDiT()
    turbo = TurboPreview("turbo.safetensors", lora_factory=lambda path, sd: holder.lora, loader=lambda path, dtype: dit)
    turbo.generate(holder, [preview_config(1.0)], None, save_lora=lambda p: None, lora_path="/tmp/x/lora.safetensors")
    assert dit.post_load["exclude_quant_modules"] == ["first", "tmlp*"] and dit.post_load["qtype"] == "convrot8"


def test_cast_full_precision_leaves_quantized_layers():
    from extensions.slider_tuple.turbo import cast_full_precision

    class Quantized(torch.nn.Module):  # an int8 weight and its fp32 scale
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.zeros(2, 2, dtype=torch.int8))
            self.scale = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    class Modulation(torch.nn.Module):  # a model's own module with a raw parameter (Krea's RMSNorm, modulation)
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Parameter(torch.zeros(2, dtype=torch.float16))

    model = torch.nn.Sequential(torch.nn.Linear(2, 2).float(), Modulation(), Quantized())
    assert cast_full_precision(model, torch.bfloat16) == 3  # linear weight + bias, modulation table
    assert {p.dtype for p in list(model[0].parameters()) + list(model[1].parameters())} == {torch.bfloat16}
    assert model[2].scale.dtype == torch.float32 and model[2].weight.dtype == torch.int8


def test_force_first_sample_on_resume_only():
    from types import SimpleNamespace

    class Trainer(TurboSampleMixin, BaseTrainer):
        def __init__(self, holder, train_config):
            self.train_config = train_config
            super().__init__(holder)

    holder = ImageHolder(FakeLoRA())
    both = SimpleNamespace(skip_first_sample=True, force_first_sample=True)
    trainer = Trainer(holder, both)
    assert both.skip_first_sample is False and trainer.skip_step0_sample  # upstream's start-up check now samples
    assert trainer.sample(0) is None and holder.calls == []  # fresh run: step 0 still skipped
    assert trainer.sample(4750) == 4750 and len(holder.calls) == 1  # resumed: previews rendered
    skip_only = SimpleNamespace(skip_first_sample=True, force_first_sample=False)
    Trainer(ImageHolder(FakeLoRA()), skip_only)
    assert skip_only.skip_first_sample is True  # unchanged without force


def test_preview_lora_on_training_dit():
    """No path: render on Raw with the trainee network live, a fixed preview
    LoRA attached for sampling only, nothing swapped or offloaded."""
    trainee = FakeLoRA()
    holder = FakeHolder(trainee)
    raw, network = holder.model, holder.network
    preview_lora = FakeLoRA()
    made = []
    turbo = TurboPreview(None, steps=8, cfg=1.0, mu=None, lora="turbo_lora.safetensors", lora_strength=0.8,
                         lora_factory=lambda path, sd: pytest.fail("no trainee hooks on Raw"),
                         preview_lora_factory=lambda path, sd, s: made.append((path, sd.model.name, s)) or preview_lora)
    configs = [SimpleNamespace(network_multiplier=m, num_inference_steps=30, guidance_scale=3.0) for m in (0.0, 1.0)]
    saved = []
    turbo.generate(holder, configs, None, save_lora=saved.append, lora_path="/tmp/x/lora.safetensors")
    assert saved == [] and made == [("turbo_lora.safetensors", "raw", 0.8)]
    # Raw, the live trainee network, the pinned steps and CFG
    assert [c[:5] for c in holder.calls] == [("raw", network, None, 8, 1.0)] * 2
    assert holder.model is raw and holder.network is network and raw.moves == []
    assert preview_lora.attached and preview_lora.detached
    turbo.generate(holder, configs, None, save_lora=saved.append, lora_path="/tmp/x/lora.safetensors")
    assert len(made) == 1  # loaded once per run


def test_preview_lora_on_preview_dit():
    trainee, preview_lora = FakeLoRA(), FakeLoRA()
    holder = FakeHolder(trainee)
    turbo = TurboPreview("turbo.safetensors", lora="extra.safetensors", lora_factory=lambda path, sd: trainee,
                         preview_lora_factory=lambda path, sd, s: preview_lora,
                         loader=lambda path, dtype: FakeModule("turbo"))
    turbo.generate(holder, [SimpleNamespace(network_multiplier=1.0, num_inference_steps=30, guidance_scale=3.0)],
                   None, save_lora=lambda p: None, lora_path="/tmp/x/lora.safetensors")
    assert holder.calls[0][0] == "turbo" and preview_lora.detached and trainee.detached


def test_turbo_preview_from_raw_lora(tmp_path):
    lora = tmp_path / "turbo_lora.safetensors"
    lora.write_bytes(b"")
    turbo = turbo_preview_from_raw({"lora": str(lora), "steps": 8, "cfg": 1.0}, arch="moss_image")
    assert (turbo.path, turbo.lora, turbo.lora_strength, turbo.steps, turbo.cfg, turbo.mu) == (None, str(lora), 1.0, 8, 1.0, "keep")
    assert turbo_preview_from_raw({"lora": str(lora), "lora_strength": 0.5}).lora_strength == 0.5
    for raw, message in [({"lora": "x.ckpt"}, "lora must be"),
                         ({"lora": str(tmp_path / "no.safetensors")}, "lora: file not found")]:
        with pytest.raises(ValueError, match=message):
            turbo_preview_from_raw(raw)


def test_split_qkv_lora_matches_comfy_row_patches():
    """ComfyUI: part i of a q/k/v LoRA patches rows [i*n, (i+1)*n) of the fused
    qkv weight by alpha_i/rank_i * up_i @ down_i. The fused LoRA must equal that."""
    from extensions.slider_tuple.turbo import fuse_split_qkv_lora

    torch.manual_seed(0)
    dim, ranks, alphas = 6, (2, 3, 2), (2.0, 1.5, 8.0)
    sd = {}
    for part, r, a in zip("qkv", ranks, alphas):
        sd[f"diffusion_model.layers.0.proj.{part}.lora_down.weight"] = torch.randn(r, dim)
        sd[f"diffusion_model.layers.0.proj.{part}.lora_up.weight"] = torch.randn(dim, r)
        sd[f"diffusion_model.layers.0.proj.{part}.alpha"] = torch.tensor(a)
    sd["diffusion_model.layers.0.proj.o.lora_down.weight"] = torch.randn(2, dim)
    fused = fuse_split_qkv_lora(sd, ["layers.0.proj.qkv", "layers.0.proj.o"])
    assert sorted(fused) == sorted(["diffusion_model.layers.0.proj.qkv.lora_down.weight",
                                    "diffusion_model.layers.0.proj.qkv.lora_up.weight",
                                    "diffusion_model.layers.0.proj.qkv.alpha",
                                    "diffusion_model.layers.0.proj.o.lora_down.weight"])
    comfy = torch.zeros(3 * dim, dim)
    for i, (part, r, a) in enumerate(zip("qkv", ranks, alphas)):
        p = f"diffusion_model.layers.0.proj.{part}"
        comfy[i * dim:(i + 1) * dim] += a / r * sd[p + ".lora_up.weight"] @ sd[p + ".lora_down.weight"]
    p = "diffusion_model.layers.0.proj.qkv"
    rank = fused[p + ".lora_down.weight"].shape[0]
    ours = fused[p + ".alpha"].item() / rank * fused[p + ".lora_up.weight"] @ fused[p + ".lora_down.weight"]
    assert torch.allclose(ours, comfy, atol=1e-5)
    # a DiT with separate q/k/v keeps the LoRA as is
    assert fuse_split_qkv_lora(sd, ["layers.0.proj.q", "layers.0.proj.k", "layers.0.proj.v"]) == sd


def test_preview_sampler_scheduler_options(tmp_path):
    lora = tmp_path / "turbo_lora.safetensors"
    lora.write_bytes(b"")
    turbo = turbo_preview_from_raw({"lora": str(lora), "sampler": "er_sde", "scheduler": "beta"}, arch="moss_image")
    assert turbo.sampling_attrs == {"sample_sampler": "er_sde", "sample_scheduler": "beta"}
    assert turbo_preview_from_raw({"lora": str(lora), "sampler": "er_sde"}, arch="krea2").sampling_attrs == \
        {"sample_sampler": "er_sde"}
    for raw, arch, message in [({"lora": str(lora), "sampler": "dpmpp"}, "moss_image", "sampler: one of"),
                               ({"lora": str(lora), "scheduler": "karras"}, "moss_image", "scheduler: one of")]:
        with pytest.raises(ValueError, match=message):
            turbo_preview_from_raw(raw, arch=arch)

    # set on the holder while sampling, restored after
    holder = FakeHolder(FakeLoRA())
    holder.sample_sampler, holder.sample_scheduler = "euler", "vendor"
    seen = []
    original = holder.generate_single_image
    holder.generate_single_image = lambda *a: seen.append((holder.sample_sampler, holder.sample_scheduler)) or original(*a)
    turbo.preview_lora_factory = lambda path, sd, s: FakeLoRA()
    turbo.generate(holder, [SimpleNamespace(network_multiplier=1.0, num_inference_steps=8, guidance_scale=1.0)],
                   None, save_lora=lambda p: None, lora_path="/tmp/x/lora.safetensors")
    assert seen == [("er_sde", "beta")]
    assert (holder.sample_sampler, holder.sample_scheduler) == ("euler", "vendor")


def test_stock_sampling_with_weight_zero_reuse():
    """turbo_preview without path or lora: the training DiT, the live trainee
    network, each sample's own steps/CFG and schedule_mu; weight 0 once."""
    turbo = turbo_preview_from_raw({}, arch="moss_image")
    assert (turbo.path, turbo.lora, turbo.steps, turbo.cfg, turbo.mu, turbo.reuse_weight_zero) == \
        (None, None, None, None, "keep", True)
    assert turbo_preview_from_raw({"steps": 50, "cfg": 3.0}, arch="moss_image").steps == 50
    holder = ImageHolder(FakeLoRA())
    holder.model_config.model_kwargs["schedule_mu"] = 0.7
    raw, network = holder.model, holder.network
    grid = lambda: [SimpleNamespace(**vars(preview_config(m)), sample_steps=None) for m in (0.0, 1.0)]
    saved = []
    turbo.generate(holder, grid(), None, save_lora=saved.append, lora_path="/tmp/x/lora.safetensors")
    assert saved == [] and raw.moves == []
    assert [c[:5] for c in holder.calls] == [("raw", network, 0.7, 25, 4.0)] * 2
    turbo.generate(holder, grid(), None, save_lora=saved.append, lora_path="/tmp/x/lora.safetensors")
    assert len(holder.calls) == 3  # weight 0 reused
    assert holder.model_config.model_kwargs["schedule_mu"] == 0.7
