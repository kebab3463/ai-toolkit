"""Turbo preview sampling while training on Raw (F7).

ai-toolkit samples previews with the model it trains, so a Raw run previews Raw
with CFG, which says little about the Turbo model the LoRA is deployed on.
TurboPreview renders the training samples with a Turbo DiT instead: the full
weight-swap design (the handoff's fallback, chosen because the deployed model
is a ComfyUI int8_convrot checkpoint, and a Raw->Turbo delta extracted from an
8-bit file would mix quantization error into a small weight difference).

At each sample event: the current trainee LoRA is saved, the Raw DiT is moved
off the GPU (optional), the Turbo DiT moved on and swapped in as ``sd.model``,
the training network detached from sampling, the schedule pinned (mu), steps
and CFG overridden, and the LoRA attached to the Turbo DiT as inference hooks
whose strength follows each image's ``--m``. Everything is restored
afterwards, also on errors.
"""

import os
from typing import Callable, Hashable, Optional

import torch
from toolkit.basic import flush

COMFY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def strip_comfy_prefix(state_dict: dict) -> dict:
    """ComfyUI checkpoints key the DiT as model.diffusion_model.<name>; Krea 2's DiT expects <name>."""
    out = {}
    for key, value in state_dict.items():
        for prefix in COMFY_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        out[key] = value
    return out


def is_comfy_quantized(path: str) -> bool:
    """True if a .safetensors file carries ComfyUI comfy_quant markers (header only)."""
    import json
    import struct

    if not (os.path.isfile(path) and path.endswith(".safetensors")):
        return False
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return any(k.endswith(".comfy_quant") for k in header)


def load_krea2_dit(path: str, dtype: torch.dtype = torch.bfloat16):
    """A Krea 2 SingleStreamDiT from a single-file checkpoint (official or ComfyUI
    layout). ComfyUI int8_convrot layers stay pre-quantized (the toolkit's
    convrot8 backend); later aitk_post_load(qtype="convrot8") skips them."""
    from extensions_built_in.diffusion_models.krea2.krea2 import (
        KREA2_MMDIT_CONFIG,
    )
    from extensions_built_in.diffusion_models.krea2.src.mmdit import (
        SingleMMDiTConfig,
        SingleStreamDiT,
    )
    from safetensors.torch import load_file

    state_dict = strip_comfy_prefix(load_file(path))
    return SingleStreamDiT.load_from_state_dict(
        state_dict, dtype, config=SingleMMDiTConfig(**KREA2_MMDIT_CONFIG)
    )


def load_moss_image_dit(path: str, dtype: torch.dtype = torch.bfloat16):
    """A MOSS-Image transformer from a single-file checkpoint (vendor-merged bf16
    or a ComfyUI int8_convrot repack). Pre-quantized layers stay as they are;
    TurboPreview.load quantizes the rest like the holder would."""
    from extensions_built_in.diffusion_models.moss_image.src.transformer import MossImageTransformer

    return MossImageTransformer.load_model(path, dtype=dtype, qtype="convrot8", quantize_on_load=False)


# Per arch: how to load the preview DiT and how it samples by default. Krea 2
# previews on the distilled Turbo (8 steps, CFG off, fixed mu); MOSS-Image has
# no distilled model, so its SFT previews sample like the vendor pipeline (30
# steps, CFG 3; the shift and the CFG block skip come from the holder).
PREVIEW_ARCHS = {
    "krea2": {"loader": load_krea2_dit, "steps": 8, "cfg": 1.0, "mu": 1.15},
    "moss_image": {"loader": load_moss_image_dit, "steps": 30, "cfg": 3.0, "mu": None},
}


def cast_full_precision(model: torch.nn.Module, dtype: torch.dtype) -> int:
    """Cast every floating tensor outside the quantized layers to ``dtype``;
    returns how many changed. Int8 checkpoints store their unquantized
    tensors in whatever the converter used (float32 in FinePorn_V6.1_INT8,
    float16 in Muse-BSY-Krea2-V4.0), and any of those meeting bf16
    activations either fails in a linear or promotes the stream to float32.
    Norm scales and modulation tables count too, not only Linear weights.
    Quantized layers (any module holding an integer tensor) keep their
    weights and scales as stored."""
    changed = 0
    for module in model.modules():
        tensors = list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))
        if any(not t.is_floating_point() and not t.is_complex() and t.dtype != torch.bool for _, t in tensors):
            continue  # a quantized layer: int8 weight + its scales
        for _, tensor in tensors:
            if tensor.is_floating_point() and tensor.dtype != dtype:
                tensor.data = tensor.data.to(dtype)
                changed += 1
    return changed


def _default_lora_factory(lora_path: str, holder):
    from toolkit.inference_lora import InferenceLoRA

    lora = InferenceLoRA(lora_path).load(holder)
    if lora.unmatched:
        raise RuntimeError(
            f"turbo preview: {len(lora.unmatched)} trainee LoRA modules do not match the Turbo DiT"
        )
    return lora


_SPLIT_QKV = ("q", "k", "v")
_LORA_DOWN = (".lora_down.weight", ".lora_A.weight")
_LORA_UP = (".lora_up.weight", ".lora_B.weight")


def fuse_split_qkv_lora(state_dict: dict, module_names) -> dict:
    """LoRAs on separate ``<p>.q``/``.k``/``.v`` projections onto a DiT that
    fuses them as ``<p>.qkv`` (MOSS-Image), mapped as ComfyUI maps them
    (comfy/lora.py, MOSSImage: part i patches rows [i*n, (i+1)*n) of the
    fused weight, scaled by its own alpha/rank). Written as one LoRA on
    ``<p>.qkv``: the down projections stacked, each pre-scaled by its part's
    alpha/rank, the up projections block-diagonal, alpha = the stacked rank
    (scale 1). Its output is exactly the three parts' outputs in their rows.
    ``module_names``: the DiT's module names. Other keys pass through."""
    from toolkit.inference_lora import _DIT_PREFIXES

    def bare(name):
        for prefix in _DIT_PREFIXES:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    fused_modules = {bare(n) for n in module_names if n.endswith(".qkv") or n.endswith("_qkv")}
    out = dict(state_dict)
    bases = {k[: -len(suffix)] for k in state_dict for suffix in _LORA_DOWN if k.endswith(suffix)}
    for base in sorted(bases):
        if not base.endswith(".q"):
            continue
        prefix = base[: -len("q")]
        if bare(prefix + "qkv") not in fused_modules:
            continue
        parts = []
        for part in _SPLIT_QKV:
            pb = prefix + part
            down = next((state_dict[pb + s] for s in _LORA_DOWN if pb + s in state_dict), None)
            up = next((state_dict[pb + s] for s in _LORA_UP if pb + s in state_dict), None)
            if down is None or up is None:
                break
            alpha = state_dict.get(pb + ".alpha")
            rank = down.shape[0]
            scale = float(alpha.flatten()[0]) / rank if alpha is not None else 1.0
            parts.append((pb, down.float() * scale, up.float()))
        if len(parts) != 3:
            continue  # incomplete triple: left as is (reported unmatched)
        downs = [d for _, d, _ in parts]
        ups = [u for _, _, u in parts]
        rank = sum(d.shape[0] for d in downs)
        fused_up = torch.zeros(sum(u.shape[0] for u in ups), rank)
        row = col = 0
        for u in ups:
            fused_up[row:row + u.shape[0], col:col + u.shape[1]] = u
            row, col = row + u.shape[0], col + u.shape[1]
        for pb, _, _ in parts:
            for suffix in (*_LORA_DOWN, *_LORA_UP, ".alpha"):
                out.pop(pb + suffix, None)
        out[prefix + "qkv.lora_down.weight"] = torch.cat(downs, 0)
        out[prefix + "qkv.lora_up.weight"] = fused_up
        out[prefix + "qkv.alpha"] = torch.tensor(float(rank))
    return out


class _FusingHolder:
    """The holder as InferenceLoRA sees it, plus split-q/k/v fusion after the
    holder's own key conversion."""

    def __init__(self, holder):
        self._holder = holder
        self.model = holder.model
        self.text_encoder = getattr(holder, "text_encoder", None)

    def convert_lora_weights_before_load(self, state_dict):
        convert = getattr(self._holder, "convert_lora_weights_before_load", None)
        if callable(convert):
            state_dict = convert(state_dict)
        return fuse_split_qkv_lora(state_dict, [n for n, _ in self.model.named_modules()])


def _default_preview_lora_factory(lora_path: str, holder, strength: float):
    from toolkit.inference_lora import InferenceLoRA

    lora = InferenceLoRA(lora_path, strength=strength).load(_FusingHolder(holder))
    if lora.convert_error:
        raise RuntimeError(f"turbo preview: converting {lora_path}: {lora.convert_error}")
    if lora.unmatched:
        raise RuntimeError(
            f"turbo preview: {len(lora.unmatched)} modules of the preview LoRA {lora_path} "
            f"do not match the preview DiT (first: {lora.unmatched[:3]})"
        )
    if not lora.entries:
        raise RuntimeError(f"turbo preview: no LoRA weights found in {lora_path}")
    return lora


def weight_zero_key(gen_config) -> Hashable:
    """What makes two weight-0 previews the same image (the LoRA contributes
    nothing at weight 0, so only the inputs of the base Turbo render count).
    Steps and CFG are not part of it: TurboPreview pins them for every image."""
    return (
        getattr(gen_config, "seed", None),
        getattr(gen_config, "prompt", None),
        getattr(gen_config, "negative_prompt", None),
        getattr(gen_config, "width", None),
        getattr(gen_config, "height", None),
        getattr(gen_config, "network_multiplier", None),
    )


KEEP = "keep"  # TurboPreview.mu: leave the holder's schedule_mu as it is


class TurboPreview:
    """Renders samples on a preview DiT with the trainee LoRA attached.

    ``path``: a separate preview DiT (Turbo, SFT), swapped in for the training
    DiT while sampling. ``path=None``: sample on the training DiT itself, the
    trainee LoRA live as in stock sampling. ``lora``: a fixed LoRA (e.g. a
    Turbo/distillation LoRA) added to the preview DiT at ``lora_strength``
    while sampling only, so previews are DiT + preview LoRA + trainee LoRA.
    Neither: stock sampling on the training DiT, plus weight-0 reuse.
    ``steps``/``cfg`` None keep each sample config's own; ``mu`` KEEP keeps
    the holder's schedule_mu (None removes it)."""

    def __init__(
        self,
        path: Optional[str],
        steps: int = 8,
        cfg: float = 1.0,
        mu: Optional[float] = 1.15,
        offload_raw: bool = True,
        reuse_weight_zero: bool = False,
        lora_factory: Callable = _default_lora_factory,
        loader: Callable = None,
        lora: Optional[str] = None,
        lora_strength: float = 1.0,
        preview_lora_factory: Callable = _default_preview_lora_factory,
        sampler: Optional[str] = None,
        scheduler: Optional[str] = None,
    ):
        self.path = path
        self.lora = lora
        self.lora_strength = lora_strength
        self.preview_lora_factory = preview_lora_factory
        self.preview_lora = None  # loaded once: its modules (preview DiT or Raw) persist
        # holder attributes set while sampling (MOSS-Image, Krea 2: sample_sampler /
        # sample_scheduler, e.g. ComfyUI's er_sde + simple); None keeps the model's
        self.sampling_attrs = {k: v for k, v in (("sample_sampler", sampler),
                                                 ("sample_scheduler", scheduler)) if v is not None}
        self.steps = steps
        self.cfg = cfg
        self.mu = mu
        self.offload_raw = offload_raw
        # weight-0 previews are the base Turbo model, identical at every sample
        # step: render each once per process and reuse it
        self.reuse_weight_zero = reuse_weight_zero
        self.weight_zero_images = {}
        self.lora_factory = lora_factory
        self.loader = loader
        self.dit = None

    def load(self, dtype=torch.bfloat16, qtype: str = "convrot8", exclude=None):
        """Load the Turbo DiT into CPU memory (once). ``exclude``: module name
        patterns kept in full precision (the holder's
        get_quantization_exclude_modules, as for its own DiT)."""
        if self.dit is None:
            dit = (
                self.loader(self.path, dtype)
                if self.loader
                else load_krea2_dit(self.path, dtype)
            )
            if hasattr(dit, "aitk_post_load"):
                dit = dit.aitk_post_load(
                    qtype=qtype, dtype=dtype, device=torch.device("cpu"),
                    exclude_quant_modules=exclude,
                )
            cast_full_precision(dit, dtype)
            dit.eval().requires_grad_(False)
            self.dit = dit
        return self.dit

    def generate(
        self,
        sd,
        configs,
        sampler,
        save_lora: Callable[[str], None],
        lora_path: str,
        generate_images: Callable = None,
    ):
        """Render ``configs`` with the Turbo DiT and the current trainee LoRA.

        ``sd`` is the training model holder; ``generate_images`` is its original
        generate_images (pass it when sd.generate_images is wrapped to call us);
        ``save_lora(path)`` writes the trainee LoRA (network.save_weights)."""
        generate_images = generate_images or sd.generate_images
        device = sd.device_torch
        raw, network = sd.model, sd.network
        mkw = sd.model_config.model_kwargs
        had_mu, old_mu = "schedule_mu" in mkw, mkw.get("schedule_mu")
        original_single = sd.generate_single_image
        own_attr = "generate_single_image" in vars(sd)
        lora = None
        separate_dit = self.path is not None
        missing = [k for k in self.sampling_attrs if not hasattr(sd, k)]
        if missing:
            raise ValueError(f"turbo_preview sampler/scheduler: {type(sd).__name__} has no {missing} "
                             "(supported for MOSS-Image and Krea 2)")
        old_attrs = {k: getattr(sd, k) for k in self.sampling_attrs}
        try:
            for k, v in self.sampling_attrs.items():
                setattr(sd, k, v)
            if separate_dit:
                os.makedirs(os.path.dirname(lora_path) or ".", exist_ok=True)
                save_lora(lora_path)
                if self.offload_raw:
                    raw.to("cpu")
                    flush()
                exclude = sd.get_quantization_exclude_modules() if hasattr(sd, "get_quantization_exclude_modules") else None
                turbo = self.load(exclude=exclude).to(device)
                sd.model = turbo
                sd.network = None  # the trainee LoRA lives on the Raw DiT's modules; hooks carry it here
            # else: sample on Raw itself; the trainee network stays live and
            # stock generate_images sets its multiplier per image
            if self.mu is None:
                mkw.pop("schedule_mu", None)
            elif self.mu != KEEP:
                mkw["schedule_mu"] = self.mu
            if self.lora is not None:
                if self.preview_lora is None:
                    self.preview_lora = self.preview_lora_factory(self.lora, sd, self.lora_strength)
                self.preview_lora.attach()
            if separate_dit:
                lora = self.lora_factory(lora_path, sd)
                lora.attach()

            def single_image(pipeline, gen_config, *args, **kwargs):
                strength = float(gen_config.network_multiplier)
                key = (
                    weight_zero_key(gen_config)
                    if self.reuse_weight_zero and strength == 0
                    else None
                )
                if key is not None and key in self.weight_zero_images:
                    return self.weight_zero_images[key]
                if lora is not None:
                    lora.strength = strength
                image = original_single(pipeline, gen_config, *args, **kwargs)
                if key is not None:
                    self.weight_zero_images[key] = image
                return image

            sd.generate_single_image = single_image
            for config in configs:
                if self.steps is not None:
                    config.num_inference_steps = self.steps
                if self.cfg is not None:
                    config.guidance_scale = self.cfg
            generate_images(configs, sampler=sampler)
        finally:
            if own_attr:
                sd.generate_single_image = original_single
            else:
                del sd.generate_single_image
            if lora is not None:
                lora.detach()
            if self.preview_lora is not None:
                self.preview_lora.detach()
            for k, v in old_attrs.items():
                setattr(sd, k, v)
            if had_mu:
                mkw["schedule_mu"] = old_mu
            else:
                mkw.pop("schedule_mu", None)
            sd.network = network
            sd.model = raw
            if self.dit is not None:
                self.dit.to("cpu")
            if separate_dit and self.offload_raw:
                raw.to(device)
            flush()


TURBO_PREVIEW_KEYS = {
    "path",
    "steps",
    "cfg",
    "mu",
    "offload_raw",
    "reuse_weight_zero",
    "lora",
    "lora_strength",
    "sampler",
    "scheduler",
}


def turbo_preview_from_raw(
    raw: Optional[dict], arch: str = "krea2", **defaults
) -> Optional[TurboPreview]:
    """A TurboPreview from a ``turbo_preview:`` config block, or None without one.

        turbo_preview:
          path: /path/to/preview_dit.safetensors  # Krea 2 Turbo / MOSS-Image SFT DiT (official or ComfyUI int8_convrot)
          steps: 8                 # per-arch default (PREVIEW_ARCHS): Krea 2 8, MOSS-Image 30
          cfg: 1.0                 # Krea 2 1.0 (0-normalized: off), MOSS-Image 3.0
          mu: 1.15                 # Krea 2 only: fixed schedule shift of the distilled Turbo
          offload_raw: true        # move the training DiT to CPU while sampling
          reuse_weight_zero: true  # render each weight-0 preview once per run
          lora: /path/to/turbo_lora.safetensors  # optional: a fixed LoRA on the preview DiT, sampling only
          lora_strength: 1.0
          sampler: er_sde          # "euler" (the model's own) or ComfyUI's "er_sde"
          scheduler: simple        # "vendor" (the model's own), or ComfyUI's "simple" / "beta"

    Without ``path``, previews render on the training DiT itself (Raw); then
    ``lora`` is required: e.g. a Turbo LoRA, giving Raw + Turbo LoRA + trainee.

    ``arch`` is the training model's (``model.arch``, variant suffix ignored);
    it picks the loader and the sampling defaults. ``defaults`` override those
    defaults (not the block's own values)."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("turbo_preview must be a mapping with at least a path")
    unknown = set(raw) - TURBO_PREVIEW_KEYS
    if unknown:
        raise ValueError(
            f"turbo_preview: unknown keys {sorted(unknown)} (known: {sorted(TURBO_PREVIEW_KEYS)})"
        )
    family = (arch or "krea2").split(":")[0]
    if family not in PREVIEW_ARCHS:
        raise ValueError(f"turbo_preview: no preview DiT loader for arch {arch!r} "
                         f"(supported: {sorted(PREVIEW_ARCHS)})")
    spec = PREVIEW_ARCHS[family]
    # the per-arch steps/cfg/mu are the preview DiT's; on the training DiT
    # (no path) each sample keeps its own sample-block settings
    own_dit = raw.get("path") is not None
    options = {
        "steps": spec["steps"] if own_dit else None,
        "cfg": spec["cfg"] if own_dit else None,
        "mu": spec["mu"] if own_dit else KEEP,
        "offload_raw": True,
        "reuse_weight_zero": True,
        **defaults,
        **raw,
    }
    path = options.pop("path", None)
    lora = options.pop("lora", None)
    for key, value in (("path", path), ("lora", lora)):
        if value is None:
            continue
        if not (isinstance(value, str) and value.endswith(".safetensors")):
            raise ValueError(f"turbo_preview.{key} must be a .safetensors file path")
        if not os.path.isfile(value):
            raise ValueError(f"turbo_preview.{key}: file not found: {value}")
    if options.get("sampler") is not None or options.get("scheduler") is not None:
        from toolkit.comfy_flow_sampling import SAMPLERS, SCHEDULERS

        if options.get("sampler") not in (None, *SAMPLERS):
            raise ValueError(f"turbo_preview.sampler: one of {SAMPLERS}")
        if options.get("scheduler") not in (None, "vendor", *SCHEDULERS):
            raise ValueError(f"turbo_preview.scheduler: one of {('vendor',) + SCHEDULERS}")
    if options["steps"] is not None and int(options["steps"]) < 1:
        raise ValueError("turbo_preview.steps must be >= 1")
    return TurboPreview(
        path,
        steps=None if options["steps"] is None else int(options["steps"]),
        cfg=None if options["cfg"] is None else float(options["cfg"]),
        mu=options["mu"] if options["mu"] in (None, KEEP) else float(options["mu"]),
        offload_raw=bool(options["offload_raw"]),
        reuse_weight_zero=bool(options["reuse_weight_zero"]),
        loader=spec["loader"],
        lora=lora,
        lora_strength=float(options.get("lora_strength", 1.0)),
        sampler=options.get("sampler"),
        scheduler=options.get("scheduler"),
    )


class TurboSampleMixin:
    """Renders a trainer's samples on a Turbo DiT. Mix in before the trainer
    class (``class T(TurboSampleMixin, DiffusionTrainer)``) and set
    ``self.turbo_preview``; without one, sampling is the trainer's own."""

    turbo_preview: Optional[TurboPreview] = None
    turbo_preview_folder = "turbo_preview"  # under the run folder: the trainee LoRA saved for each sample event
    skip_step0_sample = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolve_first_sample()

    def resolve_first_sample(self):
        """skip_first_sample + force_first_sample: skip the step-0 previews of a
        fresh run (the base model), but sample on resume, e.g. a step whose
        previews were lost to a crash. Upstream lets skip win over force, so
        here skip is handed to sample() as "skip step 0 only"."""
        train = getattr(self, "train_config", None)
        if train is not None and train.skip_first_sample and train.force_first_sample:
            train.skip_first_sample = False
            self.skip_step0_sample = True

    def sample(self, step=None, is_first=False):
        if self.skip_step0_sample and step is not None and step <= 1 and not is_first:
            from toolkit.print import print_acc

            print_acc("Skipping step-0 samples (skip_first_sample; force_first_sample samples on resume)")
            return None
        if self.turbo_preview is None:
            return super().sample(step, is_first)
        from toolkit.print import print_acc

        turbo = self.turbo_preview
        original = self.sd.generate_images
        lora_path = os.path.join(
            self.save_root,
            self.turbo_preview_folder,
            "turbo_preview_lora.safetensors",
        )

        def turbo_generate(configs, sampler=None, **kwargs):
            on = "preview DiT" if turbo.path else "training DiT"
            extra = f" + {os.path.basename(turbo.lora)} @ {turbo.lora_strength:g}" if turbo.lora else ""
            print_acc(
                f"Turbo preview ({len(configs)} images, "
                f"{turbo.steps or 'sample-config'} steps, {on}{extra})"
            )
            turbo.generate(
                self.sd,
                configs,
                sampler,
                save_lora=lambda path: self.network.save_weights(
                    path, dtype=torch.float32
                ),
                lora_path=lora_path,
                generate_images=original,
            )

        self.sd.generate_images = turbo_generate
        try:
            return super().sample(step, is_first)
        finally:
            del self.sd.generate_images
