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
from typing import Callable, Optional

import torch

from toolkit.basic import flush

COMFY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def strip_comfy_prefix(state_dict: dict) -> dict:
    """ComfyUI checkpoints key the DiT as model.diffusion_model.<name>; Krea 2's DiT expects <name>."""
    out = {}
    for key, value in state_dict.items():
        for prefix in COMFY_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
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
    from safetensors.torch import load_file

    from extensions_built_in.diffusion_models.krea2.krea2 import KREA2_MMDIT_CONFIG
    from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleMMDiTConfig, SingleStreamDiT

    state_dict = strip_comfy_prefix(load_file(path))
    return SingleStreamDiT.load_from_state_dict(state_dict, dtype, config=SingleMMDiTConfig(**KREA2_MMDIT_CONFIG))


def _default_lora_factory(lora_path: str, holder):
    from toolkit.inference_lora import InferenceLoRA

    lora = InferenceLoRA(lora_path).load(holder)
    if lora.unmatched:
        raise RuntimeError(f"turbo preview: {len(lora.unmatched)} trainee LoRA modules do not match the Turbo DiT")
    return lora


class TurboPreview:
    def __init__(self, path: str, steps: int = 8, cfg: float = 1.0, mu: Optional[float] = 1.15,
                 offload_raw: bool = True, lora_factory: Callable = _default_lora_factory, loader: Callable = None):
        self.path = path
        self.steps = steps
        self.cfg = cfg
        self.mu = mu
        self.offload_raw = offload_raw
        self.lora_factory = lora_factory
        self.loader = loader
        self.dit = None

    def load(self, dtype=torch.bfloat16, qtype: str = "convrot8"):
        """Load the Turbo DiT into CPU memory (once)."""
        if self.dit is None:
            dit = self.loader(self.path, dtype) if self.loader else load_krea2_dit(self.path, dtype)
            if hasattr(dit, "aitk_post_load"):
                dit = dit.aitk_post_load(qtype=qtype, dtype=dtype, device=torch.device("cpu"))
            dit.eval().requires_grad_(False)
            self.dit = dit
        return self.dit

    def generate(self, sd, configs, sampler, save_lora: Callable[[str], None], lora_path: str,
                 generate_images: Callable = None):
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
        os.makedirs(os.path.dirname(lora_path) or ".", exist_ok=True)
        save_lora(lora_path)
        try:
            if self.offload_raw:
                raw.to("cpu")
                flush()
            turbo = self.load().to(device)
            sd.model = turbo
            sd.network = None  # the trainee LoRA lives on the Raw DiT's modules; hooks carry it here
            if self.mu is None:
                mkw.pop("schedule_mu", None)
            else:
                mkw["schedule_mu"] = self.mu
            lora = self.lora_factory(lora_path, sd)
            lora.attach()

            def single_image(pipeline, gen_config, *args, **kwargs):
                lora.strength = float(gen_config.network_multiplier)
                return original_single(pipeline, gen_config, *args, **kwargs)

            sd.generate_single_image = single_image
            for config in configs:
                config.num_inference_steps = self.steps
                config.guidance_scale = self.cfg
            generate_images(configs, sampler=sampler)
        finally:
            if own_attr:
                sd.generate_single_image = original_single
            else:
                del sd.generate_single_image
            if lora is not None:
                lora.detach()
            if had_mu:
                mkw["schedule_mu"] = old_mu
            else:
                mkw.pop("schedule_mu", None)
            sd.network = network
            sd.model = raw
            if self.dit is not None:
                self.dit.to("cpu")
            if self.offload_raw:
                raw.to(device)
            flush()
