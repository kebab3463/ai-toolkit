"""CPU stand-ins for the model, network and batch that guided slider losses touch.

The stub model's prediction is linear in the latents and in the LoRA multiplier:

    v = base(x) + m * lora(x)

with ``base`` frozen and ``lora`` trainable (1x1 convs over channels), so losses
and LoRA gradients can be compared exactly without a GPU or real weights.
"""

from types import SimpleNamespace
from typing import List, Optional

import torch

from toolkit.prompt_utils import PromptEmbeds


class StubNetwork:
    def __init__(self):
        self.is_active = False
        self.multiplier = 1.0


class StubScheduler:
    def __init__(self):
        self.set_train_timesteps_calls = 0

    def set_train_timesteps(self, *args, **kwargs):
        self.set_train_timesteps_calls += 1


class StubModel:
    """Implements the subset of BaseModel used by toolkit/guidance.py."""

    def __init__(self, channels: int = 4, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.device_torch = torch.device("cpu")
        self.torch_dtype = "fp32"
        self.is_flow_matching = True
        self.noise_scheduler = StubScheduler()
        self.network = StubNetwork()
        self.unet = torch.nn.Module()
        self.base = torch.nn.Conv2d(channels, channels, 1, bias=False)
        self.lora = torch.nn.Conv2d(channels, channels, 1, bias=False)
        with torch.no_grad():
            self.base.weight.copy_(torch.randn(self.base.weight.shape, generator=gen))
            self.lora.weight.copy_(0.1 * torch.randn(self.lora.weight.shape, generator=gen))
        self.base.requires_grad_(False)

    def lora_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.lora.parameters())

    def zero_lora_grad(self):
        for p in self.lora_parameters():
            p.grad = None

    def add_noise(self, latents, noise, timesteps):
        # rectified flow: x_t = (1 - t) x + t eps, t in [0, 1]
        t = (timesteps.float() / 1000.0).view(-1, *([1] * (latents.dim() - 1)))
        return (1.0 - t) * latents + t * noise

    def condition_noisy_latents(self, latents, batch):
        return latents

    def encode_images(self, pixels, device=None, dtype=None):
        # stand-in VAE: 8x average pooling, padded from 3 to the model's channels
        x = torch.nn.functional.avg_pool2d(pixels.float(), 8)
        pad = self.base.in_channels - x.shape[1]
        return torch.cat([x, x[:, :1].expand(-1, pad, -1, -1)], dim=1) if pad > 0 else x

    def _multiplier_tensor(self, batch_size: int) -> torch.Tensor:
        m = self.network.multiplier if self.network.is_active else 0.0
        if isinstance(m, (int, float)):
            m = [float(m)] * batch_size
        return torch.tensor(m, dtype=torch.float32).view(-1, 1, 1, 1)

    def predict_noise(self, latents, conditional_embeddings=None, timestep=None, guidance_scale=1.0, **kwargs):
        m = self._multiplier_tensor(latents.shape[0])
        return self.base(latents) + m * self.lora(latents)


def make_batch(
    latents: torch.Tensor,
    unconditional_latents: Optional[torch.Tensor] = None,
    guidance_type: str = "polarity",
    network_weight: float = 1.0,
):
    dataset_config = SimpleNamespace(guidance_type=guidance_type, network_weight=network_weight)
    file_items = [SimpleNamespace(dataset_config=dataset_config, network_weight=network_weight) for _ in range(latents.shape[0])]
    return SimpleNamespace(
        file_items=file_items,
        latents=latents,
        unconditional_latents=unconditional_latents,
        get_network_weight_list=lambda: [x.network_weight for x in file_items],
    )


def make_train_config(timestep_type: str = "linear"):
    return SimpleNamespace(linear_timesteps=False, linear_timesteps2=False, timestep_type=timestep_type, dtype="fp32")


def make_embeds(batch_size: int, tokens: int = 3, dim: int = 8, seed: int = 0) -> PromptEmbeds:
    gen = torch.Generator().manual_seed(seed)
    return PromptEmbeds(torch.randn(batch_size, tokens, dim, generator=gen))


def make_pair(batch_size: int = 2, channels: int = 4, size: int = 8, seed: int = 1):
    """Latents for a 'before'/'after' pair plus shared noise and timesteps."""
    gen = torch.Generator().manual_seed(seed)
    shape = (batch_size, channels, size, size)
    after = torch.randn(shape, generator=gen)
    before = torch.randn(shape, generator=gen)
    noise = torch.randn(shape, generator=gen)
    timesteps = torch.randint(1, 1000, (batch_size,), generator=gen).float()
    return after, before, noise, timesteps


class DataModelStub:
    """The model attributes AiToolkitDataset / FileItemDTO read for a plain image
    dataset (no caching, no controls), so the real dataloader runs on CPU."""

    is_xl = is_vega = is_ssd = False
    is_audio_model = is_multimodal_llm = False
    encode_control_in_text_embeddings = encode_first_frame_in_text_embeddings = False
    text_embedding_uses_target_size = use_raw_control_images = False
    load_rgba = dopsd_self_ref = False
    te_padding_side = "right"
    vae = None
    unet = None

    def get_bucket_divisibility(self):
        return 16

    def get_latent_space_version(self):
        return "sd1"

    def get_text_embedding_space_version(self):
        return "sd1"
