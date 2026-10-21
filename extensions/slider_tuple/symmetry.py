"""Symmetry loss for sliders: only the odd part of the LoRA's effect may change the prediction.

With the LoRA at multiplier s, a linear slider's prediction is v(s) = v0 + s d:
turning it up and down moves the image by opposite amounts. Centering cancels
the members' shared content in d, but not in the part of the response that is
even in s (from the network's nonlinearity, it grows with the LoRA). That part
is trained toward the members' common content, so every image drifts toward
the training set whichever way the slider is turned (NOTES.md, "Run 3 results").
This loss removes it on the training images themselves:

    L_sym = strength * mean( ((v(+s) + v(-s)) / 2 - v(0))^2 )

with the batch's own noisy latents, caption, noise and timesteps, v(0) the
frozen model and s per item drawn from [0, max_multiplier]. The odd part
cancels inside (v(+s) + v(-s)) / 2, so the slider itself is not penalized.

Memory: the exact gradient is accumulated with one gradient-carrying forward at
a time. The residual r = (v(+s) + v(-s)) / 2 - v(0) takes v(+s) from its own
gradient pass (v(0), v(-s) from no-grad passes), then dL/dtheta = strength / N * <r, dv(+s) + dv(-s)> is
backpropagated through v(+s) and v(-s) separately. Four forwards (v0 and v(-s)
without gradients, v(+s) and v(-s) with), one backward graph alive at a time.
"""

from typing import Sequence

import torch

from toolkit.train_tools import get_torch_dtype


def symmetry_loss(
    sd,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    embeds,
    multipliers: Sequence[float],
    strength: float,
    pred_kwargs: dict,
    batch,
) -> tuple:
    """Returns (detached weighted loss, unweighted mse float); backward already done.

    embeds: the batch's PromptEmbeds; multipliers: s >= 0 per batch item."""
    dtype = get_torch_dtype(sd.torch_dtype)
    device = sd.device_torch
    s = [float(m) for m in multipliers]

    def predict():
        return sd.predict_noise(
            latents=noisy, conditional_embeddings=embeds, timestep=timesteps, guidance_scale=1.0, **pred_kwargs
        )

    with torch.no_grad():
        latents = latents.to(device, dtype=dtype).detach()
        noise = noise.to(device, dtype=dtype).detach()
        noisy = sd.condition_noisy_latents(sd.add_noise(latents, noise, timesteps).detach(), batch)
        embeds = embeds.to(device, dtype=dtype)
        sd.network.is_active = False
        v0 = predict().detach().float()
        sd.network.is_active = True
        sd.network.multiplier = [-m for m in s]
        v_minus = predict().detach().float()

    # d/dtheta of strength * mean(r^2) = strength / N * <r, dv(+s) + dv(-s)>, r fixed at this theta
    sd.network.multiplier = s
    pred_plus = predict()
    residual = (pred_plus.detach().float() + v_minus) / 2 - v0
    mse = residual.pow(2).mean()
    scale = strength / residual.numel()
    (scale * (residual.to(pred_plus.dtype) * pred_plus).sum()).backward()
    del pred_plus
    sd.network.multiplier = [-m for m in s]
    pred_minus = predict()
    (scale * (residual.to(pred_minus.dtype) * pred_minus).sum()).backward()
    sd.network.multiplier = s
    return (mse * strength).detach(), mse.item()
