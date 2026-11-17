"""ComfyUI's flow-model samplers and schedulers, ported for training previews.

Ported from ComfyUI (comfy/samplers.py, comfy/k_diffusion/sampling.py,
comfy/model_sampling.py) for CONST (rectified-flow) models. ComfyUI derives
schedules from the model's sigma table, which depends on its sampling class:

  - ``discrete_flow`` (ModelSamplingDiscreteFlow, multiplier 1): 1000 entries,
    time_snr_shift(shift, i / 1000). MOSS-Image: shift 5.
  - ``flux`` (ModelSamplingFlux): 10000 entries, flux_time_shift(mu, 1,
    i / 10000). Krea 2: mu 1.15 in ComfyUI.

Schedulers ``simple`` and ``beta`` (alpha = beta = 0.6); sampler ``er_sde``
(ER-SDE-Solver-3, s_noise 1, max_stage 3) with ComfyUI's first-sigma offset.

The model is ``velocity(x, sigma) -> v`` (noise - clean, sigma 1 = noise);
denoised = x - sigma * v, as CONST.calculate_denoised. Noise comes from the
caller's generator, so images match ComfyUI's in behaviour, not bit for bit.
"""

import math
from typing import Callable, Optional

import numpy
import torch

SCHEDULERS = ("simple", "beta")
SAMPLERS = ("euler", "er_sde")


def time_snr_shift(alpha: float, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def flux_time_shift(mu: float, sigma: float, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class FlowSampling:
    """ComfyUI's model_sampling for a flow model: the sigma table and
    percent_to_sigma."""

    def __init__(self, kind: str, shift: float):
        if kind == "discrete_flow":
            self.table = time_snr_shift(shift, torch.arange(1, 1001, dtype=torch.float32) / 1000)
            self._sigma = lambda t: time_snr_shift(shift, t)
        elif kind == "flux":
            self.table = flux_time_shift(shift, 1.0, torch.arange(1, 10001, dtype=torch.float32) / 10000)
            self._sigma = lambda t: flux_time_shift(shift, 1.0, t)
        else:
            raise ValueError(f"unknown flow sampling kind {kind!r}")
        self.kind, self.shift = kind, shift

    def percent_to_sigma(self, percent: float) -> float:
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return self._sigma(1.0 - percent)

    def sigmas(self, scheduler: str, steps: int) -> torch.Tensor:
        if scheduler == "simple":
            return simple_sigmas(self.table, steps)
        if scheduler == "beta":
            return beta_sigmas(self.table, steps)
        raise ValueError(f"unknown scheduler {scheduler!r} (ComfyUI ports: {SCHEDULERS})")


def simple_sigmas(table: torch.Tensor, steps: int) -> torch.Tensor:
    stride = len(table) / steps
    sigmas = [float(table[-(1 + int(x * stride))]) for x in range(steps)]
    return torch.FloatTensor(sigmas + [0.0])


def beta_sigmas(table: torch.Tensor, steps: int, alpha: float = 0.6, beta: float = 0.6) -> torch.Tensor:
    import scipy.stats

    total = len(table) - 1
    ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
    ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total)
    sigmas, last = [], -1
    for t in ts:
        if t != last:
            sigmas.append(float(table[int(t)]))
        last = t
    return torch.FloatTensor(sigmas + [0.0])


def _half_log_snr(sigma: torch.Tensor) -> torch.Tensor:
    # CONST: log(alpha_t / sigma_t) = log((1 - sigma) / sigma)
    return sigma.logit().neg()


def _offset_first_sigma(sigmas: torch.Tensor, sampling: FlowSampling, percent: float = 1e-4) -> torch.Tensor:
    if len(sigmas) > 1 and sigmas[0] >= 1:
        sigmas = sigmas.clone()
        sigmas[0] = sampling.percent_to_sigma(percent)
    return sigmas


@torch.no_grad()
def sample_er_sde(
    velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    sigmas: torch.Tensor,
    sampling: FlowSampling,
    generator: Optional[torch.Generator] = None,
    s_noise: float = 1.0,
    max_stage: int = 3,
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """ComfyUI's sample_er_sde (default noise scaler, 200 integration points).
    ``callback(step_index, sigma, denoised)``."""

    def noise_scaler(v):
        return v * ((v ** 0.3).exp() + 10.0)

    points = 200.0
    point_index = torch.arange(0, points, dtype=torch.float32, device=x.device)
    sigmas = _offset_first_sigma(sigmas, sampling)
    er_lambdas = _half_log_snr(sigmas).neg().exp()  # sigma_t / alpha_t

    old_denoised = old_denoised_d = None
    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        denoised = x - sigma * velocity(x, sigma)
        if callback is not None:
            callback(i, sigma, denoised)
        stage = min(max_stage, i + 1)
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lam_s, lam_t = er_lambdas[i], er_lambdas[i + 1]
            alpha_s = sigmas[i] / lam_s
            alpha_t = sigmas[i + 1] / lam_t
            r_alpha = alpha_t / alpha_s
            r = noise_scaler(lam_t) / noise_scaler(lam_s)

            x = r_alpha * r * x + alpha_t * (1 - r) * denoised  # stage 1 (Euler)
            if stage >= 2:
                dt = lam_t - lam_s
                step = -dt / points
                lam_pos = lam_t + point_index * step
                scaled = noise_scaler(lam_pos)
                s = torch.sum(1 / scaled) * step
                denoised_d = (denoised - old_denoised) / (lam_s - er_lambdas[i - 1])
                x = x + alpha_t * (dt + s * noise_scaler(lam_t)) * denoised_d
                if stage >= 3:
                    s_u = torch.sum((lam_pos - lam_s) / scaled) * step
                    denoised_u = (denoised_d - old_denoised_d) / ((lam_s - er_lambdas[i - 2]) / 2)
                    x = x + alpha_t * ((dt ** 2) / 2 + s_u * noise_scaler(lam_t)) * denoised_u
                old_denoised_d = denoised_d
            if s_noise > 0:
                device = generator.device if generator is not None else x.device
                noise = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype).to(x.device)
                x = x + alpha_t * noise * s_noise * (lam_t ** 2 - lam_s ** 2 * r ** 2).sqrt().nan_to_num(nan=0.0)
        old_denoised = denoised
    return x


def run_sampler(sampler: str, velocity, latents, sigmas, sampling: FlowSampling, generator=None, callback=None):
    """``euler`` or ``er_sde`` over ``sigmas`` (descending, ending in 0)."""
    if sampler == "er_sde":
        return sample_er_sde(velocity, latents, sigmas, sampling, generator=generator, callback=callback)
    if sampler == "euler":
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            v = velocity(latents, sigma)
            if callback is not None:
                callback(i, sigma, latents - sigma * v)
            latents = latents + (sigma_next - sigma) * v
        return latents
    raise ValueError(f"unknown sampler {sampler!r} (supported: {SAMPLERS})")
