# slider_tuple: orientation notes (M0)

Checked 2026-10-16 against upstream `0d19b26` (= `origin/main`) and the `mydev`
branch rebased on it. Everything in the handoff's codebase map holds; this file
records only what differs from the handoff or what it does not say.

## Workflow differences from the handoff

- **Branch.** Work happens on `mydev`, not a separate `slider-tuples` branch.
  `mydev` is `origin/main` plus local commits.
- **Upstream remote.** There is no `upstream` remote. This fork's `origin/main`
  is kept identical to `ostris/ai-toolkit` `main` (it was force-synced to
  `0d19b26`), so "rebase on `upstream/main`" means rebase on `origin/main`
  after syncing the fork.
- **Configs live in this folder** (`extensions/slider_tuple/configs/`), not in
  `config/`: the repo's `.gitignore` ignores `/config/*` (personal configs).
  Copy a config to `config/` before editing paths if you prefer.
- **Tests:** `.venv/bin/python -m pytest extensions/slider_tuple/tests`
  (pytest is a uv dev dependency; the repo root is on the pytest pythonpath).

## Line references

All handoff line numbers are correct at `0d19b26`. On `mydev`,
`SDTrainer.py` is shifted by about +93 lines (local grad-norm logging work).
Search by name.

| Item | `0d19b26` | `mydev` |
| --- | --- | --- |
| `SDTrainer.calculate_loss` | 542 | 635 |
| `SDTrainer.get_guided_loss` | 1118 | 1211 |
| `SDTrainer.train_single_accumulation` | 1462 | 1555 |
| guided-loss branch (`unconditional_latents` / `do_guided_loss`) | 2222 | 2315 |
| DOP / blank-prompt preservation | 2283 | 2376 |
| `toolkit/guidance.py` `get_guided_loss_polarity` / `get_guidance_loss` | 405 / 716 | same |
| `config_modules.py` sample `--m` parsing | **1425, 1446** (handoff says ~1491) | 1511, 1532 |

## Behavior the handoff does not mention

- **`polarity` runs its own backward.** `get_guided_loss_polarity` calls
  `loss.backward()` (or `scaler.scale(loss).backward()`) internally and returns
  a detached loss with `requires_grad_(True)`, so the trainer's later backward
  is a no-op. Any `slider_tuple` loss that delegates to or mimics it must keep
  that contract, and it is what makes per-member backward passes
  (`slider_backward_per_member`) fit naturally.
- **`polarity` sums, not averages, its two members:**
  `loss = mse(pos) + mse(neg)`, each a per-element MSE, then `.mean()` over
  everything. The F2 equivalence test must compare against the **sum** over
  members, not the mean.
- **`polarity` touches scheduler state.** Under flow matching it calls
  `sd.noise_scheduler.set_train_timesteps(1000, timestep_type=..., latents=...)`
  on every step (dynamic shift by latent size). The timesteps passed in were
  sampled before that call by the trainer.
- **The guided path rebuilds its own noisy latents** from `batch.latents`,
  `noise` and `timesteps`; it ignores the `noisy_latents` argument. A
  per-loss timestep window (F5) can therefore resample `timesteps` inside
  the `get_guided_loss` override without touching the main loop.
- **The guided path is entered only when** `batch.unconditional_latents is not
  None or self.do_guided_loss`. Tuple datasets have no `unconditional_path`,
  so the trainer must set `self.do_guided_loss = True` for them.
- **`mask_multiplier` reaches `get_guidance_loss` but is dropped** before
  `get_guided_loss_polarity` (only `direct` and `targeted_flow` take it).
  `prior_pred` is likewise passed through `**kwargs` and unused by
  `polarity`.
- **`guidance_type` is not validated.** `DatasetConfig.guidance_type` is a
  plain `kwargs.get` (default `'targeted'`), so `slider_tuple` passes config
  loading without touching `GuidanceType`.
- **Paired images are never latent-cached.** With `cache_latents_to_disk`,
  only the main image's latents are cached; the `unconditional_path` image is
  reloaded and VAE-encoded on every step (`BaseSDTrainProcess.py`, search
  `unconditional_tensor`). Expect extra VAE time per step and keep the VAE
  available on the GPU.
- **Extension discovery imports eagerly.** `get_all_extensions_process_dict()`
  calls `get_process()` on every extension at job start, so the trainer
  module must import cleanly without a GPU and without side effects.
- **Krea 2 quantization.** `quantize: true` already applies
  `Krea2Model.get_quantization_exclude_modules()`; no extra config needed.
  The UI's Krea 2 Raw preset uses `timestep_type: linear`,
  `quantize` + `quantize_te`, `low_vram`, CFG 4. The Turbo preset samples
  with **9** steps and CFG 1 (the handoff says 8; Krea's own guidance should
  decide F6/F7 defaults).
- **Krea 2 needs `model.dtype: bf16` (upstream bug).** `ModelConfig.dtype`
  defaults to `float16` and `vae_dtype` follows it, while the trainer builds
  the model with `train.dtype` (bf16). The Krea 2 pipeline then calls
  `decode_latents(..., dtype=model.torch_dtype)`, which casts latents to bf16
  and feeds them to the fp16 VAE: `RuntimeError: Input type (BFloat16) and
  bias type (Half)`. Reproduced on `0d19b26` at the first decode. The UI's job
  defaults don't set `model.dtype` either. Workaround: set `model.dtype: bf16`
  (all configs here do). Upstreamable fix: cast to `self.vae_torch_dtype` in
  `toolkit/models/v2/vae/qwen_image.py` `decode_latents`, or pass it from the
  pipeline. Candidate for M8.
- **No peak-VRAM logging.** The trainer's `performance_log_every` reports
  timings only. For M1, watch VRAM with
  `/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv -l 1`
  (`nvidia-smi` is not on `PATH` in WSL).
- **LoRA per-sample multipliers** (`network_mixins.py` ~L824) accept a float,
  a list (one per batch item) or a tensor, as the handoff says.

## Base model: BigLove Krea Life 2 fine-tune

From M2 on, configs train on `models/BigLoveKreaLife2_raw_AI-Toolkit` (a
fine-tune of Krea 2 Raw; `/models` is gitignored) instead of `krea/Krea-2-Raw`.

- Format check against the official `raw.safetensors` header: the same 430
  tensors, names and shapes, so the stock Krea 2 loader reads it directly
  (folder path, single `.safetensors` inside). Only the transformer is
  replaced; Qwen3-VL and the Qwen-Image VAE still load from their defaults.
- The fine-tune stores all tensors in bf16, while the official checkpoint
  keeps 174 small tensors (norm scales, modulation, `tmlp`) in fp32. No
  practical difference: Krea 2 uses the default `aitk_cast_on_load`, which
  casts everything to the load dtype (bf16) for both checkpoints.
- M1's 200-step run used the official Raw checkpoint, and the smoke pairs
  were rendered with it; that is fine for plumbing checks.
- **Deployment pairing.** The model author ships this checkpoint as the Raw
  counterpart of the BigLoveKreaLife2 Turbo model, for training LoRAs that
  are used on that Turbo model. The user deploys BigLoveKreaLife2 Turbo as an
  int8_convrot quant, so the handoff's plan (train on Raw, judge on Turbo)
  applies with BigLove Raw/Turbo in place of the official pair.
- **For F7 (Turbo previews):** use the BigLove pair, not the official one.
  Either extract a BigLove Raw->Turbo delta LoRA from the two checkpoints,
  or (simpler, and exactly what is deployed) swap in the user's BigLove Turbo
  int8_convrot file for sampling only. ai-toolkit can load ComfyUI
  int8_convrot checkpoints (`toolkit/util/comfy_quant_import.py`; that format
  is the toolkit's `convrot8` storage). Location of the Turbo file: TBD.
- **Training quantization.** int8_convrot == `qtype: convrot8`, which trains
  with a real int8 tensor-core forward (bit-identical to inference) and a
  straight-through backward. Candidate replacement for `qfloat8` on the DiT,
  pending a 20-step speed/VRAM A/B with `scripts/compare_runs.py`.

## Milestone results

### M1: baseline polarity smoke test (2026-10-16, RTX 5090 32 GB)

Config `configs/krea2_polarity_smoke.yaml`, 5 synthetic pairs from
`scripts/make_smoke_pairs.py`, LoRA rank 16, 200 steps at 1024 px.

| Run | Result |
| --- | --- |
| fp8 (`quantize` + `quantize_te`, qfloat8) | Completed 200 steps. ~6.3 s/step (from sample timestamps, sampling excluded). Peak VRAM: not yet recorded. |
| bf16, `quantize: false` | **Fail.** VRAM saturates and no step completes in reasonable time. |
| bf16, `quantize: false`, `low_vram: true` | **Fail.** Runs, but >5 min/step; unusable. |

Samples (same seed, `--m -1/0/1`): identical at step 0; at step 200 `m=0`
still matches the base render while `m=-1` and `m=1` move in opposite
directions. With these synthetic pairs the LoRA mostly picks up clothing
differences rather than body shape, as expected when identity and outfit
details are not held fixed across a pair. Plumbing confirmed; no quality
claim.

Consequences: fp8 quantization is mandatory for Krea 2 slider training on
32 GB, and each extra grad-carrying member per step costs VRAM, so
`slider_backward_per_member` (F2) matters for k > 2.

### M2: slider_tuple trainer reproduces sd_trainer (2026-10-16)

`scripts/compare_runs.py` on the two smoke configs (BigLove Krea Life 2 Raw,
fp8 qfloat8 DiT + TE, 1024 px, `SEED=42`, 20 steps, sampling off).

| | `sd_trainer` | `slider_tuple_trainer` |
| --- | --- | --- |
| Median s/step | 5.55 | 5.59 |
| Peak GPU memory (whole GPU, nvidia-smi) | 25,651 MiB | 27,485 MiB |

Per-step `loss/loss` agrees within 0.00–0.07% over all 19 logged steps,
including the same high-timestep spike at step 19 (1.0994 vs 1.0994); the
residual is GPU nondeterminism. **Pass.** The 1.8 GB memory gap is not
attributable to the trainer (it only delegates on this path): `memory.used`
is whole-GPU (allocator pool plus any other GPU users under WSL), and WSL's
nvidia-smi cannot report per-process memory. `compare_runs.py` now also
records torch's in-process `max_memory_allocated` / `max_memory_reserved`. (The CPU test already shows the
override is bit-identical to upstream `polarity` on a stub model.)

### Training quantization A/B: qfloat8 vs convrot8 (2026-10-16)

DiT `qtype` only (TE stays qfloat8), BigLove Raw, 1024 px, 20 steps,
`SEED=42`, via `compare_runs.py --set-b model.qtype=convrot8`.

| | qfloat8 | convrot8 |
| --- | --- | --- |
| Median s/step | 5.57 | **4.54** (-18%) |
| Torch peak allocated / reserved | 24.30 / 26.01 GiB | **20.83 / 23.50 GiB** |
| Whole-GPU peak (nvidia-smi) | 25,651 MiB | 24,945 MiB |
| Layer output error vs bf16, Gaussian input | 2.67% | **1.32%** |
| Layer output error vs bf16, 8 outlier channels x30 | 2.67% | **1.48%** |

Layer error: mean relative output error over 9 real BigLove layers
(blocks 0 and 14, attention `wq`/`wo` and MLP `up`/`gate`/`down`), 4096
tokens, each quantized with `toolkit.util.quantize.quantize_module`.
convrot8 also quantizes activations (W8A8) yet halves the error; qfloat8
is weight-only and bounded by fp8's 3 mantissa bits.

**Per-step losses are not comparable across qtypes.** Even with `SEED=42`
the two runs drew different noise/timesteps (step-1 loss 0.273 vs 0.098,
where the zero-initialized LoRA cannot be the cause), so something on one
quantization path consumes the torch RNG. Both runs trained the same 256
LoRA modules. Same-qtype comparisons (M2) do line up. Not chased further;
fix it (e.g. reseed after model load) before relying on cross-qtype loss
comparisons.

**Decision:** both smoke configs now use `qtype: convrot8` for the DiT
(faster, less memory, more accurate, and the same numerics as the
deployed int8_convrot Turbo model).

### M3: per-image slider scales (F1) (2026-10-16)

Dataset keys (`slider_tuple_trainer` only, default off):

```yaml
datasets:
  - folder_path: /data/tuples
    slider_scales: true             # read per-image scales
    slider_scale_normalize: max_abs # optional: divide by the largest |scale|
```

Per image, first match wins: `<stem>.scale` next to the image (one float),
then `scales.json` in `folder_path` (`{"<stem>": float}`), then the
dataset's `network_weight`. Sidecar and `scales.json` must agree where both
define a stem. With `polarity`, an image's scale `s` trains the LoRA at `+s`
on it and `-s` on its `unconditional_path` partner.

Implemented without touching upstream: the trainer pairs the raw dataset
dicts with `self.dataset_configs` (same split, same order) and sets
`FileItemDTO.network_weight` before the first batch. For M8, the generic
part (sidecar/`scales.json` lookup into `network_weight`) could move into
`FileItemDTO`/`DatasetConfig` as a small upstream PR.

Verified (CPU, `tests/test_scales.py`): batch multipliers from
`DataLoaderBatchDTO.get_network_weight_list()` equal the sidecar values
through the real dataloader, including with a worker process; missing
sidecars fall back without error. **Accepted.** No GPU run needed until
tuples (M4).
