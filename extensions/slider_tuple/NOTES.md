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
