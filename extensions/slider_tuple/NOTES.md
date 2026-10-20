# slider_tuple: orientation notes (M0)

Checked 2026-10-16 against upstream `0d19b26` (= `origin/main`) and the `mydev`
branch rebased on it. Everything in the handoff's codebase map holds; this file
records only what differs from the handoff or what it does not say. For the
step-by-step workflow (data -> training -> sweep), see `README.md`.

**Guiding doc.** Since 2026-10-19 development follows
`docs/Handoff Tuple Slider Training for Krea 2 in an ai-toolkit Fork.md`, which
replaces the earlier "Image-Pair Slider Training" handoff. Its F2 (centered
multipliers from measured levels) superseded the first M4 implementation; see
"M4 revised" below. Codebase map, M0-M3, F3 and F4 are unchanged.

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
  pending a 20-step speed/VRAM A/B with `scripts/dev/compare_runs.py`.

## Milestone results

### M1: baseline polarity smoke test (2026-10-16, RTX 5090 32 GB)

Config `configs/dev/krea2_polarity_smoke.yaml`, 5 synthetic pairs from
`scripts/dev/make_smoke_pairs.py`, LoRA rank 16, 200 steps at 1024 px.

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

`scripts/dev/compare_runs.py` on the two smoke configs (BigLove Krea Life 2 Raw,
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

### M4: tuple datasets and `guidance_type: slider_tuple` (F2) (2026-10-16)

Usage (see `tuples.py` / `manifest.py` docstrings for every key):

```yaml
      datasets:
        - slider_tuples: datasets/tuples/tuples.jsonl
          guidance_type: slider_tuple
          resolution: [1024]
          cache_latents_to_disk: true
      slider:
        members_per_step: all        # or an int >= 2
        level_norm: median_span      # kappa: median_span | max_span | none
        min_span: 0.3                # in kappa units
        weight_cap_pct: 95
        backward_per_member: false
        debug_dump_steps: 0
```

(Keys as of the revised F2; see "M4 revised" below.)

How it works: each tuple's first member (the anchor) becomes a dataloader item
via a generated JSON dataset (`<run>/slider_tuple/dataset_<i>.json`, the
`{path: {"caption": ...}}` format upstream reads), so upstream buckets, crops,
flips and caches latents and text embeddings for it. At loss time the trainer
loads the other members through the anchor file item's own resize/crop/flip
path and caches their latents in memory per crop. Members are sampled per
step, ordered by descending level, centered (see "M4 revised") and arranged in
slots so a pair is polarity's `[positive, negative]` batch.

Design notes and deviations:

- **VAE latents are sampled, not the mean.** `QwenImageVAE.encode_images`
  uses `latent_dist.sample()`. Member latents are encoded once and cached,
  like upstream's own latent cache does for anchors; polarity instead
  re-encodes its partner every step. Consequence: GPU loss curves cannot match
  polarity step for step (polarity consumes the RNG before noise/timesteps).
- **Batch members per step.** All tuples in a batch train the same number of
  members: `min(members_per_step, member count)` over the batch.
- **Restrictions (v1):** tuple datasets cannot be mixed with other datasets
  or be `is_reg`; they need buckets; augmentations, `alpha_mask`, video and an
  explicit `folder_path` / `dataset_path` are rejected. Manifest masks are
  parsed but unused until M5.
- **Upstream oddity:** `FileItemDTO.__init__` sets `flip_y` from the
  `flip_x` kwarg (`data_transfer_object/data_loader.py`). Members copy
  whatever flips the anchor file item ends up with, so they stay aligned
  either way. Candidate one-line upstream fix (M8).

First implementation, CPU verification (80 tests): a +/-1 tuple run through the real dataloader and
the trainer equals `get_guided_loss_polarity` in loss and LoRA gradient on the
same pair, noise and timestep (batch 1 and 2); members load bit-identical to
upstream's image path for every flip combination; zero-sum weights give
`sum(w*s) = 0` (e.g. -0.6/+1.4 -> 1.4/0.6) and zero LoRA gradient for
identical-image tuples; per-member backward equals the single forward.

GPU check (`compare_runs.py`, 20 steps, `SEED=42`, convrot8, BigLove Raw, same
5 pairs; polarity side reused from the convrot8 A/B run):

| | polarity (`sd_trainer`) | slider_tuple |
| --- | --- | --- |
| Median s/step | 4.54 | 4.45 |
| Torch peak allocated / reserved | 20.83 / 23.50 GiB | 20.82 / 23.50 GiB |
| Mean / median loss (19 steps) | 0.212 / 0.187 | 0.341 / 0.238 |
| Same, without its one 2.01 step | | 0.248 / 0.208 |

Per-step losses differ from step 1 (different noise/timesteps, see the VAE
note above), so only the distributions are comparable. The mean gap comes from
a single high-loss step; a permutation test finds no difference (p = 0.64 for
medians, 0.23 for means; per-step SD 0.14-0.43). The step-0 debug dump shows
the members crop-aligned (identical framing; differences confined to the
person and the renders' own background variation). Same cost as polarity
(member latents are encoded once instead of every step). **Accepted** on the
CPU equivalence tests plus this GPU smoke run; an exact fixed-input check on
the real model (both losses on one batch, noise and timestep) is possible if
stronger GPU evidence is wanted.

### M4 revised: centered multipliers from measured levels (2026-10-19)

Per the revised handoff's F2:

- **Manifest:** members have `level` (measured), `role` (`real` | `edit`, at
  most one real), optional `quality` in [0, 1] and `level_var` > 0; tuples may
  have `quality` and `original` (untouched photo, metrics/dumps only). The
  old `scale` / `weight` keys and the "scale 0 = untrained reference" idea are
  gone: every member trains.
- **Centering** (`centering.py`): per step, over the sampled members,
  s_i = (a_i - a_w) / kappa (w-weighted mean), so sum(w_i s_i) = 0 exactly.
  w_i = quality * 1/level_var, mean 1 per tuple; loss weight omega * w_i with
  omega = tuple quality. A pair is +-delta/(2 kappa).
- **kappa** (`level_norm`) over the training set; `min_span` (kappa units)
  drops tuples and constrains sampled subsets; `weight_cap_pct` caps tuple
  leverage omega * sum(w s^2). Dropped/capped tuples are printed at startup and
  left out of the anchor dataset.
- **Removed:** `zero_sum_weights`, `force_opposite_signs` (centering makes
  both redundant). `slider_scale_normalize` is for F1 datasets only.
- **Test hook:** `center_offset` (warns at startup).
- **Logs:** `slider/weighted_scale_sum`, `slider/span` (kappa units),
  `slider/loss_rank<r>` (rank 0 = highest level, normally the real photo),
  `slider/members_per_step`; anchors still log `slider/anchor_loss`.
- **Config naming:** the handoff's flat `slider_*` names live in the
  process-level `slider:` block without the prefix (already in use; the
  handoff marks names as proposals).

Acceptance (CPU, 118 tests total): (1) pair 2 kappa apart, unit weights ==
polarity at weight 1 in loss and LoRA gradient, through the real dataloader
and trainer (batch 1 and 2); (2) identical images -> zero LoRA gradient for
any weights; (3) levels (1.4, 0.9, -0.2), weights (1, 0.5, 1) -> scales
(0.74, 0.24, -0.86)/kappa with zero weighted sum; (4) `center_offset` makes the
shared-content gradient nonzero, and a pair offset by delta/2kappa gets no
gradient from its lower member; (5) dropped and capped tuples reported at
startup; (6) debug dump shows all members (with rank/role/level/multiplier)
plus the original. The M4 GPU smoke result above still holds: for the +-1
smoke pairs with `level_norm: none` the revised math gives the same
multipliers.

### M5: masked loss (F3) and anchor preservation (F4) (2026-10-16)

**One mask per tuple (2026-10-20).** At LoRA init the per-pixel gradient is
sum_i w_i s_i M_i^2 (v0 - y_i). With y_i = c + d_i (shared content c), the
shared term is M^2 (v0 - c) sum_i w_i s_i = 0 only if every member has the same
mask M; with per-member masks it is sum_i w_i s_i M_i^2 (v0 - c), nonzero
wherever masks differ between members with nonzero multipliers. Body masks
differ exactly along the silhouette, so per-member masks leak shared content
there. Hence: a tuple-level `"mask"` in `tuples.jsonl` (applies to members
without their own), built as the union of the members' bodies at 1.0, the
(identical) pasted-back head at 0, background 0.1-0.3. The trainer warns at
startup about tuples whose members are masked differently. Test:
`test_different_member_masks_leak_shared_content`.

**F3, masks.** Per-member `"mask"` in `tuples.jsonl` (grayscale, same size as
the image: ~1 body, 0 head, 0.1-0.3 background). Masks follow the anchor's
flip/resize/crop and the dataset's `invert_mask` / `mask_min_value` like
upstream `load_mask_image` (without its random blur), are resized to the
latent the way SDTrainer builds `mask_multiplier`, and are normalized to
mean 1 **per member** (upstream normalizes over the whole batch; per member
keeps each member's loss scale independent of the others' mask coverage).

Upstream patch (separate commit, upstreamable): `polarity` now honors
`mask_multiplier`; the unconditional image reuses the conditional image's
mask (same stem and framing). A separate partner mask would need dataloader
changes; tuple datasets already support one mask per member.

**F4, anchors.** `slider:` keys `anchor_strength` (0 = off), `anchor_prompts`,
`anchor_prob`, `anchor_max_multiplier`, `anchor_images`. Anchor prompts are
encoded before SDTrainer unloads the text encoder; the anchor loss uses the
step's noise and full-range timesteps, its own backward, and logs
`slider/anchor_loss` (unweighted MSE).

CPU verification (100 tests): all-ones mask == unmasked; changing latents
under a zero mask region leaves the LoRA gradient unchanged (and does change
it without the mask), for both slider_tuple and patched polarity; member
masks equal upstream `load_mask_image`; the anchor target is the LoRA-off
prediction and the anchor loss falls under gradient descent.

**Data the three-way comparison needs** (tuples only / + masks / + masks +
anchors): real tuples per the handoff's dataset contract (15-20 to start),
real per-member masks, and the user's anchor prompts. The comparison's
LPIPS-on-anchor-prompts criterion needs the F6 sweep (M6).

GPU smoke (`compare_runs.py`, `krea2_tuple_smoke_m5.yaml` vs the M4 tuple run,
20 steps, `SEED=42`, placeholder band masks, 8 scene anchor prompts, strength 1):

| | tuples | + masks + anchors |
| --- | --- | --- |
| Median s/step | 4.45 | 7.04 (+58%, the anchor's two extra forwards) |
| Torch peak allocated / reserved | 20.82 / 23.50 GiB | 20.85 / 23.55 GiB |
| Mean / median loss | 0.341 / 0.238 | 0.338 / 0.213 |

The anchor draws prompts and multipliers from Python's `random`, so torch's
noise/timestep stream stayed aligned with the tuple run (same step-2 spike,
2.014 vs 2.020) and the 1-20% per-step differences are the masks and
anchors themselves. `slider/anchor_loss` is logged every step at ~5e-5
(0.03% of the total) and flat over 20 steps: a fresh LoRA barely moves
anchor predictions yet, so the anchor has little to correct. Whether
`anchor_strength` 1 is enough is for the full-length M5 comparison;
`anchor_prob` < 1 trades preservation for speed. Plumbing **passes**.

### M6 (in progress): F6 linearity sweep (2026-10-19)

`scripts/slider_sweep.py` + `sweep.py`. Renders prompts x seeds x multipliers
(default 0...1.5 step 0.25 plus the -0.5 / -1 diagnostic) with the LoRA
attached through `toolkit/inference_lora.py` hooks (strength changes per
image; a slider_tuple LoRA resolves 256/256 modules on Krea 2). Default is the
Turbo deployment path (8 steps, CFG off, `model_kwargs.schedule_mu: 1.15`,
which the Krea 2 pipeline already supports); `--raw` gives Raw sampling.
Metrics: `--measure module:function` (user's attribute score), background
LPIPS outside the union of Sapiens2 person mattes (`facebook/sapiens2-matting-1b`,
downloaded on first use), and `--face module:function` (face embedding).
Summary: slope, R^2, Spearman, fitted intercept vs the m = 0 value, worst
side effects; the negative diagnostic is reported separately with flags when a
side effect at -1 is worse than at +1 by `--margin`.

- **Identity drift = face LPIPS on head crops, not a face embedding.** The
  sweep only compares each image with its own m = 0 render, so a perceptual
  distance on the head region (square crop around the union of both
  Keypoint R-CNN head ellipses, 256 px) answers "did the face change" without
  a new dependency (insightface models are non-commercial; facenet-pytorch
  pins an old torch). `--face module:function` stays for an embedding later.
- **"Value at m = 0 vs base":** with hook-attached LoRAs, m = 0 is the base
  model exactly, so the summary compares the linear fit's intercept with the
  measured m = 0 value (`intercept_minus_m0`), i.e. how linear the response
  is near zero.

CPU verification: metric/statistics tests, including "identical renders ->
slope 0, LPIPS 0, face similarity 1" (the F6 acceptance criterion), and the
script's measure/summary path end to end on pre-rendered images. GPU
acceptance run (`--zero-lora`): pending.

`scripts/dev/tuples_to_pairs.py` prepares the M5 tuples-vs-pairs comparison
(each tuple reduced to its highest and lowest member, everything else equal).

### Paste-back of head and background (2026-10-20)

The handoff's dataset contract wants head and background pasted back from
the original into every member. `scripts/paste_back.py` does it for folders
of edits (one subfolder per level) plus originals, writing a tuple dataset
(members in the original's frame, `original`, one tuple mask, manifest,
report, previews). Algorithm in `paste_back.py`; lessons from the first real
run (20 heavy/thin pairs, all fixed and regression-tested):

- **Don't trust ECC.** It diverged to 250-12000 px transforms on
  person-filled / low-texture frames and, on heads, settled about 1 px off
  while reporting 0.998. Registration candidates are bounded and scored by
  our own background NCC; head alignment uses template match + masked-NCC
  search, with ECC affine kept only if it raises that NCC.
- **Profile and back views** collapse ear/eye distances; head size takes the
  largest estimate including face-to-shoulder distance.
- **Keep only the member's own body (+ the original's)**, never all members'
  bodies, or the thinner member shows a ghost of the heavier outline.
- **Tone-match each member to the original** (low-frequency field from
  shared background) before compositing; keep blends narrow (4 levels) with
  the seam 2% of the short side outside the bodies, or wide low-frequency
  bands smear nearby body content into the background.

Data (`datasets/weight_slider_data` -> `datasets/weight_slider_tuples`):
registrations <= 11 px, all 40 heads pasted (head NCC 0.90-0.999), faces
reviewed visually on all 20 pairs. Known limits: long hair outside the head
ellipse stays the editor's per member; where the original's body covered
background that a thinner member shows, only the editor's reconstruction
exists (visible as a narrow strip in heavy - thin differences, e.g.
`v3jf2w2lrvih1`).

Captions (2026-10-20): the user's long captions
(`weight_slider_tuples/captions_json/*.json`, `caption_long`; identical to
`captions/*.txt`), 695-948 characters, ~200 tokens, well under Krea 2's
512-token cap. Held out by `split_holdout.py` (seed 0):
`8545b11fbbac7e7435d3e564e78ad083`, `tvbeats-MmdfvjpuR-E-unsplash`;
18 tuples train. Commands: README.md, section 2.

**To do after the first training tests:** a body-weight `--measure` for the
sweep (e.g. body width from the Sapiens2 matte relative to head size), so
the sweep reports linearity and not just side effects.

### F7: Turbo preview sampling while training on Raw (2026-10-20)

`turbo.py` + `slider: turbo_preview: <file>`. At each sample event the trainee
LoRA is saved, the Raw DiT moved to the CPU (`turbo_preview_offload`), the
Turbo DiT swapped in as `sd.model`, the training network detached from
sampling, and the saved LoRA attached as `InferenceLoRA` hooks whose strength
follows each prompt's `--m`; steps / CFG / mu come from `turbo_preview_steps`
(8) / `_cfg` (1) / `_mu` (1.15). Everything is restored in `finally`.

- **Full weight swap, not a Raw->Turbo delta:** the deployed Turbo is a
  ComfyUI int8_convrot file, and a delta from an 8-bit file would mix
  quantization error into a small weight difference.
- **ComfyUI int8_convrot loads as is:** `load_krea2_dit` strips the
  `model.diffusion_model.` prefix and the layers stay pre-quantized
  (`OstrisLinear` with `cr8_qdata` / `cr8_scales`; 224 modules). The
  BigLove Turbo file loads on CPU in 7.3 s, memory-mapped (~1.2 GB RSS).
  `slider_sweep.py --model <that file>` uses the same loader.

GPU acceptance (2026-10-20, `configs/dev/krea2_turbo_preview_f7.yaml`, smoke
pairs, `SEED=42`): **passes.**

- Step 0 (LoRA still zero): all three `--m -1/0/1` previews are
  pixel-identical to a plain Turbo render of the same prompt and seed from
  `slider_sweep.py --model <turbo file>` (LPIPS 0.00000, max |diff| 0), so the
  trainer's preview path and the sweep sample the same way.
- Step 20: `--m 0` still pixel-identical; `--m -1` / `+1` differ (LPIPS 0.134
  / 0.094), so the hooks carry the trainee LoRA at each prompt's strength.
- Cost: 3 Turbo previews in 11-12 s (~3.9 s/image incl. the swap); training
  speed unchanged (4.6 s/step vs 4.45-4.54 before), so moving Raw to the CPU and
  back costs nothing visible per step. Peak memory was not recorded.

### First weight-slider run (2026-10-20)

`configs/krea2_weight_slider.yaml` on `datasets/weight_slider_tuples`
(18 heavy/thin pairs, levels ±1, `level_norm: none`, rank 16, lr 1e-4,
1000 steps, anchors at `anchor_prob` 0.5, Turbo previews). Debug dumps show
the direction right (heavy +1, thin −1) and the tuple masks right. Total loss
is flat (0.271 -> 0.270, normal: noise dominates the slider signal), the
member losses equal (no real-vs-edit gap), and `slider/anchor_loss` rose
5e-5 -> 3.4e-4, so the LoRA did move. Turbo samples: weak effect (the man
at +1 a little heavier at step 1000, the woman hardly), plus scene drift at
±1 (the storefront changes).

### Body-width measure and what it found in the weight data (2026-10-20)

`body_measure.py`: mean silhouette width (Sapiens2 matte, connected region
around the torso) over rows 0.15-1.5 torso lengths below the shoulder
midpoint, divided by the shoulder-to-hip distance (Keypoint R-CNN). A 2D
proxy instead of the body-warp handoff's SMPL fitting: no new dependencies,
and the sweep only compares an image with its own m = 0 render, where pose
and framing are nearly fixed. Single-image SMPL estimators also regress
toward average shape, which hurts most on exactly these bodies. The fitting
stays the next step if the 2D measure proves too noisy or for absolute
levels (body-warp handoff, W0 + fit/measure only).

`scripts/measure_tuples.py` on the 20 tuples:

- heavy > thin in **19/19** measurable tuples (one heavy and two originals
  have hidden hips). Spearman(level, score) over all members is only 0.25,
  as expected: absolute widths vary with pose and view across photos.
- **The heavy edits barely changed the body:** heavy - original median
  -0.001 (-0.1%), range -0.03..+0.05; thin - original median -0.097 (-9.2%),
  range -0.23..-0.03. Visual check of the previews agrees (e.g.
  `789572eb...`: heavy and original look the same). The originals are
  already heavy, so the editor made little change upward.
- **Consequence for training:** the pairs are effectively "original vs
  thinner". With ±1 levels the LoRA-off point is the heavy-thin midpoint,
  so `--m 1` is trained to move only about half the original-thin difference,
  ~5% width. That fits the weak samples. Options: heavier edits, measured
  levels (`--out --add-original`, `level_norm: median_span`, so m = 1 is one
  median span), and sweeping to m = 2-3 to see whether the direction is
  learned at all.

### First weight sweeps: Raw vs Turbo (2026-10-20)

`slider_sweep.py --measure body`, the first weight LoRA (step 1000), the 2
held-out captions x 3 seeds, m in {0, ±0.5, ±1, 1.5, ±2, ±3}; Turbo = BigLove
int8_convrot (8 steps, CFG off), Raw = BigLove Raw (`--raw`, 25 steps, CFG 4).
Per-multiplier table (now part of every `summary.txt`):

| m | Turbo delta (right sign) | Turbo bg / face LPIPS | Raw delta (right sign) | Raw bg / face LPIPS |
| --- | --- | --- | --- | --- |
| -3 | +0.088 (0.00) | 0.47 / 0.47 | +0.054 (0.33) | 0.46 / 0.34 |
| -1 | -0.011 (0.50) | 0.11 / 0.16 | +0.017 (0.17) | 0.07 / 0.08 |
| +1 | +0.014 (0.67) | 0.11 / 0.16 | +0.015 (0.50) | 0.06 / 0.09 |
| +1.5 | +0.096 (1.00) | 0.19 / 0.24 | +0.047 (0.83) | 0.15 / 0.14 |
| +2 | +0.187 (1.00) | 0.28 / 0.30 | +0.087 (0.83) | 0.26 / 0.22 |
| +3 | +0.459 (1.00) | 0.46 / 0.48 | +0.180 (1.00) | 0.56 / 0.42 |

(delta: body-width change against the same prompt/seed at m = 0; m = 0
scores 0.70-0.75; per-image noise `delta_sd` ~0.03-0.07.)

- **It learned a "heavier" direction, and it transfers to Turbo:** reliable
  from m = 1.5 (every image wider), visibly heavier at +2 on the contact sheets.
  Turbo responds about twice as strongly as Raw at the same m, so M6b
  (Raw->Turbo transfer) is not the bottleneck here.
- **Threshold, not linear:** nothing measurable at |m| <= 1 (deltas inside
  the noise, right sign ~50%), then accelerating. Pooled R^2 0.60 / 0.66.
- **No "thinner" direction:** negative m never narrows the body. Large
  negative m changes the image (framing, blur) and the score goes *up*
  (pose changes).
- **Side effects grow as fast as the effect:** background and face LPIPS are
  already 0.1-0.16 at m = 1 on Turbo, where the body hasn't changed; faces age
  at +3, and Raw at +3 breaks the frame (picture-in-picture). Within the
  trained range the LoRA mostly changes things *other* than the body.
- Fits the data finding above: the only real signal in the pairs is
  original -> thinner (~9% width), split as ±1 around the midpoint. Yet
  the LoRA learned the opposite side better, so the data alone doesn't explain
  the asymmetry. Evidence is thin (2 prompts x 3 seeds).

### Run 2: measured levels (prepared 2026-10-20)

**Data design (from the user):** `heavy/` is not a heavier edit. It is the
original sent through the same edit pipeline with a "keep everything the same"
prompt, so both members carry the generator's look and centering cancels the
real-vs-synthetic difference. That is the right control. It also means
`--add-original` (a real photo as a third member) would reintroduce what the
pass-through removes, so run 2 doesn't use it. The weight data is effectively
a "thinner" slider, which fits the handoff's top-level-plus-edits-downward
design; heavier-than-original is extrapolation.

**Run 2** (`configs/krea2_weight_slider_measured.yaml`): the same as run 1
except measured levels (`datasets/weight_slider_tuples/measured/`, body width
of the pass-through and thin members) and `level_norm: median_span`. Same two
held-out tuples (`split_holdout.py --ids`), so the sweeps compare directly.
Planner output: kappa = 0.114, 16 tuples kept, `2060bd6e...` dropped
(span 0.028 < 0.3 kappa), `v3jf2w2lrvih1` leverage-capped (95th percentile;
strongest edit). Multipliers per tuple run from ±0.16 (weak edits) to ±0.94.

Expectations: the LoRA trains at about half run 1's multipliers for a
typical tuple, so it must grow ~2x to reproduce an edit at m = 1. The same
steps / lr may look weaker per unit m; if so, raise steps or lr, since the
direction is unchanged. Tuple influence grows with s^2, so the 3-4 weak-edit
tuples (span 0.03-0.04, comparable to the measure's ±0.03 noise) contribute
little.
