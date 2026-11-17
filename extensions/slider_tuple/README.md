# slider_tuple: slider LoRAs for Krea 2 from image tuples

Trains "slider" LoRAs (one attribute, controlled by LoRA strength) on Krea 2
Raw from sets of images of the same person at different attribute levels,
and judges them on the Turbo model they are deployed on. This file is the
how-to for a new slider; `NOTES.md` is the development log (why things are
the way they are, test results), and
`docs/Handoff Tuple Slider Training for Krea 2 in an ai-toolkit Fork.md` the
design.

All commands run from the repo root on the `mydev` branch.

## Contents

1. [One-time setup](#1-one-time-setup)
2. [Prepare the data](#2-prepare-the-data)
3. [Train](#3-train)
4. [Evaluate](#4-evaluate)
5. [Reference](#5-reference): manifest format, `slider:` keys, scripts,
   folder layout, known issues

## 1. One-time setup

```bash
uv sync                      # .venv with the pinned torch / diffusers (+ pytest)
.venv/bin/python -m pytest -q extensions/slider_tuple/tests   # CPU-only, ~15 s
```

- `.env` in the repo root: `HF_TOKEN` (and `HF_HOME` if the Hugging Face cache
  lives elsewhere). Scripts load it themselves.
- **Raw model** (training): `models/BigLoveKreaLife2_raw_AI-Toolkit/`, a folder
  with one DiT `.safetensors` in the `krea/Krea-2-Raw` format. The text
  encoder and VAE download from their default repos on first use. Any Krea 2
  Raw fine-tune in that format works; `krea/Krea-2-Raw` for the official one.
- **Turbo model** (previews and evaluation): the deployed checkpoint, e.g. the
  ComfyUI int8_convrot file
  `/home/csprunger/.nvme_cache/diffusion_models/Krea2/BigLoveKreaLife2_int8.safetensors`.
  Use the Turbo that belongs to the Raw you train on.
- **Mask and metric models** download on first use: Sapiens2 matting
  (`facebook/sapiens2-matting-1b`, 6.2 GB, into `models/sapiens2/`),
  torchvision Keypoint R-CNN (~230 MB) and AlexNet for LPIPS (~230 MB,
  torch hub cache).
- GPU memory (WSL): `/usr/lib/wsl/lib/nvidia-smi` (not on `PATH`).

## 2. Prepare the data

What a good tuple is (from the handoff's dataset contract):

- One person per tuple, **2+ images at different attribute levels**, same
  pose, framing, clothing, light and background, everything except the
  attribute. Edited images (e.g. heavier / thinner versions of one photo)
  work; the edit should change only the attribute.
- **Put every member through the same edit pipeline**, including one
  "pass-through" edit that asks for no change. That one stands in for the
  original at its level. All members then carry the generator's look
  equally, and centering cancels it. Paired with a real photo, the LoRA
  would also learn "real -> synthetic". The weight data's `heavy/` folder is
  such a pass-through (NOTES.md, "Run 2").
- Faces and backgrounds identical across members: paste-back (below) does
  that for edits of an original photo.
- **One caption per tuple, without the attribute.** Describe everything
  else in detail, so the LoRA does not learn it. Don't mention body size
  for a weight slider.
- 15-20 tuples is a usable start; ~10% are held out for evaluation.

### 2a. Edits of original photos: paste-back (the usual path)

Folder layout, files matched by name (any of png/jpg/webp):

```
datasets/<slider>_data/
  orig/<name>.png        the untouched photo
  heavy/<name>.png       one folder per level (any names)
  thin/<name>.png
  captions/<name>.json   or <name>.txt (optional, can come later; any folder)
```

```bash
.venv/bin/python extensions/slider_tuple/scripts/paste_back.py \
    datasets/weight_slider_data datasets/weight_slider_tuples \
    --member heavy=1 --member thin=-1 --original orig \
    --captions datasets/weight_slider_tuples/captions_json --caption-key caption_long --preview
```

(This is the exact command that built `datasets/weight_slider_tuples`; the
images were made first without `--captions`, then the re-run with captions
only rewrote the manifest.)

- `--member NAME=LEVEL` per member folder. Levels are numbers on one scale:
  `+1/-1` for "more/less", or measured values (e.g. estimated BMI) if you
  have them (then use `level_norm: median_span`, see §3).
- Captions: `<name>.txt`, or `<name>.json` with the caption under
  `--caption-key` (default `caption_long`; e.g. `caption_short`). Tuples without
  one get `--default-caption`, and the final line lists them.
- Output: `heavy/`, `thin/`, `original/`, `masks/` (one loss mask per tuple:
  body 1, background `--background` 0.2, head `--head-weight` 0), `tuples.jsonl`,
  `report.csv`, `preview/`.
- **Mask weights:** the head and background are the same in every member
  after paste-back, so a weight above 0 there trains "keep this as is": the
  LoRA pays for changing faces or backgrounds. Head 0 leaves faces out of the
  loss entirely. To change the weights of existing masks without re-running
  anything, use `scripts/reweight_masks.py` (e.g. `--background 0.5
  --head-weight 0.3`; keep the head below the background). It writes new
  masks and a manifest next to the old ones.
- **Check by eye** `preview/<name>.jpg`: raw vs pasted members, the heavy-thin
  difference (should be body only) and the face close-ups. In `report.csv`,
  `head_pasted` False means the edit's own head was kept (the final line
  lists those); a large `registration_max_shift_px` means the editor moved
  the frame, a large `head_shift_px` that it moved the head.
- Re-running is cheap: finished images are reused, so new captions only
  rewrite `tuples.jsonl`. Use `--overwrite` after changing the images or the
  paste-back settings. That re-run also resets the levels, so once a
  manifest has measured levels (§2d), add captions with
  `scripts/set_captions.py <manifest> <captions folder>` instead. It changes
  only the captions and can be re-run as more caption files arrive; tuples
  without one keep theirs.

### 2b. Already aligned images (no originals)

Write `tuples.jsonl` yourself (format in §5) or convert a before/after folder
pair with `scripts/pairs_to_manifest.py positive_dir negative_dir out.jsonl`,
then add one loss mask per tuple:

```bash
.venv/bin/python extensions/slider_tuple/scripts/make_tuple_masks.py \
    datasets/<slider>/tuples.jsonl datasets/<slider>/tuples_masked.jsonl --preview
```

### 2c. Hold out tuples for evaluation

```bash
.venv/bin/python extensions/slider_tuple/scripts/split_holdout.py datasets/weight_slider_tuples/tuples.jsonl
```

Writes `train.jsonl` and `held_out.jsonl` next to it (10%, seeded, so a re-run
after a caption change holds out the same tuples; `--ids a,b` to choose).
Re-run it whenever `tuples.jsonl` changes.

### 2d. Measure the data (weight sliders; do this before training)

Check that the edits actually change the attribute, and by how much:

```bash
.venv/bin/python extensions/slider_tuple/scripts/measure_tuples.py \
    datasets/weight_slider_tuples/tuples.jsonl --preview
```

- The score is body width in torso lengths (`body_measure.py`): mean
  silhouette width from below the shoulders to the upper thighs, divided by
  the shoulder-to-hip distance. It is reliable *within* a tuple or a sweep
  (same pose and framing), not as an absolute weight across photos.
- It prints each tuple's scores, how many tuples the score orders like their
  levels, and whether the original lies between its members. It writes
  `measurements.csv` (also chest / waist / hip / thigh widths) and
  `measure_preview/<id>.jpg` (measured band shaded) next to the manifest.
- **Look at the size of the changes, not just the order.** In the weight
  data, thin measured 2-19% narrower than the pass-through edit, so a few
  pairs barely differ, and ±1 levels train them as hard as the strong ones.
  (The pass-through measured the same as the original, median −0.1%, which
  confirms it changes nothing.)
- **Measured levels instead of ±1** (the run-2 recipe, commands at the top of
  `configs/krea2_weight_slider_measured.yaml`):
  `--out <folder>/tuples.jsonl`, then `split_holdout.py` on it (`--ids` to
  keep the same held-out tuples as an earlier run), then train with
  `level_norm: median_span`. Each tuple then trains in proportion to how much
  its edit changed the body, and m = 1 is one median tuple's change.
  Tuples with an unmeasurable image (shoulders or hips hidden) are left out
  and listed; the trainer drops spans under `min_span` × κ at startup.
- `--add-original` also adds the real original as a `real` member. Use it only
  when the other members are real photos too, or when real-vs-synthetic
  differences don't matter. Next to pass-through edits it works against them.

**Your own measurements (a CSV).** For a measure made outside this extension
(mask areas, a classifier score, ...), with one row per tuple and one column
per member folder:

```bash
.venv/bin/python extensions/slider_tuple/scripts/levels_from_csv.py \
    datasets/slider_tuples/unmeasured_tuples.jsonl datasets/slider_tuples/measurements.csv \
    --column neg=thin --column pos=heavy --out datasets/slider_tuples/tuples.jsonl
```

- `--transform log` (default) sets level = log(value), so a tuple's span is
  its log ratio. Use it for areas, counts and widths, which scale with the
  photo's size and framing. Use `linear` for scores already comparable across
  images.
- Then run `split_holdout.py` and train with `level_norm: median_span`, as
  above. Each tuple trains at ±(span / 2) / κ: ±0.5 for a median tuple, more
  for a strong edit, less for a weak one.
- The script prints the ratio and span range and how many tuples the
  trainer's `min_span` will drop. `--min-ratio` drops near-no-op edits
  itself. Tuples without a row are left out and listed.

### 2e. Intermediate members from interpolated frames (optional)

Turning pairs into larger tuples: render a sequence that morphs each top
member into its bottom one (e.g. video frames, named `<tuple id>_<index>.png`),
then measure every frame and pick a few evenly spaced in body width:

```bash
.venv/bin/python extensions/slider_tuple/scripts/measure_images.py datasets/weight_slider_tuples/mmh3 \
    --top datasets/weight_slider_tuples/heavy --bottom datasets/weight_slider_tuples/thin \
    --pick 2 --preview --export
```

- Scores are smoothed along the frame index (the measure's per-image noise
  is ~0.03), and picks are the frames closest to evenly spaced targets
  strictly between the tuple's actual top and bottom members.
- Check `mmh3_measure/groups.csv` (flags: span within noise, scores not
  monotone along the frames, frames not covering the range) and
  `preview/<id>.png` (score curve with picks, thumbnails).
- `--export` copies the picks to `mmh3_measure/export/mid1/<id>.png`,
  `mid2/...` (mid1 nearest the top). These are member folders in the layout
  `paste_back.py` reads.
- **Before using picks as members, give them the other members' processing
  and framing.** Run them through the same editor as the edits (for the
  weight data: Qwen Image 2.1, with the original as reference to restore the
  frame edges the 9:16 video crop removed). Then put the folders next to the
  others and paste back with all members, e.g. `--member heavy=1
  --member mid1=0.33 --member mid2=-0.33 --member thin=-1`. Re-measure with
  `measure_tuples.py --out` (§2d): the editor pass can change the body a
  little, so the measured levels after it are the ones to train on.
  Paste-back's registration only accepts near-identity alignments (±5%
  scale), so raw cropped frames would be rejected.
- Picking before the editor pass means only the picks need editing.

## 3. Train

```bash
cp extensions/slider_tuple/configs/krea2_weight_slider.yaml config/my_slider.yaml   # config/ is gitignored
#   (or krea2_weight_slider_measured.yaml for measured levels)
# edit the lines marked EDIT, then
.venv/bin/python run.py config/my_slider.yaml
```

What to edit (all commented in the file):

- `name`: run folder `output/<name>/` and LoRA file name.
- `slider_tuples`: your `train.jsonl`.
- `level_norm`: `none` for `+1/-1` levels (strength 1 = the edit),
  `median_span` for measured levels (strength 1 = a typical tuple's range).
- **Anchors**, things the slider must leave alone at any strength up to
  `anchor_max_multiplier`:
  - `anchor_images`: a folder of photos, each with its own caption next to it
    (`<name>.txt` or `.json`, field `anchor_caption_key`). Each anchor is noised
    from its photo and conditioned on its caption. For a women-only slider,
    use photos of men in the same kinds of scenes as the training data
    (`configs/experiments/weight_r5_women.yaml`). A folder without captions
    keeps the older behavior (its images noised for `anchor_prompts`).
  - `anchor_prompts`: prompts without images, noised from the training image
    of the step. Weak: the image and the prompt don't match. Prefer
    captioned images.
- `symmetry_strength` (see NOTES.md, "Run 3 results"): removes the part of
  the LoRA's effect that is the same at +m and −m, the drift both directions
  share. Costs ~5 s on the steps where it runs (`symmetry_prob`).
- `turbo_preview`: your Turbo checkpoint (remove the key to preview on Raw).
- `sample.prompts`: held-out captions with `--m -1`, `--m 0`, `--m 1`
  (same seed, `walk_seed: false`, so only the strength differs).

**From the web UI instead:** register the config as a UI job, then start it
from the browser (Jobs → the job → start). It shows status, progress, loss
graphs and the samples as a grid (one row per sample step, one column per
prompt), and stop / sample now / save now work:

```bash
.venv/bin/python extensions/slider_tuple/scripts/ui_jobs.py add config/my_slider.yaml
.venv/bin/python extensions/slider_tuple/scripts/ui_jobs.py list
```

- `add` takes any number of configs. A run that already finished in the UI's
  training folder (`output/<name>/<name>.safetensors`) is added as completed,
  so past experiments and their samples show up too. Everything else is added
  stopped, ready to start.
- The stored job config sets `training_folder` to the UI's and writes
  `sample.prompts` as `sample.samples`, the form the UI's grid reads. Python
  takes either.
- After editing a config, `add --update` replaces a job's config (not while
  it is running or queued). The UI reads its database live, so no restart.
- The trainer is built on the UI's `DiffusionTrainer`, which reports to the UI
  only when the UI started the run; from the command line nothing changes.

During training:

- **Startup log:** tuples kept / dropped (`min_span`), the masked-loss count,
  warnings about members masked differently.
- **`output/<name>/slider_debug/`:** the first steps' members with their
  multipliers. Check that "+" is the direction you want.
- **Samples** in `output/<name>/samples/` every `sample_every` steps, on Turbo
  (8 steps, CFG off). `--m 0` must not change between sample events.
- **Metrics** (`loss_log.db`, also in the UI): `slider/weighted_scale_sum` ~0,
  `slider/loss_rank<r>` per member rank, `slider/anchor_loss`.
- Speed on an RTX 5090: ~4.5 s/step plain, ~7 s/step with anchors at
  `anchor_prob: 1`, 21-24 GiB.

The LoRA is `output/<name>/<name>.safetensors`, plus step snapshots
`<name>_<step>.safetensors`.

## 4. Evaluate

The sweep renders held-out prompts × seeds × strengths on Turbo and measures
linearity and side effects:

```bash
.venv/bin/python extensions/slider_tuple/scripts/slider_sweep.py \
    --model /home/csprunger/.nvme_cache/diffusion_models/Krea2/BigLoveKreaLife2_int8.safetensors \
    --lora output/my_slider/my_slider.safetensors \
    --prompts datasets/weight_slider_tuples/held_out.jsonl \
    --multipliers 0,0.5,1,1.5,2,3 --diagnostic=-0.5,-1,-2,-3 \
    --measure body --out output/sweeps/my_slider_turbo
```

- Outputs: `images/`, `sheets/prompt_NNN.png` (contact sheet per prompt, the
  first thing to look at), `sweep.csv`, `summary.txt` / `summary.json`.
- **Read the per-multiplier table** at the end of `summary.txt` before the
  pooled slope. Per strength it lists `delta`, the measure change against the
  same prompt and seed at m = 0; `delta_sd`; `right_sign`, the share of images
  that moved in the direction of m; and the mean LPIPS. A working direction
  has `right_sign` near 1 and `delta` well above `delta_sd`. Then compare that
  with the LPIPS at the same strength to see the cost. The pooled
  slope / R² mixes in thresholds and breakdowns at large m.
- **Parity** (end of `summary.txt`, or `compare_sweeps.py --parity`): the
  measure change split into the part that flips with the sign of m (`odd`:
  the slider) and the part that is the same for +m and −m (`even`). A linear
  slider has `even` ≈ 0. A growing `even` part together with equal background
  LPIPS at ±m is shared drift: the LoRA pulling every image toward the
  training data's common content, whichever way it is turned. In the weight
  runs it grew faster than the slider with training and marked where
  checkpoints turned bad (NOTES.md, "Run 3").
- **Learning-rate schedules** (this fork's `train.stages`: per-stage `steps`,
  `lr`, `lr_scheduler` (cosine / linear / step / ...), `max_grad_norm`,
  `timestep_type`) work with the slider trainer. Decided in advance, e.g. learn
  at 3e-4, then decay (`configs/experiments/weight_s512_stages.yaml`). Decided
  from what the previews show: `scripts/branch_run.py output/<run> <step> --lr
  1e-4 --scheduler cosine --steps 250 --register` continues from that
  checkpoint (weights only; fresh optimizer; its step-0 previews show the
  starting point). Best saved for polishing a slider that already works: a
  lower lr moves along the same path more slowly and less noisily, it doesn't
  remove the drift.
- `--resummarize` rewrites the summary and sheets from `sweep.csv` without
  loading models (e.g. after a tooling update).
- **In-training sweeps (faster; the default for experiments):** make the
  run's sample previews the sweep grid, so every sample step renders a sweep
  of that checkpoint on Turbo during training, with no separate sweep job:
  `scripts/sweep_grid.py CONFIG --prompts held_out.jsonl` (2 prompts x 3 seeds
  x 7 strengths = 42 images, ~3 min per sample step). After or during
  training, `scripts/sweep_from_samples.py output/<run> --measure body` writes
  `output/sweeps/<run>/<run>_<step>/` per sample step for `compare_sweeps.py`.
  The UI shows the grid live.
- **Compare runs:** sweep each run with the same prompts, seeds and
  multipliers, then
  `scripts/compare_sweeps.py output/sweeps/run_a output/sweeps/run_b` prints
  the per-multiplier tables side by side (`--csv` for a file).
- **Is it still learning?** Several LoRAs render with one model load, each
  into `<out>/<file stem>/`: sweep a run's step checkpoints with
  `--lora output/<run>/<run>_*.safetensors --out output/sweeps/<run>_steps`,
  then `compare_sweeps.py output/sweeps/<run>_steps/* output/sweeps/<run>`
  (step folders sort by step, a folder without a step number goes last). If
  `delta` still grows from step to step at the same m, train longer or with a
  higher lr. If it stopped growing, more of the same won't help.
- **Side effects, built in:** `lpips_bg` (change outside the people) and
  `face_lpips` (change on the head crop), each against the same prompt and
  seed at strength 0. Flags in `summary.txt` mark side effects worse at -1
  than at +1.
- **Attribute linearity** needs a measurement: `--measure body` (built-in
  body width, as in §2d, sharing the side-effect models) or
  `--measure mymodule:score`, a function taking a PIL image and returning a
  number, or None when it can't measure. Without it the summary reports side
  effects only. Unmeasurable images are left out of the fit and counted.
- **What slope to expect** (body width): a slider that fully learned its data
  changes the score by about half the data's mean member-to-member difference
  per unit of `--m` with `level_norm: none` and ±1 levels, or by one median
  span with `median_span`. For the first weight data that is ~0.05 per
  unit, i.e. ~5% width, which is why `--m 1` looks subtle; sweep to 2-3.
  The first weight LoRA's results are in NOTES.md ("First weight sweeps").
- **Raw vs Turbo:** run the same sweep with `--model <Raw folder> --raw`. A
  slope on Raw but not on Turbo means the LoRA doesn't transfer; no slope
  on either means it didn't learn.
- `--multipliers` is the main range and `--diagnostic` the extra one; both are
  rendered. For a two-way slider like weight, put the negative side in
  `--diagnostic` as above (the default assumes a positive-only slider).
- Re-runs reuse rendered images; delete `images/` to re-render.
- Sanity check of the tooling: `--zero-lora` must give slope ~0 and LPIPS ~0.
- `scripts/image_lpips.py ref.png a.png ...` compares any images directly.

## 5. Reference

### Manifest (`tuples.jsonl`, one tuple per line)

```json
{"id": "p001", "caption": "photo of ...", "quality": 1.0,
 "original": "original/p001.png", "mask": "masks/p001.png",
 "members": [
   {"image": "real/p001.png",  "level": 1.0,  "role": "real"},
   {"image": "edit1/p001.png", "level": 0.4,  "role": "edit", "quality": 0.8, "level_var": 0.05},
   {"image": "edit2/p001.png", "level": -0.5, "role": "edit"}]}
```

- Paths are relative to the manifest's folder.
- `level` (required) is the attribute on one scale across the dataset.
- Optional: `role` (`real` | `edit`, at most one `real`), `quality` (0-1,
  weights the tuple / member), `level_var` (>0, uncertain levels weigh less),
  `original` (metrics and debug dumps only).
- `mask` is grayscale, same size as the images: 1 trains fully, 0 ignores.
  Prefer one tuple-level `mask`: per-member masks leak shared content
  where they differ (NOTES.md, "One mask per tuple").
- Multipliers are centered per step: s_i = (level_i − weighted mean) / κ.

### `slider:` keys (process level)

| Key | Default | Meaning |
| --- | --- | --- |
| `level_norm` | `median_span` | κ: `median_span` / `max_span` of tuple spans, or `none` (κ = 1) |
| `members_per_step` | `all` | members per tuple per step (int ≥ 2 samples subsets) |
| `min_span` | 0.3 | drop tuples (and subsets) spanning less, in κ units |
| `weight_cap_pct` | 95 | cap each tuple's leverage at this percentile |
| `backward_per_member` | false | one backward per member (less memory, slower) |
| `shared_residual_weight` | 1.0 | λ: < 1 trains each member's deviation from the tuple's shared residual more than the shared part; 0 removes the drive of the shared drift (loss.py) |
| `debug_dump_steps` | 0 | write member sheets for the first N steps |
| `anchor_strength` | 0 (off) | weight of the anchor loss |
| `anchor_prompts` | [] | prompts whose predictions the LoRA must not change (noised from the batch) |
| `anchor_prob` | 1.0 | chance per step to add the anchor loss |
| `anchor_max_multiplier` | 1.0 | anchor strengths drawn from ±this |
| `anchor_images` | none | folder of anchor photos: all captioned = paired anchors (photo + its caption); none captioned = noised for `anchor_prompts` |
| `anchor_caption_key` | `caption_long` | field read from `.json` anchor captions |
| `symmetry_strength` | 0 (off) | weight of mean(((v(+s) + v(−s)) / 2 − v(0))²) on the training batch |
| `symmetry_prob` | 1.0 | chance per step (4 extra forwards, ~5 s on a 5090) |
| `symmetry_max_multiplier` | 2.0 | s drawn from [0, this] |
| `turbo_preview` | none | Turbo `.safetensors` for training samples |
| `turbo_preview_steps` / `_cfg` / `_mu` | 8 / 1.0 / 1.15 | Turbo sampling |
| `turbo_preview_offload` | true | move Raw to the CPU while previewing |
| `center_offset` | 0 | test hook only; breaks centering |

Dataset entries take `slider_tuples: <manifest>` with
`guidance_type: slider_tuple`; augmentations and `folder_path` are rejected.

### Scripts

| Script | Use |
| --- | --- |
| `scripts/paste_back.py` | edits + originals → aligned tuple dataset with masks (§2a) |
| `scripts/make_tuple_masks.py` | one loss mask per tuple for an existing manifest (§2b) |
| `scripts/pairs_to_manifest.py` | before/after folders → `tuples.jsonl` (§2b) |
| `scripts/split_holdout.py` | train / held-out manifests (§2c) |
| `scripts/reweight_masks.py` | change background / head / body weights of existing masks (§2a) |
| `scripts/measure_tuples.py` | body-width measure of every member; measured levels (§2d) |
| `scripts/set_captions.py` | captions into an existing manifest from `.txt` / `.json` files, levels kept (§2a) |
| `scripts/levels_from_csv.py` | measured levels from your own per-member measurements in a CSV (§2d) |
| `scripts/measure_images.py` | measure a folder of frames; pick evenly spaced intermediates per tuple (§2e) |
| `scripts/slider_sweep.py` | linearity and side-effect sweep on Turbo (§4) |
| `scripts/sweep_grid.py` | make a config's sample previews a sweep grid (in-training sweeps, §4) |
| `scripts/sweep_from_samples.py` | sweep tables from a run's sample images, one folder per step (§4) |
| `scripts/compare_sweeps.py` | per-multiplier tables of several sweeps side by side (§4) |
| `scripts/image_lpips.py` | LPIPS between images |
| `scripts/branch_run.py` | continue a run from one of its checkpoints with a new lr / schedule (§4) |
| `scripts/ui_jobs.py` | register configs (and finished runs) as web UI jobs (§3) |
| `scripts/dev/compare_runs.py` | two configs, same seed, short runs: loss / speed / memory A/B |
| `scripts/dev/make_smoke_pairs.py` | synthetic pairs for plumbing tests (not training data) |
| `scripts/dev/make_band_masks.py` | placeholder masks for plumbing tests |
| `scripts/dev/tuples_to_pairs.py` | reduce tuples to their extremes (tuples-vs-pairs comparison) |

`configs/krea2_weight_slider.yaml` (±1 levels) and
`configs/krea2_weight_slider_measured.yaml` (measured levels) are the training
templates;
`configs/experiments/` holds run-specific A/B configs (each header says what it
changes and why); `configs/dev/` holds the milestone smoke and acceptance configs (NOTES.md).

### ComfyUI nodes (checking edits while making them)

`comfyui/` is a ComfyUI node pack running the same body-width measure, so you
can judge an edit inside the editing workflow and redo weak ones right away:

```bash
ln -s /home/csprunger/ai-toolkit/extensions/slider_tuple/comfyui \
    /home/csprunger/ComfyUI/custom_nodes/ComfyUI-SliderMeasure   # then restart ComfyUI
```

- **Slider Edit Strength (body width):** reference (the pass-through edit, or
  the original) + edited image(s) → `change` (edit − reference; negative =
  thinner), `relative_change`, `ok`, a `verdict` per pair (`OK`, `TOO WEAK`,
  `WRONG DIRECTION`, `UNMEASURABLE`), and a side-by-side preview with the
  measured band. `ok` needs the chosen `direction` by at least `min_change`
  (default 0.08; per-image noise ~0.03, the first weight data's median edit was
  0.11). One reference can serve a whole batch of edits.
- **Body Width Score (slider_tuple):** the score of each image, plus a preview.
- Scores are identical to `measure_tuples.py` (checked on a real pair). The
  pack loads `body_measure.py` and the Sapiens2 code from this checkout by file
  path (no packages installed into ComfyUI) and reuses
  `models/sapiens2/sapiens2_1b_matting.safetensors`. The measure models are
  registered with ComfyUI's model management, which loads and offloads them.

### Code layout

| File | Role |
| --- | --- |
| `SliderTupleTrainer.py` | the trainer (`type: slider_tuple_trainer`), subclass of SDTrainer |
| `manifest.py`, `tuples.py` | manifest parsing; `slider:` config and dataset wiring |
| `centering.py`, `loss.py` | centered multipliers, member weights, the tuple loss |
| `members.py`, `anchors.py` | member images / masks / latent cache; anchor pool and loss |
| `symmetry.py` | symmetry loss (only the odd part of the LoRA's effect may change predictions) |
| `captions.py` | caption files next to images (`.txt` / `.json`) |
| `scales.py` | F1: per-image scales for polarity datasets |
| `turbo.py` | Turbo previews and ComfyUI checkpoint loading |
| `paste_back.py`, `tuple_masks.py` | paste-back and mask building |
| `sweep.py` | sweep metrics and summaries |
| `body_measure.py` | body width in torso lengths (`--measure body`) |
| `comfyui/` | ComfyUI nodes for the body-width measure (edit strength check) |

### Known issues

- Krea 2 needs `model.dtype: bf16` in every config (upstream bug; the
  float16 default breaks decoding).
- Training needs 8-bit DiT weights on 32 GB (`qtype: convrot8`); unquantized
  bf16 does not fit and `low_vram` is far too slow.
- Paste-back: long hair outside the head ellipse stays the editor's per
  member; background that only the original's body covered keeps the
  editor's reconstruction.
- Not yet checked: loading these LoRAs in ComfyUI on the int8_convrot Turbo
  (the sweep and previews use the toolkit's own hooks).
