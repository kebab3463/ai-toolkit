import os
import random
from typing import Dict, List, Optional

import torch
from PIL import Image, ImageOps

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds
from toolkit.train_tools import get_torch_dtype

from .anchors import anchor_loss, center_crop_geometry, list_anchor_images
from .loss import slider_tuple_loss
from .manifest import SliderTuple
from .members import MemberLatentCache, dump_tuple, load_member_image, load_member_mask
from .centering import TuplePlan, centered_scales, plan_tuples, sample_members, weighted_scale_sum
from .symmetry import symmetry_loss
from .scales import SliderDatasetOptions, apply_slider_scales, slider_options_by_config
from .tuples import SliderTrainConfig, load_tuple_dataset, write_anchor_dataset
from .turbo import TurboPreview

# guidance types handled here; every other guidance_type goes to upstream unchanged
SLIDER_GUIDANCE_TYPES = ("slider_tuple",)


class SliderTupleTrainer(DiffusionTrainer):
    """SDTrainer with tuple-based image slider losses.

    Built on DiffusionTrainer (SDTrainer + the web UI's job status: progress,
    stop, sample / save now), which is inert unless the run was started from
    the UI (sqlite_db_path + AITK_JOB_ID). Selected with
    ``type: slider_tuple_trainer`` in a job config. Datasets whose
    ``guidance_type`` is not one of ``SLIDER_GUIDANCE_TYPES`` (e.g. ``polarity``)
    train exactly as they would under ``sd_trainer``.

    Dataset keys (all default off): ``slider_scales``, ``slider_scale_normalize``
    (scales.py); ``slider_tuples`` with ``guidance_type: slider_tuple``
    (tuples.py). Process-level ``slider:`` block: see tuples.SliderTrainConfig.
    """

    def __init__(self, process_id: int, job, config, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.slider_dataset_options: Dict[int, SliderDatasetOptions] = slider_options_by_config(
            self.get_conf("datasets", None) or [], self.dataset_configs
        )
        self.init_slider_state(SliderTrainConfig.from_raw(self.get_conf("slider", None)))
        self.setup_tuple_datasets()
        if self.slider_config.anchors_enabled and self.slider_config.anchor_images:
            self.slider_anchor_images = list_anchor_images(self.slider_config.anchor_images)

    def init_slider_state(self, slider_config: SliderTrainConfig):
        self.slider_config = slider_config
        self.slider_tuples_by_anchor: Dict[str, SliderTuple] = {}
        self.slider_plans: Dict[str, TuplePlan] = {}
        self.slider_kappa = 1.0
        self.slider_member_latents = MemberLatentCache()
        self.slider_anchor_embeds = []
        self.slider_anchor_images: List[str] = []
        self.slider_anchor_latents = MemberLatentCache()
        self.slider_turbo: Optional[TurboPreview] = None
        if slider_config.turbo_preview:
            if not os.path.isfile(slider_config.turbo_preview):
                raise ValueError(f"slider.turbo_preview: file not found: {slider_config.turbo_preview}")
            self.slider_turbo = TurboPreview(
                slider_config.turbo_preview,
                steps=slider_config.turbo_preview_steps,
                cfg=slider_config.turbo_preview_cfg,
                mu=slider_config.turbo_preview_mu,
                offload_raw=slider_config.turbo_preview_offload,
            )

    def sample(self, step=None, is_first=False):
        """Upstream sampling, rendered on the Turbo DiT when slider.turbo_preview is set."""
        if self.slider_turbo is None:
            return super().sample(step, is_first)
        original = self.sd.generate_images
        lora_path = os.path.join(self.save_root, "slider_tuple", "turbo_preview_lora.safetensors")

        def turbo_generate(configs, sampler=None, **kwargs):
            print_acc(f"slider_tuple: Turbo preview ({len(configs)} images, {self.slider_turbo.steps} steps)")
            self.slider_turbo.generate(
                self.sd, configs, sampler,
                save_lora=lambda path: self.network.save_weights(path, dtype=torch.float32),
                lora_path=lora_path, generate_images=original,
            )

        self.sd.generate_images = turbo_generate
        try:
            return super().sample(step, is_first)
        finally:
            del self.sd.generate_images

    def setup_tuple_datasets(self):
        tuple_sets = []
        for i, config in enumerate(self.dataset_configs):
            options = self.slider_dataset_options[id(config)]
            if options.tuples is not None:
                tuple_sets.append((i, config, load_tuple_dataset(config, options.tuples, f"dataset {i} ({options.tuples})")))
        if not tuple_sets:
            return
        if len(tuple_sets) != len(self.dataset_configs):
            raise ValueError("slider_tuple: tuple datasets cannot be mixed with other datasets in one job")

        # a resolution list loads the same manifest more than once; plan each tuple once
        unique: Dict[str, SliderTuple] = {}
        for _, _, tuples in tuple_sets:
            for t in tuples:
                known = unique.get(t.id)
                if known is not None and [(m.image, m.level) for m in known.members] != [(m.image, m.level) for m in t.members]:
                    raise ValueError(f"slider_tuple: tuple id {t.id!r} names two different tuples")
                unique.setdefault(t.id, t)
        anchors = [t.anchor.image for t in unique.values()]
        if len(set(anchors)) != len(anchors):
            raise ValueError("slider_tuple: two tuples share an anchor (first) image")

        cfg = self.slider_config
        self.slider_plans, report = plan_tuples(list(unique.values()), cfg.level_norm, cfg.min_span, cfg.weight_cap_pct)
        self.slider_kappa = report.kappa
        for line in report.lines():
            print_acc(line)

        for i, config, tuples in tuple_sets:
            kept = [unique[t.id] for t in tuples if t.id in self.slider_plans]
            if not kept:
                raise ValueError(f"slider_tuple: every tuple of dataset {i} was dropped")
            config.dataset_path = write_anchor_dataset(kept, os.path.join(self.save_root, "slider_tuple", f"dataset_{i}.json"))
        self.slider_tuples_by_anchor = {t.anchor.image: t for t in unique.values() if t.id in self.slider_plans}
        # tuple datasets have no unconditional_path, so enter the guided-loss branch explicitly
        self.do_guided_loss = True

        tuples = list(self.slider_tuples_by_anchor.values())
        n_masks = sum(m.mask is not None for t in tuples for m in t.members)
        print_acc(f"slider_tuple: {len(tuples)} tuples, {sum(len(t.members) for t in tuples)} members; {cfg}")
        if n_masks:
            print_acc(f"slider_tuple: masked loss on for {n_masks} members (unmasked members use weight 1)")
        mismatched = [t.id for t in tuples if t.mask_mismatch]
        if mismatched:
            shown = ", ".join(mismatched[:10]) + (f" (+{len(mismatched) - 10} more)" if len(mismatched) > 10 else "")
            print_acc(
                f"slider_tuple: WARNING {len(mismatched)} tuples mask their members differently: {shown}. "
                "Centering cancels shared content only where every member has the same mask weight, so "
                "shared content leaks into the slider where these masks differ. Prefer one tuple-level "
                "mask (e.g. the union of the members' bodies)."
            )
        if cfg.center_offset != 0:
            print_acc(
                f"slider_tuple: WARNING center_offset = {cfg.center_offset:g} (test hook): multipliers are no "
                "longer centered, so content shared by a tuple's members no longer cancels"
            )

    def hook_before_train_loop(self):
        # before super() and before any batch is drawn, so dataloader workers
        # (started lazily on first iteration) see the per-image weights
        self.apply_slider_scales()
        # SDTrainer's hook may unload the text encoder (cache_text_embeddings), so encode first
        self.encode_anchor_prompts()
        super().hook_before_train_loop()

    def encode_anchor_prompts(self):
        if not self.slider_config.anchors_enabled:
            return
        with torch.no_grad():
            self.sd.text_encoder_to(self.device_torch)
            self.slider_anchor_embeds = [
                self.encode_static_prompt(prompt).to(self.device_torch, dtype=self.sd.torch_dtype).detach()
                for prompt in self.slider_config.anchor_prompts
            ]
        source = f"{len(self.slider_anchor_images)} anchor images" if self.slider_anchor_images else "the batch latents"
        print_acc(f"slider_tuple: anchor loss on {len(self.slider_anchor_embeds)} prompts over {source}")

    def apply_slider_scales(self):
        for loader in (self.data_loader, self.data_loader_reg):
            if loader is None:
                continue
            for dataset in loader.dataset.datasets:
                by_config: Dict[int, list] = {}
                configs = {}
                for file_item in dataset.file_list:
                    key = id(file_item.dataset_config)
                    by_config.setdefault(key, []).append(file_item)
                    configs[key] = file_item.dataset_config
                for key, file_items in by_config.items():
                    options = self.slider_dataset_options.get(key, SliderDatasetOptions())
                    summary = apply_slider_scales(file_items, options, configs[key])
                    if summary is not None:
                        norm = f", divided by {summary.normalized_by:g}" if summary.normalized_by else ""
                        print_acc(
                            f"slider_tuple: {configs[key].folder_path}: {summary.count} images, scales from "
                            f"{summary.from_sidecar} sidecars / {summary.from_json} scales.json / "
                            f"{summary.from_default} network_weight{norm}; "
                            f"range [{summary.min_scale:g}, {summary.max_scale:g}]"
                        )

    def get_guided_loss(
        self,
        noisy_latents: torch.Tensor,
        conditional_embeds: PromptEmbeds,
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: DataLoaderBatchDTO,
        noise: torch.Tensor,
        unconditional_embeds: Optional[PromptEmbeds] = None,
        **kwargs,
    ):
        guidance_type = batch.file_items[0].dataset_config.guidance_type
        if guidance_type not in SLIDER_GUIDANCE_TYPES:
            return super().get_guided_loss(
                noisy_latents=noisy_latents,
                conditional_embeds=conditional_embeds,
                match_adapter_assist=match_adapter_assist,
                network_weight_list=network_weight_list,
                timesteps=timesteps,
                pred_kwargs=pred_kwargs,
                batch=batch,
                noise=noise,
                unconditional_embeds=unconditional_embeds,
                **kwargs,
            )
        return self.get_slider_tuple_loss(conditional_embeds, timesteps, pred_kwargs, batch, noise)

    def get_slider_tuple_loss(self, conditional_embeds, timesteps, pred_kwargs, batch, noise):
        cfg = self.slider_config
        tuples = [self.slider_tuples_by_anchor[item.path] for item in batch.file_items]
        plans = [self.slider_plans[t.id] for t in tuples]

        # every tuple in the batch trains the same number of members (one slot each)
        k = min(
            len(t.members) if cfg.members_per_step == "all" else min(cfg.members_per_step, len(t.members))
            for t in tuples
        )
        picks = [sample_members(t, k, cfg.min_span * self.slider_kappa) for t in tuples]

        # centered multipliers over the members sampled in this step
        slot_scales = [[0.0] * len(tuples) for _ in range(k)]
        slot_weights = [[0.0] * len(tuples) for _ in range(k)]
        step_scales, wsums, spans = [], [], []
        for b, (t, p, plan) in enumerate(zip(tuples, picks, plans)):
            levels = [t.members[i].level for i in p]
            w = [plan.member_weights[i] for i in p]
            scales = centered_scales(levels, w, self.slider_kappa, cfg.center_offset)
            for j in range(k):
                slot_scales[j][b] = scales[j]
                slot_weights[j][b] = plan.tuple_weight * w[j]
            step_scales.append(dict(zip(p, scales)))
            wsums.append(weighted_scale_sum(scales, w))
            spans.append((max(levels) - min(levels)) / self.slider_kappa)
        self.additional_logs["slider/weighted_scale_sum"] = sum(wsums) / len(wsums)
        self.additional_logs["slider/span"] = sum(spans) / len(spans)
        self.additional_logs["slider/members_per_step"] = k

        if self.step_num < cfg.debug_dump_steps:
            self.dump_tuples(tuples, picks, step_scales, batch)

        member_losses: List[List[float]] = []
        loss = slider_tuple_loss(
            self.sd,
            slot_latents=self.gather_member_latents(tuples, picks, batch),
            slot_scales=slot_scales,
            slot_weights=slot_weights,
            noise=noise,
            timesteps=timesteps,
            conditional_embeds=conditional_embeds,
            pred_kwargs=pred_kwargs,
            batch=batch,
            train_config=self.train_config,
            backward_per_member=cfg.backward_per_member,
            slot_masks=self.gather_member_masks(tuples, picks, batch),
            member_losses=member_losses,
        )
        self.log_losses_by_rank(picks, plans, member_losses)
        if cfg.anchors_enabled and random.random() < cfg.anchor_prob:
            loss = self.add_anchor_loss(loss, batch, noise, timesteps, pred_kwargs)
        if cfg.symmetry_enabled and random.random() < cfg.symmetry_prob:
            loss = self.add_symmetry_loss(loss, batch, noise, timesteps, conditional_embeds, pred_kwargs)
        return loss

    def add_symmetry_loss(self, loss, batch, noise, timesteps, conditional_embeds, pred_kwargs):
        """symmetry.py on the batch's own (anchor member) latents and caption."""
        cfg = self.slider_config
        weighted, mse = symmetry_loss(
            self.sd,
            latents=batch.latents,
            noise=noise,
            timesteps=timesteps,
            embeds=conditional_embeds,
            multipliers=[random.uniform(0.0, cfg.symmetry_max_multiplier) for _ in batch.file_items],
            strength=cfg.symmetry_strength,
            pred_kwargs=pred_kwargs,
            batch=batch,
        )
        self.additional_logs["slider/symmetry_loss"] = mse
        total = (loss.detach() + weighted).detach()
        total.requires_grad_(True)
        return total

    def log_losses_by_rank(self, picks, plans, member_losses):
        """Mean member MSE per rank in its tuple (rank 0 = highest level, normally the real photo)."""
        by_rank: Dict[int, List[float]] = {}
        for j, slot in enumerate(member_losses):
            for b, value in enumerate(slot):
                by_rank.setdefault(plans[b].ranks[picks[b][j]], []).append(value)
        for rank, values in by_rank.items():
            self.additional_logs[f"slider/loss_rank{rank}"] = sum(values) / len(values)

    def add_anchor_loss(self, loss, batch, noise, timesteps, pred_kwargs):
        cfg = self.slider_config
        n = len(batch.file_items)
        weighted, mse = anchor_loss(
            self.sd,
            latents=self.anchor_latents(batch),
            noise=noise,
            timesteps=timesteps,
            anchor_embeds=[random.choice(self.slider_anchor_embeds) for _ in range(n)],
            multipliers=[random.uniform(-cfg.anchor_max_multiplier, cfg.anchor_max_multiplier) for _ in range(n)],
            strength=cfg.anchor_strength,
            pred_kwargs=pred_kwargs,
            batch=batch,
        )
        self.additional_logs["slider/anchor_loss"] = mse
        total = (loss.detach() + weighted).detach()
        total.requires_grad_(True)
        return total

    def anchor_latents(self, batch) -> torch.Tensor:
        """The batch's own latents, or a random anchor image per item, center-cropped
        to that item's size (cached like member latents)."""
        if not self.slider_anchor_images:
            return batch.latents
        dtype = get_torch_dtype(self.train_config.dtype)
        requests = []
        for item in batch.file_items:
            path = random.choice(self.slider_anchor_images)
            with Image.open(path) as img:
                size = ImageOps.exif_transpose(img).size  # the loader applies EXIF rotation first
            requests.append((path, center_crop_geometry(size, (item.crop_width, item.crop_height))))

        def encode(pixels):
            with torch.no_grad():
                return self.sd.encode_images(pixels.to(self.device_torch, dtype=dtype))

        latents = self.slider_anchor_latents.get(requests, encode)
        return torch.stack([x.to(self.device_torch, dtype=dtype) for x in latents])

    def gather_member_latents(self, tuples: List[SliderTuple], picks: List[List[int]], batch) -> List[torch.Tensor]:
        """(B, ...) latents per slot: the anchor's from the batch (upstream cached
        it), every other member's from the crop-aligned member cache."""
        dtype = get_torch_dtype(self.train_config.dtype)
        device = self.device_torch

        def encode(pixels):
            with torch.no_grad():
                return self.sd.encode_images(pixels.to(device, dtype=dtype))

        requests = [
            (t.members[i].image, item)
            for t, p, item in zip(tuples, picks, batch.file_items)
            for i in p
            if i != 0
        ]
        cached = iter(self.slider_member_latents.get(requests, encode) if requests else [])
        k = len(picks[0])
        slots = [[None] * len(tuples) for _ in range(k)]
        for b, p in enumerate(picks):
            for j, i in enumerate(p):
                slots[j][b] = batch.latents[b] if i == 0 else next(cached)
        return [torch.stack([x.to(device, dtype=dtype) for x in slot]) for slot in slots]

    def gather_member_masks(self, tuples: List[SliderTuple], picks: List[List[int]], batch) -> List[Optional[torch.Tensor]]:
        """(B, 1, H, W) loss masks per slot, aligned with each member's crop;
        members without a mask get ones, slots with no mask at all get None."""
        slots = []
        for j in range(len(picks[0])):
            members = [(t.members[p[j]], item) for t, p, item in zip(tuples, picks, batch.file_items)]
            if all(m.mask is None for m, _ in members):
                slots.append(None)
                continue
            masks = [
                load_member_mask(m.mask, item) if m.mask is not None
                else torch.ones(1, item.crop_height, item.crop_width)
                for m, item in members
            ]
            slots.append(torch.stack(masks))
        return slots

    def dump_tuples(self, tuples, picks, step_scales, batch):
        folder = os.path.join(self.save_root, "slider_debug")
        for t, p, scales, item, plan in zip(tuples, picks, step_scales, batch.file_items, [self.slider_plans[t.id] for t in tuples]):
            images = [load_member_image(m.image, item) for m in t.members]
            labels = [
                f"r{plan.ranks[i]} {m.role} a={m.level:g}" + (f" s={scales[i]:+.2f}" if i in scales else " (not sampled)")
                for i, m in enumerate(t.members)
            ]
            if t.original is not None:
                images.append(load_member_image(t.original, item))
                labels.append("original (metrics only)")
            path = dump_tuple(os.path.join(folder, f"step{self.step_num:06d}_{t.id}.png"), images, labels)
            print_acc(f"slider_tuple: debug dump {path}")
