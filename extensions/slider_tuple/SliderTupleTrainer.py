import os
from typing import Dict, List, Optional

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds
from toolkit.train_tools import get_torch_dtype

from .loss import slider_tuple_loss
from .manifest import SliderTuple
from .members import MemberLatentCache, dump_tuple, load_member_image, load_member_mask
from .sampling import member_weights, sample_members, weighted_scale_sum
from .scales import SliderDatasetOptions, apply_slider_scales, slider_options_by_config
from .tuples import SliderTrainConfig, prepare_tuple_dataset

# guidance types handled here; every other guidance_type goes to upstream unchanged
SLIDER_GUIDANCE_TYPES = ("slider_tuple",)


class SliderTupleTrainer(SDTrainer):
    """SDTrainer with tuple-based image slider losses.

    Selected with ``type: slider_tuple_trainer`` in a job config. Datasets whose
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
        self.slider_config = SliderTrainConfig.from_raw(self.get_conf("slider", None))
        self.slider_tuples_by_anchor: Dict[str, SliderTuple] = {}
        self.slider_member_latents = MemberLatentCache()
        self.setup_tuple_datasets()

    def setup_tuple_datasets(self):
        tuple_configs = []
        for i, config in enumerate(self.dataset_configs):
            options = self.slider_dataset_options[id(config)]
            if options.tuples is None:
                continue
            by_anchor = prepare_tuple_dataset(
                config,
                options.tuples,
                options.normalize,
                os.path.join(self.save_root, "slider_tuple", f"dataset_{i}.json"),
                where=f"dataset {i} ({options.tuples})",
            )
            for anchor, tup in by_anchor.items():
                known = self.slider_tuples_by_anchor.get(anchor)
                # a resolution list repeats the same manifest; different tuples may not share an anchor
                if known is not None and [(m.image, m.scale) for m in known.members] != [(m.image, m.scale) for m in tup.members]:
                    raise ValueError(f"slider_tuple: {anchor} is the anchor of two different tuples")
            self.slider_tuples_by_anchor.update(by_anchor)
            tuple_configs.append(config)

        if not tuple_configs:
            return
        if len(tuple_configs) != len(self.dataset_configs):
            raise ValueError("slider_tuple: tuple datasets cannot be mixed with other datasets in one job")
        # tuple datasets have no unconditional_path, so enter the guided-loss branch explicitly
        self.do_guided_loss = True

        tuples = list(self.slider_tuples_by_anchor.values())
        n_members = sum(len(t.members) for t in tuples)
        n_masks = sum(m.mask is not None for t in tuples for m in t.members)
        print_acc(
            f"slider_tuple: {len(tuples)} tuples, {n_members} members "
            f"({sum(len(t.trainable) for t in tuples)} trainable); {self.slider_config}"
        )
        if n_masks:
            print_acc(f"slider_tuple: masked loss on for {n_masks} members (unmasked members use weight 1)")

    def hook_before_train_loop(self):
        # before super() and before any batch is drawn, so dataloader workers
        # (started lazily on first iteration) see the per-image weights
        self.apply_slider_scales()
        super().hook_before_train_loop()

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

        # every tuple in the batch trains the same number of members (one slot each)
        k = min(
            len(t.trainable) if cfg.members_per_step == "all" else min(cfg.members_per_step, len(t.trainable))
            for t in tuples
        )
        picks = [sample_members(t, k, cfg.force_opposite_signs) for t in tuples]

        slot_latents = self.gather_member_latents(tuples, picks, batch)
        slot_scales = [[t.members[p[j]].scale for t, p in zip(tuples, picks)] for j in range(k)]
        per_tuple_weights = [
            member_weights(
                [t.members[i].scale for i in p], [t.members[i].weight for i in p], cfg.zero_sum_weights
            )
            for t, p in zip(tuples, picks)
        ]
        slot_weights = [[w[j] for w in per_tuple_weights] for j in range(k)]

        wsum = [weighted_scale_sum([t.members[i].scale for i in p], w) for t, p, w in zip(tuples, picks, per_tuple_weights)]
        self.additional_logs["slider/weighted_scale_sum"] = sum(wsum) / len(wsum)
        self.additional_logs["slider/members_per_step"] = k

        if self.step_num < cfg.debug_dump_steps:
            self.dump_tuples(tuples, picks, batch)

        return slider_tuple_loss(
            self.sd,
            slot_latents=slot_latents,
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
        )

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

    def dump_tuples(self, tuples, picks, batch):
        folder = os.path.join(self.save_root, "slider_debug")
        for t, p, item in zip(tuples, picks, batch.file_items):
            images = [load_member_image(m.image, item) for m in t.members]
            labels = [f"s={m.scale:+g}" + (" train" if i in p else "") for i, m in enumerate(t.members)]
            path = dump_tuple(os.path.join(folder, f"step{self.step_num:06d}_{t.id}.png"), images, labels)
            print_acc(f"slider_tuple: debug dump {path}")
