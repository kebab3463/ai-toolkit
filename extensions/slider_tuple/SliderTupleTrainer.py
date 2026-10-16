from typing import Dict, Optional

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds

from .scales import SliderDatasetOptions, apply_slider_scales, slider_options_by_config

# guidance types handled here; every other guidance_type goes to upstream unchanged
SLIDER_GUIDANCE_TYPES = ("slider_tuple",)


class SliderTupleTrainer(SDTrainer):
    """SDTrainer with tuple-based image slider losses.

    Selected with ``type: slider_tuple_trainer`` in a job config. Datasets whose
    ``guidance_type`` is not one of ``SLIDER_GUIDANCE_TYPES`` (e.g. ``polarity``)
    train exactly as they would under ``sd_trainer``.

    Dataset keys (all default off, see scales.py): ``slider_scales``,
    ``slider_scale_normalize``.
    """

    def __init__(self, process_id: int, job, config, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.slider_dataset_options: Dict[int, SliderDatasetOptions] = slider_options_by_config(
            self.get_conf("datasets", None) or [], self.dataset_configs
        )

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

        raise NotImplementedError(
            f"guidance_type '{guidance_type}' is reserved for tuple datasets and is not implemented yet"
        )
