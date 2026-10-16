from typing import Optional

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds

# guidance types handled here; every other guidance_type goes to upstream unchanged
SLIDER_GUIDANCE_TYPES = ("slider_tuple",)


class SliderTupleTrainer(SDTrainer):
    """SDTrainer with tuple-based image slider losses.

    Selected with ``type: slider_tuple_trainer`` in a job config. Datasets whose
    ``guidance_type`` is not one of ``SLIDER_GUIDANCE_TYPES`` (e.g. ``polarity``)
    train exactly as they would under ``sd_trainer``.
    """

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
