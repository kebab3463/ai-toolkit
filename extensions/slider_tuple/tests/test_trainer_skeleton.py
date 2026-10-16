"""M2: the slider_tuple trainer registers as an extension and leaves upstream guidance untouched."""

import pytest
import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from extensions.slider_tuple import AI_TOOLKIT_EXTENSIONS
from extensions.slider_tuple.SliderTupleTrainer import SLIDER_GUIDANCE_TYPES, SliderTupleTrainer
from toolkit.guidance import get_guided_loss_polarity

from .stubs import StubModel, make_batch, make_embeds, make_pair, make_train_config


def make_trainer(sd: StubModel) -> SliderTupleTrainer:
    # skip SDTrainer.__init__ (it needs a job, a model on disk and a GPU)
    trainer = SliderTupleTrainer.__new__(SliderTupleTrainer)
    trainer.sd = sd
    trainer.train_config = make_train_config()
    return trainer


def test_extension_registers_trainer():
    uids = [ext.uid for ext in AI_TOOLKIT_EXTENSIONS]
    assert uids == ["slider_tuple_trainer"]
    process_cls = AI_TOOLKIT_EXTENSIONS[0].get_process()
    assert process_cls is SliderTupleTrainer
    assert issubclass(process_cls, SDTrainer)


def test_polarity_delegates_to_upstream_unchanged():
    after, before, noise, timesteps = make_pair()
    embeds = make_embeds(after.shape[0])

    # reference: upstream polarity called directly
    sd_ref = StubModel()
    ref_loss = get_guided_loss_polarity(
        noisy_latents=None,
        conditional_embeds=embeds,
        match_adapter_assist=False,
        network_weight_list=[0.7, 0.7],
        timesteps=timesteps,
        pred_kwargs={},
        batch=make_batch(after, before),
        noise=noise,
        sd=sd_ref,
        train_config=make_train_config(),
    )
    ref_grad = sd_ref.lora.weight.grad.clone()

    # same inputs through the slider_tuple trainer
    sd = StubModel()
    trainer = make_trainer(sd)
    loss = trainer.get_guided_loss(
        noisy_latents=None,
        conditional_embeds=embeds,
        match_adapter_assist=False,
        network_weight_list=[0.7, 0.7],
        timesteps=timesteps,
        pred_kwargs={},
        batch=make_batch(after, before),
        noise=noise,
        mask_multiplier=torch.ones(after.shape[0], 1, 1, 1),
        prior_pred=None,
    )

    assert torch.equal(loss, ref_loss)
    assert torch.equal(sd.lora.weight.grad, ref_grad)
    assert ref_grad.abs().sum() > 0
    # upstream contract: backward already ran, returned loss is detached but marked as requiring grad
    assert loss.requires_grad and loss.grad_fn is None
    assert sd.network.multiplier == [0.7, 0.7, -0.7, -0.7]


@pytest.mark.parametrize("guidance_type", SLIDER_GUIDANCE_TYPES)
def test_slider_guidance_types_are_reserved(guidance_type):
    after, before, noise, timesteps = make_pair()
    trainer = make_trainer(StubModel())
    with pytest.raises(NotImplementedError, match=guidance_type):
        trainer.get_guided_loss(
            noisy_latents=None,
            conditional_embeds=make_embeds(after.shape[0]),
            match_adapter_assist=False,
            network_weight_list=[1.0, 1.0],
            timesteps=timesteps,
            pred_kwargs={},
            batch=make_batch(after, None, guidance_type=guidance_type),
            noise=noise,
        )
