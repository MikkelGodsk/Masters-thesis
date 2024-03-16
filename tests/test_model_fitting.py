import os

import torch
from transformers import TrainingArguments
from transformers.trainer_utils import TrainOutput
from trl import SFTTrainer
from datasets import load_dataset, Dataset

from lima_utils import TemplateFormatter
from model_factories import Factory

from tests.cache_dir import *


def overfit_to_small_batch(model_name, dataset_name, mock_model_name, checkpoint_dir, use_lora:bool=False, use_quantization:bool=False, backprop_trick:bool=False):
    # Spawn model and tokenizer
    factory = Factory.spawn_factory(model_name, cache_dir=cache_dir, model_name=mock_model_name)
    if use_lora and use_quantization:
        factory.setup_peft()
    if backprop_trick:
        factory.setup_mebp(optimizer_cls=torch.optim.Adam, initial_lr=5e-5)
    #factory.setup_model(device_map='cpu')
    m = factory.spawn_model()
    model, optimizer_and_lr_scheduler = m[0], m[1:]   # I use `optimizer_and_lr_scheduler` here to avoid interfering with the `optimizer` argument.
    tokenizer = factory.spawn_tokenizer()

    # Dataset setup
    ds = load_dataset(dataset_name, 'plain_text', cache_dir=cache_dir)
    template_formatter = TemplateFormatter(ds, tokenizer)
    train_ds = template_formatter.train_ds
    train_ds = Dataset.from_dict(train_ds[0:2])

    # Trainer setup
    training_args = TrainingArguments(
        num_train_epochs=1000,#2000,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        per_device_train_batch_size=2,
        do_train=True,
        overwrite_output_dir=True,
        save_strategy="no",
        output_dir=checkpoint_dir,
        report_to="none",
        optim='adamw_torch',
        learning_rate=5e-5,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=template_formatter.collator,
        dataset_text_field="text",
        train_dataset=train_ds,
        max_seq_length=128,
        optimizers=optimizer_and_lr_scheduler,   # If backprop_trick is False, this is set to (None, None) by the factory...
    )
    train_output: TrainOutput = trainer.train()
    return train_output.training_loss


def test_overfit_to_small_batch_opt():
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints', 'test_overfit_to_small_batch_opt')
    model_name:str="facebook/opt-125m"
    mock_model_name:str = model_name #'hf-internal-testing/tiny-random-OPTForCausalLM'   # Mock model doesn't seem to work for training - Has another embedding dimension...
    dataset_name:str="GAIR/lima"
    train_loss = overfit_to_small_batch(model_name, dataset_name, mock_model_name, checkpoint_dir)
    assert torch.isclose(torch.tensor(train_loss), torch.tensor(0.0), atol=1e-2)


def test_overfit_to_small_batch_opt_backprop_trick():   # takes approx. 50 mins. to run on a GTX 960
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints', 'test_overfit_to_small_batch_opt_backprop_trick')
    model_name:str="facebook/opt-125m"
    mock_model_name:str = model_name #'hf-internal-testing/tiny-random-OPTForCausalLM'   # Mock model doesn't seem to work for training - Has another embedding dimension...
    dataset_name:str="GAIR/lima"
    train_loss = overfit_to_small_batch(model_name, dataset_name, mock_model_name, checkpoint_dir, backprop_trick=True)
    assert torch.isclose(torch.tensor(train_loss), torch.tensor(0.0), atol=1e-2)