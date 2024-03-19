"""Tests whether the train dataloader supplied by SFTTrainer is set up correctly."""

import os

from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer  # Use `.get_train_dataloader()`

from src.model_factories import Factory
from lima_utils import TemplateFormatter

OUTPUT_DIR = os.getenv("OUTPUT_DIR_MSC")
cache_dir = os.path.join(OUTPUT_DIR, "cache_dir", "huggingface")


# According to `https://github.com/huggingface/transformers/issues/15308`, HuggingFace uses this repo `https://huggingface.co/hf-internal-testing` for internal testing


def get_train_dataloader_and_tokenizer(
    model_name: str,
    dataset_name: str,
    checkpoint_dir: str,
    mock_model_name: str | None = None,
):
    # Setup model and tokenizer
    factory = Factory.spawn_factory(
        architecture_type=model_name, cache_dir=cache_dir
    )  # For tokenizer, and maybe model
    if mock_model_name is not None:
        mock_factory = Factory.spawn_factory(
            architecture_type=model_name,
            model_name=mock_model_name,
            cache_dir=cache_dir,
        )
        m = mock_factory.spawn_model()
    else:
        m = factory.spawn_model()
    model, optimizer_and_lr_scheduler = m[0], m[1:]
    tokenizer = factory.spawn_tokenizer()

    # Setup dataset
    ds = load_dataset(dataset_name, "plain_text", cache_dir=cache_dir)
    template_formatter = TemplateFormatter(ds, tokenizer)
    train_ds = template_formatter.train_ds

    # Setup trainer
    training_args = TrainingArguments(
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        do_train=True,
        overwrite_output_dir=True,
        save_strategy="epoch",
        output_dir=checkpoint_dir,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=template_formatter.collator,
        dataset_text_field="text",
        train_dataset=train_ds,
        max_seq_length=2048,
        optimizers=optimizer_and_lr_scheduler,  # If backprop_trick is False, this is set to (None, None) by the factory...
    )
    dataloader = trainer.get_train_dataloader()
    return dataloader, tokenizer


def check_special_tokens(dataloader, tokenizer):
    """Asserts that the bos, sep, and eos tokens are in the tokenized instructions.
    Also asserts that they are not repeated twice in a row.
    """
    for batch in dataloader:
        for tokenized_instruction in batch["input_ids"]:
            if tokenizer.bos_token_id is not None:
                assert tokenizer.bos_token_id in tokenized_instruction
            if tokenizer.sep_token_id is not None:
                assert tokenizer.sep_token_id in tokenized_instruction
            if tokenizer.eos_token_id is not None:
                if tokenizer.pad_token_id in tokenized_instruction:
                    assert tokenizer.eos_token_id in tokenized_instruction
            prev_token = -1
            for token in tokenized_instruction:
                if (
                    token
                    in [
                        tokenizer.bos_token_id,
                        tokenizer.sep_token_id,
                        tokenizer.eos_token_id,
                    ]
                ) and (token is not None):
                    assert token != prev_token
                prev_token = token


def test_lima_train_dataloader_for_opt():
    """Tests whether the train dataloader supplied by SFTTrainer is set up correctly."""
    model_name: str = "facebook/opt-125m"
    mock_model_name: str = "hf-internal-testing/tiny-random-OPTForCausalLM"
    dataset_name: str = "GAIR/lima"
    checkpoint_dir = os.path.join(
        OUTPUT_DIR, "checkpoints", "test_lima_train_dataloader_for_opt"
    )
    train_dataloader, tokenizer = get_train_dataloader_and_tokenizer(
        model_name, dataset_name, checkpoint_dir, mock_model_name=mock_model_name
    )
    check_special_tokens(train_dataloader, tokenizer)


def test_lima_train_dataloader_for_llama2():
    """Tests whether the train dataloader supplied by SFTTrainer is set up correctly."""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    mock_model_name: str = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    dataset_name: str = "GAIR/lima"
    checkpoint_dir = os.path.join(
        OUTPUT_DIR, "checkpoints", "test_lima_train_dataloader_for_llama2"
    )
    train_dataloader, tokenizer = get_train_dataloader_and_tokenizer(
        model_name, dataset_name, checkpoint_dir, mock_model_name=mock_model_name
    )
    check_special_tokens(train_dataloader, tokenizer)
