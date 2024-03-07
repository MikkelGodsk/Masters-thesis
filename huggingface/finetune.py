from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import PeftModel  # https://discuss.huggingface.co/t/typeerror-llamaforcausallm-init-got-an-unexpected-keyword-argument-load-in-4bit/41245
from trl import SFTTrainer
from trl.trainer import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
from peft import LoraConfig
import torch
import wandb
import numpy as np
import os
from time import time

from profiler_callbacks import TorchProfilerCallback, MemoryHistoryCallback, WandBProfilerCallback, WandBTimerCallback
from lima_utils import *


SPECIAL_TOKENS_DICT = {'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'mask_token': '<mask>'}


def main(model_name:str="facebook/opt-125m", dataset_name:str="GAIR/lima", max_new_tokens: int=2048, num_epochs:int=2, use_lora:bool=True, use_quantization:bool=True, profile:bool=False, output_dir: str=None, test: bool=False, n_test_batches: int=10):
    """Finetunes the given model on the given dataset.

    Args:
        model: The model to train. Can be "facebook/opt-125m", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf" etc...
        dataset: The dataset to finetune on. Can only be "GAIR/lima" for now.
        max_new_tokens: The maximum number of tokens to generate in the finetuning process. NOTE: My PC can handle max_new_tokens=256 with OPT-125m, but go for 128 for testing...
        num_epochs: The number of epochs to train for.
        profile: Whether to do a detailed profiling of the GPU memory and the stack during training. Note that a rudimentory GPU memory profiler is always sampling the first 5 steps of an epoch. 
        test: Whether we are running test code. If so, you may also want to set n_test_batches to a small number.
        n_test_batches: Number of batches used for testing if test is True. Set e.g. to 1 for overfitting to a single batch (to check if the model is set up correctly).
    """

    # Environment setup
    OUTPUT_DIR = os.getenv("OUTPUT_DIR_MSC")
    assert (OUTPUT_DIR is not None) or (output_dir is not None), """
        Please set the environment variable OUTPUT_DIR_MSC to the directory where you want to store the output. E.g.: export OUTPUT_DIR_MSC=/work3/<name>/myfolder
        Alternatively, you can pass the output directory as an argument to the function."""
    if OUTPUT_DIR is None: OUTPUT_DIR = output_dir

    experiment_name = f"{dataset_name.split('/')[1]}-{model_name.split('/')[1]}" + ("-test" if test else "") + f"-{int(time())}"

    os.environ['WANDB_PROJECT'] = 'lima_ft_'+model_name.split('/')[1]
    os.environ['WANDB_DIR'] = os.path.join(OUTPUT_DIR, 'logs')                      # Becomes 'OUTPUT_DIR/logs/wandb'
    os.environ['WANDB_CACHE_DIR'] = os.path.join(OUTPUT_DIR, 'cache_dir', 'wandb')  # Becomes 'OUTPUT_DIR/cache_dir/wandb'
    cache_dir = os.path.join(OUTPUT_DIR, 'cache_dir', 'huggingface')                # Becomes "OUTPUT_DIR/cache_dir/huggingface"
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints', experiment_name)       # Becomes "OUTPUT_DIR/checkpoints/<experiment_name>"
    logging_dir = os.path.join(OUTPUT_DIR, 'logs', experiment_name)                 # Becomes "OUTPUT_DIR/logs/<experiment_name>"

    # Model setup
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=cache_dir, 
        quantization_config=quantization_config if use_quantization else None,
        device_map="auto",  # Not sure if the device map enables FSDP by default...
    )
    model = PeftModel.from_pretrained(model, model_name, config=lora_config) if use_lora else model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, **SPECIAL_TOKENS_DICT)

    # Dataset setup
    ## Data collator: To tell the trainer how to split and mask the dataset (in teacher forcing)
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)
        # DataCollatorForCompletionOnlyLM finds the instruction template (by looping through the tokens - as we could easily implement ourselves) and cuts. Then it cuts again after the assistant template.
        # The cuts are, however, not made in the input_ids... Here the model is being given the answers too.... But the labels are set to -100 for everything until the answer begins, making the loss function ignore them...
        # Hence it implements teacher forcing when trained....
        # The trainer's dataloader (evidently) calls its DataCollatorForCompletionOnlyLM.torch_call method either during dataset building or during fetching
        # UserWarning: It seems like the UserWarning with not being able to find the instruction and response templates were fixed when setting max_seq_length=2048....

    ds = load_dataset(dataset_name, 'plain_text', cache_dir=cache_dir)
    train_ds = process_ds(ds["train"])
    if test: 
        train_ds = Dataset.from_dict(train_ds[0:n_test_batches])
    eval_ds = ds["test"]["conversations"]

    # Trainer setup
    training_args = TrainingArguments(
        num_train_epochs=10 if test else num_epochs,
        per_device_train_batch_size=1 if test else 16,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        logging_steps=n_test_batches if test else 100,
        do_train=True,
        output_dir=checkpoint_dir,
        logging_dir=logging_dir,
        overwrite_output_dir=True,
        report_to="wandb",
        run_name=experiment_name,
        save_strategy="epoch"
    )
    callbacks = (   # If profile is True, the setup is TorchProfilerCallback, MemoryHistoryCallback, and ExampleCallback. If profile is false, the setup is WandBProfilerCallback and ExampleCallback.
        [
            WandBProfilerCallback(profile_epochs=[], profile_n_steps=5),
        ] if not profile else [
            TorchProfilerCallback(logging_dir, profile_epochs=[], profile_n_steps=5), 
            MemoryHistoryCallback(logging_dir, profile_epochs=[], profile_n_steps=5),
        ]) + [
            ExampleCallback(train_ds, eval_ds, max_new_tokens=max_new_tokens),
            WandBTimerCallback(),
        ]
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        dataset_text_field="text",
        train_dataset=train_ds,
        max_seq_length=max_new_tokens,
        callbacks=callbacks,
    )
    trainer.train()


if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)