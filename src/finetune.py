from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig  # https://huggingface.co/docs/peft/developer_guides/quantization  https://discuss.huggingface.co/t/typeerror-llamaforcausallm-init-got-an-unexpected-keyword-argument-load-in-4bit/41245
from trl import SFTTrainer
from transformers.integrations import WandbCallback
from trl.trainer import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
import torch
import os
from time import time

from profiler_callbacks import TorchProfilerCallback, MemoryHistoryCallback, WandBProfilerCallback, WandBTimerCallback
from lima_utils import *


def get_experiment_name(
        model_name:str, 
        dataset_name:str, 
        test:bool, 
        use_lora:bool, 
        use_quantization:bool,
        fp32:bool,
        tf32:bool,
        backprop_trick:bool,
        optimizer:str,
        foreach:bool,
    ):
    name = ""
    name += dataset_name.split('/')[1]
    name += "-" + model_name.split('/')[1]
    name += "-test" if test else ""
    name += "-lora" if use_lora else ""
    name += "-quant" if use_quantization else ""
    name += "-amp_fp32" if fp32 else ""
    name += "-amp_tf32" if tf32 else ""
    name += "-backprop_trick" if backprop_trick else ""
    name += f"-{optimizer}"
    name += f"-foreach-{foreach}" if backprop_trick else ""	
    name += f"-{int(time())}"
    return name

def main(model_name:str="facebook/opt-125m", dataset_name:str="GAIR/lima", 
         max_seq_length: int=1024, 
         num_epochs:int=2, 
         use_lora:bool=False, use_quantization:bool=False, gradient_checkpointing:bool=True, 
         fp16:bool=False, tf32:bool=False,   # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#floating-data-types
         profile:bool=False, profiler_repeat_every_n_steps:int=-1,
         resume_from_checkpoint:str=None,
         output_dir: str=None, 
         test: bool=False, n_test_batches: int=10,
         optimizer:str='adamw_torch',
         foreach:bool=True,  # Don't use this...
         no_eval:bool=False,
         backprop_trick:bool=False):
    """Finetunes the given model on the given dataset.

    Args:
        model_name: The name of the model to train. Can be "facebook/opt-125m", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf" etc...
        dataset_name: The name of the dataset to finetune on. Can only be "GAIR/lima" for now.
        max_seq_length: The maximum number of tokens to generate in the finetuning process. NOTE: My PC can handle max_seq_length=256 with OPT-125m, but go for 128 for testing...
        num_epochs: The number of epochs to train for.
        use_lora: Whether to use LoRA for training.
        use_quantization: Whether to use quantization for training.
        fp16: Whether to use fp16 floating data type.
        tf32: Whether to use tf32 floating data type.
        profile: Whether to do a detailed profiling of the GPU memory and the stack during training. Note that a rudimentary GPU memory profiler is always sampling the first 5 steps of an epoch. 
        profiler_repeat_every_n_steps: Number of steps to repeat the profiling.
        resume_from_checkpoint: If not None, the path to a checkpoint to resume training from (the entire experiment will simply proceed from there...).
        output_dir: The directory where the output will be stored.
        test: Whether we are running test code. If so, you may also want to set n_test_batches to a small number.
        n_test_batches: Number of batches used for testing if test is True. Set e.g. to 1 for overfitting to a single batch (to check if the model is set up correctly).
        backprop_trick: Whether to use the backprop trick for training.
        optimizer: The optimizer to use. Can only be "adam" or "sgd" if the backprop_trick is enabled.
    """

    # Environment setup
    OUTPUT_DIR = os.getenv("OUTPUT_DIR_MSC")
    assert (OUTPUT_DIR is not None) or (output_dir is not None), """
        Please set the environment variable OUTPUT_DIR_MSC to the directory where you want to store the output. E.g.: export OUTPUT_DIR_MSC=/work3/<name>/myfolder
        Alternatively, you can pass the output directory as an argument to the function."""
    if OUTPUT_DIR is None: OUTPUT_DIR = output_dir

    if resume_from_checkpoint is None:
        experiment_name = get_experiment_name(model_name, dataset_name, test, use_lora, use_quantization, fp16, tf32, backprop_trick, optimizer, foreach)
    else:
        experiment_name = resume_from_checkpoint

    os.environ['WANDB_PROJECT'] = 'lima_ft_'+model_name.split('/')[1]
    os.environ['WANDB_DIR'] = os.path.join(OUTPUT_DIR, 'logs')                      # Becomes 'OUTPUT_DIR/logs/wandb'
    os.environ['WANDB_CACHE_DIR'] = os.path.join(OUTPUT_DIR, 'cache_dir', 'wandb')  # Becomes 'OUTPUT_DIR/cache_dir/wandb'
    cache_dir = os.path.join(OUTPUT_DIR, 'cache_dir', 'huggingface')                # Becomes "OUTPUT_DIR/cache_dir/huggingface"
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints', experiment_name)       # Becomes "OUTPUT_DIR/checkpoints/<experiment_name>"
    logging_dir = os.path.join(OUTPUT_DIR, 'logs', experiment_name)                 # Becomes "OUTPUT_DIR/logs/<experiment_name>"

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Model setup
    lora_config = LoraConfig(   # https://huggingface.co/docs/peft/developer_guides/lora
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    quantization_config = BitsAndBytesConfig(   # Also explains LoRA: https://huggingface.co/docs/peft/developer_guides/quantization
        load_in_8bit=True,
        #load_in_4bit=True,
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
    ## Prepare the model for kbit training (if use_quantization is True) and get the LoRA model (if use_lora is True)
    model = prepare_model_for_kbit_training(model) if use_quantization else model
    model = get_peft_model(model, lora_config) if use_lora else model
    
    # Setup backprop trick if needed
    if backprop_trick:
        from backprop_trick import MotherOptimizer
        optimizer = optimizer.lower()
        if optimizer == 'adam':
            optimizer_cls = torch.optim.Adam
        elif optimizer == 'sgd':
            optimizer_cls = torch.optim.SGD
        optimizer = MotherOptimizer(model.parameters(), optimizer_cls, lr=1e-3, foreach=foreach)
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = 1e-3
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Dataset setup
    ds = load_dataset(dataset_name, 'plain_text', cache_dir=cache_dir)
    template_formatter = TemplateFormatter(ds, tokenizer)
    train_ds, test_ds = template_formatter.train_ds, template_formatter.test_ds
    if test: 
        train_ds = Dataset.from_dict(train_ds[0:n_test_batches])

    # Trainer setup
    if tf32: 
        torch.backends.cuda.matmul.allow_tf32 = True
    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1 if test else 16,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=n_test_batches if test else 100,
        do_train=True,
        output_dir=checkpoint_dir,
        logging_dir=logging_dir,
        overwrite_output_dir=True,
        report_to="wandb",
        run_name=experiment_name,
        save_strategy="epoch",
        fp16=fp16,
        tf32=tf32,
        optim=optimizer if not backprop_trick else "adamw_torch",
    )
    callbacks = []
    if profile:
        # If profile is True, the setup is TorchProfilerCallback, MemoryHistoryCallback, WandBTimerCallback, and ExampleCallback.
        callbacks.append(TorchProfilerCallback(logging_dir, profile_epochs=[], profile_n_steps=5, repeat_every_n_steps=profiler_repeat_every_n_steps))
        callbacks.append(MemoryHistoryCallback(logging_dir, profile_epochs=[], profile_n_steps=5, repeat_every_n_steps=profiler_repeat_every_n_steps))
    else:
        # If profile is false, the setup is WandBProfilerCallback,.WandBTimerCallback and ExampleCallback.
        callbacks.append(WandBProfilerCallback(profile_epochs=[], profile_n_steps=5, repeat_every_n_steps=profiler_repeat_every_n_steps))
    if not no_eval:
        callbacks.append(ExampleCallback(template_formatter, max_seq_length=max_seq_length))
    callbacks.append(WandBTimerCallback())
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=template_formatter.collator,
        dataset_text_field="text",
        train_dataset=train_ds,
        max_seq_length=max_seq_length,
        callbacks=callbacks,
        optimizers=(optimizer, lr_scheduler) if backprop_trick else (None, None),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint is not None)


if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)