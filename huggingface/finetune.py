from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig  # https://huggingface.co/docs/peft/developer_guides/quantization  https://discuss.huggingface.co/t/typeerror-llamaforcausallm-init-got-an-unexpected-keyword-argument-load-in-4bit/41245
from trl import SFTTrainer
from trl.trainer import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
import torch
import os
from time import time

from profiler_callbacks import TorchProfilerCallback, MemoryHistoryCallback, WandBProfilerCallback, WandBTimerCallback
from lima_utils import *

"""
    NOTE ON MEMORY:
    - If the sequence length is large and the batch size is large, then the cross-entropy computation may be the most expensive operation! 
    E.g. Opt-125m:
        a head with 50272 output features x 16 batch size x 1024 sequence length x 4 bytes per float = 3.294.625.792 bytes = 3.29 GB (and that's just the logit size!)
        the model's hidden dimensions are 768... so the hidden states are 768 x 16 x 1024 x 4 = 6.29 MB...

    NOTE ON ERRORS:
    - If you get the error ```/opt/conda/conda-bld/pytorch_1708025845868/work/aten/src/ATen/native/cuda/Indexing.cu:1290: indexSelectLargeIndex: block: [176,0,0], thread: [127,0,0] Assertion `srcIndex < srcSelectDimSize` failed.```
    it might be because the pad_token is set to something other than the eos_token. This is a known issue with Llama-2-7b. The solution is to set pad_token = eos_token. This is done in the SPECIAL_TOKENS_DICT above.
    Also it might be an issue with max_seq_length being too long. For opt-125m, it seems to happen if max_seq_length=4096, but not if max_seq_length=1024.

    - If you get the UserWarning that "### Human: " could not be found, maybe the formatting of the dataset is wrong. It seems like the DataCollatorForCompletionOnlyLM is not able to find the instruction and response templates. 
    This may also sometimes be fixed by setting max_seq_length=2048.
    
    NOTE ON Data collator:
    - The DataCollatorForCompletionOnlyLM is used to mask the response in the dataset. It is used to tell the model how mask out the response, when given the entire sequence.
"""


SPECIAL_TOKENS_DICT = {} #{'pad_token': '</s>', 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'mask_token': '<mask>'}
# Llama-2-7b has an issue. It seems like it might be the pad-token that causes it, but it's not clear... The issue: /opt/conda/conda-bld/pytorch_1708025845868/work/aten/src/ATen/native/cuda/Indexing.cu:1290: indexSelectLargeIndex: block: [97,0,0], thread: [95,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
## It seems to work to set pad_token = eos_token, as done here: https://discuss.huggingface.co/t/finetuning-quantised-llama-2-with-lora/49289


def get_experiment_name(
        model_name:str, 
        dataset_name:str, 
        test:bool, 
        use_lora:bool, 
        use_quantization:bool,
        fp32:bool,
        tf32:bool,

    ):
    name = ""
    name += dataset_name.split('/')[1]
    name += "-" + model_name.split('/')[1]
    name += "-test" if test else ""
    name += "-lora" if use_lora else ""
    name += "-quant" if use_quantization else ""
    name += "-amp_fp32" if fp32 else ""
    name += "-amp_tf32" if tf32 else ""
    name += f"-{int(time())}"
    return name

def main(model_name:str="facebook/opt-125m", dataset_name:str="GAIR/lima", max_seq_length: int=1024, num_epochs:int=2, 
         use_lora:bool=False, use_quantization:bool=False, gradient_checkpointing:bool=True, 
         fp16:bool=False, tf32:bool=False,   # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#floating-data-types
         profile:bool=False, profiler_repeat_every_n_steps:int=-1,
         output_dir: str=None, 
         test: bool=False, n_test_batches: int=10):
    """Finetunes the given model on the given dataset.

    Args:
        model_name: The name of the model to train. Can be "facebook/opt-125m", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf" etc...
        dataset_name: The name of the dataset to finetune on. Can only be "GAIR/lima" for now.
        max_seq_length: The maximum number of tokens to generate in the finetuning process. NOTE: My PC can handle max_seq_length=256 with OPT-125m, but go for 128 for testing...
        num_epochs: The number of epochs to train for.
        use_lora: Whether to use LoRA for training.
        use_quantization: Whether to use quantization for training.
        fp32: Whether to use fp32 floating data type.
        tf32: Whether to use tf32 floating data type.
        profile: Whether to do a detailed profiling of the GPU memory and the stack during training. Note that a rudimentary GPU memory profiler is always sampling the first 5 steps of an epoch. 
        profiler_repeat_every_n_steps: Number of steps to repeat the profiling.
        output_dir: The directory where the output will be stored.
        test: Whether we are running test code. If so, you may also want to set n_test_batches to a small number.
        n_test_batches: Number of batches used for testing if test is True. Set e.g. to 1 for overfitting to a single batch (to check if the model is set up correctly).
    """

    # Environment setup
    OUTPUT_DIR = os.getenv("OUTPUT_DIR_MSC")
    assert (OUTPUT_DIR is not None) or (output_dir is not None), """
        Please set the environment variable OUTPUT_DIR_MSC to the directory where you want to store the output. E.g.: export OUTPUT_DIR_MSC=/work3/<name>/myfolder
        Alternatively, you can pass the output directory as an argument to the function."""
    if OUTPUT_DIR is None: OUTPUT_DIR = output_dir

    experiment_name = get_experiment_name(model_name, dataset_name, test, use_lora, use_quantization, fp16, tf32)

    os.environ['WANDB_PROJECT'] = 'lima_ft_'+model_name.split('/')[1]
    os.environ['WANDB_DIR'] = os.path.join(OUTPUT_DIR, 'logs')                      # Becomes 'OUTPUT_DIR/logs/wandb'
    os.environ['WANDB_CACHE_DIR'] = os.path.join(OUTPUT_DIR, 'cache_dir', 'wandb')  # Becomes 'OUTPUT_DIR/cache_dir/wandb'
    cache_dir = os.path.join(OUTPUT_DIR, 'cache_dir', 'huggingface')                # Becomes "OUTPUT_DIR/cache_dir/huggingface"
    checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints', experiment_name)       # Becomes "OUTPUT_DIR/checkpoints/<experiment_name>"
    logging_dir = os.path.join(OUTPUT_DIR, 'logs', experiment_name)                 # Becomes "OUTPUT_DIR/logs/<experiment_name>"

    # Tokenizer setup
    if model_name == "facebook/opt-125m":
        pass
        #SPECIAL_TOKENS_DICT['pad_token'] = '<pad>'
    elif model_name == "meta-llama/Llama-2-7b-hf":
        SPECIAL_TOKENS_DICT['pad_token'] = '</s>'
    elif model_name == "meta-llama/Llama-2-7b-chat-hf":
        SPECIAL_TOKENS_DICT['pad_token'] = '</s>'

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, **SPECIAL_TOKENS_DICT)

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
    

    # Dataset setup
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)  # Data collator tells how to split and mask the dataset (in teacher forcing)
    ds = load_dataset(dataset_name, 'plain_text', cache_dir=cache_dir)
    train_ds = process_ds(ds["train"])
    if test: 
        train_ds = Dataset.from_dict(train_ds[0:n_test_batches])
    eval_ds = ds["test"]["conversations"]

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
    )
    callbacks = (   # If profile is True, the setup is TorchProfilerCallback, MemoryHistoryCallback, and ExampleCallback. If profile is false, the setup is WandBProfilerCallback and ExampleCallback.
        [
            WandBProfilerCallback(profile_epochs=[], profile_n_steps=5, repeat_every_n_steps=profiler_repeat_every_n_steps),
        ] if not profile else [
            TorchProfilerCallback(logging_dir, profile_epochs=[], profile_n_steps=5, repeat_every_n_steps=profiler_repeat_every_n_steps), 
            MemoryHistoryCallback(logging_dir, profile_epochs=[], profile_n_steps=5, repeat_every_n_steps=profiler_repeat_every_n_steps),
        ]) + [
            ExampleCallback(train_ds, eval_ds, max_new_tokens=max_seq_length),
            WandBTimerCallback(),
        ]
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        dataset_text_field="text",
        train_dataset=train_ds,
        max_seq_length=max_seq_length,
        callbacks=callbacks,
    )
    trainer.train()


if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(main)