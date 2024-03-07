from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from transformers.integrations import WandbCallback
from trl import SFTTrainer
from trl.trainer import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
import torch
import wandb
import numpy as np
import os
from time import time

from profiler_callbacks import TorchProfilerCallback, MemoryHistoryCallback, WandBProfilerCallback
from lima_utils import *

"""
TODO:  Reproduce LIMA experiment &
- Get WandB to log on work3!!   WORKS
- Overfit to single batch       WORKS
- Validation set                WORKS
- Use click or something else
- Checkpointing....
- Accelerate for multi gpu??
- Gradient trick??
- DPO?
- CPU off loading
- PEFT
"""

test = True    # Flag for whether we are testing the code or not. If true, we pass a single batch and overfit to that...
n_test_batches = 10  # E.g.: Can we overfit to a single batch? (For testing/debugging purposes). We should be able to if the model is set up correctly...

model_name = "facebook/opt-125m"
dataset_name = "GAIR/lima"
experiment_name = f"{dataset_name.split('/')[1]}-{model_name.split('/')[1]}" + ("-test" if test else "") + f"-{int(time())}"

OUTPUT_DIR = os.getenv("OUTPUT_DIR_MSC")
assert OUTPUT_DIR is not None

os.environ['WANDB_PROJECT'] = 'lima_ft_playground'
os.environ['WANDB_DIR'] = os.path.join(OUTPUT_DIR, 'logs')   #'/work3/s184399/logs/wandb'
os.environ['WANDB_CACHE_DIR'] = os.path.join(OUTPUT_DIR, 'cache_dir', 'wandb')  # '/work3/s184399/cache_dir/wandb'
cache_dir = os.path.join(OUTPUT_DIR, 'cache_dir', 'huggingface') # "/work3/s184399/cache_dir/huggingface"
checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints', experiment_name) # "/work3/s184399/checkpoints/lima-opt-125m"
logging_dir = os.path.join(OUTPUT_DIR, 'logs', experiment_name)  # "/work3/s184399/logs/lima-opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")  # Not sure if the device map enables FSDP by default...
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# Dataset setup
## Data collator: To tell the trainer how to split and mask the dataset (in teacher forcing)
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)
    # DataCollatorForCompletionOnlyLM finds the instruction template (by looping through the tokens - as we could easily implement ourselves) and cuts. Then it cuts again after the assistant template.
    # The cuts are, however, not made in the input_ids... Here the model is being given the answers too.... But the labels are set to -100 for everything until the answer begins, making the loss function ignore them...
    # Hence it implements teacher forcing when trained....
    # The trainer's dataloader (evidently) calls its DataCollatorForCompletionOnlyLM.torch_call method either during dataset building or during fetching

ds = load_dataset(dataset_name, 'plain_text', cache_dir=cache_dir)
train_ds = process_ds(ds["train"])
if test: train_ds = Dataset.from_dict(train_ds[0:n_test_batches])
eval_ds = ds["test"]["conversations"]
len_eval = len(eval_ds)
len_train_ds = len(train_ds)

training_args = TrainingArguments(
    num_train_epochs=100 if test else 1,
    per_device_train_batch_size=1,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    logging_steps=100,
    do_train=True,
    output_dir=checkpoint_dir,
    logging_dir=logging_dir,
    overwrite_output_dir=True,
    report_to="wandb",
    run_name=experiment_name,
    save_strategy="epoch"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=collator,
    dataset_text_field="text",
    train_dataset=train_ds,
    max_seq_length=256, #2048,   # My PC can handle max_seq_length=256 with OPT-125m !!!   # It seems like the UserWarning with not being able to find the instruction and response templates were fixed when setting max_seq_length=2048....
    callbacks=[
        WandBProfilerCallback(logging_dir, profile_epochs=[], profile_n_steps=5),
        #TorchProfilerCallback(logging_dir, profile_epochs=[0], profile_n_steps=5, do_stack_profile=False, upload_to_wandb=True), 
        #MemoryHistoryCallback(logging_dir, profile_epochs=[0], profile_n_steps=5),
        #ExampleCallback(train_ds, eval_ds, max_new_tokens=128), #1024), 
    ],
)
trainer.train()