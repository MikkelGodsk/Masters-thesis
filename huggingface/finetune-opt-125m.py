from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
from transformers.integrations import WandbCallback
from trl import SFTTrainer
from datasets import load_dataset
import torch
import wandb
import numpy as np
import os

"""
TODO:
- Overfit to single example
- Validation set
- Accelerate for multi gpu??
- Gradient trick??
- DPO?
- CPU off loading
- PEFT
"""

os.environ['WANDB_CACHE_DIR'] = '/work3/s184399/logs/wandb'
os.environ['WANDB_PROJECT'] = 'lima_ft'


model_name = "facebook/opt-125m"   # Small enough to do fairly fast generation on CPU (0.8s per small sentence) => Might be good for development, debugging and testing...
dataset_name = "GAIR/lima"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset
ds = load_dataset(dataset_name, 'plain_text')

def process_ds(ds):
    return ds.map(lambda x: {"prompt": x["conversations"][0], "completion": x["conversations"][1]}, remove_columns=["conversations", "source"])

train_ds = process_ds(ds["train"])
eval_ds = ds["test"]["conversations"]
len_eval = len(eval_ds)

# Define training args
class ExampleCallback(WandbCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]

        prompt = eval_ds[np.random.randint(len_eval)][0]
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")  # Why is this not a dictionary???
        tokenized_completion = model.generate(tokenized_prompt.cuda(), max_new_tokens=100)
        completion = tokenizer.decode(token_ids=tokenized_completion.squeeze(0), skip_special_tokens=True)

        # Log to wandb
        text_table = wandb.Table(columns=["prompt", "completion"])
        text_table.add_data(prompt, completion)
        wandb.log({"examples": text_table})

training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=16,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    logging_steps=1,
    do_train=True,
    output_dir="/work3/s184399/checkpoints/lima-opt-125m",
    logging_dir="/work3/s184399/logs/lima-opt-125m",
    overwrite_output_dir=True,
    report_to="wandb",
    run_name="finetune-opt-125m",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    max_seq_length=1024,
    callbacks=[ExampleCallback],
)
trainer.train()