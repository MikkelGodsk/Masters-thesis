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

"""
TODO:  Reproduce LIMA experiment &
- Get WandB to log on work3!!   WORKS
- Overfit to single batch       WORKS
- Validation set                WORKS
- Accelerate for multi gpu??
- Gradient trick??
- DPO?
- CPU off loading
- PEFT
"""

test = True    # Flag for whether we are testing the code or not. If true, we pass a single batch and overfit to that...

os.environ['WANDB_PROJECT'] = 'lima_ft_llama-2'
os.environ['WANDB_DIR'] = '/work3/s184399/logs/wandb'
os.environ['WANDB_CACHE_DIR'] = '/work3/s184399/cache_dir/wandb'

model_name = "meta-llama/Llama-2-7b"
dataset_name = "GAIR/lima"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")  # Not sure if the device map enables FSDP by default...
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset setup
## Data collator: To tell the trainer how to split and mask the dataset (in teacher forcing)
instruction_template = "### Human:"
response_template = "### Assistant:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)
    # DataCollatorForCompletionOnlyLM finds the instruction template (by looping through the tokens - as we could easily implement ourselves) and cuts. Then it cuts again after the assistant template.
    # The cuts are, however, not made in the input_ids... Here the model is being given the answers too.... But the labels are set to -100 for everything until the answer begins, making the loss function ignore them...
    # Hence it implements teacher forcing when trained....
    # The trainer's dataloader (evidently) calls its DataCollatorForCompletionOnlyLM.torch_call method either during dataset building or during fetching

def process_ds(ds):
    def aux(example):
        x = example["conversations"]
        assert len(x) == 2
        #return {"prompt": x[0], "completion": x[1]}
        return {'text': f"{instruction_template} {x[0]}{response_template} {x[1]}"}
    
    return ds.filter(
        lambda x: x["source"] != "multi_turn"    # We only want the instruction tuning bit
    ).map(
        aux, remove_columns=["conversations", "source"]
    )

ds = load_dataset(dataset_name, 'plain_text')
train_ds = process_ds(ds["train"])
if test: train_ds = Dataset.from_dict(train_ds["train"][0:1])  # Single batch for debugging purposes
eval_ds = ds["test"]["conversations"]
len_eval = len(eval_ds)
len_train_ds = len(train_ds)


# Define training args
class ExampleCallback(WandbCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]

        # Sample training examples (mostly for debugging purposes...)
        example = train_ds[np.random.randint(len_train_ds)]
        text = example["text"]
        prompt, suggested_completion = text.split(response_template)
        prompt = f"{prompt}{response_template} "
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")  # Why is this not a dictionary???
        tokenized_completion = model.generate(tokenized_prompt.cuda(), max_new_tokens=1024)
        completion = tokenizer.decode(token_ids=tokenized_completion.squeeze(0), skip_special_tokens=True)
        raw_completion = completion
        
        # Formatting to look nice
        completion = completion.split(response_template)[1]
        prompt = prompt.replace(instruction_template, '').replace(response_template, '')

        # Log to wandb
        text_table = wandb.Table(columns=["prompt", "suggested completion", "completion", "raw_completion"])
        text_table.add_data(prompt, suggested_completion, completion, raw_completion)
        wandb.log({"Training examples": text_table})
        
        # Evaluate on eval set
        eval_table = wandb.Table(columns=["prompt", "completion"])
        for prompt in eval_ds:
            prompt = prompt[0]
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_completion = model.generate(tokenized_prompt.cuda(), max_new_tokens=1024)
            completion = tokenizer.decode(token_ids=tokenized_completion.squeeze(0), skip_special_tokens=True)
            eval_table.add_date(prompt, completion)
        wandb.log({"eval": eval_table})

training_args = TrainingArguments(
    num_train_epochs=200 if test else 3,
    per_device_train_batch_size=1 if test else 16,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    logging_steps=25,
    do_train=True,
    output_dir="/work3/s184399/checkpoints/lima-llama-2-7b",
    logging_dir="/work3/s184399/logs/lima-llama-2-7b",
    overwrite_output_dir=True,
    report_to="wandb",
    run_name="llama-2-7b code test" if test else "finetune-llama-2-7b",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=collator,
    dataset_text_field="text",
    train_dataset=train_ds,
    max_seq_length=1024,
    callbacks=[ExampleCallback],
)
trainer.train()