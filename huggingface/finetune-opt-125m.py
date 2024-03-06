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

model_name = "facebook/opt-125m"   # Small enough to do fairly fast generation on CPU (0.8s per small sentence) => Might be good for development, debugging and testing...
dataset_name = "GAIR/lima"
experiment_name = f"{dataset_name.split('/')[1]}-{model_name.split('/')[1]}" + ("-test" if test else "")

OUTPUT_DIR = os.getenv("OUTPUT_DIR_MSC")
assert OUTPUT_DIR is not None
print(f"Outputting to: {OUTPUT_DIR}")

os.environ['WANDB_PROJECT'] = 'lima_ft'
os.environ['WANDB_DIR'] = os.path.join(OUTPUT_DIR, 'logs', 'wandb')   #'/work3/s184399/logs/wandb'
os.environ['WANDB_CACHE_DIR'] = os.path.join(OUTPUT_DIR, 'cache_dir', 'wandb')  # '/work3/s184399/cache_dir/wandb'
cache_dir = os.path.join(OUTPUT_DIR, 'cache_dir', 'huggingface') # "/work3/s184399/cache_dir/huggingface"
checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints', experiment_name) # "/work3/s184399/checkpoints/lima-opt-125m"
logging_dir = os.path.join(OUTPUT_DIR, 'logs', experiment_name)  # "/work3/s184399/logs/lima-opt-125m"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

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
        return {'text': f"{instruction_template}\n{x[0]}\n\n{response_template}\n{x[1]}"}   # This article might have an opinion on this: https://www.philschmid.de/instruction-tune-llama-2
    
    return ds.filter(
        lambda x: x["source"] != "multi_turn"    # We only want the instruction tuning bit
    ).map(
        aux, remove_columns=["conversations", "source"]
    )

ds = load_dataset(dataset_name, 'plain_text', cache_dir=cache_dir)
train_ds = process_ds(ds["train"]).train_test_split(train_size=1 if test else 0.8)   # Test size can be given as fraction or absolute number.... Also useful for testing.
train_split, val_split = train_ds["train"], train_ds["test"]
eval_ds = ds["test"]["conversations"]
len_eval = len(eval_ds)
len_train_split = len(train_split)

#code_test_ds = Dataset.from_dict(train_ds["train"][0:1])  # Single batch
#if test: train_split = code_test_ds

# Define training args
class ExampleCallback(WandbCallback):   # Source: https://docs.wandb.ai/guides/integrations/huggingface#custom-logging-log-and-view-evaluation-samples-during-training
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_table = wandb.Table(columns=["prompt", "suggested completion", "completion", "raw_completion"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        # print(kwargs.keys())   # Kwargs contains keys: ['model', 'tokenizer', 'optimizer', 'lr_scheduler', 'train_dataloader', 'eval_dataloader']

        #prompt = eval_ds[np.random.randint(len_eval)][0]
        #train_dataloader = kwargs["train_dataloader"]   # The train_dataloader always picks the same order...
        example = train_split[np.random.randint(len_train_split)]   # Maybe I should use the dataloader and DataCollator instead???
        text = example["text"]
        prompt, suggested_completion = text.split(response_template)
        prompt = f"{prompt}{response_template} "
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")  # Why is this not a dictionary???
        tokenized_completion = model.generate(tokenized_prompt.cuda(), max_new_tokens=256 if test else 1024)
        completion = tokenizer.decode(token_ids=tokenized_completion.squeeze(0), skip_special_tokens=True)
        raw_completion = completion
        
        # Formatting to look nice
        completion = completion.split(response_template)[1]
        prompt = prompt.replace(instruction_template, '').replace(response_template, '')

        # Log to wandb
        self.text_table.add_data(prompt, suggested_completion, completion, raw_completion)
        wandb.log({"examples": self.text_table})

training_args = TrainingArguments(
    num_train_epochs=200 if test else 15,  # 15 epochs in the paper...
    per_device_train_batch_size=1 if test else 16,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    logging_steps=25,
    do_train=True,
    output_dir=checkpoint_dir,
    logging_dir=logging_dir,
    overwrite_output_dir=True,
    report_to="wandb",
    run_name="opt-125m code test" if test else "finetune-opt-125m",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=collator,
    dataset_text_field="text",
    train_dataset=train_split, #train_ds,
    eval_dataset=val_split,
    max_seq_length=256 if test else 2048,   # It seems like the UserWarning with not being able to find the instruction and response templates were fixed when doubling max_seq_length and changing the template slightly....
    callbacks=[ExampleCallback],
)
trainer.train()

# Set max_seq_length=2048 both in trainer and in callback (in callback maybe just 1024...)