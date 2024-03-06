import wandb
from transformers.integrations import WandbCallback
from tqdm import tqdm
import numpy as np


instruction_template = "### Human:"
response_template = "### Assistant:"


def to_template(prompt, response=""):
    #return f"{instruction_template} {prompt}{response_template} {response}"
    return f"{instruction_template}\n{prompt}\n\n{response_template}\n{response}"   # This article might have an opinion on this: https://www.philschmid.de/instruction-tune-llama-2

def format_prompt(prompt):
    return prompt.replace(instruction_template, '').replace(response_template, '')

def format_completion(completion):
    return completion.split(response_template)[1]

def process_ds(ds):
    def aux(example):
        x = example["conversations"]
        assert len(x) == 2
        #return {"prompt": x[0], "completion": x[1]}
        return {'text': to_template(*x)} #f"{instruction_template} {x[0]}{response_template} {x[1]}"}
    
    return ds.filter(
        lambda x: x["source"] != "multi_turn"    # We only want the instruction tuning bit
    ).map(
        aux, remove_columns=["conversations", "source"]
    )

# Define training args
class ExampleCallback(WandbCallback):   # Source: https://docs.wandb.ai/guides/integrations/huggingface#custom-logging-log-and-view-evaluation-samples-during-training
    def __init__(self, train_ds, eval_ds, *args, max_new_tokens=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.max_new_tokens = max_new_tokens
        self.len_train_ds = len(train_ds)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Kwargs contains keys: ['model', 'tokenizer', 'optimizer', 'lr_scheduler', 'train_dataloader', 'eval_dataloader']
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]

        # Sample training examples (mostly for debugging purposes...)
        example = self.train_ds[np.random.randint(self.len_train_ds)]
        text = example["text"]
        prompt, suggested_completion = text.split(response_template)
        prompt = f"{prompt}{response_template} "
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")  # Why is this not a dictionary???
        tokenized_completion = model.generate(tokenized_prompt.cuda(), max_new_tokens=self.max_new_tokens)
        completion = tokenizer.decode(token_ids=tokenized_completion.squeeze(0), skip_special_tokens=True)
        raw_completion = completion
        
        # Formatting to look nice
        prompt = format_prompt(prompt)
        completion = format_completion(completion)

        # Log to wandb
        text_table = wandb.Table(columns=["prompt", "suggested completion", "completion", "raw_completion"])
        text_table.add_data(prompt, suggested_completion, completion, raw_completion)
        wandb.log({"Training examples": text_table})
        
        # Evaluate on eval set
        eval_table = wandb.Table(columns=["prompt", "completion"])
        for prompt in tqdm(self.eval_ds[0:10], desc="Evaluating model on eval set"):
            prompt = to_template(prompt[0])
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_completion = model.generate(tokenized_prompt.cuda(), max_new_tokens=self.max_new_tokens)
            completion = tokenizer.decode(token_ids=tokenized_completion.squeeze(0), skip_special_tokens=True)
            prompt = format_prompt(prompt)
            completion = format_completion(completion)
            eval_table.add_data(prompt, completion)
        wandb.log({"eval": eval_table})


