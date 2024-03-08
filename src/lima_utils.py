import wandb
from transformers import TrainerCallback
from tqdm import tqdm
import numpy as np
from functools import partial
from trl.trainer import DataCollatorForCompletionOnlyLM


instruction_template = "### Human:"
response_template = "### Assistant:"

def filter_example(x):
    return x["source"] != "multi_turn"

def process_example(x, tokenizer):
    x = x['conversations']
    assert len(x) in [1,2], "The multi-turn format is not supported"
    x_out = [{'role': 'user', 'content': x[0]}]   # Instruction
    if len(x) == 2:
        x_out.append({'role': 'assistant', 'content': x[1]})   # Response
    return {'text': tokenizer.apply_chat_template(x_out, tokenize=False, add_generation_prompt=True)}  # NOTE: Calling `apply_chat_template` on the instance again is no bueno


class TemplateFormatter:   # If more datasets enter the game, we should use inheritance...
    def __init__(self, ds, tokenizer):
        self.ds = ds
        self.tokenizer = tokenizer

        if tokenizer.name_or_path == "meta-llama/Llama-2-7b-hf":
            instruction_template = '[INST]'
            response_template = '[/INST]'
        elif tokenizer.name_or_path == "facebook/opt-125m":
            # With opt, we need to change the chat template, because it's pretty bad...
            from chat_templates.new_opt_chat_template import new_opt_chat_template
            tokenizer.add_special_tokens({'sep_token': '<SEP>'})
            tokenizer.chat_template = new_opt_chat_template
            instruction_template = None
            response_template = tokenizer.sep_token

        self.instruction_template = instruction_template
        self.response_template = response_template

        self.map_kwargs = {'remove_columns': ["conversations", "source"]}
        self.data_processor = partial(process_example, tokenizer=self.tokenizer)

        self.train_ds, self.test_ds = self.process_lima()
        self.collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, 
            mlm=False, 
            instruction_template=instruction_template, 
            response_template=response_template
        )

    def format_prompt(prompt):
        return prompt.replace(instruction_template, '').replace(response_template, '')

    def format_completion(completion):
        return completion.split(response_template)[1]

    def process_lima(self):
        train_ds = self.ds['train'].filter(filter_example).map(self.data_processor, **self.map_kwargs)
        test_ds = self.ds['test'].filter(filter_example).map(self.data_processor, **self.map_kwargs)
        return train_ds, test_ds
    
    def get_instruction_and_response(self, x):
        """
            Mostly for testing, the callback and as an example...
        """
        if isinstance(x, str):
            x = {'text': [x]}
        x_tokenized = self.tokenizer(x['text'], return_tensors='pt')
        collated_x = self.collator.torch_call([x_tokenized['input_ids'][0]])
        input_ids = collated_x['input_ids']
        labels = collated_x['labels']
        instruction = self.tokenizer.decode(input_ids[labels == self.collator.ignore_index])
        response = self.tokenizer.decode(input_ids[labels != self.collator.ignore_index])
        return instruction, response
    
    def remove_prompt_from_completion(self, tokenized_prompt, tokenized_completion):
        return tokenized_completion[len(tokenized_prompt):]


# Define training args
class ExampleCallback(TrainerCallback):   # Source: https://docs.wandb.ai/guides/integrations/huggingface#custom-logging-log-and-view-evaluation-samples-during-training
    def __init__(self, template_formatter, *args, max_new_tokens=1024, log_n_examples=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_ds = template_formatter.train_ds
        self.eval_ds = template_formatter.test_ds
        self.template_formatter = template_formatter
        self.max_new_tokens = max_new_tokens
        self.len_train_ds = len(self.train_ds)
        self.len_eval_ds = len(self.eval_ds)
        self.log_n_examples = log_n_examples

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Kwargs contains keys: ['model', 'tokenizer', 'optimizer', 'lr_scheduler', 'train_dataloader', 'eval_dataloader']
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        
        model.eval()

        # Sample training examples (mostly for debugging purposes...)
        text_table = wandb.Table(columns=["prompt", "suggested completion", "completion"])
        for example in tqdm(self.train_ds[np.random.randint(self.len_train_ds, size=self.log_n_examples)], desc="Sampling training examples"):
            prompt, suggested_completion = self.template_formatter.get_instruction_and_response(example)
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_completion = model.generate(tokenized_prompt.cuda(), max_new_tokens=self.max_new_tokens)
            print(tokenized_completion)
            tokenized_completion = self.template_formatter.remove_prompt_from_completion(tokenized_prompt, tokenized_completion)
            print(tokenized_completion)
            completion = tokenizer.decode(token_ids=tokenized_completion.squeeze(0), skip_special_tokens=False)
            text_table.add_data(prompt, suggested_completion, completion)
        wandb.log({"Training examples": text_table})
        
        # Evaluate on eval set
        eval_table = wandb.Table(columns=["prompt", "completion"])
        for prompt in tqdm(self.eval_ds[np.random.randint(self.len_eval_ds, size=self.log_n_examples)], desc="Evaluating model on eval set"):
            prompt, _ = self.template_formatter.get_instruction_and_response(prompt)
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_completion = model.generate(tokenized_prompt.cuda(), max_new_tokens=self.max_new_tokens)
            tokenized_completion = self.template_formatter.remove_prompt_from_completion(tokenized_prompt, tokenized_completion)
            completion = tokenizer.decode(token_ids=tokenized_completion.squeeze(0), skip_special_tokens=False)  # Keep in mind that the prompt is included in the completion
            eval_table.add_data(prompt, completion)
        wandb.log({"eval": eval_table})

        model.train()

