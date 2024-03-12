"""
Utilitites for the LIMA dataset.
--------------------------------

The module :mod:`lima_utils` ensures that the LIMA dataset is correctly formatted when used with the TRL Trainer.
"""
import wandb
from transformers import TrainerCallback
from tqdm import tqdm
import numpy as np
from functools import partial
from trl.trainer import DataCollatorForCompletionOnlyLM
import torch


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
    """Ensures that the data has the correct format for the TRL library.
    
    To ensure compatibility with the TRL library, we need to pass the trainer the class's `.train_ds` and `.test_ds`.

    Example:
        To ensure correct formatting of the dataset, use the TemplateFormatter like this::

            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)   # Potentially with additional special tokens
            ds = load_dataset("lima", "plain_text")
            template_formatter = TemplateFormatter(ds, tokenizer)
            train_ds, test_ds = template_formatter.train_ds, template_formatter.test_ds    # These are the datasets to pass the SFTTrainer
            ...
            trainer = SFTTrainer(
                ...
                train_dataset=train_ds,
                ...
            )
        
    
    Attributes:
        ds (datasets.DatasetDict): The dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        instruction_template (str): The instruction template.
        response_template (str): The response template.
        map_kwargs (dict): The kwargs for the `map` method.
        data_processor (function): The function to process the data.
        train_ds (datasets.Dataset): The training dataset.
        test_ds (datasets.Dataset): The test dataset.
        collator (trl.trainer.DataCollatorForCompletionOnlyLM): The data collator.

    Methods:
        process_lima: Process the dataset.
        get_instruction_and_response: Get the instruction and response from a given example.
        remove_prompt_from_completion: Remove the prompt from the completion.
        correct_tokenized_prompt: Put a prompt in the correct format even after splitting...
    """
    def __init__(self, ds, tokenizer):
        self.ds = ds
        self.tokenizer = tokenizer

        self.tokenizer, self.instruction_template, self.response_template = self.prepare_tokenizer_and_templates(tokenizer)

        self.map_kwargs = {'remove_columns': ["conversations", "source"]}
        self.data_processor = partial(process_example, tokenizer=self.tokenizer)

        self.train_ds, self.test_ds = self.process_lima()
        self.collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, 
            mlm=False, 
            instruction_template=self.instruction_template, 
            response_template=self.response_template
        )

    def prepare_tokenizer_and_templates(self, tokenizer):
        """Adds special tokens to the tokenizer."""
        if tokenizer.name_or_path in ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"]:
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            instruction_template = '[INST]'
            response_template = '[/INST]'
        elif tokenizer.name_or_path == "facebook/opt-125m":
            # With opt, we need to change the chat template, because it's pretty bad...
            from chat_templates.new_opt_chat_template import new_opt_chat_template
            tokenizer.add_special_tokens({'sep_token': '<SEP>'})
            tokenizer.chat_template = new_opt_chat_template
            instruction_template = None
            response_template = tokenizer.sep_token
        return tokenizer, instruction_template, response_template

    def process_lima(self):
        train_ds = self.ds['train'].filter(filter_example).map(self.data_processor, **self.map_kwargs)
        test_ds = self.ds['test'].filter(filter_example).map(self.data_processor, **self.map_kwargs)
        return train_ds, test_ds
    
    def get_instruction_and_response(self, x):
        """Splits the input into an instruction and a response using the data collator - i.e. in the same way as the SFTTrainer does it.

        Args:
            x (str or dict): The input.

        Returns:
            tuple: The instruction and the response (both as strings).
        """
        if isinstance(x, str):
            x = {'text': [x]}
        x_tokenized = self.tokenizer(x['text'], return_tensors='pt')
        collated_x = self.collator.torch_call([x_tokenized['input_ids'][0]])
        input_ids = collated_x['input_ids']
        labels = collated_x['labels']
        tokenized_instruction = input_ids[..., labels == self.collator.ignore_index]
        tokenized_response = input_ids[..., labels != self.collator.ignore_index]
        instruction = self.tokenizer.decode(tokenized_instruction)
        response = self.tokenizer.decode(tokenized_response)
        return instruction, response, tokenized_instruction, tokenized_response
    
    def remove_prompt_from_completion(self, tokenized_prompt, tokenized_completion):
        """Takes a completion (which includes the prompt), and then removes the prompt from it.

        Args:
            tokenized_prompt (torch.Tensor): The tokenized prompt.
            tokenized_completion (torch.Tensor): The tokenized completion.
        
        Returns:
            torch.Tensor: The tokenized completion without the prompt.
        """
        skip_tokens = tokenized_prompt.shape[-1]
        return tokenized_completion[..., skip_tokens:]
    
    def clean_tokenized_prompt(self, tokenized_prompt):
        """Cleans the sep/eos token from the prompt, if it exists."""
        keep_idx = (tokenized_prompt != self.tokenizer.eos_token_id)
        if self.tokenizer.sep_token_id is not None:
            keep_idx = keep_idx & (tokenized_prompt != self.tokenizer.sep_token_id)
        return tokenized_prompt[keep_idx]


# If more datasets enter the game, put this in a "trainer_callbacks.py" file alongside the other callbacks for profiling...
class ExampleCallback(TrainerCallback):   # Source: https://docs.wandb.ai/guides/integrations/huggingface#custom-logging-log-and-view-evaluation-samples-during-training
    """A callback for the `Trainer` the model responses to randomly selected training examples and eval examples.

    Attributes:
        template_formatter (TemplateFormatter): The template formatter.
        max_seq_length (int): The maximum sequence length.
        log_n_examples (int): The number of examples to log.

    Methods:
        on_log: Logs the model responses to randomly selected training examples and eval examples.
    """
    def __init__(self, template_formatter, *args, max_seq_length=1024, log_n_examples=10, **kwargs):
        """Initializes the callback.
        
        Args:
            template_formatter (TemplateFormatter): The template formatter.
            *args: Any residual arguments we don't need in the callback.
            max_seq_length (int): The maximum sequence length.
            log_n_examples (int): The number of examples to log.
            **kwargs: Any residual keyword arguments we don't need in the callback.
        """
        super().__init__(*args, **kwargs)
        self.train_ds = template_formatter.train_ds
        self.eval_ds = template_formatter.test_ds
        self.template_formatter = template_formatter
        self.max_seq_length = max_seq_length
        self.len_train_ds = len(self.train_ds)
        self.len_eval_ds = len(self.eval_ds)
        self.log_n_examples = log_n_examples

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Logs the model responses to randomly selected training examples and eval examples.

        Args:
            args: The arguments.
            state: The state.
            control: The control.
            logs: The logs.
            **kwargs: The keyword arguments.
        """
        # Kwargs contains keys: ['model', 'tokenizer', 'optimizer', 'lr_scheduler', 'train_dataloader', 'eval_dataloader']
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        
        with torch.no_grad():
            # Sample training examples (mostly for debugging purposes...)
            text_table = wandb.Table(columns=["prompt", "suggested completion", "model's completion"])
            for example in tqdm(self.train_ds[np.random.randint(self.len_train_ds, size=self.log_n_examples)]['text'], desc="Sampling training examples"):
                prompt, suggested_completion, tokenized_instruction, _ = self.template_formatter.get_instruction_and_response(example)
                #tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
                tokenized_completion = model.generate(tokenized_instruction.unsqueeze(0).cuda(), max_length=self.max_seq_length)
                tokenized_completion = self.template_formatter.remove_prompt_from_completion(tokenized_instruction, tokenized_completion)
                completion = tokenizer.decode(token_ids=tokenized_completion.squeeze(0), skip_special_tokens=False)
                text_table.add_data(prompt, suggested_completion, completion)
            wandb.log({"Training examples": text_table})
            
            # Evaluate on eval set
            eval_table = wandb.Table(columns=["prompt", "model's completion"])
            for prompt in tqdm(self.eval_ds[np.random.randint(self.len_eval_ds, size=self.log_n_examples)]['text'], desc="Evaluating model on eval set"):
                prompt, suggested_completion, tokenized_instruction, _ = self.template_formatter.get_instruction_and_response(example)
                #tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
                tokenized_completion = model.generate(tokenized_instruction.unsqueeze(0).cuda(), max_length=self.max_seq_length)
                tokenized_completion = self.template_formatter.remove_prompt_from_completion(tokenized_instruction, tokenized_completion)
                completion = tokenizer.decode(token_ids=tokenized_completion.squeeze(0), skip_special_tokens=False)  # Keep in mind that the prompt is included in the completion
                eval_table.add_data(prompt, completion)
            wandb.log({"eval": eval_table})


