import pytest

import os
from transformers import AutoTokenizer
from datasets import load_dataset
import torch

from model_factories import Factory
from tests.cache_dir import cache_dir




class MockWandbTable:
    def __init__(self):
        self.logs = []

    def add_data(self, *x):
        self.logs.append(x)


class MockWandb:
    def __init__(self):
        self.logs = []

    def Table(self, *args, **kwargs):
        return MockWandbTable()

    def log(self, x):
        self.logs.append(x)


class MockModel:
    def __init__(self):
        pass

    def generate(self, *args, **kwargs):
        return torch.Tensor([4123,4363,2341,7654,1923,15,32534,63464,346,2])


def test_wandb_example_callback():
    import lima_utils
    lima_utils.wandb = MockWandb()

    model_name = "meta-llama/Llama-2-7b-hf"
    dataset_name = "GAIR/lima"

    model = MockModel()
    tokenizer = Factory.spawn_factory(model_name, cache_dir).spawn_tokenizer()
    ds = load_dataset(dataset_name, 'plain_text', cache_dir=cache_dir)
    template_formatter = lima_utils.TemplateFormatter(ds, tokenizer)
    callback = lima_utils.ExampleCallback(template_formatter)

    assert len(lima_utils.wandb.logs) == 0

    callback.on_log(args="", state="", control="", model=model, tokenizer=tokenizer)  # args, state, control, logs=None, **kwargs
    prev_prompt = ""
    for prompt, suggested_competion, _ in lima_utils.wandb.logs[0]["Training examples 0"].logs:
        assert prompt != prev_prompt
        assert f'{tokenizer.bos_token}{tokenizer.bos_token}' not in prompt  # No double bos tokens
        for special_token in tokenizer.all_special_tokens:
            if special_token != tokenizer.bos_token:
                assert special_token not in prompt
            if special_token != tokenizer.eos_token:
                assert special_token not in suggested_competion
        prev_prompt = prompt

    for prompt, suggested_competion in lima_utils.wandb.logs[1]["Eval examples 0"].logs:
        assert prompt != prev_prompt
        assert f'{tokenizer.bos_token}{tokenizer.bos_token}' not in prompt  # No double bos tokens
        for special_token in tokenizer.all_special_tokens:
            if special_token != tokenizer.bos_token:
                assert special_token not in prompt
            if special_token != tokenizer.eos_token:
                assert special_token not in suggested_competion
        prev_prompt = prompt

