import shutil, os

from transformers import LlamaForCausalLM, LlamaTokenizerFast, OPTForCausalLM, GPT2TokenizerFast

from src.model_factories import Factory
from tests.cache_dir import cache_dir


def test_opt_factory():
    factory = Factory.spawn_factory('OPT', cache_dir)
    factory.setup_model()
    factory.setup_tokenizer()
    factory.setup_mebp()
    model, optimizer, lr_scheduler = factory.spawn_model()
    tokenizer = factory.spawn_tokenizer()
    assert isinstance(model, OPTForCausalLM)
    param = next(iter(model.parameters()))
    assert len(param._post_accumulate_grad_hooks) > 0
    
    assert isinstance(tokenizer, GPT2TokenizerFast)
    assert tokenizer.sep_token == '<SEP>'
    from chat_templates.new_opt_chat_template import new_opt_chat_template
    assert tokenizer.chat_template == new_opt_chat_template
    assert tokenizer.instruction_template is None
    assert tokenizer.response_template == tokenizer.sep_token


def test_llama_factory():
    factory = Factory.spawn_factory('LLAMA2', cache_dir)
    factory.setup_model()
    factory.setup_tokenizer()
    factory.setup_mebp()
    model, optimizer, lr_scheduler = factory.spawn_model()
    tokenizer = factory.spawn_tokenizer()
    assert isinstance(model, LlamaForCausalLM)
    param = next(iter(model.parameters()))
    assert len(param._post_accumulate_grad_hooks) > 0
    #assert model.config.pad_token_id == 31999
    #assert model.model.embed_tokens.padding_idx == 31999
    # My computer is too small to test any training, unfortunately... But according to the PyTorch doc, nn.Embedding's padding vector should not change (gotta trust the docs, right?)

    assert isinstance(tokenizer, LlamaTokenizerFast)
    assert tokenizer.pad_token == tokenizer.eos_token #'<pad>'
    assert tokenizer.instruction_template == '[INST]'
    assert tokenizer.response_template == '[/INST]'