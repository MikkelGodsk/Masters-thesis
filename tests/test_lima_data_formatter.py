import pytest
import warnings
import re

from datasets import load_dataset

from src.lima_utils import TemplateFormatter
from src.model_factories import Factory
from tests.cache_dir import cache_dir


def fail_on_warning(func):
    def func_wrapper(*args, **kwargs):
        warnings.filterwarnings("error", category=UserWarning)
        func(*args, **kwargs)
        warnings.filterwarnings("ignore", category=UserWarning)

    return func_wrapper


def test_fail_on_warning():
    @fail_on_warning
    def func():
        warnings.warn("This is a warning", UserWarning)

    with pytest.raises(UserWarning):
        func()


llama_instruction_template = r"\[INST\]\s{{0,1}}{:s}\s{{0,1}}\[/INST\]"  # This pattern was suggested by ChatGPT
llama_response_template = r"\s{{0,1}}{:s}\s{{0,1}}"
opt_instruction_template = r"{:s}\s{{0,1}}"#<SEP>"
opt_response_template = r"\s{{0,1}}{:s}\s{{0,1}}"


def check_tokenized_instructions_and_completions(tokenizer, padded_instructions, padded_tokenized_completions):
    # Check that the instructions start with a BOS token, and that the completions end with an EOS token.
    for instruction in padded_instructions["input_ids"]:
        for token in instruction:
            if token == tokenizer.pad_token_id:         # Skip padding tokens
                continue
            elif token == tokenizer.bos_token_id:       # Stop when we find the BOS token
                break
            else:                                       # If we find anything else, raise an error
                raise AssertionError("Instructions do not start with BOS token")
        #if hasattr(tokenizer, "sep_token_id"):   # Doesn't matter, it is trained with the SEP token...
        #    assert tokenizer.sep_token_id not in instruction, "Instructions contain SEP token"
    for completion in padded_tokenized_completions["input_ids"]:
        if len(completion) > 0:
            assert completion[-1] == tokenizer.eos_token_id, "Completions do not end with EOS token"


@fail_on_warning
def test_data_formatter_exact_match():
    global \
        correct_llama_train_instruction, \
        correct_llama_train_response, \
        correct_llama_test_instruction
    global \
        correct_opt_train_instruction, \
        correct_opt_train_response, \
        correct_opt_test_instruction
    dataset_name: str = "GAIR/lima"
    opt_factory = Factory.spawn_factory("opt", cache_dir)
    tokenizer_opt = opt_factory.spawn_tokenizer()
    llama_factory = Factory.spawn_factory("llama2", cache_dir)
    tokenizer_llama = llama_factory.spawn_tokenizer()
    ds = load_dataset(dataset_name, "plain_text")

    # NOTE: The instruction templates can be read from `tokenizer.default_chat_template`
    llama_formatter = TemplateFormatter(ds, tokenizer_llama)
    opt_formatter = TemplateFormatter(ds, tokenizer_opt)

    llama_train_ds = llama_formatter.train_ds[0:5]['text']
    llama_test_ds = llama_formatter.test_ds[0:5]['text']
    opt_train_ds = opt_formatter.train_ds[0:5]['text']
    opt_test_ds = opt_formatter.test_ds[0:5]['text']

    llama_padded_instructions, llama_padded_tokenized_completions = llama_formatter.get_instruction_and_response(llama_train_ds)
    opt_padded_instructions, opt_padded_tokenized_completions = opt_formatter.get_instruction_and_response(opt_train_ds)
    
    # Test the tokenized text
    check_tokenized_instructions_and_completions(tokenizer_llama, llama_padded_instructions, llama_padded_tokenized_completions)
    check_tokenized_instructions_and_completions(tokenizer_opt, opt_padded_instructions, opt_padded_tokenized_completions)

    # Test the logged text
    llama_prompts = tokenizer_llama.batch_decode(
        sequences=llama_padded_instructions['input_ids'], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    llama_suggested_completions = tokenizer_llama.batch_decode(
        sequences=llama_padded_tokenized_completions['input_ids'], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    opt_prompts = tokenizer_opt.batch_decode(
        sequences=opt_padded_instructions['input_ids'], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    opt_suggested_completions = tokenizer_opt.batch_decode(
        sequences=opt_padded_tokenized_completions['input_ids'], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    for llama_p, llama_c, (correct_p, correct_c) in zip(llama_prompts, llama_suggested_completions, ds['train']['conversations'][0:5]):
        assert re.search(llama_instruction_template.format(re.escape(correct_p)), llama_p) is not None
        assert re.search(llama_response_template.format(re.escape(correct_c)), llama_c) is not None

    for opt_p, opt_c, (correct_p, correct_c) in zip(opt_prompts, opt_suggested_completions, ds['train']['conversations'][0:5]):
        assert re.search(opt_instruction_template.format(re.escape(correct_p)), opt_p) is not None
        assert re.search(opt_response_template.format(re.escape(correct_c)), opt_c) is not None
    
    llama_padded_instructions_test, llama_padded_tokenized_completions_test = llama_formatter.get_instruction_and_response(llama_test_ds)
    opt_padded_instructions_test, opt_padded_tokenized_completions_test = opt_formatter.get_instruction_and_response(opt_test_ds)

    # Test the tokenized text
    check_tokenized_instructions_and_completions(tokenizer_llama, llama_padded_instructions_test, llama_padded_tokenized_completions_test)
    check_tokenized_instructions_and_completions(tokenizer_opt, opt_padded_instructions_test, opt_padded_tokenized_completions_test)

    # Test the logged text
    llama_prompts = tokenizer_llama.batch_decode(
        sequences=llama_padded_instructions_test['input_ids'], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    llama_suggested_completions = tokenizer_llama.batch_decode(
        sequences=llama_padded_tokenized_completions_test['input_ids'], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    opt_prompts = tokenizer_opt.batch_decode(
        sequences=opt_padded_instructions_test['input_ids'], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    opt_suggested_completions = tokenizer_opt.batch_decode(
        sequences=opt_padded_tokenized_completions_test['input_ids'], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    for llama_p, llama_c, (correct_p, ) in zip(llama_prompts, llama_suggested_completions, ds['test']['conversations'][0:5]):
        assert re.search(llama_instruction_template.format(re.escape(correct_p)), llama_p) is not None
        assert llama_c == ""

    for opt_p, opt_c, (correct_p, ) in zip(opt_prompts, opt_suggested_completions, ds['test']['conversations'][0:5]):
        assert re.search(opt_instruction_template.format(re.escape(correct_p)), opt_p) is not None
        assert opt_c == ""


@fail_on_warning
def test_data_formatter_all_examples_no_warning():
    # Stress test: Can we use it without failing even once?
    dataset_name: str = "GAIR/lima"

    opt_factory = Factory.spawn_factory("facebook/opt-125m", cache_dir)
    tokenizer_opt = opt_factory.spawn_tokenizer()
    llama_factory = Factory.spawn_factory("meta-llama/Llama-2-7b-hf", cache_dir)
    tokenizer_llama = llama_factory.spawn_tokenizer()
    ds = load_dataset(dataset_name, "plain_text")

    # NOTE: The instruction templates can be read from `tokenizer.default_chat_template`
    llama_formatter = TemplateFormatter(ds, tokenizer_llama)
    opt_formatter = TemplateFormatter(ds, tokenizer_opt)

    for example in llama_formatter.train_ds["text"]:
        x_tokenized = llama_formatter.tokenizer(example, return_tensors="pt")
        llama_formatter.collator.torch_call([x_tokenized["input_ids"][0]])

    for example in opt_formatter.train_ds["text"]:
        x_tokenized = opt_formatter.tokenizer(example, return_tensors="pt")
        opt_formatter.collator.torch_call([x_tokenized["input_ids"][0]])

    for example in llama_formatter.test_ds["text"]:
        x_tokenized = llama_formatter.tokenizer(example, return_tensors="pt")
        llama_formatter.collator.torch_call([x_tokenized["input_ids"][0]])

    for example in opt_formatter.test_ds["text"]:
        x_tokenized = opt_formatter.tokenizer(example, return_tensors="pt")
        opt_formatter.collator.torch_call([x_tokenized["input_ids"][0]])
