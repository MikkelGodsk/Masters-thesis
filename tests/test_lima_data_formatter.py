import pytest
import warnings

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


correct_llama_train_instruction = "<s> [INST] Can brain cells move? By movement I mean long distance migration (preferably within the brain only). [/INST]"
correct_llama_train_response = """The question is relatively broad and one should take into account that the brain not only consists of neurons, but also glial cells (supportive cells) and pre-mitotic neuronal stem cells. Furthermore, as critical fellow-scientists have indicated, developmental stage is very important, as the developing embryonic brain is very different from the adult brain.
However, after sifting through various publications, the answer to the question is actually remarkably simple: Yes, brain cells migrate.
In  the adult brain glial cells migrate in the brain (Klämbt, 2009). Glial cells are involved in a myriad of functions, but a notable example of migrating glial cells are the oligodendrocytes that migrate relative long distances to find their target axons onto which they wrap themselves to form the insulating myelin sheath (Tsai and Miller, 2002).
Neuronal stem cells migrate over long distances in response to injury (Imitola et al., 2004) and they migrate from specific stem-cell locations (e.g., hippocampus and subventricular zone) to other regions (Clarke, 2003).
Post-mitotic, but non-differentiated neurons have been shown to migrate in the adult brain in fish (Scott et al., 2012), and in mammals and non-human primates as well (Sawada et al., 2011).
Not surprisingly, glial cells, stem cells and neurons also migrate during embryonic development. Most notably, post-mitotic neurons destined to fulfill peripheral functions have to migrate over relatively long distances from the neural crest to their target locations (Neuroscience, 2nd ed, Neuronal Migration). </s>"""
correct_llama_test_instruction = "<s> [INST] I'm writing a NeurIPS paper about a new model architecture for processing and generating long texts. Here are some facts about the paper:\n* The main trick is to replace some of the attention heads with an exponential moving average, where the decay rate is learned for each head. We call this architecture ExeMA.\n* On language modeling, the perplexity difference between our model and a vanilla transformer is negligible, but that's because next-token prediction is almost always a local task, so perplexity won't be sensitive enough to detect any improvements in long-range understanding.\n* However, on the SCROLLS benchmark, our model improves by 10% over the baseline.\n* We also have a new metric for measuring coherence in generated text (CoGnaTe), where our model generates text that is 43% more coherent than the baseline.\nHelp me write the paper's introduction. [/INST]"


correct_opt_train_instruction = "</s>Can brain cells move? By movement I mean long distance migration (preferably within the brain only). <SEP>"
correct_opt_train_response = """The question is relatively broad and one should take into account that the brain not only consists of neurons, but also glial cells (supportive cells) and pre-mitotic neuronal stem cells. Furthermore, as critical fellow-scientists have indicated, developmental stage is very important, as the developing embryonic brain is very different from the adult brain.
However, after sifting through various publications, the answer to the question is actually remarkably simple: Yes, brain cells migrate.
In  the adult brain glial cells migrate in the brain (Klämbt, 2009). Glial cells are involved in a myriad of functions, but a notable example of migrating glial cells are the oligodendrocytes that migrate relative long distances to find their target axons onto which they wrap themselves to form the insulating myelin sheath (Tsai and Miller, 2002).
Neuronal stem cells migrate over long distances in response to injury (Imitola et al., 2004) and they migrate from specific stem-cell locations (e.g., hippocampus and subventricular zone) to other regions (Clarke, 2003).
Post-mitotic, but non-differentiated neurons have been shown to migrate in the adult brain in fish (Scott et al., 2012), and in mammals and non-human primates as well (Sawada et al., 2011).
Not surprisingly, glial cells, stem cells and neurons also migrate during embryonic development. Most notably, post-mitotic neurons destined to fulfill peripheral functions have to migrate over relatively long distances from the neural crest to their target locations (Neuroscience, 2nd ed, Neuronal Migration). </s>"""
correct_opt_test_instruction = "</s>I'm writing a NeurIPS paper about a new model architecture for processing and generating long texts. Here are some facts about the paper:\n* The main trick is to replace some of the attention heads with an exponential moving average, where the decay rate is learned for each head. We call this architecture ExeMA.\n* On language modeling, the perplexity difference between our model and a vanilla transformer is negligible, but that's because next-token prediction is almost always a local task, so perplexity won't be sensitive enough to detect any improvements in long-range understanding.\n* However, on the SCROLLS benchmark, our model improves by 10% over the baseline.\n* We also have a new metric for measuring coherence in generated text (CoGnaTe), where our model generates text that is 43% more coherent than the baseline.\nHelp me write the paper's introduction. <SEP>"


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
    # tokenizer_opt = AutoTokenizer.from_pretrained("facebook/opt-125m")
    # tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    opt_factory = Factory.spawn_factory("opt", cache_dir)
    tokenizer_opt = opt_factory.spawn_tokenizer()
    llama_factory = Factory.spawn_factory("llama2", cache_dir)
    tokenizer_llama = llama_factory.spawn_tokenizer()
    ds = load_dataset(dataset_name, "plain_text")

    # NOTE: The instruction templates can be read from `tokenizer.default_chat_template`
    llama_formatter = TemplateFormatter(ds, tokenizer_llama)
    opt_formatter = TemplateFormatter(ds, tokenizer_opt)

    (
        llama_train_instruction,
        llama_train_response,
        _llama_tokenized_train_instruction,
        _llama_tokenized_train_response,
    ) = llama_formatter.get_instruction_and_response(llama_formatter.train_ds[0])
    (
        opt_train_instruction,
        opt_train_response,
        _opt_tokenized_train_instruction,
        _opt_tokenized_train_response,
    ) = opt_formatter.get_instruction_and_response(opt_formatter.train_ds[0])
    assert llama_train_instruction == correct_llama_train_instruction
    assert llama_train_response == correct_llama_train_response
    assert opt_train_instruction == correct_opt_train_instruction
    assert opt_train_response == correct_opt_train_response

    llama_test_instruction, llama_test_response, llama_tokenized_test_instruction, _ = (
        llama_formatter.get_instruction_and_response(llama_formatter.test_ds[0])
    )
    opt_test_instruction, opt_test_response, opt_tokenized_test_instruction, _ = (
        opt_formatter.get_instruction_and_response(opt_formatter.test_ds[0])
    )
    assert llama_test_instruction == correct_llama_test_instruction
    assert opt_test_instruction == correct_opt_test_instruction
    assert llama_test_response == ""
    assert opt_test_response == ""


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
