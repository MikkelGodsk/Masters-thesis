"""
Correctly configures the models and tokenizers for the different architectures.
-------------------------------------------------------------------------------

The reason for this module is that we don't want to deal with correctly setting up the models in the main script. 
Each model has its own quirks and needs for being set up correctly. This module takes care of that.

To introduce a new model, you need to do the following:
    
        1. Create a new subclass of `Factory` and override the methods as needed. Take inspiration from the existing subclasses.
        2. Register the new subclass in the `factory_dict` dictionary.

To use the module, you can do the following::
    
            factory = Factory.spawn_factory('LLaMA')
            factory.setup_peft(lora_config, quantization_config)
            factory.setup_model(**model_args)
            factory.setup_tokenizer(**tokenizer_args)
            factory.setup_mebp(torch.optim.SGD, torch.optim.lr_scheduler.CosineAnnealingLR)
            model = factory.spawn_model()
            tokenizer = factory.spawn_tokenizer()

Note that none of the `setup_...` calls are mandatory. If you don't call them, the default values will be used (i.e. no PEFT, no MEbP, default model settings, default tokenizer settings etc.).
Here the idea is that the model should be correctly configured when just doing `factory.spawn_model()`, i.e. all the necessary model-specific configurations are done automatically, abstracted entirely away from the user.
"""

from typing import Optional, Dict, Any, Type, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, BitsAndBytesConfig
from transformers import PretrainedConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig

from transformers import LlamaConfig, OPTConfig

from backprop_trick import MotherOptimizer


factory_dict = {}  # Maps model architectures to the appropriate subclass


class Factory:
    """Base setup methods that might be common across different models or left empty to be defined in subclasses.
    The factories have the following purposes:
        
        - Correctly set up the model (with PEFT etc. if needed).
        - Correctly set up the tokenizer.
        - Add `.instruction_template` and `.completion_template` to the tokenizer as attributes. 
        - Spawining a fresh instance of the model and the tokenizer.

    For each subclass, add it to factory_dict as such::

        factory_dict['LLaMA'] = LlamaFactory

    Example usage::

        factory = FactoryBase.spawn_factory('LLaMA', cache_dir, pretrained_config)
        factory.setup_peft(lora_config, quantization_config)
        factory.setup_model(**model_args)
        factory.setup_tokenizer(**tokenizer_args)
        factory.setup_mebp(torch.optim.SGD, torch.optim.lr_scheduler.CosineAnnealingLR)
        model = factory.spawn_model()
        tokenizer = factory.spawn_tokenizer()
    """
    model_name: str
    cache_dir: str
    pretrained_config: Optional[PretrainedConfig] = None
    lora_config: Optional[LoraConfig] = None
    quantization_config: Optional[BitsAndBytesConfig] = None
    device_map: str
    model_kwargs: Optional[Dict]
    tokenizer_kwargs: Optional[Dict]
    use_mebp: bool = False
    optimizer_cls: Type[torch.optim.Optimizer] = None
    lr_scheduler_cls: Type[torch.optim.lr_scheduler._LRScheduler] = None
    optimizer_kwargs: Optional[Dict[str, Any]]
    lr_scheduler_kwargs: Optional[Dict[str, Any]]
    initial_lr: float

    def __init__(self, cache_dir: str, pretrained_config: Optional[PretrainedConfig] = None):
        self.cache_dir = cache_dir
        self.pretrained_config = pretrained_config
        self._model_was_set_up = False
        self._tokenizer_was_set_up = False

    def setup_peft(self, lora_config: LoraConfig = None, quantization_config: BitsAndBytesConfig = None) -> None:
        self.lora_config = lora_config                      # For reference:  https://huggingface.co/docs/peft/developer_guides/lora
        self.quantization_config = quantization_config      # For reference (also explains LoRA): https://huggingface.co/docs/peft/developer_guides/quantization
        if lora_config is None and quantization_config is None:
            # Set default parameters
            self.lora_config = LoraConfig(
                r=16,
                lora_alpha=8,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                #load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )

    def _auto_setup(self) -> None:
        """If the model or tokenizer is not set up, then we set them up with the default values.
        """
        if not self._model_was_set_up:
            self.setup_model()
        if not self._tokenizer_was_set_up:
            self.setup_tokenizer()

    def setup_model(self, device_map:str="auto", model_kwargs: Optional[Dict] = {}) -> None:
        self.device_map = device_map
        self.model_kwargs = model_kwargs

    def setup_tokenizer(self, tokenizer_kwargs: Optional[Dict] = {}):
        self.tokenizer_kwargs = tokenizer_kwargs

    def setup_mebp(self, 
                   optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
                   lr_scheduler_cls: Type[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.CosineAnnealingLR,
                   optimizer_kwargs: Optional[Dict[str, Any]] = {},
                   lr_scheduler_kwargs: Optional[Dict[str, Any]] = {},
                   initial_lr: float = 1e-3,
                   ) -> None:
        self.use_mebp = True
        self.optimizer_cls = optimizer_cls
        self.lr_scheduler_cls = lr_scheduler_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.initial_lr = initial_lr
        if 'T_max' not in lr_scheduler_kwargs.keys() and lr_scheduler_cls == torch.optim.lr_scheduler.CosineAnnealingLR:
            self.lr_scheduler_kwargs['T_max'] = 10

    def _wrap_mebp(self, model: torch.nn.Module) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer, lr_scheduler = None, None
        if self.use_mebp:
            optimizer = MotherOptimizer(
                model.parameters(), 
                self.optimizer_cls, 
                lr=self.initial_lr,
                **self.optimizer_kwargs
            )
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = self.initial_lr  # Needed for lr scheduler
            lr_scheduler = self.lr_scheduler_cls(optimizer, **self.lr_scheduler_kwargs)
        return model, optimizer, lr_scheduler

    def spawn_model(self) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        self._auto_setup()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            quantization_config=self.quantization_config,
            config = self.pretrained_config,
        )
        if self.quantization_config is not None:
            model = prepare_model_for_kbit_training(model)
        if self.lora_config is not None:
            model = get_peft_model(model, self.lora_config)
        return self._wrap_mebp(model)

    def spawn_tokenizer(self) -> AutoTokenizer:
        self._auto_setup()
        return AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir,
            config = self.pretrained_config,
        )

    @classmethod
    def spawn_factory(cls, architecture_type: str, *args, **kwargs) -> "Factory":
        # Dynamically select and instantiate the correct factory
        architecture_type = architecture_type.lower()
        if architecture_type in factory_dict:
            return factory_dict[architecture_type](*args, **kwargs)
        else:
            raise ValueError(f"Factory for architecture type '{architecture_type}' not found")


class Llama2Factory(Factory):
    """Sets up the LLaMa2 models and tokenizers. Refer to: https://huggingface.co/docs/transformers/main/model_doc/llama2
    """
    def __init__(self, *args, version: str = '7b', pretrained_config: Optional[LlamaConfig] = None, **kwargs):
        #pad_token_id = 32000
        #if pretrained_config is not None:
        #    pretrained_config.pad_token_id = pad_token_id   # Unfortunately this does not work... The model simply cannot figure it out, it seems...
        #else:
        #    pretrained_config = LlamaConfig(pad_token_id=pad_token_id)
        self.model_name = f'meta-llama/Llama-2-{version}-hf'
        super().__init__(*args, pretrained_config=pretrained_config, **kwargs)

    def spawn_model(self):
        self._auto_setup()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            quantization_config=self.quantization_config,
            config = self.pretrained_config,
        )
        #model.config.pad_token_id = model.config.eos_token_id

        # To be a bit on the safe side, I set up this before PEFT, although doing so after might also be okay (just one more .model call to do...)
        model.config.pad_token_id = 32000
        model.resize_token_embeddings(32001)   # https://www.reddit.com/r/LocalLLaMA/comments/15hz7gl/comment/jw4vrdx/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
        model.model.padding_idx = 32000
        model.model.embed_tokens.padding_idx = 32000

        if self.quantization_config is not None:
            model = prepare_model_for_kbit_training(model)
        if self.lora_config is not None:
            model = get_peft_model(model, self.lora_config)
        return self._wrap_mebp(model)

    def spawn_tokenizer(self):
        tokenizer = super().spawn_tokenizer()
        tokenizer.add_special_tokens({"pad_token":"<pad>"})   # Unfortunately this does not work... The model simply cannot figure it out, it seems...
        #tokenizer.pad_token = tokenizer.eos_token   # According to this reddit post, this could lead to the text generation not terminating... https://www.reddit.com/r/LocalLLaMA/comments/15hz7gl/my_finetuning_based_on_llama27bchathf_model/
                                                    # See this github issue: https://github.com/huggingface/transformers/issues/22794#issuecomment-1598977285
        tokenizer.instruction_template = '[INST]'
        tokenizer.response_template = '[/INST]'
        return tokenizer

factory_dict['llama2'] = Llama2Factory
factory_dict["meta-llama/llama-2-7b-hf"] = Llama2Factory


class OptFactory(Factory):
    """Sets up the OPT models and tokenizers. Refer to: https://huggingface.co/docs/transformers/main/model_doc/opt
    """
    # Implementations specific to OptFactory
    def __init__(self, *args, version: str = '125m', **kwargs):
        from chat_templates.new_opt_chat_template import new_opt_chat_template
        super().__init__(*args, **kwargs)
        self.model_name = f'facebook/opt-{version}'
        self.new_opt_chat_template = new_opt_chat_template

    def spawn_tokenizer(self):
        tokenizer = super().spawn_tokenizer()
        tokenizer.add_special_tokens({'sep_token': '<SEP>'})
        tokenizer.chat_template = self.new_opt_chat_template
        tokenizer.instruction_template = None
        tokenizer.response_template = tokenizer.sep_token
        return tokenizer
    
factory_dict['opt'] = OptFactory
factory_dict["facebook/opt-125m"] = OptFactory


