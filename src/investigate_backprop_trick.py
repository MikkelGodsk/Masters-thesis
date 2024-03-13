from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig  # https://huggingface.co/docs/peft/developer_guides/quantization  https://discuss.huggingface.co/t/typeerror-llamaforcausallm-init-got-an-unexpected-keyword-argument-load-in-4bit/41245
from trl import SFTTrainer
from transformers.integrations import WandbCallback
from trl.trainer import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
import torch
import os
from time import time

from profiler_callbacks import TorchProfilerCallback, MemoryHistoryCallback, WandBProfilerCallback, WandBTimerCallback
from lima_utils import *


model_name:str="facebook/opt-125m"
dataset_name:str="GAIR/lima"

OUTPUT_DIR = os.getenv("OUTPUT_DIR_MSC")
experiment_name = "profiling_backprop_trick"

os.environ['WANDB_PROJECT'] = 'lima_ft_'+model_name.split('/')[1]
os.environ['WANDB_DIR'] = os.path.join(OUTPUT_DIR, 'logs')                      # Becomes 'OUTPUT_DIR/logs/wandb'
os.environ['WANDB_CACHE_DIR'] = os.path.join(OUTPUT_DIR, 'cache_dir', 'wandb')  # Becomes 'OUTPUT_DIR/cache_dir/wandb'
cache_dir = os.path.join(OUTPUT_DIR, 'cache_dir', 'huggingface')                # Becomes "OUTPUT_DIR/cache_dir/huggingface"
checkpoint_dir = os.path.join(OUTPUT_DIR, 'checkpoints', experiment_name)       # Becomes "OUTPUT_DIR/checkpoints/<experiment_name>"
logging_dir = os.path.join(OUTPUT_DIR, 'logs', experiment_name)                 # Becomes "OUTPUT_DIR/logs/<experiment_name>"

if not os.path.isdir(logging_dir): os.mkdir(logging_dir)

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    cache_dir=cache_dir, 
    quantization_config=None,
    device_map="auto",  # Not sure if the device map enables FSDP by default...
)

ds = load_dataset(dataset_name, 'plain_text', cache_dir=cache_dir)
template_formatter = TemplateFormatter(ds, tokenizer)
train_ds, test_ds = template_formatter.train_ds, template_formatter.test_ds
if True: 
    train_ds = Dataset.from_dict(train_ds[0:5])

if True:
    from backprop_trick import MotherOptimizer
    optimizer = MotherOptimizer(model.parameters(), torch.optim.SGD, lr=1e-3)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = 1e-3
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


x = train_ds[0]
x_tokenized = tokenizer.encode(x['text'], return_tensors="pt")
collated_x = template_formatter.collator.torch_call([x_tokenized[0]])
collated_x = {k:v.cuda() for k, v in collated_x.items()}

torch.cuda.empty_cache()
torch.cuda.memory._record_memory_history(enabled='all')   # Maybe running both profilers at once will introduce a lot of overhead?
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
) as prof:
    out = model.forward(**collated_x)
    loss = out.loss
    loss.backward(retain_graph=True)

import pickle
out_path = logging_dir
s = torch.cuda.memory._snapshot()
prof.export_chrome_trace(os.path.join(out_path, f"stack_trace_decorator_{time()}.json"))
with open(os.path.join(out_path, f"cuda_memory_trace_decorator_{time()}.pickle"), "wb") as f:
    pickle.dump(s, f)
torch.cuda.memory._record_memory_history(enabled=None)

for param in model.parameters():
    if param.grad is not None:
        print(param)

#print(model.model.embed_tokens.grad)

if False:
    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        logging_steps=5,
        do_train=True, 
        output_dir=checkpoint_dir,
        logging_dir=logging_dir,
        overwrite_output_dir=True,
        save_strategy="epoch",
        )
    callbacks = []
    callbacks.append(TorchProfilerCallback(logging_dir, profile_epochs=[], profile_n_steps=5, repeat_every_n_steps=5))
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=template_formatter.collator,
        dataset_text_field="text",
        train_dataset=train_ds,
        max_seq_length=128,
        callbacks=callbacks,
        optimizers=(optimizer, lr_scheduler) if True else (None, None),
    )
    trainer.train()