import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

dataset_dir = 'geometry-of-truth/datasets'
dtype = np.float32
model_name = 'meta-llama/Llama-2-7b-hf'

OUTPUT_DIR = os.getenv("OUTPUT_DIR_MSC")
cache_dir = os.path.join(
    OUTPUT_DIR, "cache_dir", "huggingface"
)


class Hook:
  # Inspired by https://github.com/saprmarks/geometry-of-truth/blob/main/generate_acts.py
  def __init__(self):
    self.activations = []
    self.idx = -1   # The index of the token we look at the internal state for

  def __call__(self, module, args, output, **kwargs):
    o = output[0][..., self.idx,:].detach().cpu().numpy().astype(dtype)
    self.activations.append(o)


def compute_activations(statements: List[str], model: torch.nn.Module, tokenizer) -> np.ndarray:
  """
    Returns:
    - Activations of shape [num_layers, num_samples, n_hidden_dim]
  """
  hooks = []
  handles = []
  for i, layer in enumerate(model.model.layers):
    hook = Hook()
    handle = layer.register_forward_hook(hook)
    hooks.append(hook)
    handles.append(handle)

  for statement in tqdm(statements):
    tokens = tokenizer.encode(statement, return_tensors='pt').cuda()
    _ = model(tokens)

  for handle in handles:
    handle.remove()

  activations = []
  for hook in hooks:
    activations.append(np.vstack(hook.activations))

  return np.stack(activations, axis=0)


model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)

common_claim_df = pd.read_csv(os.path.join(dataset_dir, 'common_claim_true_false.csv'))
activations = compute_activations(common_claim_df['statement'], model, tokenizer)
np.savez(os.path.join(OUTPUT_DIR, 'common_claim_true_false.npz'), activations=activations, labels=common_claim_df['label'])