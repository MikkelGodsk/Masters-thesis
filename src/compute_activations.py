import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List


dtype = np.float32


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


def main(
  dataset:str='common_claim_true_false',
  model:str='meta-llama/Llama-2-7b-hf',
):
  """Computes the internal activations for each instance in the given dataset.
  The output file is stored as `OUTPUT_DIR_MSC/{model}-{dataset}.npz`, where `OUTPUT_DIR_MSC` is an environment variable.
  In the file, the activations are stored with the key `activations` and the labels are stored with the key `labels`.
  
  Args:
      dataset (str, optional): The dataset to use. Pick one of the "true_false" datasets from the folder `src/geometry-of-truth/datasets` (NOTE: See the readme for how to install this correctly. It just needs to be cloned). Defaults to 'common_claim_true_false.npz'.
      model (str, optional): The model name. Defaults to 'meta-llama/Llama-2-7b-hf'.
  """
  dataset_dir = 'geometry-of-truth/datasets'
  OUTPUT_DIR = os.getenv("OUTPUT_DIR_MSC")
  cache_dir = os.path.join(
      OUTPUT_DIR, "cache_dir", "huggingface"
  )
  df = pd.read_csv(os.path.join(dataset_dir, dataset+'.csv'))
  model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", cache_dir=cache_dir)
  tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, cache_dir=cache_dir)
  activations = compute_activations(df['statement'], model, tokenizer)
  np.savez(os.path.join(OUTPUT_DIR, dataset+'.npz'), activations=activations, labels=df['label'])


if __name__ == '__main__':
  from jsonargparse import CLI
  CLI(main)