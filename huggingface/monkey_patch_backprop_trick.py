import torch
from functools import partial

# NOTE: Does not work yet...
# TODO: Need to implement an LR scheduler that is setup with the model's optimizers...

"""
How to use with Hugging Face Trainer:
```
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")
model = patch_model(model, torch.optim.AdamW, optimizer_defaults={"lr": 1e-3})
...
trainer = SFTTrainer(
    ...
    optimizers=(DummyOptimizer(), None),  # This is necessary to avoid the HF trainer from creating its own optimizer
)
```
"""


class DummyOptimizer(torch.optim.Optimizer):
  """ A dummy optimizer that does nothing. To be injected into HF trainer"""
  def __init__(self):
    super().__init__([torch.ones([1])], {})
  def step(self):
    pass
  def zero_grad(self):
    pass


def _fusion_step_hook(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer):
    """
        An auxiliary function to be used as a hook for the model's parameters. It is only meant to be used in conjunction with the patch_model function.
    """
    optimizer.step()
    param.grad = None


def patch_model(model: torch.nn.Module, optimizer_cls, optimizer_defaults={}):
    """
        Sets up the hooks of the model. This is meant to be called only once in the constructor.

        Args:
            model: The model to setup the hooks for
            optimizer_cls: The optimizer class to use
            optimizer_defaults: The default parameters to pass to the optimizer as a dictionary of kwargs.
    """
    model.optimizer_list = [(p, optimizer_cls([p], **optimizer_defaults, foreach=False)) for p in model.parameters()]
    for p, optimizer in model.optimizer_list:
        hook = partial(_fusion_step_hook, optimizer=optimizer)  # Partially evaluate with the matching optimizer reference
        p.register_post_accumulate_grad_hook(hook)
    return model