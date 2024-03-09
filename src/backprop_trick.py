import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Any, Dict, Generator, Iterable, List, Tuple, Type, TypeAlias, Union

# NOTE: Does not work yet...
# TODO: Need to implement an LR scheduler that is setup with the model's optimizers...

"""
  Resources used in the implementation (for future reference):
  - SFTTrainer:                       https://github.com/huggingface/trl/blob/0f13e51efab6bea6b51200ea66396a0716d63182/trl/trainer/sft_trainer.py#L55
  - Trainer._inner_training_loop:     https://github.com/huggingface/transformers/blob/092f1fdaa4224fdd88c616dc9678e6fcb37bfffd/src/transformers/trainer.py#L1631
  - torch.LambdaLR:                   https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR
  - torch.LRScheduler:                https://pytorch.org/ignite/_modules/ignite/handlers/param_scheduler.html#LRScheduler

  LR_Scheduler has its .step method called if it's not a ReduceLROnPlateau

  NOTE: Things to turn off
    - Gradient accumulation (i.e. args.gradient_accumulation_steps = 1)... This method mitigates the need for it!
    - Gradient clipping (args.max_grad_norm = 0 or None)... Gradient clipping must be implemented as a callback instead.

  NOTE: Distributed training
    There might be some things that could be made faster with Nvidia Apex? But I'll start simple...
    The Apex rabbithole starts here: 
      - https://github.com/huggingface/transformers/blob/092f1fdaa4224fdd88c616dc9678e6fcb37bfffd/src/transformers/trainer.py#L1770C9-L1781C18
      - https://nvidia.github.io/apex/optimizers.html
    Note: `create_optimizer_and_scheduler` only creates optimizer and scheduler if they haven't been passed already... But for some reason, they are created later if FSDP is used... Maybe something to take into account later on....
"""


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

def unzip(zipped_list):
    """
        Unzips a list of tuples into two lists
    """
    return list(zip(*zipped_list))


def fusion_step_hook(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer):
    """
        An auxiliary function to be used as a hook for the model's parameters. It is only meant to be used in conjunction with the patch_model function.
    """
    optimizer.step()
    param.grad = None


ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


### Changes to the algorithm:
#### The VirtualParameterGroup now sets only the parameters in the optimizer...
#### The VirtualParameterGroup doesn't care about the child parameter groups... Those are owned solely by the optimizer... But whenever queried, it goes in and reads them...
class VirtualParameterGroup(Dict):  # https://stackoverflow.com/questions/3387691/how-to-perfectly-override-a-dict
  """
    This class will be used for communication of hyperparameters between the mother optimizer and the child optimizers.
    It will look like an ordinary parameter group dictionary to any outside viewers, 
    but upon updating its members, it will notify all child-optimizers of the change.
  """
  def __init__(self, associated_child_optimizers: List[torch.optim.Optimizer]):
    self.associated_child_optimizers = associated_child_optimizers
    for optim in self.associated_child_optimizers:
      assert len(optim.param_groups) == 1, "The child optimizers must have only one parameter group."
    # not calling super().__init__ because we don't want to initialize the dictionary with anything... Just keep the type...

  def join_parameters(self):
    """
      This function returns a list of all the parameters in the child parameter groups.
    """
    return [param for optimizer in self.associated_child_optimizers for param in optimizer.param_groups[0]['params']]

  def keys(self):
    return self.associated_child_optimizers[0].param_groups[0].keys()
  
  def values(self):
    return [self.join_parameters()] + [v for k, v in self.associated_child_optimizers[0].param_groups[0].items() if k != 'params']
  
  def items(self):
    return [('params', self.join_parameters())] + [(k, v) for k, v in self.associated_child_optimizers[0].param_groups[0].items() if k != 'params']

  def __setitem__(self, key, value):
    """
      When updating hyperparameters, we notify all child optimizers.
      If the user tries to set 'params', an error is thrown.
    """
    if key == 'params':
      return
    for child_optimizer in self.associated_child_optimizers:
      child_optimizer.param_groups[0][key] = value

  def __getitem__(self, key: Any) -> Any:
    if key == 'params':
      return self.join_parameters()
    return self.associated_child_optimizers[0].param_groups[0][key]

  def __delitem__(self, key):
    """
      When deleting hyperparameters, we notify all child optimizers.
    """
    if key == 'params':
      return
    
    for child_optimizer in self.associated_child_optimizers:
      child_optimizer.param_groups[0].pop(key)


def get_parameter_list(param_groups: VirtualParameterGroup):
    """
      This function returns a list of parameters from the params argument.
    """
    parameter_list = []
    for param_group in param_groups:
      parameter_list.extend(param_group.params)
    return parameter_list


def cast_parameter_iterables_to_lists(params: ParamsT) -> List[Dict[str, Any]]:
    """
      We since we will need to traverse the list of parameters twice
    """
    new_params = []
    type_ = None
    for i, param in enumerate(params):
      if isinstance(param, torch.Tensor):
        if type_ is None:
          type_ = torch.Tensor
        assert type_ is not Dict, "The params argument must be an iterable of torch.Tensor or Dict objects"
        new_params.append(param)
      elif isinstance(param, Dict):
        if type_ is None:
          type_ = Dict
        assert type_ is not torch.Tensor, "The params argument must be an iterable of torch.Tensor or Dict objects"
        new_params.append({**param})
        new_params[-1]['params'] = list(new_params[-1]['params'])
    return new_params


class MotherOptimizer(torch.optim.Optimizer):
  """ 
    This optimizer holds all the "child" optimizers, which are stepped automatically. It then functions as an interface with the LR scheduler and the HF trainer.
  """
  def __init__(self, 
               params: ParamsT, 
               optimizer_cls: Type[torch.optim.Optimizer], 
               **optimizer_defaults: Dict[str, Any]):
    self._prepare_model(params, optimizer_cls, **optimizer_defaults)         # creates self.optimizer_list, self.param_groups, and registers the fusion_step_hook with all parameters...
    super().__init__(self.param_groups, {})

  def step(self, *args, **kwargs):
    pass

  def zero_grad(self, *args, **kwargs):
    pass
  
  def get_optimizers_and_virtual_param_groups(self, 
                                      params: ParamsT, 
                                      optimizer_cls: Type[torch.optim.Optimizer], 
                                      **optimizer_defaults: Dict[str, Any]
                                      ) -> Tuple[List[torch.optim.Optimizer], List[VirtualParameterGroup]]:
    """
      This function creates the child optimizers and the virtual parameter groups.
    """
    params = cast_parameter_iterables_to_lists(params)   # This call checks if the input is valid...
    optimizers = []
    virtual_param_groups = []
    for param_group in params:
      if isinstance(param_group, torch.Tensor):
        o = optimizer_cls([param_group], **optimizer_defaults)
        optimizers.append(o)
      elif isinstance(param_group, Dict):
        hyperparameters = {k: v for k, v in param_group.items() if k not in ['params']}
        associated_optimizers = []
        for param in param_group['params']:
          o = optimizer_cls([param], **hyperparameters)
          optimizers.append(o)
          associated_optimizers.append(o)
        virtual_param_groups.append(VirtualParameterGroup(associated_optimizers))

    if isinstance(param_group, torch.Tensor):
      virtual_param_groups = [VirtualParameterGroup(optimizers)]
    return optimizers, virtual_param_groups

  def _prepare_model(self,
                     params: ParamsT, 
                     optimizer_cls: Type[torch.optim.Optimizer], 
                     **optimizer_defaults: Dict[str, Any]):
    """
        Sets up the hooks of the model. This is meant to be called only once in the constructor.

        Args:
            model: The model to setup the hooks for
            optimizer_cls: The optimizer class to use
            optimizer_defaults: The default parameters to pass to the optimizer as a dictionary of kwargs.
    """
    from functools import partial
    self.child_optimizers, self.param_groups = self.get_optimizers_and_virtual_param_groups(params, optimizer_cls, **optimizer_defaults)

    # Setup hooks
    for optimizer in self.child_optimizers:
        p = optimizer.param_groups[0]['params'][0]  # As the optimizers have one parameter each!
        hook = partial(fusion_step_hook, optimizer=optimizer)  # Partially evaluate with the matching optimizer reference
        p.register_post_accumulate_grad_hook(hook)
    

"""
  Timing experiment to see if the MotherOptimizer is faster than the regular optimizer...
"""
if __name__ == '__main__':
  class ANN(nn.Module):
    def __init__(self, net_to_copy=None):
      super().__init__()
      if net_to_copy is None:
        self.dense_1 = nn.Linear(50,50)
        self.dense_2 = nn.Linear(50,50)
        self.dense_3 = nn.Linear(50,1)
      else:
        from copy import deepcopy
        self.dense_1 = deepcopy(net_to_copy.dense_1)
        self.dense_2 = deepcopy(net_to_copy.dense_2)
        self.dense_3 = deepcopy(net_to_copy.dense_3)

    def forward(self, x):
      z0 = x
      z1 = F.relu(self.dense_1(z0))
      z2 = F.relu(self.dense_2(z1))
      z3 = self.dense_3(z2)
      return z3
  
  def timing(net, optimizer, max_iter=1000):
    loss_fun = nn.MSELoss()
    for j in range(max_iter):
      # Sample random datapoint
      x = torch.randn((5,50)).cuda()
      y = torch.randn([5,1]).cuda()

      # Do inference
      pred = net(x)
      loss = loss_fun(pred, y)
      loss.backward()
      optimizer.step()
      net.zero_grad()

  net1 = ANN().cuda()
  net2 = ANN().cuda()
  param_groups_1 = [
      {'params': list(net1.dense_1.parameters()) + list(net1.dense_2.parameters()), 'lr': 0.1, 'betas': (0.9, 0.999)},
      {'params': net1.dense_3.parameters(), 'lr': 0.01, 'betas': (0.1, 0.12)},
  ]
  param_groups_2 = [
      {'params': list(net2.dense_1.parameters()) + list(net2.dense_2.parameters()), 'lr': 0.1, 'betas': (0.9, 0.999)},
      {'params': net2.dense_3.parameters(), 'lr': 0.01, 'betas': (0.1, 0.12)},
  ]
  optim1 = torch.optim.Adam(param_groups_1, lr=1e-3)
  optim2 = MotherOptimizer(param_groups_2, torch.optim.Adam, lr=1e-3)

  from timeit import repeat
  from statistics import mean, stdev
  print(f"Timing the first optimizer...")
  times = repeat(lambda: timing(net1, optim1), number=10, repeat=10)
  print(f"Mean: {mean(times)}, standard deviation: {stdev(times)}")
  print(f"Timing the second optimizer...")
  times = repeat(lambda: timing(net2, optim2), number=10, repeat=10)
  print(f"Mean: {mean(times)}, standard deviation: {stdev(times)}")
  # The backprop trick seems to take approx 70% longer than the other method...
