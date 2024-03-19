import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List


class ANN(nn.Module):
    def __init__(self, net_to_copy=None):
        super().__init__()
        if net_to_copy is None:
            self.dense_1 = nn.Linear(50, 50)
            self.dense_2 = nn.Linear(50, 50)
            self.dense_3 = nn.Linear(50, 1)
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


# Utilities for testing whether two approaches are consistent...
def net_eq(self, other):  # Overloading __eq__ gives issues
    """
    Defines two networks as the same if all their parameters are the same
    regardless of which class they are...
    """
    for s_param, o_param in zip(self.parameters(), other.parameters()):
        if not torch.all(s_param == o_param):
            return False
    return True


def consistency_check(
    net1: nn.Module,
    net2: nn.Module,
    optimzer1: torch.optim.Optimizer,
    optimizer2: torch.optim.Optimizer,
    shape_in: List[int],
    shape_out: List[int],
):
    """
    Here lambda_net1 and lambda_net2 create the nets.
    In lambda_net2, we pass net1 for copying...
    """
    # Initialize networks
    loss_fun = nn.MSELoss()
    net1.to("cpu")
    net2.to("cpu")  # Allows for deepcopy...
    assert net_eq(net1, net2)

    # Make sure that the network weights are not just referencing each other
    p1: nn.Parameter = next(iter(net1.parameters()))
    p2: nn.Parameter = next(iter(net2.parameters()))
    p1.requires_grad = False
    p2.requires_grad = False
    p1[0, 0] = 0.0
    assert not net_eq(net1, net2)
    p1[0, 0] = p2[0, 0]
    assert net_eq(net1, net2)
    p1.requires_grad = True
    p2.requires_grad = True

    # Now train
    p_init = deepcopy(p2)
    for j in range(100):
        x = torch.randn(shape_in)
        y = torch.randn(shape_out)

        # Inference and backward
        pred1 = net1(x)
        loss1: torch.Tensor = loss_fun(pred1, y)
        loss1.backward()
        optimzer1.step()
        net1.zero_grad()

        pred2 = net2(x)
        loss2: torch.Tensor = loss_fun(pred2, y)
        loss2.backward()
        optimizer2.step()
        net2.zero_grad()

        # Make sure the nets actually trained
        assert not torch.all(
            p_init == next(iter(net1.parameters()))
        ), "The network did not train"
        assert not torch.all(
            p_init == next(iter(net2.parameters()))
        ), "The network did not train"

    # See if their parameters are equal
    assert net_eq(net1, net2), f"Failed in iteration {j}"
