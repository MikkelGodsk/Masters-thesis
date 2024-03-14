import pytest

from src.backprop_trick import *
from tests.backprop_trick_test_utils import *


def test_mother_optimizer():
    net = ANN()
    correct_param_groups = [
        {'params': list(net.dense_1.parameters()) + list(net.dense_2.parameters()), 'lr': 0.1, 'betas': (0.9, 0.999)},
        {'params': net.dense_3.parameters(), 'lr': 0.01, 'betas': (0.1, 0.12)},
    ]
    mother_optimizer = MotherOptimizer(correct_param_groups, torch.optim.Adam, lr=9999) # The lr is not used
    param_group_1, param_group_2 = mother_optimizer.param_groups[0], mother_optimizer.param_groups[1]

    # Test keys
    assert set(param_group_1.keys()).issuperset(set(['params', 'lr', 'betas']))

    # Test values
    assert set(param_group_1.values()[1:]).issuperset({0.1, (0.9, 0.999)})
    assert torch.all(param_group_1.values()[0][0] == correct_param_groups[0]['params'][0])

    # Test items
    assert set(param_group_1.items()[1:]).issuperset({('lr', 0.1), ('betas', (0.9, 0.999))})
    assert param_group_1.items()[0][0] == 'params'
    assert torch.all(param_group_1.items()[0][1][0] == correct_param_groups[0]['params'][0])

    # Test __getitem__
    assert param_group_1['lr'] == 0.1

    # Test __setitem__
    param_group_1['lr'] = 100000
    assert param_group_1['lr'] == 100000
    assert mother_optimizer.child_optimizers[0].param_groups[0]['lr'] == 100000
    assert mother_optimizer.child_optimizers[1].param_groups[0]['lr'] == 100000
    assert mother_optimizer.child_optimizers[-1].param_groups[0]['lr'] == 0.01

    param_group_1['hello'] = 'world'
    assert param_group_1['hello'] == 'world'
    assert mother_optimizer.child_optimizers[0].param_groups[0]['hello'] == 'world'
    assert mother_optimizer.child_optimizers[1].param_groups[0]['hello'] == 'world'
    assert 'hello' not in mother_optimizer.child_optimizers[-1].param_groups[0].keys()

    # Test __delitem__
    del param_group_1['lr']
    assert 'lr' not in param_group_1
    assert 'lr' not in mother_optimizer.child_optimizers[0].param_groups[0]
    assert 'lr' not in mother_optimizer.child_optimizers[1].param_groups[0]
    assert 'lr' in mother_optimizer.child_optimizers[-1].param_groups[0]

    # Are the parameter groups correct?
    dense_3_params = list(net.dense_3.parameters())
    assert len(mother_optimizer.param_groups) == 2
    assert isinstance(mother_optimizer.param_groups[0], VirtualParameterGroup)
    assert isinstance(mother_optimizer.param_groups[0].associated_child_optimizers[0].param_groups[0]['params'][0], torch.Tensor)
    assert torch.all(mother_optimizer.param_groups[0].associated_child_optimizers[0].param_groups[0]['params'][0] == correct_param_groups[0]['params'][0])
    assert torch.all(mother_optimizer.param_groups[0].associated_child_optimizers[1].param_groups[0]['params'][0] == correct_param_groups[0]['params'][1])
    assert torch.all(mother_optimizer.param_groups[0].associated_child_optimizers[2].param_groups[0]['params'][0] == correct_param_groups[0]['params'][2])    
    assert torch.all(mother_optimizer.param_groups[0].associated_child_optimizers[3].param_groups[0]['params'][0] == correct_param_groups[0]['params'][3])
    assert torch.all(mother_optimizer.param_groups[1].associated_child_optimizers[0].param_groups[0]['params'][0] == dense_3_params[0])
    assert torch.all(mother_optimizer.param_groups[1].associated_child_optimizers[1].param_groups[0]['params'][0] == dense_3_params[1])


def test_mother_optimizer_init_no_parameter_groups_given():
    net = ANN()
    mother_optimizer = MotherOptimizer(net.parameters(), torch.optim.Adam, lr=0.01)
    assert len(mother_optimizer.child_optimizers) == 6
    params = net.parameters()
    for i, param in enumerate(params):
        assert torch.all(mother_optimizer.child_optimizers[i].param_groups[0]['params'][0] == param)


def test_backprop_trick_small_ann_no_parameter_groups():
    net1 = ANN()
    net2 = ANN(net1)
    optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.01)
    optimizer2 = MotherOptimizer(net2.parameters(), torch.optim.Adam, lr=0.01)
    consistency_check(net1, net2, optimizer1, optimizer2, (5,50), (5,1))


def test_backprop_trick_small_ann_with_parameter_groups():
    net1 = ANN()
    net2 = ANN(net1)
    net1_param_group = [
        {'params': list(net1.dense_1.parameters()) + list(net1.dense_2.parameters()), 'lr': 0.1, 'betas': [0.9, 0.999]},
        {'params': net1.dense_3.parameters(), 'lr': 0.01, 'betas': [0.1, 0.12]},
    ]
    net2_param_group = [
        {'params': list(net2.dense_1.parameters()) + list(net2.dense_2.parameters()), 'lr': 0.1, 'betas': [0.9, 0.999]},
        {'params': net2.dense_3.parameters(), 'lr': 0.01, 'betas': [0.1, 0.12]},
    ]
    optimizer1 = torch.optim.Adam(net1_param_group, lr=0.01)
    optimizer2 = MotherOptimizer(net2_param_group, torch.optim.Adam, lr=0.01)
    consistency_check(net1, net2, optimizer1, optimizer2, (5,50), (5,1))


def test_backprop_trick_one_layer_does_not_require_grads():
    net1 = ANN()
    net2 = ANN(net1)
    net1_param_group = [
        {'params': list(net1.dense_1.parameters()) + list(net1.dense_2.parameters()), 'lr': 0.1, 'betas': [0.9, 0.999]},
        {'params': net1.dense_3.parameters(), 'lr': 0.01, 'betas': [0.1, 0.12]},
    ]
    net2_param_group = [
        {'params': list(net2.dense_1.parameters()) + list(net2.dense_2.parameters()), 'lr': 0.1, 'betas': [0.9, 0.999]},
        {'params': net2.dense_3.parameters(), 'lr': 0.01, 'betas': [0.1, 0.12]},
    ]
    net1.dense_1.requires_grad = False
    net2.dense_1.requires_grad = False
    optimizer1 = torch.optim.Adam(net1_param_group, lr=0.01)
    optimizer2 = MotherOptimizer(net2_param_group, torch.optim.Adam, lr=0.01)
    consistency_check(net1, net2, optimizer1, optimizer2, (5,50), (5,1))
