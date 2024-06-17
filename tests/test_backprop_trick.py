import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset, Dataset

from src.backprop_trick import MotherOptimizer, VirtualParameterGroup
from tests.backprop_trick_test_utils import ANN, consistency_check


def test_mother_optimizer():
    net = ANN()
    correct_param_groups = [
        {
            "params": list(net.dense_1.parameters()) + list(net.dense_2.parameters()),
            "lr": 0.1,
            "betas": (0.9, 0.999),
        },
        {"params": net.dense_3.parameters(), "lr": 0.01, "betas": (0.1, 0.12)},
    ]
    mother_optimizer = MotherOptimizer(
        correct_param_groups, torch.optim.Adam, lr=9999
    )  # The lr is not used
    param_group_1, _ = (
        mother_optimizer.param_groups[0],
        mother_optimizer.param_groups[1],
    )

    # Test keys
    assert set(param_group_1.keys()).issuperset(set(["params", "lr", "betas"]))

    # Test values
    assert set(param_group_1.values()[1:]).issuperset({0.1, (0.9, 0.999)})
    assert torch.all(
        param_group_1.values()[0][0] == correct_param_groups[0]["params"][0]
    )

    # Test items
    assert set(param_group_1.items()[1:]).issuperset(
        {("lr", 0.1), ("betas", (0.9, 0.999))}
    )
    assert param_group_1.items()[0][0] == "params"
    assert torch.all(
        param_group_1.items()[0][1][0] == correct_param_groups[0]["params"][0]
    )

    # Test __getitem__
    assert param_group_1["lr"] == 0.1

    # Test __setitem__
    param_group_1["lr"] = 100000
    assert param_group_1["lr"] == 100000
    assert mother_optimizer.child_optimizers[0].param_groups[0]["lr"] == 100000
    assert mother_optimizer.child_optimizers[1].param_groups[0]["lr"] == 100000
    assert mother_optimizer.child_optimizers[-1].param_groups[0]["lr"] == 0.01

    param_group_1["hello"] = "world"
    assert param_group_1["hello"] == "world"
    assert mother_optimizer.child_optimizers[0].param_groups[0]["hello"] == "world"
    assert mother_optimizer.child_optimizers[1].param_groups[0]["hello"] == "world"
    assert "hello" not in mother_optimizer.child_optimizers[-1].param_groups[0].keys()

    # Test __delitem__
    del param_group_1["lr"]
    assert "lr" not in param_group_1
    assert "lr" not in mother_optimizer.child_optimizers[0].param_groups[0]
    assert "lr" not in mother_optimizer.child_optimizers[1].param_groups[0]
    assert "lr" in mother_optimizer.child_optimizers[-1].param_groups[0]

    # Are the parameter groups correct?
    dense_3_params = list(net.dense_3.parameters())
    assert len(mother_optimizer.param_groups) == 2
    assert isinstance(mother_optimizer.param_groups[0], VirtualParameterGroup)
    assert isinstance(
        mother_optimizer.param_groups[0]
        .associated_child_optimizers[0]
        .param_groups[0]["params"][0],
        torch.Tensor,
    )
    assert torch.all(
        mother_optimizer.param_groups[0]
        .associated_child_optimizers[0]
        .param_groups[0]["params"][0]
        == correct_param_groups[0]["params"][0]
    )
    assert torch.all(
        mother_optimizer.param_groups[0]
        .associated_child_optimizers[1]
        .param_groups[0]["params"][0]
        == correct_param_groups[0]["params"][1]
    )
    assert torch.all(
        mother_optimizer.param_groups[0]
        .associated_child_optimizers[2]
        .param_groups[0]["params"][0]
        == correct_param_groups[0]["params"][2]
    )
    assert torch.all(
        mother_optimizer.param_groups[0]
        .associated_child_optimizers[3]
        .param_groups[0]["params"][0]
        == correct_param_groups[0]["params"][3]
    )
    assert torch.all(
        mother_optimizer.param_groups[1]
        .associated_child_optimizers[0]
        .param_groups[0]["params"][0]
        == dense_3_params[0]
    )
    assert torch.all(
        mother_optimizer.param_groups[1]
        .associated_child_optimizers[1]
        .param_groups[0]["params"][0]
        == dense_3_params[1]
    )


def test_mother_optimizer_init_no_parameter_groups_given():
    net = ANN()
    mother_optimizer = MotherOptimizer(net.parameters(), torch.optim.Adam, lr=0.01)
    assert len(mother_optimizer.child_optimizers) == 6
    params = net.parameters()
    for i, param in enumerate(params):
        assert torch.all(
            mother_optimizer.child_optimizers[i].param_groups[0]["params"][0] == param
        )


def test_backprop_trick_small_ann_no_parameter_groups():
    net1 = ANN()
    net2 = ANN(net1)
    optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.01)
    optimizer2 = MotherOptimizer(net2.parameters(), torch.optim.Adam, lr=0.01)
    consistency_check(net1, net2, optimizer1, optimizer2, (5, 50), (5, 1))


def test_backprop_trick_small_ann_with_parameter_groups():
    net1 = ANN()
    net2 = ANN(net1)
    net1_param_group = [
        {
            "params": list(net1.dense_1.parameters()) + list(net1.dense_2.parameters()),
            "lr": 0.1,
            "betas": [0.9, 0.999],
        },
        {"params": net1.dense_3.parameters(), "lr": 0.01, "betas": [0.1, 0.12]},
    ]
    net2_param_group = [
        {
            "params": list(net2.dense_1.parameters()) + list(net2.dense_2.parameters()),
            "lr": 0.1,
            "betas": [0.9, 0.999],
        },
        {"params": net2.dense_3.parameters(), "lr": 0.01, "betas": [0.1, 0.12]},
    ]
    optimizer1 = torch.optim.Adam(net1_param_group, lr=0.01)
    optimizer2 = MotherOptimizer(net2_param_group, torch.optim.Adam, lr=0.01)
    consistency_check(net1, net2, optimizer1, optimizer2, (5, 50), (5, 1))


def test_backprop_trick_one_layer_does_not_require_grads():
    net1 = ANN()
    net2 = ANN(net1)
    net1_param_group = [
        {
            "params": list(net1.dense_1.parameters()) + list(net1.dense_2.parameters()),
            "lr": 0.1,
            "betas": [0.9, 0.999],
        },
        {"params": net1.dense_3.parameters(), "lr": 0.01, "betas": [0.1, 0.12]},
    ]
    net2_param_group = [
        {
            "params": list(net2.dense_1.parameters()) + list(net2.dense_2.parameters()),
            "lr": 0.1,
            "betas": [0.9, 0.999],
        },
        {"params": net2.dense_3.parameters(), "lr": 0.01, "betas": [0.1, 0.12]},
    ]
    net1.dense_1.requires_grad = False
    net2.dense_1.requires_grad = False
    optimizer1 = torch.optim.Adam(net1_param_group, lr=0.01)
    optimizer2 = MotherOptimizer(net2_param_group, torch.optim.Adam, lr=0.01)
    consistency_check(net1, net2, optimizer1, optimizer2, (5, 50), (5, 1))


def backprop_trick_opt_no_grad_clip(optim_cls):
    model_name: str = "facebook/opt-125m"
    dataset_name = "GAIR/lima"
    initial_lr = 1e+1

    ds = load_dataset(dataset_name, "plain_text")
    train_ds = Dataset.from_dict(ds['train'][0:30])
    model_1 = AutoModelForCausalLM.from_pretrained(model_name)
    model_2 = AutoModelForCausalLM.from_pretrained(model_name)
    model_3 = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, (p1, p2) in enumerate(zip(model_1.parameters(), model_2.parameters())):
        assert (p1==p2).all()


    optimizer_1 = optim_cls(model_1.parameters(), lr=initial_lr)
    lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=10)

    optimizer_2 = MotherOptimizer(
        model_2.parameters(),
        optim_cls,
        lr=initial_lr,
    )
    for param_group in optimizer_2.param_groups:
        param_group["initial_lr"] = initial_lr  # Needed for lr scheduler
    lr_scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=10)

    training_args = TrainingArguments(
        do_eval=False,
        output_dir=".",
        num_train_epochs=1,
        full_determinism=True,
        use_cpu=True,
        max_grad_norm=None,    # It is normalized over all parameters!  https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
    )

    trainer_1 = SFTTrainer(
        model=model_1,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="conversations",
        train_dataset=train_ds,
        max_seq_length=128,
        optimizers=(optimizer_1, lr_scheduler_1),  # If backprop_trick is False, this is set to (None, None) by the factory...
    )
    trainer_1.train()

    trainer_2 = SFTTrainer(
        model=model_2,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="conversations",
        train_dataset=train_ds,
        max_seq_length=128,
        optimizers=(optimizer_2, lr_scheduler_2),  # If backprop_trick is False, this is set to (None, None) by the factory...
    )
    trainer_2.train()   # So the error happens in here....



    ##############################################
    # Test: Are the two trained models the same? #
    ##############################################
    for i, (p1, p2) in enumerate(zip(model_1.parameters(), model_2.parameters())):
        assert torch.equal(p1, p2), f"Parameter {i} is not equal!"

    #######################################################
    # Test: Are we sure the models were actually trained? #
    #######################################################
    for i, (p1, p3) in enumerate(zip(model_1.parameters(), model_3.parameters())):
        assert not torch.equal(p1, p3), f"Parameter {i} is equal when it shouldn't be!"


def test_backprop_trick_opt_no_grad_clip():
    backprop_trick_opt_no_grad_clip(torch.optim.SGD)
    backprop_trick_opt_no_grad_clip(torch.optim.AdamW)