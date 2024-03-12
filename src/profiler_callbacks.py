"""
Contains some useful callbacks for profiling in the Hugging Face Trainer. 
-------------------------------------------------------------------------

Module :mod:`profiler_callbacks` contains the callbacks for performing memory profiling, stack profiling, and timing of the training.
The following of the callbacks log to W&B:
    
        - :class:`WandBProfilerCallback`
        - :class:`WandBTimerCallback`

The following of the callbacks log to disk:
        
            - :class:`TorchProfilerCallback`
            - :class:`MemoryHistoryCallback`
"""

import os
import torch
import pickle
import html2image  # pip install --upgrade html2image
import time
from datetime import datetime
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers import TrainerCallback
import wandb
from typing import List, Optional
from math import floor


TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def get_timestamp():
    return datetime.now().strftime(TIME_FORMAT_STR)

def torch_get_devices():
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


class ProfilerCallbackBase(TrainerCallback):
    """
        Base class for the profilers. Since they have to be scheduled identically, it makes sense to have a base class for them.
        The subclasses are meant to be used like this::

            trainer = SFTTrainer(
                ...
                callbacks=[
                    TorchProfilerCallback(logging_dir, profile_epochs=[1], profile_n_steps=5), 
                    MemoryHistoryCallback(logging_dir, profile_epochs=[1], profile_n_steps=5),
                    ...
                ],
            )
        
        I.e., they are scheduled to log the first `profile_n_steps` of each epoch in `profile_epochs`. This to ensure that the files to not become too large.

        NOTE: Be very careful that they do not log during the logging, if you are logging using model inference! This will make the files very large and slow to open!

        Docs: For HuggingFace callbacks, see https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
    """
    def __init__(self, logging_dir: str, *args, profile_epochs: List[int]=[], profile_n_steps:int=1, repeat_every_n_steps=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_epochs = profile_epochs
        self.profile_n_steps = profile_n_steps
        self.repeat_every_n_steps = repeat_every_n_steps
        self.step_counter = 0
        self.is_running = False
        self.profile_dir = os.path.join(logging_dir, "profile")
        os.makedirs(self.profile_dir, exist_ok=True)

    def start_(self):
        if not self.is_running:
            self.start()
            self.is_running = True

    def stop_(self, epoch=-1):
        if self.is_running:
            epoch = int(floor(epoch))
            timestamp = get_timestamp()
            epoch_str = f"epoch_{epoch}" if epoch>0 else ""
            file_identifier = f"{epoch_str}-{timestamp}"
            self.stop(epoch, file_identifier)
            self.is_running = False

    def step_(self):
        if self.is_running:
            self.step()

    def start(self):  # Should be overwritten
        raise NotImplementedError
    
    def step(self):  # May be overwritten
        pass
    
    def stop(self, file_identifier):  # Should be overwritten
        # file_identifier is part of the filename. E.g. name the file: f"stack_trace_{file_identifier}.json"
        raise NotImplementedError

    # Fyi, kwargs has keys: ['model', 'tokenizer', 'optimizer', 'lr_scheduler', 'train_dataloader', 'eval_dataloader']
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (floor(state.epoch) in self.profile_epochs) or (len(self.profile_epochs) == 0):    # state.epoch is a float, not an int...
            self.start_()
            self.step_counter = 0

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (not self.is_running) and (self.repeat_every_n_steps >= 0):
            if self.step_counter % self.repeat_every_n_steps == 0:
                self.start_()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (floor(state.epoch) in self.profile_epochs) or (len(self.profile_epochs) == 0):   # state.epoch is a float, not an int...
            self.step_()
            if self.step_counter > self.profile_n_steps-1:
                self.stop_(epoch=state.epoch)
            self.step_counter += 1

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (floor(state.epoch)-1 in self.profile_epochs) or (len(self.profile_epochs) == 0):   # state.epoch is a float, not an int... So we need to subtract 1 as it counts a bit up for every step...
            self.stop_()  # Stop just in case it is still running

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.stop_()  # Stop just in case it is still running


class TorchProfilerCallback(ProfilerCallbackBase):
    """
    This callback logs the PyTorch profiler data for a given epoch. It logs the first n steps of an epoch.
    NOTE: Be careful that this one does not log during the logging, if you are logging using model inference!

    To open the profiler stack trace data in Chrome, go to chrome://tracing and drag in the JSON file.
    To open the memory timeline, open the HTML file in a browser or view it in W&B.

    How to use with Hugging Face Trainer::

        trainer = SFTTrainer(
            ...
            callbacks=[
                ProfilerCallback(logging_dir, profile_epochs=[1], profile_n_steps=1, upload_to_wandb=True), 
            ],
        )

    Docs: https://pytorch.org/tutorials/recipes/recipes/profiler.html
    and https://pytorch.org/docs/stable/profiler.html
    """
    def __init__(self, logging_dir: str, *args, profile_epochs: List[int]=[], profile_n_steps:int=1, do_memory_profile=True, do_stack_profile=True, upload_to_wandb=True, save_files=True, verbose=True, **kwargs):
        super().__init__(logging_dir, *args, profile_epochs=profile_epochs, profile_n_steps=profile_n_steps, **kwargs)
        self.upload_to_wandb = upload_to_wandb
        self.save_files = save_files
        self.do_memory_profile = do_memory_profile
        self.do_stack_profile = do_stack_profile
        self.verbose = verbose
        self.profiler = self.setup_profiler()

    def setup_profiler(self):
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )

    def start(self):
        if self.verbose:
            print("\n\nStarting profiler\n\n")
        self.profiler.start()
    
    def step(self):
        self.profiler.step()

    def stop(self, epoch, file_identifier):
        """
            Stop the profiler and save the data to disk. Is safe to call whenever. Does nothing if the profiler is not running.
        """
        if self.verbose:
            print("\n\nStopping profiler\n")
        self.profiler.stop()

        if self.do_stack_profile and self.save_files:
            trace_file = os.path.join(self.profile_dir, f"stack_trace_{file_identifier}.json")
            self.profiler.export_chrome_trace(trace_file)
            print(f"Stack profiler data saved to {trace_file} (use Google Chrome and go to 'chrome://tracing' to read this file)")

        if self.do_memory_profile:
            memory_file = os.path.join(self.profile_dir, f"memory_timeline_{file_identifier}.json")
            if self.save_files:
                self.profiler.export_memory_timeline(memory_file)
                print(f"Memory profiler data saved to {memory_file}")

            for device_str in torch_get_devices():
                memory_file_img = os.path.join(self.profile_dir, f"memory_timeline_{file_identifier}_{device_str.replace(':','_')}.svg")
                fig = extract_memory_timeline_svg(self.profiler, memory_file_img, device=device_str, title=f"Memory timeline at epoch {epoch} on device {device_str}", save_file=self.save_files)
                if self.save_files:
                    print(f"Memory profiler data saved to {memory_file_img}")
                if self.upload_to_wandb: 
                    wandb.log({f"Memory timeline": wandb.Image(fig)})
        
        if self.verbose or self.save_files:
            print("\n\n")


class WandBProfilerCallback(TorchProfilerCallback):
    def __init__(self, *args, profile_epochs: List[int]=[], profile_n_steps:int=5, **kwargs):
        super().__init__(
            "", 
            *args, 
            profile_epochs=profile_epochs, 
            profile_n_steps=profile_n_steps, 
            do_memory_profile=True, 
            do_stack_profile=False, 
            upload_to_wandb=True, 
            save_files=False, 
            verbose=False, 
            **kwargs
        )


class WandBTimerCallback(TrainerCallback):
    """
        Written mostly by GitHub Copilot
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_start = 0
        self.epoch_start = 0
        self.step_start = 0

    def setup(self):
        wandb.define_metric("wall_time")
        wandb.define_metric("epoch_time", step_metric="wall_time")
        wandb.define_metric("train_time", step_metric="wall_time")
        wandb.define_metric("step_time", step_metric="wall_time")
        wandb.define_metric("epoch", step_metric="wall_time")
        wandb.define_metric("step", step_metric="wall_time")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.setup()   # It needs wandb.init to have been called, but it is only called in the trainer...
        self.train_start = time.time()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        now = time.time()
        print(f"Training ended in {now - self.train_start} seconds")

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.epoch_start = time.time()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        now = time.time()
        wandb.log({"epoch_time": now - self.epoch_start, "wall_time": now - self.train_start})
        wandb.log({"epoch": state.epoch, "wall_time": now - self.train_start})
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.step_start = time.time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        now = time.time()
        wandb.log({"step_time": now - self.step_start, "wall_time": now - self.train_start})
        wandb.log({"step": state.global_step, "wall_time": now - self.train_start})


class MemoryHistoryCallback(ProfilerCallbackBase):
    """
    To view the memory trace, go to https://pytorch.org/memory_viz and drag in the pickle file.
    NOTE: The stack trace grows upwards here!
    
    You might get this issue::
    
        huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
        To disable this warning, you can either:
            - Avoid using `tokenizers` before the fork if possible
            - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)

    I'm not sure what to do about it, so maybe this callback should only be used if you tend to run out of memory...

    Docs: https://pytorch.org/docs/stable/torch_cuda_memory.html
    and https://pytorch.org/blog/understanding-gpu-memory-1/
    """
    def __init__(self, logging_dir:str, *args, profile_epochs: List[int]=[], profile_n_steps:int=1, **kwargs):
        super().__init__(logging_dir, *args, profile_epochs=profile_epochs, profile_n_steps=profile_n_steps, **kwargs)

    def start(self):
        print("\n\nStarting CUDA memory recorder\n\n")
        torch.cuda.empty_cache()
        torch.cuda.memory._record_memory_history(enabled='all')

    def stop(self, epoch, file_identifier):
        print("\n\nStopping CUDA memory recorder")
        file_name = os.path.join(self.profile_dir, f"memory_trace_{file_identifier}.pickle")
        s = torch.cuda.memory._snapshot()
        with open(file_name, "wb") as f:
            pickle.dump(s, f)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"CUDA memory recording saved to {file_name} (go to https://pytorch.org/memory_viz to read this file - note that the stack grows upwards in the trace!)\n\n")


# Apparently the rule in Python is to define the functions before they're called, not before they're used in another function...
def extract_memory_timeline_svg(profiler, file_path: Optional[str] = None, device: Optional[str] = None, figsize=(20, 12), title=None, save_file=True):
    """
        This function is modified directly from the PyTorch source code here:
        - https://github.com/pytorch/pytorch/blob/76f3663efea524adcb60f515b471c412aa78b95e/torch/profiler/profiler.py#L258
        - https://github.com/pytorch/pytorch/blob/360761f7d039445e7b00493c2990ace9f94a5a9e/torch/profiler/_memory_profiler.py#L1133

        It is a bit of a hack, but it works. The original function is not meant to be used like this, but it is the only way to get the memory timeline as an SVG file. Copyright disclaimer from PyTorch:


        From PyTorch:

        Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
        Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
        Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
        Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
        Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
        Copyright (c) 2011-2013 NYU                      (Clement Farabet)
        Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
        Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
        Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

        From Caffe2:

        Copyright (c) 2016-present, Facebook Inc. All rights reserved.

        All contributions by Facebook:
        Copyright (c) 2016 Facebook Inc.

        All contributions by Google:
        Copyright (c) 2015 Google Inc.
        All rights reserved.

        All contributions by Yangqing Jia:
        Copyright (c) 2015 Yangqing Jia
        All rights reserved.

        All contributions by Kakao Brain:
        Copyright 2019-2020 Kakao Brain

        All contributions by Cruise LLC:
        Copyright (c) 2022 Cruise LLC.
        All rights reserved.

        All contributions from Caffe:
        Copyright(c) 2013, 2014, 2015, the respective contributors
        All rights reserved.

        All other contributions:
        Copyright(c) 2015, 2016 the respective contributors
        All rights reserved.

        Caffe2 uses a copyright model similar to Caffe: each contributor holds
        copyright over their contributions to Caffe2. The project versioning records
        all such contribution and copyright details. If a contributor wants to further
        mark their specific copyright on a particular contribution, they should
        indicate their copyright solely in the commit message of the change when it is
        committed.

        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

        3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
        and IDIAP Research Institute nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
        ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
        LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
        CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
        INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
        CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
        ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
        POSSIBILITY OF SUCH DAMAGE.
    """
    import enum
    # From file
    class Category(enum.Enum):
        INPUT = enum.auto()
        TEMPORARY = enum.auto()
        ACTIVATION = enum.auto()
        GRADIENT = enum.auto()
        AUTOGRAD_DETAIL = enum.auto()
        PARAMETER = enum.auto()
        OPTIMIZER_STATE = enum.auto()

    _CATEGORY_TO_COLORS = {
        Category.PARAMETER: "darkgreen",
        Category.OPTIMIZER_STATE: "goldenrod",
        Category.INPUT: "black",
        Category.TEMPORARY: "mediumpurple",
        Category.ACTIVATION: "red",
        Category.GRADIENT: "mediumblue",
        Category.AUTOGRAD_DETAIL: "royalblue",
        None: "grey",
    }

    _CATEGORY_TO_INDEX = {c: i for i, c in enumerate(_CATEGORY_TO_COLORS)}


    # From profiler
    if device is None and profiler.use_device and profiler.use_device != "cuda":
        device = profiler.use_device + ":0"

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
    mem_tl = MemoryProfileTimeline(profiler._memory_profile())

    # From MemoryProfileTimeline
    import matplotlib.pyplot as plt
    import numpy as np
    device_str = device

    mt = mem_tl._coalesce_timeline(device_str)
    times, sizes = np.array(mt[0]), np.array(mt[1])
    # For this timeline, start at 0 to match Chrome traces.
    t_min = min(times)
    times -= t_min
    stacked = np.cumsum(sizes, axis=1) / 1024**3
    device = torch.device(device_str)
    max_memory_allocated = torch.cuda.max_memory_allocated(device)
    max_memory_reserved = torch.cuda.max_memory_reserved(device)

    # Plot memory timeline as stacked data
    fig = plt.figure(figsize=figsize, dpi=80)
    axes = fig.gca()
    for category, color in _CATEGORY_TO_COLORS.items():
        i = _CATEGORY_TO_INDEX[category]
        axes.fill_between(
            times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
        )
    fig.legend(["Unknown" if i is None else i.name for i in _CATEGORY_TO_COLORS])
    # Usually training steps are in magnitude of ms.
    axes.set_xlabel("Time (ms)")
    axes.set_ylabel("Memory (GB)")
    title = "\n\n".join(
        ([title] if title else [])
        + [
            f"Max memory allocated: {max_memory_allocated/(1024**3):.2f} GiB \n"
            f"Max memory reserved: {max_memory_reserved/(1024**3):.2f} GiB"
        ]
    )
    axes.set_title(title)

    # Embed the memory timeline image into the HTML file
    if file_path is not None and save_file:
        fig.savefig(file_path, format="svg")

    return fig

