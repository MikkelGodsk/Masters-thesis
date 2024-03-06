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
from typing import List
from math import floor

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def get_timestamp():
    return datetime.now().strftime(TIME_FORMAT_STR)

def extract_png_from_html(html_content: str):
    """
        Source: ChatGPT
    """
    import base64
    from PIL import Image
    import io
    # Extract base64 string from your HTML content
    # html_content is your HTML file content as a string
    start = html_content.find('base64,') + 7
    end = html_content.find('"', start)
    base64_image = html_content[start:end]

    # Decode and load the image
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    return image


class ProfilerCallbackBase(TrainerCallback):
    """
        Base class for the profilers. Since they have to be scheduled identically, it makes sense to have a base class for them.
        The subclasses are meant to be used like this:
        ```
        trainer = SFTTrainer(
            ...
            callbacks=[
                TorchProfilerCallback(logging_dir, profile_epochs=[1], profile_n_steps=5), 
                MemoryHistoryCallback(logging_dir, profile_epochs=[1], profile_n_steps=5),
                ...
            ],
        )
        ```
        I.e., they are scheduled to log the first `profile_n_steps` of each epoch in `profile_epochs`. This to ensure that the files to not become too large.

        NOTE: Be very careful that they do not log during the logging, if you are logging using model inference! This will make the files very large and slow to open!

        Docs: For HuggingFace callbacks, see https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
    """
    def __init__(self, logging_dir: str, *args, profile_epochs: List[int]=[1], profile_n_steps:int=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_epochs = profile_epochs
        self.profile_n_steps = profile_n_steps
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
            self.stop(epoch)
            self.is_running = False

    def step_(self):
        if self.is_running:
            self.step()

    def start(self):  # Should be overwritten
        raise NotImplementedError
    
    def step(self):  # May be overwritten
        pass
    
    def stop(self):  # Should be overwritten
        raise NotImplementedError

    # Fyi, kwargs has keys: ['model', 'tokenizer', 'optimizer', 'lr_scheduler', 'train_dataloader', 'eval_dataloader']
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if floor(state.epoch) in self.profile_epochs:    # state.epoch is a float, not an int...
            self.start_()
            self.step_counter = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if floor(state.epoch) in self.profile_epochs:   # state.epoch is a float, not an int...
            self.step_()
            if self.step_counter > self.profile_n_steps-1:
                self.stop_(epoch=state.epoch)
            self.step_counter += 1

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if floor(state.epoch)-1 in self.profile_epochs:   # state.epoch is a float, not an int... So we need to subtract 1 as it counts a bit up for every step...
            self.stop_()  # Stop just in case it is still running

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.stop_()  # Stop just in case it is still running


class TorchProfilerCallback(ProfilerCallbackBase):
    """
    This callback logs the PyTorch profiler data for a given epoch. It logs the first n steps of an epoch.
    NOTE: Be careful that this one does not log during the logging, if you are logging using model inference!

    To open the profiler stack trace data in Chrome, go to chrome://tracing and drag in the JSON file.
    To open the memory timeline, open the HTML file in a browser or view it in W&B.

    How to use with Hugging Face Trainer:
    ```
    trainer = SFTTrainer(
        ...
        callbacks=[
            ProfilerCallback(logging_dir, profile_epochs=[1], profile_n_steps=1, upload_to_wandb=True), 
        ],
    )
    ```

    Docs: https://pytorch.org/tutorials/recipes/recipes/profiler.html
    and https://pytorch.org/docs/stable/profiler.html
    """
    def __init__(self, logging_dir: str, profile_epochs: List[int]=[1], profile_n_steps:int=1, upload_to_wandb: bool=True, *args, **kwargs):
        super().__init__(logging_dir, *args, profile_epochs=profile_epochs, profile_n_steps=profile_n_steps, **kwargs)
        self.upload_to_wandb = upload_to_wandb
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
        print("\n\nStarting profiler\n\n")
        self.profiler.start()
    
    def step(self):
        self.profiler.step()

    def stop(self, epoch=-1):
        """
            Stop the profiler and save the data to disk. Is safe to call whenever. Does nothing if the profiler is not running.
        """
        print("\n\nStopping profiler\n\n")
        self.profiler.stop()
        timestamp = get_timestamp()
        epoch_str = f"_epoch-{epoch}" if epoch>0 else ""
        trace_file = os.path.join(self.profile_dir, f"stack_trace{epoch_str}_{timestamp}.json")
        memory_file = os.path.join(self.profile_dir, f"memory_timeline{epoch_str}_{timestamp}.json")
        memory_file_html = os.path.join(self.profile_dir, f"memory_timeline{epoch_str}_{timestamp}.html")
        self.profiler.export_chrome_trace(trace_file)
        self.profiler.export_memory_timeline(memory_file)
        self.profiler.export_memory_timeline(memory_file_html)
        print(f"\n\nProfiler data saved to {trace_file}\nand {memory_file}\nand {memory_file_html}\n")
        if self.upload_to_wandb:   # This does not always seem to work...
            print("\nTaking a 15 second nap to let the file be written to disk before uploading to W&B")
            time.sleep(15)  # Wait for the file to be written to disk
            with open(memory_file_html, "r") as f:
                html_content = f.read()
            image = extract_png_from_html(html_content)
            wandb.log({f"Sample of memory timeline at epoch {epoch}": wandb.Image(image)})



class MemoryHistoryCallback(ProfilerCallbackBase):
    """
    To view the memory trace, go to https://pytorch.org/memory_viz and drag in the pickle file.
    
    You might get this issue: 
    ```
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    ```
    I'm not sure what to do about it, so maybe this callback should only be used if you tend to run out of memory...

    Docs: https://pytorch.org/docs/stable/torch_cuda_memory.html
    and https://pytorch.org/blog/understanding-gpu-memory-1/
    """
    def __init__(self, logging_dir:str, *args, profile_epochs: List[int]=[1], profile_n_steps:int=1, **kwargs):
        super().__init__(logging_dir, *args, profile_epochs=profile_epochs, profile_n_steps=profile_n_steps, **kwargs)

    def start(self):
        print("\n\nStarting CUDA memory recorder\n\n")
        torch.cuda.empty_cache()
        torch.cuda.memory._record_memory_history(enabled='all')

    def stop(self, epoch=-1):
        print("\n\nStopping CUDA memory recorder\n\n")
        timestamp = get_timestamp()
        epoch_str = f"_epoch-{epoch}" if epoch>0 else ""
        file_name = os.path.join(self.profile_dir, f"memory_trace{epoch_str}_{timestamp}.pickle")
        s = torch.cuda.memory._snapshot()
        with open(file_name, "wb") as f:
            pickle.dump(s, f)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"\n\nCUDA memory recording saved to {file_name}\n")

