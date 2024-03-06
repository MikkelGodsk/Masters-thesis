import os
import torch
from pickle import dump
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


class ProfilerCallback(TrainerCallback):
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

    For HuggingFace callbacks: https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerCallback
    """
    def __init__(self, logging_dir: str, profile_epochs: List[int]=[1], profile_n_steps:int=1, upload_to_wandb: bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_running = False
        self.upload_to_wandb = upload_to_wandb
        self.profile_epochs = profile_epochs
        self.profile_n_steps = profile_n_steps
        self.profile_dir = os.path.join(logging_dir, "profile")
        os.makedirs(self.profile_dir, exist_ok=True)
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
        if not self.is_running:
            print("\n\nStarting profiler\n\n")
            self.profiler.start()
            self.is_running = True
    
    def stop(self, epoch=-1):
        """
            Stop the profiler and save the data to disk. Is safe to call whenever. Does nothing if the profiler is not running.
        """
        if self.is_running:
            print("\n\nStopping profiler\n\n")
            self.profiler.stop()
            timestamp = get_timestamp()
            epoch_str = f"_{epoch}" if epoch>0 else ""
            self.profiler.export_chrome_trace(os.path.join(self.profile_dir, f"stack_trace{epoch_str}_{timestamp}.json"))
            self.profiler.export_memory_timeline(os.path.join(self.profile_dir, f"memory_timeline{epoch_str}_{timestamp}.json"))
            self.profiler.export_memory_timeline(os.path.join(self.profile_dir, f"memory_timeline{epoch_str}_{timestamp}.html"))
            self.is_running = False
            if self.upload_to_wandb:
                print("\nTaking a 5 second nap to let the file be written to disk before uploading to W&B")
                time.sleep(5)  # Wait for the file to be written to disk
                with open(os.path.join(self.profile_dir, f"memory_timeline{epoch_str}_{timestamp}.html"), "r") as f:
                    html_content = f.read()
                image = extract_png_from_html(html_content)
                wandb.log({f"Sample of memory timeline at epoch {epoch}": wandb.Image(image)})

    # Fyi, kwargs has keys: ['model', 'tokenizer', 'optimizer', 'lr_scheduler', 'train_dataloader', 'eval_dataloader']
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if floor(state.epoch) in self.profile_epochs:    # state.epoch is a float, not an int...
            self.start()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if floor(state.epoch) in self.profile_epochs:   # state.epoch is a float, not an int...
            self.profiler.step()
            if self.profiler.step_num > self.profile_n_steps-1:
                self.stop(epoch=state.epoch)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if floor(state.epoch)-1 in self.profile_epochs:   # state.epoch is a float, not an int... So we need to subtract 1 as it counts a bit up for every step...
            self.stop()  # Stop just in case it is still running

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.stop()


class MemoryHistoryCallback(TrainerCallback):   # Not sure if this one works.... Leaving it here as it might come in handy later...
    """
    To view the memory trace, go to https://pytorch.org/memory_viz and drag in the pickle file.
    """
    def __init__(self, *args, profile_epochs: List[int]=[1], profile_n_steps:int=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_epochs = profile_epochs
        self.profile_n_steps = profile_n_steps
        self.is_started = False
        self.step_counter = 0

    def start(self):
        if not self.is_started:
            torch.cuda.empty_cache()
            torch.cuda.memory._record_memory_history(enabled='all')
            self.is_started = True

    def stop(self):
        if self.is_started:
            timestamp = get_timestamp()
            file_prefix = f"memory_trace_{timestamp}"
            s = torch.cuda.memory._snapshot()
            with open(f"{file_prefix}.pickle", "wb") as f:
                dump(s, f)
            torch.cuda.memory._record_memory_history(enabled=None)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if floor(state.epoch) in self.profile_epochs:
            self.start()
            self.step_counter = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if floor(state.epoch) in self.profile_epochs:
            if self.step_counter > self.profile_n_steps-1:
                self.stop(epoch=state.epoch)
            self.step_counter += 1

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if floor(state.epoch)-1 in self.profile_epochs:
            self.stop()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.stop()