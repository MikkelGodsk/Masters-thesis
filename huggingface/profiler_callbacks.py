import os
import torch
from pickle import dump
import html2image  # pip install --upgrade html2image
from datetime import datetime
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers import TrainerCallback
import wandb

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
    Sources: https://pytorch.org/tutorials/recipes/recipes/profiler.html
    and https://pytorch.org/docs/stable/profiler.html
    
    This callback logs the PyTorch profiler data to wandb.
    Profiles an entire epoch.
    """
    def __init__(self, logging_dir, profile_epoch=0, upload_to_wandb=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_dir = os.path.join(logging_dir, "profile")
        os.makedirs(self.profile_dir, exist_ok=True)
        self.profile_epoch = profile_epoch
        self.is_running = False
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
        if not self.is_running:
            print("\n\nStarting profiler\n\n")
            self.profiler.start()
            self.is_running = True
    
    def stop(self):
        """
            Stop the profiler and save the data to disk. Is safe to call whenever. Does nothing if the profiler is not running.
        """
        if self.is_running:
            print("\n\nStopping profiler\n\n")
            self.profiler.stop()
            timestamp = get_timestamp()
            self.profiler.export_chrome_trace(os.path.join(self.profile_dir, f"stack_trace_{timestamp}.json"))
            self.profiler.export_memory_timeline(os.path.join(self.profile_dir, f"memory_timeline_{timestamp}.html"))
            self.is_running = False
            if self.upload_to_wandb:
                with open(os.path.join(self.profile_dir, f"memory_timeline_{timestamp}.html"), "r") as f:
                    html_content = f.read()
                image = extract_png_from_html(html_content)
                wandb.log({"embedded_png": wandb.Image(image)})

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch == self.profile_epoch: 
            self.start()

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        if state.epoch == self.profile_epoch:
            self.profiler.step()

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if state.epoch == self.profile_epoch:
            self.stop()

    def on_train_end(self, args, state, control, **kwargs):
        self.stop()


class MemoryHistoryCallback(TrainerCallback):   # Not sure if this one works.... Leaving it here as it might come in handy later...
    def __init__(self, *args, profile_epoch=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_epoch = profile_epoch
        self.is_started = False

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

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch == self.profile_epoch: 
            self.start()

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        if state.epoch == self.profile_epoch:
            self.stop()

    def on_train_end(self, args, state, control, **kwargs):
        self.stop()