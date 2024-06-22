# Masters-thesis
Code for my master's thesis.

## Setup


## Note on reproducibility
The code should be able to run as is, however in the jobscripts you might need to change some of the paths to be certain that everything is in order. If the jobscript is correct, the Python code should be able to run without any issues.
Things to take into account:
- Ensure that the environment is correctly set up.
- Ensure that you have set up Wandb.
- `source $HOME/miniconda3/bin/activate` is needed at DTU HPC in order to enable Miniconda in the job
- `export OUTPUT_DIR_MSC=/work3/s184399/msc` is needed to set the output directory. This is where everything will be outputted to (logs, checkpoints, plots etc.)
- `cd $HOME/msc/src` to finally CD into the directory with the code.


## Experiments
For the experiments, I will (mostly) be listing the jobscripts used for running them. Please read the note on reproducibility.

### Memory efficient fine-tuning
These jobscripts are in the `jobscripts/memory_efficiency` folder.

#### Experiment with a small 3-layer dense ANN
To run this experiment, simply run the `backprop_trick_time_small_ann.sh` job.

#### Experiments with OPT-125m and Llama-2
To run the experiments, use the following files:
- For the standard backprop + SGD: `finetune-opt-125m-backprop-test-False-sgd.sh`
- For the standard backprop + Adam: `finetune-opt-125m-backprop-test-False-adam.sh`
- For the memory efficient back propagation + SGD: `finetune-opt-125m-backprop-test-True-sgd.sh`
- For the memory efficient back propagation + Adam: `finetune-opt-125m-backprop-test-True-sgd.sh`
- For Adam + LoRA + 8 bit quantization: `finetune-opt-125m-LoRA-test-adam.sh`

The files are named similarly for the llama-versions.

### Truth representation
To download the datasets from "The Geometry of Truth", run `git clone https://github.com/saprmarks/geometry-of-truth.git` in the `src` directory.
Then proceed as follows:
- For the first experiment, you only need to put the datasets in a folder "Detection-datasets" in your Google drive. To run the experiment, run the file "ProbeVsModel.ipynb" on the "Notebooks" folder.
- For the second experiment, you need to put the datasets in a folder "Detection-datasets" and compute the activations using the second part of "Detection_dataset_preparation_+_compute_activations.ipynb". Then you can run "LinearVsNonlinearProbe.ipynb".
- The third experiment was run at DTU HPC. Here you need to put the dataset in a folder "ITI-datasets" and run the files from notebooks/iti_runs. The one in notebooks has a minor error.
