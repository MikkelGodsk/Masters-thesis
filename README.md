# Masters-thesis
Code for my master's thesis


## Note on reproducibility
The code should be able to run as is, however in the jobscripts you might need to change some of the paths to be certain that everything is in order. If the jobscript is correct, the Python code should be able to run without any issues.
Things to take into account:
- Ensure that the environment is correctly set up.
- Ensure that you have set up Wandb.
- `source $HOME/miniconda3/bin/activate` is needed at DTU HPC in order to enable Miniconda in the job
- `export OUTPUT_DIR_MSC=/work3/s184399/msc` is needed to set the output directory. This is where everything will be outputted to (logs, checkpoints, plots etc.)
- `cd $HOME/msc/src` to finally CD into the directory with the code.