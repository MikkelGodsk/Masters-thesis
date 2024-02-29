#!/bin/sh

### GPU and CPU related parameters for a simple GPU job script
### select a GPU queue
#BSUB -q gpuv100
### request the number of GPUs
#BSUB -gpu "num=1:mode=exclusive_process"
###BSUB -R "select[gpu80gb]"
### request the number of CPU cores (at least 4x the number of GPUs)
#BSUB -n 4
### we want to have this on a single node
#BSUB -R "span[hosts=1]"
### we need to request CPU memory, too (note: this is per CPU core)
#BSUB -R "rusage[mem=20GB]"

### -- set the job Name --
#BSUB -J Finetune_opt-125m-lima
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 12:00

# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s184399@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

source $HOME/miniconda3/bin/activate

module load cuda/12.1
module load cudnn/v8.9.1.23-prod-cuda-12.X
conda activate msc
export CUDA_LAUNCH_BLOCKING=1

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $HOME/msc/huggingface
python finetune-opt-125m.py