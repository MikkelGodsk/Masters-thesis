#!/bin/sh

### GPU and CPU related parameters for a simple GPU job script
### select a GPU queue
#BSUB -q gpua100
### request the number of GPUs
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
### request the number of CPU cores (at least 4x the number of GPUs)
#BSUB -n 4
### we want to have this on a single node
#BSUB -R "span[hosts=1]"
### we need to request CPU memory, too (note: this is per CPU core)
#BSUB -R "rusage[mem=20GB]"

### -- set the job Name --
#BSUB -J Finetune_llama-2-7b-lima
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00

### Set start time
###BSUB -b 21:20

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
export OUTPUT_DIR_MSC=/work3/s184399/msc   # Then use os.getenv("OUTPUT_DIR") in the script
export CUDA_LAUNCH_BLOCKING=1
cd $HOME/msc/src

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python finetune.py --model "meta-llama/Llama-2-7b-hf" --num_epochs 15 --max_seq_length 2048 --tf32 True --backprop_trick True --optimizer adam