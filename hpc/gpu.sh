#!/bin/bash
#SBATCH --job-name=coref-resolver
#SBATCH --output=joaa_gpu_output_%j.log
#SBATCH --error=joaa_gpu_error_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohimenul.joaa@gmail.com

module load CUDA/12.4  # CUDA version compatible with torch 2.5.1
module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
source ~/.bashrc

#conda env list
#conda list
#which python

conda activate /data/cat/ws/afjo837h-conda_vm/py3129june5

JOBID="$SLURM_JOB_ID"

# Run your script
/data/cat/ws/afjo837h-conda_vm/py3129june5/bin/python batch_with_gpu.py --job-id "$JOBID"
