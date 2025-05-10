#!/bin/bash
#SBATCH --job-name=coref-resolver
#SBATCH --output=joaa_coref_output_%j.log
#SBATCH --error=joaa_coref_error_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=02:00:00

module load CUDA/12.4  # CUDA version compatible with torch 2.5.1
module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
source ~/.bashrc

#conda env list
#conda list
#which python

conda activate /data/cat/ws/afjo837h-conda_vm/py3129june5

# Run your script
#/data/cat/ws/afjo837h-conda_vm/py3129june5/bin/python coref/coref_batch.py
#/data/cat/ws/afjo837h-conda_vm/py3129june5/bin/python topic/topic_batch.py
#/data/cat/ws/afjo837h-conda_vm/py3129june5/bin/python reranker/mxbai_reranker.py
/data/cat/ws/afjo837h-conda_vm/py3129june5/bin/python batch_with_gpu.py
