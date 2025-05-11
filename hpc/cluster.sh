#!/bin/bash
#SBATCH --job-name=coref-resolver
#SBATCH --output=joaa_cluster_output_%j.log
#SBATCH --error=joaa_cluster_error_%j.log
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohimenul.joaa@gmail.com

module load CUDA/12.4
module load Anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
source ~/.bashrc

conda activate /data/cat/ws/afjo837h-conda_vm/py3129june5

JOBID="$SLURM_JOB_ID"

# Launch 5 tasks with delays
srun --exclusive -N1 -n1 /data/cat/ws/afjo837h-conda_vm/py3129june5/bin/python batch_with_gpu_cluster.py --job-id "$JOBID" --task-id 0 &
sleep 4m

srun --exclusive -N1 -n1 /data/cat/ws/afjo837h-conda_vm/py3129june5/bin/python batch_with_gpu_cluster.py --job-id "$JOBID" --task-id 1 &
sleep 7m

srun --exclusive -N1 -n1 /data/cat/ws/afjo837h-conda_vm/py3129june5/bin/python batch_with_gpu_cluster.py --job-id "$JOBID" --task-id 2 &
sleep 11m

srun --exclusive -N1 -n1 /data/cat/ws/afjo837h-conda_vm/py3129june5/bin/python batch_with_gpu_cluster.py --job-id "$JOBID" --task-id 3 &
sleep 15m

srun --exclusive -N1 -n1 /data/cat/ws/afjo837h-conda_vm/py3129june5/bin/python batch_with_gpu_cluster.py --job-id "$JOBID" --task-id 4 &

wait
