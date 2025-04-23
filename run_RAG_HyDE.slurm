#!/bin/bash
#SBATCH --job-name=RAG_HyDE_Run                  # Job name
#SBATCH --output=logs/RAG_HyDE_Run-%j.log        # Standard output log
#SBATCH --error=logs/RAG_HyDE_Run-%j.err         # Error log
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Number of tasks
#SBATCH --gres=gpu:1                             # Request 1 GPU
#SBATCH --mem=32G                                # Memory allocation
#SBATCH --time=08:00:00                          # Maximum runtime
#SBATCH --mail-type=ALL                          # Email notifications (BEGIN, END, FAIL)
#SBATCH --mail-user=ramy_magdy_lamei.boulos@mailbox.tu-dresden.de  # Your email
#SBATCH --account=p_scads_llm_secrets            # SLURM account

# Activate your virtual environment
source /home/rabo074f/venv_rag/bin/activate

# Navigate to your working directory
cd /home/rabo074f/venv_rag/NoobRAG

# Run the RAG script
python RAG_HyDE_Draft02.py
