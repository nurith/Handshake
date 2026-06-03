#!/bin/bash
#SBATCH --job-name=wt_nonred
#SBATCH --output=pt5_dbd%j.log        # stdout+stderr → pt5_<jobid>.log
#SBATCH --error=pt5_dbd%j.err
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --cpus-per-task=32         # your 12 workers
#SBATCH --mem=100G                  # ProtT5 is large — adjust if needed
#SBATCH --time=24:00:00            # max walltime (HH:MM:SS)
#SBATCH --partition=xxx  # change to your cluster's GPU partition name
#SBATCH --account=xxx # change to your account name

# ── Environment ───────────────────────────────────────────────
module purge
module load python                 # or: module load anaconda3 / miniconda
module load anaconda3-2021.05
# If you use a conda env:
# conda activate your_env_name

# If you use a venv: Use requirements.txt
source Bert/bert/bin/activate

# ── Diagnostics (helps debug GPU detection issues) ─────────────
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ── Run ────────────────────────────────────────────────────────
cd ~/Bert/Fine-Tuning/
# Stronger — contacts are primary signal for attention
  
    # 70% — run with identical hyperparameters
python3 'PT5_train_crossattn_submatrix_supervised.py' \
   --csv ~/Bert/Fine-Tuning/splits_bind/pairs_30_uniq.csv \
    --class-weights 1.0 3.0 --label-smoothing 0.05  \
    --contact-lambda 1.0 --attn-supervision-lambda 0.5  \
    --pos-weight-cap 10.0 --lora-rank 8 --epochs 12  \
    --probs-out ./run_30_uniq --checkpoint-dir ./checkpoints_30 \
    --contact-out ./run_30_uniq/contacts.pkl \
    --model-out ./run_30_uniq/model.pth 
