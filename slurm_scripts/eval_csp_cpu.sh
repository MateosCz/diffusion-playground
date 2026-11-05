#!/bin/bash
#SBATCH --job-name=csp_eval_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=csp_eval_cpu-%j.out
#SBATCH --error=csp_eval_cpu-%j.err

set -euo pipefail

# Submit from the repository root. Uses the cluster's default partition;
# select a CPU partition with: sbatch -p PARTITION slurm_scripts/eval_csp_cpu.sh ...
cd "${SLURM_SUBMIT_DIR:-$PWD}"
checkpoint="${1:?Usage: sbatch [-p PARTITION] slurm_scripts/eval_csp_cpu.sh CHECKPOINT [evaluation arguments]}"
shift
if [[ "$checkpoint" == -* ]]; then
    echo "Expected checkpoint path as the first argument, got: $checkpoint" >&2
    echo "Usage: sbatch [-p PARTITION] slurm_scripts/eval_csp_cpu.sh CHECKPOINT [evaluation arguments]" >&2
    exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint (working directory: $PWD)" >&2
    exit 1
fi
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg

srun python -u -m src.litTrain.evalLitCSPFlowMap \
    --ckpt "$checkpoint" \
    --device cpu \
    --split test \
    --batch-size 16 \
    --n-steps 4 \
    --best-n 1 \
    "$@"
