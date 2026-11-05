#!/bin/bash
#SBATCH --job-name=csp_eval
#SBATCH --partition=titans
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=csp_eval-%j.out
#SBATCH --error=csp_eval-%j.err

set -euo pipefail

# Submit from the repository root; remaining arguments override eval defaults.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
checkpoint="${1:?Usage: sbatch slurm_scripts/eval_csp.sh CHECKPOINT [evaluation arguments]}"
shift
if [[ "$checkpoint" == -* ]]; then
    echo "Expected checkpoint path as the first argument, got: $checkpoint" >&2
    echo "Usage: sbatch slurm_scripts/eval_csp.sh CHECKPOINT [evaluation arguments]" >&2
    exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint (working directory: $PWD)" >&2
    exit 1
fi
source .venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MPLBACKEND=Agg

srun python -u -m src.litTrain.evalLitCSPFlowMap \
    --ckpt "$checkpoint" \
    --device cuda \
    --split test \
    --batch-size 16 \
    --n-steps 4 \
    --best-n 1 \
    "$@"
