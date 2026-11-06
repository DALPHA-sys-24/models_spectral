#!/bin/bash
#SBATCH --job-name=spectral_train
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-14
#SBATCH --output=Outputs/train_%A_%a.out
#SBATCH --error=Logs/train_%A_%a.err

# =============================================================================
# ENTRAINEMENT : une tâche = un dataset x une graine
# 3 datasets × 5 graines = 15 tâches -> --array=0-14
# À lancer depuis models_spectral/ :  sbatch experiments/slurm/train.sh
# Une tâche relancée reprend son run à la dernière epoch terminée (ou le saute s'il est fini).
# =============================================================================
DATASETS=("mnist" "fashion_mnist" "cifar10")
N_RUNS=5                     # graines SEED, SEED+1, ..., SEED+N_RUNS-1

N_DATASETS=${#DATASETS[@]}
# N_TOTAL = N_DATASETS × N_RUNS (valeur actuelle : 15 → --array=0-14)

# =============================================================================
# HYPERPARAMETRES (identiques pour tous les datasets)
# =============================================================================
ARCH="spectral_resnet20"
SPECTRAL_CONFIG="configs/spectral_config.json"
EPOCHS=50
BATCH_SIZE=128
LEARNING_RATE=0.1
SEED=42
VERBOSE=2
OUT_DIR="runs"
TAG="${ARCH}_${EPOCHS}ep_x${N_RUNS}"   # nom des groupes : runs/<dataset>_<TAG>/seed<k>/

mkdir -p Outputs Logs

# =============================================================================
# CALCUL DES INDICES
# =============================================================================
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / N_RUNS))
RUN_IDX=$((SLURM_ARRAY_TASK_ID % N_RUNS))
DATASET=${DATASETS[$DATASET_IDX]}
GROUP="${DATASET}_${TAG}"

# Epochs propres à un dataset (petits datasets)
if [[ "$DATASET" == "mnist1d" ]]; then
    EPOCHS=100
fi

# =============================================================================
# CHARGEMENT DES MODULES (modules de Lyra, sans environnement virtuel)
# =============================================================================
module --force purge
ml releases/2023a
ml tis/2025
ml CUDA/12.1.1
ml Python/3.11.3-GCCcore-12.3.0
ml SciPy-bundle/2023.07-gfbf-2023a
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
ml matplotlib/3.7.2
ml Seaborn/0.13.2-gfbf-2023a
ml scikit-learn/1.3.1-gfbf-2023a

# =============================================================================
# DEBUG INFO
# =============================================================================
echo "=========================================="
echo "SPECTRAL : ENTRAINEMENT"
echo "=========================================="
echo "Date: $(date)"
echo "Job: $SLURM_ARRAY_TASK_ID / $((N_DATASETS * N_RUNS - 1))"
echo "Dataset: $DATASET"
echo "Graine: $((SEED + RUN_IDX)) (run $((RUN_IDX + 1))/$N_RUNS)"
echo "Groupe: $OUT_DIR/$GROUP"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="

# =============================================================================
# EXECUTION
# =============================================================================
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=2

# --shard k/N : cette tâche n'entraîne que la graine SEED+k du groupe
python run_experiment.py \
    --dataset "$DATASET" \
    --arch "$ARCH" \
    --spectral_config "$SPECTRAL_CONFIG" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --seed "$SEED" \
    --n_runs "$N_RUNS" \
    --shard "$RUN_IDX/$N_RUNS" \
    --out_dir "$OUT_DIR" \
    --run_name "$GROUP" \
    --verbose "$VERBOSE"

# =============================================================================
# RESULTAT
# =============================================================================
exit_code=$?
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: $DATASET | graine $((SEED + RUN_IDX))"
else
    echo "FAILED: $DATASET | graine $((SEED + RUN_IDX)) (code $exit_code)"
fi
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
