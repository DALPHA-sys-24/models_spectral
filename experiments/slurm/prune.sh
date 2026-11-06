#!/bin/bash
#SBATCH --job-name=spectral_prune
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-164
#SBATCH --output=Outputs/prune_%A_%a.out
#SBATCH --error=Logs/prune_%A_%a.err

# =============================================================================
# PRUNING (après train.sh) : MODE="pre" ou "post"
#   post : une tâche = un dataset × un run × une valeur de p
#          3 datasets × 5 runs × 11 p = 165 tâches -> --array=0-164
#   pre  : une tâche = un dataset × un run (évaluer tous les p est rapide)
#          3 datasets × 5 runs = 15 tâches -> --array=0-14
# À lancer depuis models_spectral/ :  sbatch experiments/slurm/prune.sh
# Une tâche relancée saute ce qui est déjà fait. Ensuite : sbatch experiments/slurm/finish.sh
# =============================================================================
DATASETS=("mnist" "fashion_mnist" "cifar10")
N_RUNS=5
P_VALUES=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

MODE="post"                  # "pre" ou "post"
METHOD="element"             # "gradient", "element" ou "mixte"
POST_EPOCHS=20               # post : epochs de réentraînement des phi (même lr que le run)

N_DATASETS=${#DATASETS[@]}
N_P=${#P_VALUES[@]}
if [[ "$MODE" == "post" ]]; then
    N_PER_DATASET=$((N_RUNS * N_P))     # (run, p)
else
    N_PER_DATASET=$N_RUNS               # run
fi
# N_TOTAL = N_DATASETS × N_PER_DATASET (post : 165 → --array=0-164 ; pre : 15 → --array=0-14)

# Doivent être ceux de train.sh
ARCH="spectral_resnet20"
EPOCHS=20
SEED=42
OUT_DIR="runs"
PRUNING_DIR="runs_pruning"
TAG="${ARCH}_${EPOCHS}ep_x${N_RUNS}"

mkdir -p Outputs Logs "$PRUNING_DIR"

# =============================================================================
# CALCUL DES INDICES
# =============================================================================
DATASET_IDX=$((SLURM_ARRAY_TASK_ID / N_PER_DATASET))
SHARD=$((SLURM_ARRAY_TASK_ID % N_PER_DATASET))
DATASET=${DATASETS[$DATASET_IDX]}
if [[ "$DATASET" == "mnist1d" ]]; then
    EPOCHS=100
    TAG="${ARCH}_${EPOCHS}ep_x${N_RUNS}"
fi
GROUP="${DATASET}_${TAG}"
NAME="${GROUP}_${MODE}_${METHOD}"
if [[ "$MODE" == "post" ]]; then
    RUN_IDX=$((SHARD / N_P)); P=${P_VALUES[$((SHARD % N_P))]}
else
    RUN_IDX=$SHARD; P="tous"
fi
RUN_DIRS=()
for k in $(seq 0 $((N_RUNS - 1))); do RUN_DIRS+=("$OUT_DIR/$GROUP/seed$((SEED + k))"); done

# =============================================================================
# CHARGEMENT DES MODULES (environnement ~/venvs/spectral, voir README)
# =============================================================================
module --force purge
ml releases/2023a
ml Python/3.11.3-GCCcore-12.3.0
source ~/venvs/spectral/bin/activate

# =============================================================================
# DEBUG INFO
# =============================================================================
echo "=========================================="
echo "SPECTRAL : PRUNING $MODE ($METHOD)"
echo "=========================================="
echo "Date: $(date)"
echo "Job: $SLURM_ARRAY_TASK_ID / $((N_DATASETS * N_PER_DATASET - 1))"
echo "Dataset: $DATASET"
echo "Run: seed$((SEED + RUN_IDX))   p: $P"
echo "Dossier: $PRUNING_DIR/$NAME"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="

# =============================================================================
# EXECUTION
# =============================================================================
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=2

# Le dossier du pruning est créé une seule fois (verrou : les tâches démarrent en même temps)
(
    flock 9
    if [ ! -f "$PRUNING_DIR/$NAME/config.json" ]; then
        python pruning.py "${RUN_DIRS[@]}" \
            --mode "$MODE" \
            --method "$METHOD" \
            --p "${P_VALUES[@]}" \
            --epochs "$POST_EPOCHS" \
            --out_dir "$PRUNING_DIR" \
            --name "$NAME" \
            --prepare
    fi
) 9> "$PRUNING_DIR/.$NAME.lock"

# --shard k/N : cette tâche ne fait que sa part (un run en pre, un couple (run, p) en post)
python pruning.py \
    --resume "$PRUNING_DIR/$NAME" \
    --shard "$SHARD/$N_PER_DATASET"

# =============================================================================
# RESULTAT
# =============================================================================
exit_code=$?
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: $DATASET | $MODE $METHOD | seed$((SEED + RUN_IDX)) | p=$P"
else
    echo "FAILED: $DATASET | $MODE $METHOD | seed$((SEED + RUN_IDX)) | p=$P (code $exit_code)"
fi
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
