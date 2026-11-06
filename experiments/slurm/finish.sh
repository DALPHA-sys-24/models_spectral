#!/bin/bash
#SBATCH --job-name=spectral_finish
#SBATCH --partition=batch
#SBATCH --mem=8G
#SBATCH --time=0-00:30:00
#SBATCH --array=0-2
#SBATCH --output=Outputs/finish_%A_%a.out
#SBATCH --error=Logs/finish_%A_%a.err

# =============================================================================
# RESUMES ET FIGURES (après prune.sh, sans GPU) : une tâche = un dataset
# 3 datasets -> --array=0-2
# Réunit les résultats des tâches (summary.csv) et trace : courbes d'entraînement, précision
# selon p (pre et post sur la même figure s'ils existent), masques.
# À lancer depuis models_spectral/ :  sbatch experiments/slurm/finish.sh
# =============================================================================
DATASETS=("mnist" "fashion_mnist" "cifar10")
N_RUNS=5
METHOD="element"
CHANCE=0.1                   # 1 / nombre de classes

# Doivent être ceux de train.sh et prune.sh
ARCH="spectral_resnet20"
EPOCHS=20
OUT_DIR="runs"
PRUNING_DIR="runs_pruning"
TAG="${ARCH}_${EPOCHS}ep_x${N_RUNS}"

mkdir -p Outputs Logs

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
if [[ "$DATASET" == "mnist1d" ]]; then
    EPOCHS=100
    TAG="${ARCH}_${EPOCHS}ep_x${N_RUNS}"
fi
GROUP="${DATASET}_${TAG}"

module --force purge
ml releases/2023a
ml Python/3.11.3-GCCcore-12.3.0
source ~/venvs/spectral/bin/activate
export TF_CPP_MIN_LOG_LEVEL=2

echo "=========================================="
echo "SPECTRAL : RESUMES ET FIGURES"
echo "Date: $(date)"
echo "Dataset: $DATASET   Groupe: $OUT_DIR/$GROUP"
echo "=========================================="

exit_code=0
DIRS=(); LABELS=()
for MODE in pre post; do
    DIR="$PRUNING_DIR/${GROUP}_${MODE}_${METHOD}"
    [ -f "$DIR/config.json" ] || continue
    python pruning.py --resume "$DIR" --finish || exit_code=1
    DIRS+=("$DIR")
    [ "$MODE" == "pre" ] && LABELS+=("Pre-training") || LABELS+=("Post-training")
done

python plot_results.py history "$OUT_DIR/$GROUP"/seed* || exit_code=1
if [ ${#DIRS[@]} -gt 0 ]; then
    python plot_results.py pruning "${DIRS[@]}" --labels "${LABELS[@]}" --chance "$CHANCE" \
        --out "$PRUNING_DIR/figures/${GROUP}_${METHOD}" || exit_code=1
    python plot_results.py masks "${DIRS[0]}" --p 0.2 0.5 0.8 || exit_code=1
fi

echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: $DATASET | figures dans $OUT_DIR/$GROUP/figures et $PRUNING_DIR/figures"
else
    echo "FAILED: $DATASET (code $exit_code)"
fi
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
