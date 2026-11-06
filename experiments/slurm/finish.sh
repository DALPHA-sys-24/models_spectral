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
# Réunit les résultats des tâches (summary.csv) et trace, dans runs_pruning/figures/ :
#   <groupe>_<méthode>       précision selon p, pre et post sur la même figure (s'ils existent)
#   <groupe>_<mode>_methods  précision selon p des méthodes faites (gradient, element, mixte)
# plus les courbes d'entraînement et les masques.
# À lancer depuis models_spectral/ :  sbatch experiments/slurm/finish.sh
# (METHOD=element sbatch ... : une seule méthode ; défaut : toutes celles qui existent)
# =============================================================================
DATASETS=("mnist" "fashion_mnist" "cifar10")
N_RUNS=5
METHOD=${METHOD:-all}         # "gradient", "element", "mixte" ou "all"
CHANCE=0.1                   # 1 / nombre de classes

# Doivent être ceux de train.sh et prune.sh
ARCH="spectral_resnet20"
EPOCHS=50
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
ml tis/2025
ml CUDA/12.1.1
ml Python/3.11.3-GCCcore-12.3.0
ml SciPy-bundle/2023.07-gfbf-2023a
ml TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
ml matplotlib/3.7.2
ml Seaborn/0.13.2-gfbf-2023a
ml scikit-learn/1.3.1-gfbf-2023a
export TF_CPP_MIN_LOG_LEVEL=2

echo "=========================================="
echo "SPECTRAL : RESUMES ET FIGURES"
echo "Date: $(date)"
echo "Dataset: $DATASET   Groupe: $OUT_DIR/$GROUP"
echo "=========================================="

[[ "$METHOD" == "all" ]] && METHODS=("gradient" "element" "mixte") || METHODS=("$METHOD")
declare -A LABEL=([gradient]="Gradient" [element]="Element" [mixte]="Mixed")

exit_code=0
python plot_results.py history "$OUT_DIR/$GROUP"/seed* || exit_code=1

# Par méthode : résumé de chaque mode, puis pre et post sur une même figure
for M in "${METHODS[@]}"; do
    DIRS=(); LABELS=()
    for MODE in pre post; do
        DIR="$PRUNING_DIR/${GROUP}_${MODE}_${M}"
        [ -f "$DIR/config.json" ] || continue
        python pruning.py --resume "$DIR" --finish || exit_code=1
        DIRS+=("$DIR")
        [ "$MODE" == "pre" ] && LABELS+=("Pre-training") || LABELS+=("Post-training")
    done
    if [ ${#DIRS[@]} -gt 0 ]; then
        python plot_results.py pruning "${DIRS[@]}" --labels "${LABELS[@]}" --chance "$CHANCE" \
            --out "$PRUNING_DIR/figures/${GROUP}_${M}" || exit_code=1
        python plot_results.py masks "${DIRS[0]}" --p 0.2 0.5 0.8 || exit_code=1
    fi
done

# Par mode : les méthodes faites sur une même figure (orange, bleu, vert)
for MODE in pre post; do
    DIRS=(); LABELS=()
    for M in "${METHODS[@]}"; do
        DIR="$PRUNING_DIR/${GROUP}_${MODE}_${M}"
        [ -f "$DIR/summary.csv" ] || continue
        DIRS+=("$DIR"); LABELS+=("${LABEL[$M]}")
    done
    if [ ${#DIRS[@]} -gt 1 ]; then
        python plot_results.py pruning "${DIRS[@]}" --modes "$MODE" --labels "${LABELS[@]}" \
            --chance "$CHANCE" --out "$PRUNING_DIR/figures/${GROUP}_${MODE}_methods" || exit_code=1
    fi
done

echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: $DATASET | figures dans $OUT_DIR/$GROUP/figures et $PRUNING_DIR/figures"
else
    echo "FAILED: $DATASET (code $exit_code)"
fi
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
