#!/usr/bin/env bash
# Série complète : N runs (graines) -> pruning pré et post-training -> figures.
# Chaque étape reprend là où elle s'est arrêtée : relancer le script après une interruption.
#
#   bash experiments/serie.sh                                   # réglages ci-dessous
#   DATASET=mnist N_RUNS=3 EPOCHS=30 bash experiments/serie.sh  # réglages changés à la volée
#   GPUS="0 1" bash experiments/serie.sh                        # post-training sur deux GPU
#
# Sorties : runs/$NAME/seed*/  runs_pruning/$NAME_{pre,post}/  et leurs figures/ (PDF et PNG).
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-cifar10}
ARCH=${ARCH:-spectral_resnet20}
SPECTRAL_CONFIG=${SPECTRAL_CONFIG:-configs/spectral_config.json}
N_RUNS=${N_RUNS:-5}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-20}
LR=${LR:-0.1}
BATCH=${BATCH:-128}
P=${P:-"0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1"}
METHOD=${METHOD:-element}
POST_EPOCHS=${POST_EPOCHS:-20}
DEVICE=${DEVICE:-/GPU:0}
GPUS=${GPUS:-}
CHANCE=${CHANCE:-0.1}
NAME=${NAME:-${DATASET}_${ARCH}_${EPOCHS}ep_x${N_RUNS}}
EXTRA_TRAIN=${EXTRA_TRAIN:-}          # options de plus pour run_experiment.py

group=runs/$NAME
echo "== 1/4 entraînement : $group"
if [ -f "$group/group.json" ]; then
    python run_experiment.py --resume "$group" --device "$DEVICE"
else
    python run_experiment.py --dataset "$DATASET" --arch "$ARCH" --spectral_config "$SPECTRAL_CONFIG" \
        --epochs "$EPOCHS" --lr "$LR" --batch_size "$BATCH" --n_runs "$N_RUNS" --seed "$SEED" \
        --run_name "$NAME" --device "$DEVICE" $EXTRA_TRAIN
fi

prune() {  # $1 = pre ou post
    local out=runs_pruning/${NAME}_$1
    if [ -f "$out/config.json" ]; then
        python pruning.py --resume "$out" --device "$DEVICE" ${GPUS:+--gpus $GPUS}
    else
        python pruning.py "$group"/seed* --mode "$1" --p $P --method "$METHOD" --epochs "$POST_EPOCHS" \
            --out_dir runs_pruning --name "${NAME}_$1" --device "$DEVICE" ${GPUS:+--gpus $GPUS}
    fi
}
echo "== 2/4 pruning pré-training";  prune pre
echo "== 3/4 pruning post-training"; prune post

echo "== 4/4 figures"
python plot_results.py history "$group"/seed*
python plot_results.py pruning "runs_pruning/${NAME}_pre" "runs_pruning/${NAME}_post" \
    --modes pre post --labels Pre-training Post-training --chance "$CHANCE" \
    --out "runs_pruning/${NAME}_post/figures/accuracy_vs_p"
python plot_results.py masks "runs_pruning/${NAME}_pre" --p 0.2 0.5 0.8
echo "Fini : $group/figures, runs_pruning/${NAME}_{pre,post}/figures"
