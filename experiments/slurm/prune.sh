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
# PRUNING (après train.sh). MODE et METHOD se choisissent à la soumission, sans éditer ce fichier :
#   post : une tâche = un dataset × un run × une valeur de p
#          3 datasets × 5 runs × 11 p = 165 tâches
#          sbatch experiments/slurm/prune.sh                          (MODE=post, METHOD=element)
#          METHOD=mixte sbatch experiments/slurm/prune.sh
#   pre  : une tâche = un dataset × un run, tous les p (rapide) et, par défaut, les trois
#          méthodes (gradient, element, mixte) ; 3 datasets × 5 runs = 15 tâches
#          MODE=pre sbatch --array=0-14 experiments/slurm/prune.sh   (METHOD=all)
#          MODE=pre METHOD=element sbatch --array=0-14 experiments/slurm/prune.sh
# --array : la ligne #SBATCH ci-dessus vaut pour post ; en pre, la donner à sbatch (elle
# remplace celle du fichier). Une tâche hors de la grille s'arrête avec la bonne commande.
# Relancer des tâches précises est possible (ex. --array=7,12) : chacune saute ce qui est fait.
# Ensuite : sbatch experiments/slurm/finish.sh
# =============================================================================
DATASETS=("mnist" "fashion_mnist" "cifar10")
N_RUNS=5
P_VALUES=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)

MODE=${MODE:-post}            # "pre" ou "post" (MODE=pre sbatch ...)
# "gradient", "element", "mixte" ou "all" (les trois, l'une après l'autre dans la même tâche)
# défaut : all en pre, element en post (en post, all triple la durée de chaque tâche)
if [[ -z "$METHOD" ]]; then
    [[ "$MODE" == "pre" ]] && METHOD="all" || METHOD="element"
fi
POST_EPOCHS=20               # post : epochs de réentraînement des phi (même lr que le run)

if [[ "$MODE" != "pre" && "$MODE" != "post" ]]; then
    echo "ERREUR : MODE='$MODE' (attendu : pre ou post)"; exit 1
fi
if [[ "$METHOD" == "all" ]]; then
    METHODS=("gradient" "element" "mixte")
elif [[ "$METHOD" == "gradient" || "$METHOD" == "element" || "$METHOD" == "mixte" ]]; then
    METHODS=("$METHOD")
else
    echo "ERREUR : METHOD='$METHOD' (attendu : gradient, element, mixte ou all)"; exit 1
fi

N_DATASETS=${#DATASETS[@]}
N_P=${#P_VALUES[@]}
if [[ "$MODE" == "post" ]]; then
    N_PER_DATASET=$((N_RUNS * N_P))     # (run, p)
else
    N_PER_DATASET=$N_RUNS               # run
fi
N_TOTAL=$((N_DATASETS * N_PER_DATASET))   # post : 165 → --array=0-164 ; pre : 15 → --array=0-14

# --array conforme à la grille ? (tâche hors grille, ou plage 0-k trop courte)
GOOD="MODE=$MODE METHOD=$METHOD sbatch --array=0-$((N_TOTAL - 1)) experiments/slurm/prune.sh"
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "ERREUR : tâche $SLURM_ARRAY_TASK_ID hors de la grille ($N_TOTAL tâches en $MODE : 0-$((N_TOTAL - 1)))."
    echo "Annuler ce tableau (scancel $SLURM_ARRAY_JOB_ID) et soumettre : $GOOD"
    exit 1
fi
if [[ "${SLURM_ARRAY_TASK_MIN:-0}" == 0 && "${SLURM_ARRAY_TASK_COUNT:-0}" == $((SLURM_ARRAY_TASK_MAX + 1)) ]] \
        && (( SLURM_ARRAY_TASK_MAX < N_TOTAL - 1 )); then
    echo "ERREUR : --array=0-$SLURM_ARRAY_TASK_MAX, mais la grille $MODE a $N_TOTAL tâches : une partie ne serait pas faite."
    echo "Soumettre : $GOOD"
    exit 1
fi

# Doivent être ceux de train.sh
ARCH="spectral_resnet20"
EPOCHS=50
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
if [[ "$MODE" == "post" ]]; then
    RUN_IDX=$((SHARD / N_P)); P=${P_VALUES[$((SHARD % N_P))]}
else
    RUN_IDX=$SHARD; P="tous"
fi
RUN_DIRS=()
for k in $(seq 0 $((N_RUNS - 1))); do RUN_DIRS+=("$OUT_DIR/$GROUP/seed$((SEED + k))"); done

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
echo "SPECTRAL : PRUNING $MODE (${METHODS[*]})"
echo "=========================================="
echo "Date: $(date)"
echo "Job: $SLURM_ARRAY_TASK_ID / $((N_DATASETS * N_PER_DATASET - 1))"
echo "Dataset: $DATASET"
echo "Run: seed$((SEED + RUN_IDX))   p: $P"
echo "Dossiers: $PRUNING_DIR/${GROUP}_${MODE}_<méthode>"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU"
echo "=========================================="

# =============================================================================
# EXECUTION
# =============================================================================
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=2

exit_code=0
for M in "${METHODS[@]}"; do
    NAME="${GROUP}_${MODE}_${M}"     # un dossier par méthode
    # Le dossier du pruning est créé une seule fois (verrou : les tâches démarrent en même temps)
    (
        flock 9
        if [ ! -f "$PRUNING_DIR/$NAME/config.json" ]; then
            python pruning.py "${RUN_DIRS[@]}" \
                --mode "$MODE" \
                --method "$M" \
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
        --shard "$SHARD/$N_PER_DATASET" || exit_code=1
done

# =============================================================================
# RESULTAT
# =============================================================================
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: $DATASET | $MODE ${METHODS[*]} | seed$((SEED + RUN_IDX)) | p=$P"
else
    echo "FAILED: $DATASET | $MODE ${METHODS[*]} | seed$((SEED + RUN_IDX)) | p=$P (code $exit_code)"
fi
echo "Finished: $(date)"
echo "=========================================="
exit $exit_code
