"""
Plan d'une série d'expériences, découpée en tâches indépendantes : à lancer sur un cluster Slurm
(tableaux de tâches, une tâche par GPU) ou sur sa machine (une ou plusieurs tâches à la fois).

Une série est décrite dans un fichier JSON (exemple : experiments/exemple.json) : datasets,
nombre de graines, recette d'entraînement, prunings (pré, post, méthodes), valeurs de p, et
réglages Slurm. Le plan découpe le travail en quatre étapes, chacune faite de tâches
indépendantes (une ligne de commande par tâche) :

    1 train     une tâche par (dataset, graine)
    2 prepare   une tâche par (dataset, pruning) : crée le dossier du pruning (rapide, sans GPU)
    3 prune     pré-training : une tâche par (dataset, pruning, run) ;
                post-training : une tâche par (dataset, pruning, run, p, répétition)
    4 finish    une tâche par dataset : résumés (summary.csv) et figures (sans GPU)

Chaque tâche reprend là où elle s'est arrêtée si on la relance (run interrompu repris, lignes de
pruning déjà faites sautées) : relancer une étape ne refait que ce qui manque.

    python plan_experiments.py experiments/exemple.json          # écrit jobs/<nom>/
    bash jobs/<nom>/prefetch.sh          # télécharge les datasets (sur le nœud de connexion)
    sbatch jobs/<nom>/soumission.sh    # cluster : soumet les 4 étapes, chaînées
    bash jobs/<nom>/run_local.sh 2       # machine locale : 2 tâches à la fois sur le GPU

Fichiers écrits dans jobs/<nom>/ : <étape>.tasks (une commande par ligne), run_task.sh (lance
une ou plusieurs tâches d'une étape), slurm_<étape>.sh (tableau de tâches Slurm), soumission.sh,
run_local.sh, prefetch.sh, plan.json (copie de la description) et logs/.
"""
import json
import math
import os
import shlex
import sys

from .. import PROJECT_ROOT
from .common import dataset_tag, parser

STAGES = ["train", "prepare", "prune", "finish"]
GPU_STAGES = {"train", "prune"}
DEFAULT_P = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEFAULT_SLURM = {
    "partition": "batch",          # Lyra : file "batch", une RTX 6000 Ada (48 Go) par nœud
    "gpus": 1,
    "cpus_per_task": 8,
    "mem": "32G",
    "time": {"train": "1-00:00:00", "prepare": "00:20:00", "prune": "12:00:00", "finish": "00:30:00"},
    "per_gpu": 1,                  # tâches lancées en même temps sur un GPU (petits modèles : 2-4)
    "account": "",
    # Lyra : Python des modules, puis l'environnement où requirements_gpu.txt est installé
    # (le module TensorFlow/2.15 de Lyra a Keras 2 ; ce projet demande TensorFlow 2.21 et Keras 3)
    "setup": ["module --force purge", "ml releases/2023a", "ml Python/3.11.3-GCCcore-12.3.0",
              "source ~/venvs/spectral/bin/activate", "export TF_GPU_ALLOCATOR=cuda_malloc_async",
              "export TF_CPP_MIN_LOG_LEVEL=2"],
}


def parse_args(argv=None):
    p = parser(__doc__)
    p.add_argument("plan", help="fichier JSON de la série (voir experiments/exemple.json)")
    p.add_argument("--jobs_dir", default="jobs", help="où écrire jobs/<nom>/")
    p.add_argument("--python", default="python", help="interpréteur dans les tâches")
    return p.parse_args(argv)


def cli_args(options):
    """{"epochs": 20, "nesterov": false, "p": [0, 0.5]} -> ["--epochs", "20", "--no-nesterov", ...]."""
    out = []
    for key, value in options.items():
        if isinstance(value, bool):
            out.append(f"--{key}" if value else f"--no-{key}")
        elif isinstance(value, (list, tuple)):
            out += [f"--{key}"] + [str(v) for v in value]
        elif value is not None:
            out += [f"--{key}", str(value)]
    return out


def command(python, script, *args):
    return " ".join([python, script] + [shlex.quote(str(a)) for a in args])


def build_tasks(plan, python):
    """Lignes de commande de chaque étape, et la description des dossiers produits."""
    name = plan["name"]
    n_runs, seed = int(plan.get("n_runs", 5)), int(plan.get("seed", 42))
    out_dir, prune_dir = plan.get("out_dir", "runs"), plan.get("pruning_dir", "runs_pruning")
    p_values = plan.get("p", DEFAULT_P)
    prunings = plan.get("pruning", [{"mode": "pre"}])
    tasks = {s: [] for s in STAGES}
    layout = []
    for dataset in plan["datasets"]:
        tag = dataset_tag(dataset)
        group = f"{name}_{tag}"
        group_dir = os.path.join(out_dir, group)
        train = {**plan.get("train", {}), **plan.get("per_dataset", {}).get(dataset, {})}
        base = ["--dataset", dataset, "--n_runs", n_runs, "--seed", seed, "--out_dir", out_dir,
                "--run_name", group] + cli_args(train)
        for k in range(n_runs):
            tasks["train"].append(command(python, "run_experiment.py", *base, "--shard", f"{k}/{n_runs}"))
        run_dirs = [os.path.join(group_dir, f"seed{seed + k}") for k in range(n_runs)]

        pruned = []
        for pr in prunings:
            pr = dict(pr)
            mode, method = pr.get("mode", "pre"), pr.get("method", "element")
            label = pr.pop("name", f"{mode}_{method}")
            folder = f"{group}_{label}"
            opts = {"mode": mode, "method": method, "p": p_values, **{k: v for k, v in pr.items()
                                                                       if k not in ("mode", "method")}}
            tasks["prepare"].append(command(python, "pruning.py", *run_dirs, "--out_dir", prune_dir,
                                            "--name", folder, "--prepare", *cli_args(opts)))
            repeats = int(pr.get("repeats", 1))
            n_pre = n_runs if mode in ("pre", "both") else 0
            n_post = n_runs * len(p_values) * repeats if mode in ("post", "both") else 0
            n = max(n_pre, n_post)  # "both" : les unités post couvrent aussi les runs du pré
            path = os.path.join(prune_dir, folder)
            for k in range(n):
                tasks["prune"].append(command(python, "pruning.py", "--resume", path, "--shard", f"{k}/{n}"))
            pruned.append((path, mode, method, label))

        finish = [command(python, "pruning.py", "--resume", path, "--finish") for path, *_ in pruned]
        finish.append(command(python, "plot_results.py", "history", *run_dirs, "--formats", "pdf", "png"))
        chance = plan.get("chance", {}).get(dataset) if isinstance(plan.get("chance"), dict) else plan.get("chance")
        extra = ["--chance", chance] if chance else []
        for method in sorted({m for _, _, m, _ in pruned}):  # pré et post d'une méthode ensemble
            same = [(path, mode, label) for path, mode, m, label in pruned if m == method]
            labels = [lab for _, mode, _ in same
                      for lab in (["Pre-training", "Post-training"] if mode == "both" else
                                  ["Pre-training" if mode == "pre" else "Post-training"])]
            finish.append(command(python, "plot_results.py", "pruning", *[p for p, _, _ in same],
                                  "--labels", *labels, *extra,
                                  "--out", os.path.join(prune_dir, "figures", f"{group}_{method}")))
        tasks["finish"].append(" && ".join(finish))
        layout.append({"dataset": dataset, "runs": group_dir, "pruning": [p for p, *_ in pruned]})
    return tasks, layout


RUN_TASK = """#!/usr/bin/env bash
# Lance la part INDEX d'une étape : les tâches INDEX*PER+1 ... INDEX*PER+PER du fichier
# <étape>.tasks, en même temps (PER > 1 : plusieurs tâches sur le même GPU). Échoue si une échoue.
#   bash run_task.sh <étape> <INDEX> [PER]
set -uo pipefail
here={here}
stage=$1; index=$2; per=${{3:-1}}
cd {root}
mkdir -p "$here/logs"
pids=(); names=()
for i in $(seq 1 "$per"); do
    n=$((index * per + i))
    line=$(sed -n "${{n}}p" "$here/$stage.tasks")
    [ -z "$line" ] && continue
    echo "[$(date '+%F %T')] $stage tâche $n : $line"
    bash -c "$line" > "$here/logs/${{stage}}_$n.log" 2>&1 &
    pids+=($!); names+=("$n")
done
status=0
for j in ${{!pids[@]}}; do
    if wait "${{pids[$j]}}"; then
        echo "[$(date '+%F %T')] SUCCESS : $stage tâche ${{names[$j]}}"
    else
        code=$?; status=1
        echo "[$(date '+%F %T')] FAILED  : $stage tâche ${{names[$j]}} (code $code), journal : $here/logs/${{stage}}_${{names[$j]}}.log"
    fi
done
exit $status
"""

SLURM = """#!/usr/bin/env bash
#SBATCH --job-name={name}_{stage}
#SBATCH --array=0-{last}
#SBATCH --time={time}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
{gpu}{partition}{account}#SBATCH --output={here}/logs/slurm_{stage}_%A_%a.out
#SBATCH --error={here}/logs/slurm_{stage}_%A_%a.err
# Étape {stage} : {n} tâches, {per} par job ({jobs} jobs). Relancer ce script ne refait que ce qui manque.
# Les commandes des tâches sont dans {here}/{stage}.tasks (une par ligne).
{setup}

echo "=========================================="
echo "{name} : étape {stage}, job $SLURM_ARRAY_TASK_ID / {last}"
echo "Date : $(date)   Nœud : $(hostname)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "Pas de GPU"
echo "=========================================="
bash {here}/run_task.sh {stage} "$SLURM_ARRAY_TASK_ID" {per}
exit_code=$?
echo "Fini : $(date) (code $exit_code)"
exit $exit_code
"""

SUBMIT = """#!/usr/bin/env bash
#SBATCH --job-name={name}_soumission
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
{partition}{account}#SBATCH --output={here}/logs/soumission_%j.out
# Soumet les étapes à Slurm, chacune après la réussite de la précédente (afterok).
#   sbatch soumission.sh             # toutes les étapes (petit job qui soumet les autres)
#   sbatch soumission.sh prune       # à partir de l'étape prune (les précédentes sont faites)
#   bash soumission.sh               # même chose, directement depuis le nœud de connexion
set -euo pipefail
here={here}
from=${{1:-train}}
dep=""
started=0
for stage in {stages}; do
    [ "$stage" = "$from" ] && started=1
    [ $started -eq 0 ] && continue
    jid=$(sbatch --parsable $dep "$here/slurm_$stage.sh")
    echo "$stage : job $jid"
    dep="--dependency=afterok:$jid"
done
echo "Suivi : squeue -u $USER ; journaux : $here/logs/"
"""

LOCAL = """#!/usr/bin/env bash
# Toutes les étapes sur cette machine, J tâches à la fois (défaut 1) sur le GPU.
#   bash run_local.sh [J] [étape de départ]
set -euo pipefail
here={here}
jobs=${{1:-1}}
from=${{2:-train}}
started=0
for stage in {stages}; do
    [ "$stage" = "$from" ] && started=1
    [ $started -eq 0 ] && continue
    n=$(grep -c . "$here/$stage.tasks" || true)
    echo "== $stage : $n tâche(s), $jobs à la fois"
    seq 0 $((n - 1)) | xargs -P "$jobs" -I{{}} bash "$here/run_task.sh" "$stage" {{}} 1
done
echo "Fini. Journaux : $here/logs/"
"""


def main(argv=None):
    args = parse_args(argv)
    with open(args.plan) as f:
        plan = json.load(f)
    for key in ("name", "datasets"):
        if key not in plan:
            sys.exit(f"{args.plan} : clé \"{key}\" manquante (voir experiments/exemple.json)")
    slurm = {**DEFAULT_SLURM, **plan.get("slurm", {})}
    slurm["time"] = {**DEFAULT_SLURM["time"], **plan.get("slurm", {}).get("time", {})}
    tasks, layout = build_tasks(plan, args.python)

    here = os.path.abspath(os.path.join(args.jobs_dir, plan["name"]))
    os.makedirs(os.path.join(here, "logs"), exist_ok=True)
    q = shlex.quote

    def write(fname, text, executable=False):
        path = os.path.join(here, fname)
        with open(path, "w", newline="\n") as f:
            f.write(text)
        if executable:
            os.chmod(path, 0o755)

    for stage in STAGES:
        write(f"{stage}.tasks", "\n".join(tasks[stage]) + "\n")
        per = int(slurm["per_gpu"]) if stage in GPU_STAGES else 1
        n = len(tasks[stage])
        jobs = math.ceil(n / per)
        write(f"slurm_{stage}.sh", SLURM.format(
            name=plan["name"], stage=stage, last=jobs - 1, time=slurm["time"][stage],
            cpus=slurm["cpus_per_task"] if stage in GPU_STAGES else 2,
            mem=slurm["mem"] if stage in GPU_STAGES else "8G",
            gpu=f"#SBATCH --gres=gpu:{slurm['gpus']}\n" if stage in GPU_STAGES else "",
            partition=f"#SBATCH --partition={slurm['partition']}\n" if slurm["partition"] else "",
            account=f"#SBATCH --account={slurm['account']}\n" if slurm["account"] else "",
            here=here, n=n, per=per, jobs=jobs, setup="\n".join(slurm["setup"])), True)
    write("run_task.sh", RUN_TASK.format(here=q(here), root=q(PROJECT_ROOT)), True)
    write("soumission.sh", SUBMIT.format(
        name=plan["name"], here=q(here), stages=" ".join(STAGES),
        partition=f"#SBATCH --partition={slurm['partition']}\n" if slurm["partition"] else "",
        account=f"#SBATCH --account={slurm['account']}\n" if slurm["account"] else ""), True)
    write("run_local.sh", LOCAL.format(here=q(here), stages=" ".join(STAGES)), True)
    fetch = list(slurm["setup"]) + [f"cd {q(PROJECT_ROOT)}"] + [
        f"{args.python} -c {q(f'from specprune.data import load_dataset; load_dataset({d!r})')}"
        for d in plan["datasets"]]
    write("prefetch.sh", "#!/usr/bin/env bash\n# Télécharge les datasets une fois (les nœuds de "
          "calcul n'ont souvent pas internet ; plusieurs tâches ne doivent pas télécharger en même temps).\n"
          "set -eo pipefail\n" + "\n".join(fetch) + "\n", True)
    with open(os.path.join(here, "plan.json"), "w") as f:
        json.dump({**plan, "slurm": slurm, "layout": layout}, f, indent=2, ensure_ascii=False)

    print(f"Plan {plan['name']} : {here}")
    for stage in STAGES:
        per = int(slurm["per_gpu"]) if stage in GPU_STAGES else 1
        print(f"  {stage:<8} {len(tasks[stage]):>4} tâche(s)"
              + (f", {math.ceil(len(tasks[stage]) / per)} job(s) GPU" if stage in GPU_STAGES else ""))
    print(f"\n1. bash {here}/prefetch.sh            (datasets, une fois)\n"
          f"2. sbatch {here}/soumission.sh      (cluster)\n"
          f"   ou bash {here}/run_local.sh 1      (cette machine)")


if __name__ == "__main__":
    main()
