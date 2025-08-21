import json
import os
import re
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import typer
from loguru import logger

from transformer_spectrum.config import TRACKING_URI, SYNTHETIC_DATA_CONFIGS


app = typer.Typer(pretty_exceptions_show_locals=False)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", name).strip("_")


def _iter_artifacts(client: MlflowClient, run_id: str, path: str = ""):
    for artifact in client.list_artifacts(run_id, path):
        if artifact.is_dir:
            yield from _iter_artifacts(client, run_id, artifact.path)
        else:
            yield artifact.path


def _download_artifacts(
    client: MlflowClient,
    run_id: str,
    dest_dir: Path,
    include_svals: bool,
) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)

    def selector(path: str) -> bool:
        base = os.path.basename(path)
        if base.startswith("esd_") and base.endswith(".json"):
            return True
        if include_svals and base.startswith("svals_") and base.endswith(".json"):
            return True
        return False

    count = 0
    for art_path in _iter_artifacts(client, run_id):
        if selector(art_path):
            client.download_artifacts(run_id, art_path, str(dest_dir))
            count += 1
    return count


def _folder_name_from_run(run) -> str:
    """
    Build desired folder name from run params:
      - if loss_type == 'sgt': 'sgt_<sgt_loss_q>'
      - else: '<loss_type>'
      - fallback: 'unknown'
    """
    params = run.data.params or {}
    loss_type = (params.get("loss_type") or "").strip().lower()
    if loss_type == "sgt":
        q = (params.get("sgt_loss_q") or "").strip()
        name = f"{loss_type}_{q}" if q else loss_type
    else:
        name = loss_type or "unknown"
    return _safe_name(name)


def _unique_dir(base_dir: Path, name: str) -> Path:
    """
    Ensure unique directory path under base_dir for the given name.
    If the directory exists, append _dup1, _dup2, ... until free.
    """
    d = base_dir / name
    if not d.exists():
        return d
    i = 1
    while True:
        cand = base_dir / f"{name}_dup{i}"
        if not cand.exists():
            return cand
        i += 1


@app.command()
def main(
    output_root: Path = Path(TRACKING_URI / "collected_esd"),
    include_svals: bool = typer.Option(False, "--include-svals", help="Also collect svals_*.json files"),
    experiment: list[str] = typer.Option(
        None,
        "--experiment",
        "-e",
        help="Experiment name(s). Pass multiple --experiment to collect several.",
    ),
    n_runs: int = typer.Option(2, "--n-runs", "-n", help="If >0, expand names to <exp>_run_1..n"),
):
    mlflow.set_tracking_uri(TRACKING_URI)

    base_experiments = list(experiment) if experiment else [x["experiment_name"] for x in SYNTHETIC_DATA_CONFIGS]

    if n_runs and n_runs > 0:
        experiments = [f"{exp}_run_{i}" for exp in base_experiments for i in range(1, n_runs + 1)]
    else:
        experiments = base_experiments

    if not experiments:
        logger.warning("No experiments to collect. Exiting.")
        raise typer.Exit(code=0)

    logger.info(f"Collecting from {len(experiments)} experiment(s): {experiments}")

    client = MlflowClient()
    output_root.mkdir(parents=True, exist_ok=True)

    for exp_name in experiments:
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            logger.warning(f"Experiment not found: {exp_name}")
            continue

        exp_dir = output_root / _safe_name(exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)

        runs = client.search_runs(
            [exp.experiment_id],
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=5000,
            order_by=["attributes.start_time DESC"],
        )
        if not runs:
            logger.info(f"No runs in experiment: {exp_name}")
            continue

        for run in runs:
            # build folder name from run params (loss_type / sgt_q)
            base_folder_name = _folder_name_from_run(run)
            run_dir = _unique_dir(exp_dir, base_folder_name)
            run_dir.mkdir(parents=True, exist_ok=True)

            n_files = _download_artifacts(client, run.info.run_id, run_dir, include_svals)

            manifest = {
                "experiment_name": exp_name,
                "experiment_id": exp.experiment_id,
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName") or run.data.tags.get("run_name") or "unnamed",
                "loss_type": run.data.params.get("loss_type"),
                "sgt_loss_q": run.data.params.get("sgt_loss_q"),
                "artifact_count": n_files,
                "folder_name": run_dir.name,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            logger.info(f"{exp_name} -> {run_dir.name}: downloaded {n_files} JSON file(s)")


if __name__ == "__main__":
    app()
