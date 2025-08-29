import json
import os
import re
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import typer
from loguru import logger
from typing import Iterable, Dict, Optional

from transformer_spectrum.config import TRACKING_URI, SYNTHETIC_DATA_CONFIGS, TRACK_EPOCHS


app = typer.Typer(pretty_exceptions_show_locals=False)


def _values_at_steps(
    client: MlflowClient,
    run_id: str,
    metric_key: str,
    steps: Iterable[int],
) -> Dict[int, Optional[float]]:
    """
    Return metric values for the given steps (epochs). If a step is missing, value is None.
    Requires metrics to have been logged with `step=epoch`.
    """
    wanted = set(int(s) for s in steps)
    out: Dict[int, Optional[float]] = {s: None for s in wanted}

    # get full metric history (each item has .value, .timestamp, .step)
    # If you’re on a very old MLflow version with pagination limits,
    # swap to `get_metric_history_paginated`.
    history = client.get_metric_history(run_id, metric_key)

    # keep the latest value for each step (in case logged multiple times)
    for m in history:
        s = int(getattr(m, "step", 0) or 0)
        if s in wanted:
            out[s] = m.value

    return out


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
    n_runs: int = typer.Option(5, "--n-runs", "-n", help="If >0, expand names to <exp>_run_1..n"),
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

            epochs = TRACK_EPOCHS

            train_by_epoch = _values_at_steps(client, run.info.run_id, "train_loss", epochs)
            val_by_epoch = _values_at_steps(client, run.info.run_id, "val_loss", epochs)

            manifest = {
                "experiment_name": exp_name,
                "experiment_id": exp.experiment_id,
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName") or run.data.tags.get("run_name") or "unnamed",
                "loss_type": run.data.params.get("loss_type"),
                "sgt_loss_q": run.data.params.get("sgt_loss_q"),
                "epochs": list(epochs),
                "train_loss_at_epochs": {str(k): v for k, v in sorted(train_by_epoch.items())},
                "val_loss_at_epochs": {str(k): v for k, v in sorted(val_by_epoch.items())},
                "artifact_count": n_files,
                "folder_name": run_dir.name,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            logger.info(f"{exp_name} -> {run_dir.name}: downloaded {n_files} JSON file(s)")


if __name__ == "__main__":
    app()
