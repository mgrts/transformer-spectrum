"""
Unified CLI for transformer-spectrum experiments.

This module provides a unified command-line interface for:
- Training models (gradient-based, ES, GA)
- Generating datasets
- Running experiment sweeps
- Evaluating trained models

Usage:
    python -m transformer_spectrum.cli train --help
    python -m transformer_spectrum.cli generate --help
    python -m transformer_spectrum.cli sweep --help
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger

app = typer.Typer(
    name="transformer-spectrum",
    help="Transformer weight spectrum analysis toolkit",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


# =============================================================================
# Train Commands
# =============================================================================

@app.command("train")
def train(
    # Dataset
    dataset_path: Path = typer.Option(
        ...,
        "--dataset", "-d",
        help="Path to dataset .npy file",
        exists=True,
    ),
    # Model
    model_type: str = typer.Option(
        "transformer",
        "--model", "-m",
        help="Model type: transformer or lstm",
    ),
    embed_dim: int = typer.Option(64, "--embed-dim", help="Embedding dimension"),
    num_heads: int = typer.Option(4, "--num-heads", help="Number of attention heads"),
    num_layers: int = typer.Option(4, "--num-layers", help="Number of layers"),
    # Data
    sequence_length: int = typer.Option(300, "--seq-len", help="Total sequence length"),
    output_length: int = typer.Option(60, "--output-len", help="Prediction horizon"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    test_split: float = typer.Option(0.1, "--test-split", help="Validation split ratio"),
    # Training
    num_epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs"),
    learning_rate: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    early_stopping_patience: int = typer.Option(20, "--patience", help="Early stopping patience"),
    # Loss
    loss_type: str = typer.Option("mse", "--loss", "-l", help="Loss: mse, mae, cauchy, sgt"),
    sgt_sigma: float = typer.Option(1.0, "--sgt-sigma", help="SGT sigma parameter"),
    sgt_lambda: float = typer.Option(0.0, "--sgt-lambda", help="SGT lambda (skewness)"),
    sgt_q: float = typer.Option(2.0, "--sgt-q", help="SGT q parameter"),
    # Experiment
    experiment_name: str = typer.Option("experiment", "--name", "-n", help="MLflow experiment name"),
    seed: int = typer.Option(927, "--seed", "-s", help="Random seed"),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda, mps"),
    # Config file (overrides CLI args if provided)
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML/TOML config file",
        exists=True,
    ),
) -> None:
    """
    Train a model using gradient descent (Adam optimizer).

    Example:
        python -m transformer_spectrum.cli train -d data/processed/dataset.npy -n my_experiment
    """
    from transformer_spectrum.modeling.train import main as train_main

    if config:
        # Load from config file
        from transformer_spectrum.settings import ExperimentConfig
        if config.suffix in (".yaml", ".yml"):
            cfg = ExperimentConfig.from_yaml(config)
        elif config.suffix == ".toml":
            cfg = ExperimentConfig.from_toml(config)
        else:
            raise typer.BadParameter(f"Unsupported config format: {config.suffix}")

        # Override with CLI args if they differ from defaults
        # (This is a simplified approach - full implementation would check each arg)
        dataset_path = cfg.dataset_path or dataset_path
        experiment_name = cfg.name
        seed = cfg.seed

    logger.info(f"Starting training: {experiment_name}")

    train_main(
        dataset_path=dataset_path,
        model_type=model_type,
        sequence_length=sequence_length,
        output_length=output_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        test_split=test_split,
        seed=seed,
        early_stopping_patience=early_stopping_patience,
        loss_type=loss_type,
        sgt_loss_sigma=sgt_sigma,
        sgt_loss_lambda=sgt_lambda,
        sgt_loss_q=sgt_q,
        experiment_name=experiment_name,
    )


@app.command("train-es")
def train_es(
    # Dataset
    dataset_path: Path = typer.Option(
        ...,
        "--dataset", "-d",
        help="Path to dataset .npy file",
        exists=True,
    ),
    # Model
    model_type: str = typer.Option("transformer", "--model", "-m", help="Model type"),
    embed_dim: int = typer.Option(64, "--embed-dim", help="Embedding dimension"),
    num_heads: int = typer.Option(4, "--num-heads", help="Number of attention heads"),
    num_layers: int = typer.Option(4, "--num-layers", help="Number of layers"),
    # Data
    sequence_length: int = typer.Option(300, "--seq-len", help="Total sequence length"),
    output_length: int = typer.Option(60, "--output-len", help="Prediction horizon"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    test_split: float = typer.Option(0.1, "--test-split", help="Validation split ratio"),
    # Loss
    loss_type: str = typer.Option("mse", "--loss", "-l", help="Loss function type"),
    sgt_sigma: float = typer.Option(1.0, "--sgt-sigma", help="SGT sigma"),
    sgt_lambda: float = typer.Option(0.0, "--sgt-lambda", help="SGT lambda"),
    sgt_q: float = typer.Option(2.0, "--sgt-q", help="SGT q"),
    # ES parameters
    es_generations: int = typer.Option(150, "--generations", "-g", help="Number of ES generations"),
    es_popsize: int = typer.Option(256, "--popsize", help="Population size"),
    es_sigma: float = typer.Option(0.15, "--es-sigma", help="ES noise sigma"),
    es_lr: float = typer.Option(0.05, "--es-lr", help="ES learning rate"),
    # Experiment
    experiment_name: str = typer.Option("ES-experiment", "--name", "-n", help="Experiment name"),
    seed: int = typer.Option(927, "--seed", "-s", help="Random seed"),
) -> None:
    """
    Train a model using Evolution Strategy (ES) optimization.

    ES is a gradient-free optimization method that can explore
    the loss landscape differently than gradient descent.
    """
    from transformer_spectrum.modeling.train_es import main as train_es_main

    logger.info(f"Starting ES training: {experiment_name}")

    train_es_main(
        dataset_path=dataset_path,
        model_type=model_type,
        sequence_length=sequence_length,
        output_length=output_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        batch_size=batch_size,
        loss_type=loss_type,
        sgt_loss_sigma=sgt_sigma,
        sgt_loss_lambda=sgt_lambda,
        sgt_loss_q=sgt_q,
        test_split=test_split,
        seed=seed,
        experiment_name=experiment_name,
        es_generations=es_generations,
        es_popsize=es_popsize,
        es_sigma=es_sigma,
        es_lr=es_lr,
    )


@app.command("train-ga")
def train_ga(
    # Dataset
    dataset_path: Path = typer.Option(
        ...,
        "--dataset", "-d",
        help="Path to dataset .npy file",
        exists=True,
    ),
    # Model
    model_type: str = typer.Option("transformer", "--model", "-m", help="Model type"),
    embed_dim: int = typer.Option(64, "--embed-dim", help="Embedding dimension"),
    num_heads: int = typer.Option(4, "--num-heads", help="Number of attention heads"),
    num_layers: int = typer.Option(4, "--num-layers", help="Number of layers"),
    # Data
    sequence_length: int = typer.Option(300, "--seq-len", help="Total sequence length"),
    output_length: int = typer.Option(60, "--output-len", help="Prediction horizon"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    test_split: float = typer.Option(0.1, "--test-split", help="Validation split ratio"),
    # Loss
    loss_type: str = typer.Option("mae", "--loss", "-l", help="Loss function type"),
    sgt_sigma: float = typer.Option(1.0, "--sgt-sigma", help="SGT sigma"),
    sgt_lambda: float = typer.Option(0.0, "--sgt-lambda", help="SGT lambda"),
    sgt_q: float = typer.Option(2.0, "--sgt-q", help="SGT q"),
    # GA parameters
    ga_generations: int = typer.Option(100, "--generations", "-g", help="Number of GA generations"),
    ga_popsize: int = typer.Option(300, "--popsize", help="Population size"),
    ga_mutation_stdev: float = typer.Option(0.1, "--mutation-std", help="Mutation std dev"),
    ga_crossover_eta: float = typer.Option(10.0, "--crossover-eta", help="SBX crossover eta"),
    # Experiment
    experiment_name: str = typer.Option("GA-experiment", "--name", "-n", help="Experiment name"),
    seed: int = typer.Option(927, "--seed", "-s", help="Random seed"),
) -> None:
    """
    Train a model using Genetic Algorithm (GA) optimization.
    """
    from transformer_spectrum.modeling.train_ga import main as train_ga_main

    logger.info(f"Starting GA training: {experiment_name}")

    train_ga_main(
        dataset_path=dataset_path,
        model_type=model_type,
        sequence_length=sequence_length,
        output_length=output_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        batch_size=batch_size,
        loss_type=loss_type,
        sgt_loss_sigma=sgt_sigma,
        sgt_loss_lambda=sgt_lambda,
        sgt_loss_q=sgt_q,
        test_split=test_split,
        seed=seed,
        experiment_name=experiment_name,
        ga_generations=ga_generations,
        ga_popsize=ga_popsize,
        ga_mutation_stdev=ga_mutation_stdev,
        ga_crossover_eta=ga_crossover_eta,
    )


# =============================================================================
# Data Generation Commands
# =============================================================================

@app.command("generate-synthetic")
def generate_synthetic(
    output_path: Path = typer.Option(
        Path("data/processed/synthetic_dataset.npy"),
        "--output", "-o",
        help="Output path for generated dataset",
    ),
    n_sequences: int = typer.Option(1000, "--n-sequences", "-n", help="Number of sequences"),
    sequence_length: int = typer.Option(300, "--seq-len", help="Sequence length"),
    n_features: int = typer.Option(1, "--features", help="Number of features"),
    # SGT distribution parameters
    sigma: float = typer.Option(1.0, "--sigma", help="SGT scale parameter"),
    lam: float = typer.Option(0.0, "--lambda", help="SGT skewness parameter"),
    p: float = typer.Option(2.0, "--p", help="SGT shape parameter p"),
    q: float = typer.Option(2.0, "--q", help="SGT shape parameter q"),
    # Smoothing
    apply_smoothing: bool = typer.Option(True, "--smooth/--no-smooth", help="Apply smoothing"),
    kernel_size: int = typer.Option(99, "--kernel-size", help="Smoothing kernel size"),
    seed: int = typer.Option(927, "--seed", "-s", help="Random seed"),
) -> None:
    """
    Generate synthetic dataset with SGT-distributed noise.

    The data is generated from a Skewed Generalized T distribution
    and optionally smoothed with a combined cosine-Gaussian kernel.
    """
    from transformer_spectrum.data.synthetic.generate_data import main as generate_main

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic data: {output_path}")

    generate_main(
        output_path=output_path,
        n_sequences=n_sequences,
        sequence_length=sequence_length,
        n_features=n_features,
        sigma=sigma,
        lam=lam,
        p=p,
        q=q,
        apply_smoothing=apply_smoothing,
        kernel_size=kernel_size,
        seed=seed,
    )


@app.command("process-covid")
def process_covid(
    input_path: Path = typer.Option(
        Path("data/external/dataset.csv"),
        "--input", "-i",
        help="Path to raw COVID data CSV",
        exists=True,
    ),
    output_path: Path = typer.Option(
        Path("data/processed/dataset.npy"),
        "--output", "-o",
        help="Output path for processed dataset",
    ),
    sequence_length: int = typer.Option(300, "--seq-len", help="Sequence length"),
) -> None:
    """
    Process OWID COVID data into training format.
    """
    from transformer_spectrum.data.owid_covid.dataset import main as process_main

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing COVID data: {input_path} -> {output_path}")

    process_main(
        input_path=input_path,
        output_path=output_path,
        sequence_length=sequence_length,
    )


@app.command("download-covid")
def download_covid(
    output_path: Path = typer.Option(
        Path("data/external/dataset.csv"),
        "--output", "-o",
        help="Output path for downloaded data",
    ),
) -> None:
    """
    Download OWID COVID data.
    """
    from transformer_spectrum.data.owid_covid.load_data import download
    from transformer_spectrum.settings import DATA_URL

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading COVID data to: {output_path}")

    download(input_url=DATA_URL, output_path=output_path)


# =============================================================================
# Sweep / Batch Commands
# =============================================================================

@app.command("sweep")
def sweep(
    sweep_type: str = typer.Argument(
        ...,
        help="Sweep type: synthetic, covid, rvr",
    ),
    n_runs: int = typer.Option(5, "--runs", "-r", help="Number of runs per configuration"),
) -> None:
    """
    Run a pre-defined experiment sweep.

    Sweep types:
    - synthetic: Run experiments on synthetic data with various loss functions
    - covid: Run experiments on COVID data
    - rvr: Run experiments on RVR US hospitalization data
    """
    sweep_type = sweep_type.lower()

    if sweep_type == "synthetic":
        from transformer_spectrum.experiments.synthetic_data import main as sweep_main
        sweep_main(n_runs=n_runs)
    elif sweep_type == "covid":
        from transformer_spectrum.experiments.owid_covid_data import main as sweep_main
        sweep_main(n_runs=n_runs)
    elif sweep_type == "rvr":
        from transformer_spectrum.experiments.rvr_us_data import main as sweep_main
        sweep_main(n_runs=n_runs)
    else:
        raise typer.BadParameter(
            f"Unknown sweep type: {sweep_type}. Choose from: synthetic, covid, rvr"
        )


# =============================================================================
# Utility Commands
# =============================================================================

@app.command("info")
def info() -> None:
    """
    Show package and environment information.
    """
    import sys
    import torch
    import numpy as np

    from transformer_spectrum.settings import PROJ_ROOT

    typer.echo(f"Python: {sys.version}")
    typer.echo(f"PyTorch: {torch.__version__}")
    typer.echo(f"NumPy: {np.__version__}")
    typer.echo(f"Project root: {PROJ_ROOT}")
    typer.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if hasattr(torch.backends, "mps"):
        typer.echo(f"MPS available: {torch.backends.mps.is_available()}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
