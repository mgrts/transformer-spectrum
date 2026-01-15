"""
Centralized configuration using Pydantic settings.

This module provides structured configuration for experiments, training,
and paths. Supports loading from environment variables and YAML/TOML files.
"""

from __future__ import annotations

import os
import random
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Path Configuration (derived from project root)
# =============================================================================

def _get_project_root() -> Path:
    """Get project root directory (parent of transformer_spectrum package)."""
    return Path(__file__).resolve().parents[1]


PROJ_ROOT = _get_project_root()
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TRACKING_URI = PROJ_ROOT / "mlruns"


# =============================================================================
# Enums
# =============================================================================

class LossType(str, Enum):
    """Supported loss function types."""
    MSE = "mse"
    MAE = "mae"
    CAUCHY = "cauchy"
    SGT = "sgt"


class ModelType(str, Enum):
    """Supported model architectures."""
    TRANSFORMER = "transformer"
    LSTM = "lstm"


class OptimizerType(str, Enum):
    """Supported optimizer types."""
    ADAM = "adam"
    SGD = "sgd"
    ES = "es"  # Evolution Strategy
    GA = "ga"  # Genetic Algorithm


# =============================================================================
# Configuration Models
# =============================================================================

class SGTLossConfig(BaseModel):
    """Configuration for SGT (Skewed Generalized T) loss function."""

    sigma: float = Field(default=1.0, gt=0, description="Scale parameter (must be > 0)")
    lam: float = Field(default=0.0, ge=-1.0, le=1.0, description="Skewness parameter in (-1, 1)")
    p: float = Field(default=2.0, gt=0, description="Shape parameter p (must be > 0)")
    q: float = Field(default=2.0, gt=0, description="Shape parameter q (must be > 0)")
    eps: float = Field(default=1e-6, ge=0, description="Epsilon for numerical stability")


class LossConfig(BaseModel):
    """Loss function configuration."""

    type: LossType = Field(default=LossType.MSE, description="Loss function type")
    sgt: SGTLossConfig = Field(default_factory=SGTLossConfig, description="SGT loss parameters")
    cauchy_gamma: float = Field(default=2.0, gt=0, description="Gamma parameter for Cauchy loss")


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    type: ModelType = Field(default=ModelType.TRANSFORMER, description="Model architecture")
    embed_dim: int = Field(default=64, gt=0, description="Embedding dimension")
    num_heads: int = Field(default=4, gt=0, description="Number of attention heads")
    num_layers: int = Field(default=4, gt=0, description="Number of transformer/LSTM layers")
    dropout: float = Field(default=0.1, ge=0, le=1, description="Dropout rate")

    @field_validator("num_heads")
    @classmethod
    def heads_divide_embed(cls, v: int, info) -> int:
        """Validate that embed_dim is divisible by num_heads."""
        embed_dim = info.data.get("embed_dim", 64)
        if embed_dim % v != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({v})")
        return v


class DataConfig(BaseModel):
    """Data processing configuration."""

    sequence_length: int = Field(default=300, gt=0, description="Total sequence length")
    output_length: int = Field(default=60, gt=0, description="Prediction horizon length")
    batch_size: int = Field(default=32, gt=0, description="Training batch size")
    test_split: float = Field(default=0.1, gt=0, lt=1, description="Validation split ratio")
    num_features: int = Field(default=1, gt=0, description="Number of input features")

    @property
    def input_length(self) -> int:
        """Compute input length from sequence and output length."""
        return self.sequence_length - self.output_length

    @model_validator(mode="after")
    def validate_lengths(self) -> "DataConfig":
        """Validate that output_length < sequence_length."""
        if self.output_length >= self.sequence_length:
            raise ValueError(
                f"output_length ({self.output_length}) must be < sequence_length ({self.sequence_length})"
            )
        return self


class TrainingConfig(BaseModel):
    """Training configuration for gradient-based optimization."""

    num_epochs: int = Field(default=100, gt=0, description="Number of training epochs")
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    early_stopping_patience: int = Field(default=20, gt=0, description="Early stopping patience")

    # Epochs at which to log detailed spectral metrics
    track_epochs: tuple[int, ...] = Field(
        default=(0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
        description="Epochs to log detailed spectral metrics"
    )


class ESConfig(BaseModel):
    """Configuration for Evolution Strategy optimizer."""

    generations: int = Field(default=150, gt=0, description="Number of ES generations")
    popsize: int = Field(default=256, gt=0, description="Population size (must be even if antithetic)")
    sigma: float = Field(default=0.15, gt=0, description="Initial noise standard deviation")
    lr: float = Field(default=0.05, gt=0, description="Learning rate for mean update")
    lr_sigma: float = Field(default=0.0, ge=0, description="Learning rate for sigma adaptation")
    antithetic: bool = Field(default=True, description="Use antithetic (mirrored) sampling")
    rank_transform: bool = Field(default=True, description="Apply rank-based fitness shaping")
    max_batches_per_eval: int = Field(default=32, gt=0, description="Max batches per fitness evaluation")
    resample_each_call: bool = Field(default=True, description="Resample batches each evaluation")
    log_every: int = Field(default=5, gt=0, description="Log every N generations")
    save_top_k: int = Field(default=5, gt=0, description="Keep top K solutions in hall of fame")
    track_epochs: tuple[int, ...] = Field(
        default=(0, 10, 25, 50, 100),
        description="Generations to log detailed spectral metrics"
    )

    @model_validator(mode="after")
    def validate_antithetic(self) -> "ESConfig":
        """Validate popsize is even when using antithetic sampling."""
        if self.antithetic and self.popsize % 2 != 0:
            raise ValueError("popsize must be even when antithetic=True")
        return self


class GAConfig(BaseModel):
    """Configuration for Genetic Algorithm optimizer."""

    generations: int = Field(default=100, gt=0, description="Number of GA generations")
    popsize: int = Field(default=300, gt=0, description="Population size")
    mutation_stdev: float = Field(default=0.1, gt=0, description="Gaussian mutation std dev")
    crossover_eta: float = Field(default=10.0, gt=0, description="SBX crossover eta parameter")
    crossover_rate: float = Field(default=0.6, ge=0, le=1, description="Crossover probability")
    tournament_size: int = Field(default=3, gt=0, description="Tournament selection size")
    eval_max_batches: int = Field(default=32, gt=0, description="Max batches per evaluation")
    init_bounds: tuple[float, float] = Field(
        default=(-0.5, 0.5),
        description="Initial weight bounds (lo, hi)"
    )
    track_epochs: tuple[int, ...] = Field(
        default=(0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
        description="Generations to log detailed spectral metrics"
    )


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    name: str = Field(default="experiment", description="Experiment name for MLflow")
    seed: int = Field(default=927, ge=0, description="Random seed for reproducibility")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        description="Device to use for training"
    )

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    es: ESConfig = Field(default_factory=ESConfig)
    ga: GAConfig = Field(default_factory=GAConfig)

    # Optimizer type determines which config is used
    optimizer: OptimizerType = Field(default=OptimizerType.ADAM)

    # Paths (with defaults from project structure)
    dataset_path: Path | None = Field(default=None, description="Path to dataset .npy file")
    output_dir: Path | None = Field(default=None, description="Output directory for artifacts")

    def get_resolved_device(self) -> torch.device:
        """Resolve 'auto' device to actual device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_toml(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from TOML file."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for MLflow logging."""
        return _flatten_dict(self.model_dump(mode="json"))


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dictionary with dot notation keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# =============================================================================
# Seed & Determinism
# =============================================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable deterministic CUDA operations (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# Default Experiment Configurations (from original config.py)
# =============================================================================

# Legacy constants for backward compatibility
SEED = 927
SEQUENCE_LENGTH = 300
N_RUNS = 5
TRACK_EPOCHS = (0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
SGT_LOSS_LAMBDAS = [-0.1, -0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01, 0.1]

DEFAULT_SYNTHETIC_DATA_CONFIGS = [
    {"lam": 0.0, "q": 100.0, "sigma": 0.707, "experiment_name": "normal"},
    {"lam": 0.0, "q": 1.001, "sigma": 15.0, "experiment_name": "heavy-tailed"},
    {"lam": 0.0, "q": 100.0, "sigma": 0.707, "kernel_size": 15, "experiment_name": "normal-short-kernel"},
    {"lam": 0.0, "q": 1.001, "sigma": 15.0, "kernel_size": 15, "experiment_name": "heavy-tailed-short-kernel"},
]

DEFAULT_TRAINING_CONFIGS = [
    {"loss_type": "sgt", "sgt_loss_lambda": 0.0, "sgt_loss_q": 1.001, "sgt_loss_sigma": 15.0},
    {"loss_type": "sgt", "sgt_loss_lambda": 0.0, "sgt_loss_q": 100.0, "sgt_loss_sigma": 0.707},
    {"loss_type": "sgt", "sgt_loss_lambda": 0.0, "sgt_loss_q": 5.0, "sgt_loss_sigma": 0.707},
    {"loss_type": "sgt", "sgt_loss_lambda": 0.0, "sgt_loss_q": 20.0, "sgt_loss_sigma": 0.707},
    {"loss_type": "sgt", "sgt_loss_lambda": 0.0, "sgt_loss_q": 75.0, "sgt_loss_sigma": 0.707},
    {"loss_type": "mse"},
    {"loss_type": "mae"},
    {"loss_type": "cauchy"},
]

# Aliases for backward compatibility with old config.py naming
SYNTHETIC_DATA_CONFIGS = DEFAULT_SYNTHETIC_DATA_CONFIGS
TRAINING_CONFIGS = DEFAULT_TRAINING_CONFIGS

# URL for OWID COVID data
DATA_URL = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
