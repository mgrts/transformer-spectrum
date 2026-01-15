"""
Tests for configuration and settings.
"""

import pytest
import tempfile
from pathlib import Path

from transformer_spectrum.settings import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    LossConfig,
    TrainingConfig,
    ESConfig,
    GAConfig,
    LossType,
    ModelType,
    set_seed,
)


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_default_config(self):
        """Test creating default config."""
        cfg = ExperimentConfig()

        assert cfg.name == "experiment"
        assert cfg.seed == 927
        assert cfg.device == "auto"
        assert cfg.model.type == ModelType.TRANSFORMER
        assert cfg.loss.type == LossType.MSE

    def test_custom_config(self):
        """Test creating custom config."""
        cfg = ExperimentConfig(
            name="my-experiment",
            seed=42,
            device="cpu",
            model=ModelConfig(embed_dim=128, num_heads=8),
            loss=LossConfig(type=LossType.SGT),
        )

        assert cfg.name == "my-experiment"
        assert cfg.seed == 42
        assert cfg.model.embed_dim == 128
        assert cfg.loss.type == LossType.SGT

    def test_to_dict(self):
        """Test flattening config to dict."""
        cfg = ExperimentConfig()
        d = cfg.to_dict()

        assert "name" in d
        assert "model.embed_dim" in d
        assert "loss.type" in d

    def test_yaml_roundtrip(self):
        """Test saving and loading from YAML."""
        cfg = ExperimentConfig(
            name="test-yaml",
            seed=123,
            model=ModelConfig(embed_dim=32),
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            cfg.to_yaml(Path(f.name))
            loaded = ExperimentConfig.from_yaml(Path(f.name))

        assert loaded.name == cfg.name
        assert loaded.seed == cfg.seed
        assert loaded.model.embed_dim == cfg.model.embed_dim

        Path(f.name).unlink()


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default(self):
        """Test default model config."""
        cfg = ModelConfig()

        assert cfg.type == ModelType.TRANSFORMER
        assert cfg.embed_dim == 64
        assert cfg.num_heads == 4
        assert cfg.num_layers == 4

    def test_invalid_heads(self):
        """Test that embed_dim must be divisible by num_heads."""
        # This should work
        cfg = ModelConfig(embed_dim=64, num_heads=4)
        assert cfg.embed_dim == 64

        # This should fail
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(embed_dim=64, num_heads=5)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_input_length(self):
        """Test input_length property."""
        cfg = DataConfig(sequence_length=300, output_length=60)

        assert cfg.input_length == 240

    def test_invalid_lengths(self):
        """Test validation of lengths."""
        with pytest.raises(ValueError, match="output_length"):
            DataConfig(sequence_length=100, output_length=150)


class TestLossConfig:
    """Tests for LossConfig."""

    def test_sgt_params(self):
        """Test SGT loss parameters."""
        cfg = LossConfig(
            type=LossType.SGT,
            sgt={"sigma": 2.0, "lam": 0.1, "q": 5.0},
        )

        assert cfg.type == LossType.SGT
        assert cfg.sgt.sigma == 2.0
        assert cfg.sgt.lam == 0.1
        assert cfg.sgt.q == 5.0

    def test_sgt_validation(self):
        """Test SGT parameter validation."""
        # Valid
        cfg = LossConfig(type=LossType.SGT, sgt={"lam": 0.5})
        assert cfg.sgt.lam == 0.5

        # Invalid lambda (must be in [-1, 1])
        with pytest.raises(ValueError):
            LossConfig(type=LossType.SGT, sgt={"lam": 1.5})


class TestESConfig:
    """Tests for ES configuration."""

    def test_antithetic_validation(self):
        """Test that popsize must be even for antithetic sampling."""
        # Valid
        cfg = ESConfig(popsize=256, antithetic=True)
        assert cfg.popsize == 256

        # Invalid (odd popsize with antithetic)
        with pytest.raises(ValueError, match="even"):
            ESConfig(popsize=255, antithetic=True)

        # Valid (odd popsize without antithetic)
        cfg = ESConfig(popsize=255, antithetic=False)
        assert cfg.popsize == 255


class TestGAConfig:
    """Tests for GA configuration."""

    def test_default(self):
        """Test default GA config."""
        cfg = GAConfig()

        assert cfg.generations == 100
        assert cfg.popsize == 300
        assert cfg.crossover_rate == 0.6


class TestSetSeed:
    """Tests for seed setting."""

    def test_seed_determinism(self):
        """Test that setting seed gives deterministic random numbers."""
        import torch
        import numpy as np

        set_seed(42)
        r1 = torch.rand(5)
        n1 = np.random.rand(5)

        set_seed(42)
        r2 = torch.rand(5)
        n2 = np.random.rand(5)

        assert torch.allclose(r1, r2)
        assert np.allclose(n1, n2)

    def test_different_seeds(self):
        """Test that different seeds give different results."""
        import torch
        import numpy as np

        set_seed(42)
        r1 = torch.rand(5)

        set_seed(43)
        r2 = torch.rand(5)

        assert not torch.allclose(r1, r2)
