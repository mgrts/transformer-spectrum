# transformer-spectrum

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Transformer Spectrum Analysis**: Investigating how different loss functions affect the weight spectrum of transformer matrices.

## Overview

This project trains transformer models for autoregression on synthetic and real datasets, then analyzes how the spectral properties of weight matrices evolve during training. Key spectral metrics include:

- **Spectral Entropy**: How uniformly distributed singular values are
- **Alpha Exponent**: Power-law decay rate of singular values
- **Stable Rank**: Effective rank of weight matrices
- **Power-Law Alpha**: Heavy-tail index estimated via Hill and KS methods

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/transformer-spectrum.git
cd transformer-spectrum

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Running Your First Experiment

```bash
# Generate synthetic data
python -m transformer_spectrum.cli generate-synthetic \
    -o data/processed/synthetic.npy \
    --n-sequences 1000 \
    --seq-len 300

# Train a model
python -m transformer_spectrum.cli train \
    -d data/processed/synthetic.npy \
    -n my-first-experiment \
    --loss mse \
    --epochs 50
```

## CLI Reference

The unified CLI provides commands for data generation, training, and experiment sweeps.

### Training Commands

```bash
# Standard gradient-based training (Adam)
python -m transformer_spectrum.cli train \
    -d data/processed/dataset.npy \
    -n experiment-name \
    --model transformer \
    --loss sgt \
    --sgt-q 2.0 \
    --epochs 100

# Evolution Strategy training
python -m transformer_spectrum.cli train-es \
    -d data/processed/dataset.npy \
    -n es-experiment \
    --generations 150 \
    --popsize 256

# Genetic Algorithm training
python -m transformer_spectrum.cli train-ga \
    -d data/processed/dataset.npy \
    -n ga-experiment \
    --generations 100 \
    --popsize 300
```

### Data Generation Commands

```bash
# Generate synthetic SGT-distributed data
python -m transformer_spectrum.cli generate-synthetic \
    -o data/processed/synthetic.npy \
    --sigma 1.0 \
    --lambda 0.0 \
    --q 2.0

# Download and process COVID data
python -m transformer_spectrum.cli download-covid
python -m transformer_spectrum.cli process-covid
```

### Experiment Sweeps

```bash
# Run pre-defined experiment sweeps
python -m transformer_spectrum.cli sweep synthetic --runs 5
python -m transformer_spectrum.cli sweep covid --runs 3
python -m transformer_spectrum.cli sweep rvr --runs 3
```

### Help

```bash
python -m transformer_spectrum.cli --help
python -m transformer_spectrum.cli train --help
```

## Configuration

### Using YAML Config Files

You can define experiment configurations in YAML:

```yaml
# config/my_experiment.yaml
name: "my-experiment"
seed: 42
device: auto

model:
  type: transformer
  embed_dim: 64
  num_heads: 4
  num_layers: 4

data:
  sequence_length: 300
  output_length: 60
  batch_size: 32

loss:
  type: sgt
  sgt:
    sigma: 1.0
    lam: 0.0
    q: 2.0

training:
  num_epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 20
```

Then run:

```bash
python -m transformer_spectrum.cli train --config config/my_experiment.yaml
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_loss_functions.py -v

# Run with coverage
pytest tests/ --cov=transformer_spectrum --cov-report=html

# Run fast tests only (skip slow ones)
pytest tests/ -m "not slow"
```

## Docker

### Building the Image

```bash
docker build -t transformer-spectrum .
```

### Running Experiments

```bash
# Training with mounted volumes
docker run -v $(pwd)/data:/app/data -v $(pwd)/mlruns:/app/mlruns \
    transformer-spectrum train \
    -d /app/data/processed/dataset.npy \
    -n docker-experiment

# Interactive shell
docker run -it --rm --entrypoint /bin/bash \
    -v $(pwd)/data:/app/data \
    transformer-spectrum
```

### Kubernetes / GKE Job Example

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: transformer-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: gcr.io/your-project/transformer-spectrum:latest
        args:
          - train
          - -d
          - /data/processed/dataset.npy
          - -n
          - k8s-experiment
          - --epochs
          - "100"
        volumeMounts:
        - name: data-volume
          mountPath: /data
        - name: mlruns-volume
          mountPath: /app/mlruns
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
      - name: mlruns-volume
        persistentVolumeClaim:
          claimName: mlruns-pvc
      restartPolicy: Never
  backoffLimit: 3
```

## Development

### Code Style

```bash
# Format code
make format

# Lint
make lint

# Run tests
make test
```

## Project Structure

```
├── transformer_spectrum/     # Source code
│   ├── cli.py               # Unified CLI entrypoint
│   ├── settings.py          # Pydantic configuration models
│   ├── data/                # Data loading and processing
│   │   ├── synthetic/       # Synthetic data generation
│   │   ├── owid_covid/      # COVID data processing
│   │   └── rvr_us/          # RVR hospitalization data
│   ├── experiments/         # Experiment sweep scripts
│   ├── metrics/             # Spectral and training metrics
│   │   ├── spectral_metrics.py
│   │   ├── training_metrics.py
│   │   ├── heavy_tail_estimation.py
│   │   └── skewness_estimation.py
│   └── modeling/            # Models and training
│       ├── models.py        # Transformer, LSTM architectures
│       ├── loss_functions.py # SGT, Cauchy, etc.
│       ├── trainer.py       # Training loops
│       ├── train.py         # Gradient-based training CLI
│       ├── train_es.py      # Evolution strategy CLI
│       └── train_ga.py      # Genetic algorithm CLI
├── tests/                   # Test suite
├── data/                    # Data directories
├── models/                  # Saved model checkpoints
├── mlruns/                  # MLflow experiment tracking
├── Dockerfile              
├── requirements.txt
└── pyproject.toml
```

## Loss Functions

The project supports several loss functions for training:

| Loss | Description | Use Case |
|------|-------------|----------|
| `mse` | Mean Squared Error | Standard regression |
| `mae` | Mean Absolute Error | Robust to outliers |
| `cauchy` | Cauchy/Lorentzian loss | Heavy-tailed errors |
| `sgt` | Skewed Generalized T | Flexible tail behavior |

### SGT Loss Parameters

- `sigma`: Scale parameter (> 0)
- `lambda`: Skewness (-1 to 1, 0 = symmetric)
- `p`: Shape parameter (typically 2.0)
- `q`: Tail heaviness (larger = lighter tails)

## Spectral Metrics

All spectral metrics are computed on weight matrices during training:

- **Spectral Entropy**: `H = -Σ p_i log(p_i)` where `p_i = σ_i² / Σσ²`
- **Stable Rank**: `||W||_F² / ||W||_2² = Σσ² / σ_max²`
- **Alpha Exponent**: Slope of log-log plot of singular values
- **Power-Law Alpha**: Tail index from Hill/KS estimators

## MLflow Tracking

Experiments are tracked with MLflow. View the UI:

```bash
mlflow ui --backend-store-uri mlruns/
```

Then open http://localhost:5000

## License

MIT License - see [LICENSE](LICENSE) for details.

--------

