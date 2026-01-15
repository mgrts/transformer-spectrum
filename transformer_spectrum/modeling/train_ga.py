from pathlib import Path

import mlflow
import numpy as np
import torch
import typer
from loguru import logger

from transformer_spectrum.settings import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    SEQUENCE_LENGTH,
    TRACKING_URI,
    TRACK_EPOCHS,
)
from transformer_spectrum.metrics.heavy_tail_estimation import estimate_kappa_exponent
from transformer_spectrum.metrics.skewness_estimation import skewness, skewness_of_diff
from transformer_spectrum.modeling.data_processing import create_dataloaders
from transformer_spectrum.modeling.trainer import TrainerGA
from transformer_spectrum.modeling.utils import set_seed, get_loss_function, get_model

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
        dataset_path: Path = PROCESSED_DATA_DIR / 'synthetic_dataset.npy',
        model_type: str = 'transformer',
        sequence_length: int = SEQUENCE_LENGTH,
        output_length: int = 60,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        batch_size: int = 32,
        loss_type: str = 'mae',
        sgt_loss_sigma: float = 1.0,
        sgt_loss_lambda: float = 0.0,
        sgt_loss_q: float = 2.0,
        test_split: float = 0.1,
        seed: int = SEED,
        experiment_name: str = 'Transformer-GA-synthetic',

        # GA hyperparams
        ga_generations: int = 100,
        ga_popsize: int = 300,
        ga_mutation_stdev: float = 0.1,
        ga_crossover_eta: float = 10.0,
        cross_over_rate: float = 0.6,
        tournament_size: int = 3,
        ga_eval_max_batches: int = 32,
        init_lo: float = -0.5,
        init_hi: float = 0.5,
):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f'Using device: {device}')

    data = np.load(dataset_path)
    input_length = sequence_length - output_length
    assert output_length < sequence_length, 'Output length must be less than sequence length'

    train_loader, val_loader = create_dataloaders(
        data=data,
        input_len=input_length,
        output_len=output_length,
        batch_size=batch_size,
        test_split=test_split,
        seed=seed,
    )

    model = get_model(
        model_type=model_type,
        in_dim=data.shape[-1],
        out_dim=data.shape[-1],
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        device=device,
    )

    criterion = get_loss_function(loss_type, sgt_loss_lambda, sgt_loss_q, sgt_loss_sigma)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        output_dir = MODELS_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = output_dir / 'model.pt'

        # Dataset stats for the run
        flat_data = data.reshape(-1)
        skew = skewness(flat_data)
        diff_skew = skewness_of_diff(data)
        _, _, kappa_val, _, _ = estimate_kappa_exponent(flat_data, n=10)

        mlflow.log_metrics({
                'skewness': skew,
                'skewness_of_diff': diff_skew,
                'kappa': kappa_val,
        })

        logger.info(f'Skewness={skew:.4f}, DiffSkew={diff_skew:.4f}, Kappa={kappa_val:.4f}')

        # Log run params
        mlflow.log_params({
            'optimizer_type': 'genetic',
            'model_type': model_type,
            'loss_type': loss_type,
            'sgt_loss_lambda': sgt_loss_lambda,
            'sgt_loss_q': sgt_loss_q,
            'sgt_loss_sigma': sgt_loss_sigma,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'output_length': output_length,
            'test_split': test_split,
            'seed': seed,

            # GA params
            'ga_generations': ga_generations,
            'ga_popsize': ga_popsize,
            'ga_mutation_stdev': ga_mutation_stdev,
            'ga_crossover_eta': ga_crossover_eta,
            'cross_over_rate': cross_over_rate,
            'tournament_size': tournament_size,
            'ga_eval_max_batches': ga_eval_max_batches,
            'init_bounds_lo': init_lo,
            'init_bounds_hi': init_hi,
        })

        init_bounds = (float(init_lo), float(init_hi)) if init_lo < init_hi else None

        trainer = TrainerGA(
            experiment_name=experiment_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            model_save_path=model_save_path,
            device=str(device),
            ga_generations=ga_generations,
            ga_popsize=ga_popsize,
            ga_mutation_stdev=ga_mutation_stdev,
            ga_crossover_eta=ga_crossover_eta,
            ga_eval_max_batches=ga_eval_max_batches,
            track_epochs=TRACK_EPOCHS,
            cross_over_rate=cross_over_rate,
            tournament_size=tournament_size,
            init_bounds=init_bounds,
        )

        best_objective = trainer.fit()

        mlflow.log_metric('best_objective', best_objective)

        logger.success('Genetic Algorithm optimization complete.')
        logger.info(f'Model saved to: {model_save_path}')


if __name__ == '__main__':
    app()
