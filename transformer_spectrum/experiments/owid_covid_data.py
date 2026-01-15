import random

import typer

from transformer_spectrum.settings import (N_RUNS, PROCESSED_DATA_DIR, TRAINING_CONFIGS)
from transformer_spectrum.modeling.train import main as train_main

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(n_runs: int = N_RUNS):
    experiment_name_base = 'covid-owid'
    dataset_path = PROCESSED_DATA_DIR / 'dataset.npy'
    training_configs = TRAINING_CONFIGS

    for training_config in training_configs:
        loss_type = training_config['loss_type']

        for run_idx in range(1, n_runs + 1):
            experiment_seed = random.randint(0, 2 ** 32 - 1)
            experiment_name = f'{experiment_name_base}_run_{run_idx}'

            typer.echo(f'Starting training with config: {training_config}, run: {run_idx}, seed: {experiment_seed}')

            if loss_type.lower() == 'sgt':
                sgt_lambda = training_config['sgt_loss_lambda']
                sgt_q = training_config['sgt_loss_q']
                sgt_sigma = training_config['sgt_loss_sigma']
                train_main(
                    dataset_path=dataset_path,
                    loss_type=loss_type,
                    sgt_loss_lambda=sgt_lambda,
                    sgt_loss_q=sgt_q,
                    sgt_loss_sigma=sgt_sigma,
                    experiment_name=experiment_name,
                    seed=experiment_seed,
                )
            else:
                train_main(
                    dataset_path=dataset_path,
                    loss_type=loss_type,
                    experiment_name=experiment_name,
                    seed=experiment_seed,
                )

            typer.echo(f'Completed training with config: {training_config}, run: {run_idx}\n')


if __name__ == '__main__':
    app()
