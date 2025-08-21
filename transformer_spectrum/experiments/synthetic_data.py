import random

import typer

from transformer_spectrum.config import N_RUNS, PROCESSED_DATA_DIR, SYNTHETIC_DATA_CONFIGS, TRAINING_CONFIGS
from transformer_spectrum.data.synthetic.generate_data import main as generate_data_main
from transformer_spectrum.modeling.train import main as train_main

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(n_runs: int = N_RUNS):
    dataset_path = PROCESSED_DATA_DIR / 'synthetic_dataset.npy'
    dataset_configs = SYNTHETIC_DATA_CONFIGS
    training_configs = TRAINING_CONFIGS

    for ds_config in dataset_configs:
        lam = ds_config['lam']
        q = ds_config['q']
        sigma = ds_config['sigma']
        base_experiment_name = ds_config['experiment_name']

        typer.echo(f'==== Generating dataset with lam={lam}, q={q}, sigma={sigma} ====')

        if 'kernel_size' in ds_config:
            kernel_size = ds_config['kernel_size']
            typer.echo(f'Using kernel size: {kernel_size}')
            generate_data_main(lam=lam, q=q, sigma=sigma, kernel_size=kernel_size)
        else:
            generate_data_main(lam=lam, q=q, sigma=sigma)

        typer.echo('Dataset generation complete.\n')

        for train_config in training_configs:
            loss_type = train_config['loss_type']

            for run_idx in range(1, n_runs + 1):
                experiment_seed = random.randint(0, 2**32 - 1)
                experiment_name = f'{base_experiment_name}_run_{run_idx}'

                typer.echo(f'==== Starting training: {experiment_name} with loss_type={loss_type}, seed={experiment_seed} ====')

                if loss_type.lower() == 'sgt':
                    train_main(
                        dataset_path=dataset_path,
                        loss_type=loss_type,
                        sgt_loss_lambda=train_config['sgt_loss_lambda'],
                        sgt_loss_q=train_config['sgt_loss_q'],
                        sgt_loss_sigma=train_config['sgt_loss_sigma'],
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
                typer.echo(f'==== Completed training: {experiment_name} ====\n')


if __name__ == '__main__':
    app()
