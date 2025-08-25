import random

import typer

from transformer_spectrum.config import (N_RUNS, PROCESSED_DATA_DIR,
                                     TRAINING_CONFIGS)
from transformer_spectrum.data.rvr_us.dataset import main as create_dataset_main
from transformer_spectrum.modeling.train import main as train_main

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(n_runs: int = N_RUNS):
    time_series_list = [
        'average_inpatient_beds_occupied',
        'total_admissions_all_influenza_confirmed_past_7days'
    ]

    training_configs = TRAINING_CONFIGS
    dataset_path = PROCESSED_DATA_DIR / 'rvr_us_dataset.npy'

    for time_series in time_series_list:
        if time_series == 'average_inpatient_beds_occupied':
            experiment_base_name = 'rvr-us-bed-occupancy'
        elif time_series == 'total_admissions_all_influenza_confirmed_past_7days':
            experiment_base_name = 'rvr-us-influenza-cases'
        else:
            raise ValueError(f'Invalid time series: {time_series}')

        create_dataset_main(time_series=time_series)

        for training_config in training_configs:
            loss_type = training_config['loss_type']

            for run_idx in range(1, n_runs + 1):
                experiment_seed = random.randint(0, 2**32 - 1)
                experiment_name = f'{experiment_base_name}_run_{run_idx}'

                typer.echo(f'Starting training: {experiment_name} with config: {training_config}, seed: {experiment_seed}')

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

                typer.echo(f'Completed training: {experiment_name}\n')


if __name__ == '__main__':
    app()
