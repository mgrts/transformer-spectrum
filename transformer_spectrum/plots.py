from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer
from loguru import logger

from transformer_spectrum.settings import FIGURES_DIR, PROCESSED_DATA_DIR
from transformer_spectrum.data.rvr_us.dataset import main as create_rvr_dataset_main
from transformer_spectrum.data.synthetic.generate_data import main as generate_data_main
from transformer_spectrum.settings import SYNTHETIC_DATA_CONFIGS

app = typer.Typer(pretty_exceptions_show_locals=False)


def create_boxplot(data_dict: dict, output_path: Path, title: str, xlim: int):
    plt.figure(figsize=(15, 9))
    sns.boxplot(data=list(data_dict.values()), orient='h')

    plt.yticks(ticks=range(len(data_dict)), labels=list(data_dict.keys()))
    plt.title(title)
    plt.xlabel('Values')
    plt.ylabel('Distribution')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    sns.despine(trim=True)

    plt.xticks(ticks=np.arange(-xlim, xlim + 1, 1))
    plt.xlim(-xlim, xlim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    logger.success(f'Boxplots saved to {output_path}')


@app.command()
def synthetic(
    output_path: Path = FIGURES_DIR / 'dataset_boxplots.png',
    sample_size: int = 1000,
    xlim: int = 5,
):
    logger.info('Generating synthetic datasets and creating boxplots...')

    dataset_params = SYNTHETIC_DATA_CONFIGS
    for i, params in enumerate(dataset_params):
        params['label'] = f"Dataset {i + 1}: {params['experiment_name']}"

    datasets = {}

    for params in dataset_params:
        generate_data_main(
            lam=params['lam'],
            q=params['q'],
            sigma=params['sigma'],
            n_sequences=sample_size,
            apply_smoothing=False,
        )
        data = np.load(PROCESSED_DATA_DIR / 'synthetic_dataset.npy').flatten()
        datasets[params['label']] = data
        logger.info(f'Generated dataset: {params["label"]}')

    create_boxplot(datasets, output_path, 'Comparison of Generated Datasets', xlim)


@app.command()
def real(
    output_path: Path = FIGURES_DIR / 'real_dataset_boxplots.png',
    xlim: int = 5,
):
    logger.info('Creating real-world datasets and generating boxplots...')

    dataset_specs = [
        {'label': 'RVR - Bed Occupancy', 'time_series': 'average_inpatient_beds_occupied'},
        {'label': 'RVR - Influenza Cases', 'time_series': 'total_admissions_all_influenza_confirmed_past_7days'},
        {'label': 'OWID - COVID Cases', 'path': PROCESSED_DATA_DIR / 'dataset.npy'},
    ]

    datasets = {}

    for spec in dataset_specs:
        label = spec['label']
        if 'time_series' in spec:
            try:
                create_rvr_dataset_main(time_series=spec['time_series'])
                data = np.load(PROCESSED_DATA_DIR / 'rvr_us_data.npy').flatten()
                datasets[label] = data
                logger.info(f"Created and loaded dataset: {label}")
            except Exception as e:
                logger.warning(f"Failed to process {label}: {e}")
        else:
            try:
                data = np.load(spec['path']).flatten()
                datasets[label] = data
                logger.info(f"Loaded dataset: {label}")
            except Exception as e:
                logger.warning(f"Failed to load {label}: {e}")

    create_boxplot(datasets, output_path, 'Comparison of Real-World Datasets', xlim)


if __name__ == '__main__':
    app()
