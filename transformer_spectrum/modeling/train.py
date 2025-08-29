from pathlib import Path

import mlflow
import numpy as np
import torch
import typer
from loguru import logger

from transformer_spectrum.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    SEQUENCE_LENGTH,
    TRACKING_URI,
)
from transformer_spectrum.metrics.heavy_tail_estimation import estimate_kappa_exponent
from transformer_spectrum.metrics.skewness_estimation import skewness, skewness_of_diff
from transformer_spectrum.modeling.data_processing import create_dataloaders
from transformer_spectrum.modeling.loss_functions import CauchyLoss, SGTLoss
from transformer_spectrum.modeling.models import LSTM, TransformerWithPE
from transformer_spectrum.modeling.trainer import Trainer
from transformer_spectrum.modeling.utils import set_seed
from transformer_spectrum.modeling.visualize import log_sample_images

app = typer.Typer(pretty_exceptions_show_locals=False)


def get_loss_function(loss_type: str, sgt_loss_lambda: float = 0.0, sgt_loss_q: float = 2.0,
                      sgt_loss_sigma: float = 1.0):
    loss_type = loss_type.lower()
    if loss_type == 'sgt':
        return SGTLoss(eps=0.0, sigma=sgt_loss_sigma, p=2.0, q=sgt_loss_q, lam=sgt_loss_lambda)
    elif loss_type == 'mse':
        return torch.nn.MSELoss()
    elif loss_type == 'mae':
        return torch.nn.L1Loss()
    elif loss_type == 'cauchy':
        return CauchyLoss(gamma=2.0)
    else:
        raise ValueError(f'Unsupported loss type: {loss_type}')


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
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        test_split: float = 0.1,
        seed: int = SEED,
        early_stopping_patience: int = 20,
        loss_type: str = 'sgt',
        sgt_loss_sigma: float = 1.0,
        sgt_loss_lambda: float = 0.0,
        sgt_loss_q: float = 2.0,
        experiment_name: str = 'Transformer-SGT-synthetic',
):
    set_seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    assert output_length < sequence_length, 'Output length must be less than sequence length'

    logger.info(f'Using device: {device}')
    logger.info('Loading data...')

    data = np.load(dataset_path)
    input_length = sequence_length - output_length

    train_loader, val_loader = create_dataloaders(
        data=data,
        input_len=input_length,
        output_len=output_length,
        batch_size=batch_size,
        test_split=test_split,
        seed=seed,
    )

    if model_type == 'transformer':
        model = TransformerWithPE(
            in_dim=data.shape[-1],
            out_dim=data.shape[-1],
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        ).to(device)
    elif model_type == 'lstm':
        model = LSTM(
            input_dim=data.shape[-1],
            hidden_dim=embed_dim,
            num_layers=num_layers,
            output_dim=data.shape[-1],
        ).to(device)
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    criterion = get_loss_function(loss_type, sgt_loss_lambda, sgt_loss_q, sgt_loss_sigma)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        output_dir = MODELS_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = output_dir / 'model.pt'

        logger.info('Estimating skewness and heavy-tailness...')

        flat_data = data.reshape(-1)

        skew = skewness(flat_data)
        diff_skew = skewness_of_diff(data)
        mlflow.log_metric('skewness', skew)
        mlflow.log_metric('skewness_of_diff', diff_skew)
        logger.info(f'Skewness: {skew:.4f}, Skewness of Diff: {diff_skew:.4f}')

        _, _, kappa_val, _, _ = estimate_kappa_exponent(flat_data, n=10)
        mlflow.log_metric('kappa', kappa_val)
        logger.info(f'Kappa (n=10): {kappa_val:.4f}')

        mlflow.log_params({
            'model_type': model_type,
            'sgt_loss_lambda': sgt_loss_lambda,
            'sgt_loss_q': sgt_loss_q,
            'sgt_loss_sigma': sgt_loss_sigma,
            'input_length': input_length,
            'output_length': output_length,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'loss_type': loss_type,
            'early_stopping_patience': early_stopping_patience,
            'num_epochs': num_epochs,
            'test_split': test_split,
            'seed': seed,
        })

        trainer = Trainer(
            experiment_name=experiment_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            model_save_path=model_save_path,
            device=device,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
        )
        best_val_loss, best_train_metrics, best_val_metrics, best_spectral_metrics = trainer.train()

        mlflow.log_metrics({
            'best_train_smape': best_train_metrics.get('smape'),
            'best_val_smape': best_val_metrics.get('smape'),
        })
        for key, value in best_spectral_metrics.items():
            mlflow.log_metric(f'best_{key}', float(value))

        log_sample_images(model, val_loader, model_path=model_save_path)

        logger.success('Training complete.')


if __name__ == '__main__':
    app()
