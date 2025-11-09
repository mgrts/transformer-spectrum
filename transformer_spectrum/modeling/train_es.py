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
    TRACK_EPOCHS,
)
from transformer_spectrum.metrics.heavy_tail_estimation import estimate_kappa_exponent
from transformer_spectrum.metrics.skewness_estimation import skewness, skewness_of_diff
from transformer_spectrum.modeling.data_processing import create_dataloaders
from transformer_spectrum.modeling.trainer import TrainerES, ESConfig
from transformer_spectrum.modeling.utils import set_seed, get_loss_function, get_model
from transformer_spectrum.modeling.visualize import log_sample_images

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
        loss_type: str = 'sgt',
        sgt_loss_sigma: float = 1.0,
        sgt_loss_lambda: float = 0.0,
        sgt_loss_q: float = 2.0,
        test_split: float = 0.1,
        seed: int = SEED,
        experiment_name: str = 'Transformer-ES-synthetic',

        # ES hyperparams
        es_generations: int = 150,
        es_popsize: int = 256,
        es_sigma: float = 0.15,
        es_lr: float = 0.05,
        es_lr_sigma: float = 0.0,
        es_antithetic: bool = True,
        es_rank_transform: bool = True,
        es_max_batches_per_eval: int = 32,
        es_resample_each_call: bool = True,
        es_log_every: int = 5,
        es_save_top_k: int = 5,
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

    # ----- model -----
    model = get_model(
        model_type=model_type,
        in_dim=data.shape[-1],
        out_dim=data.shape[-1],
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        device=device,
    )

    # ----- loss -----
    criterion = get_loss_function(loss_type, sgt_loss_lambda, sgt_loss_q, sgt_loss_sigma)

    # ----- MLflow -----
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        output_dir = MODELS_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = output_dir / 'model.pt'

        # Dataset stats
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

        # Params
        mlflow.log_params({
            'optimizer_type': 'simple_es',
            'loss_type': loss_type,
            'sgt_loss_lambda': sgt_loss_lambda,
            'sgt_loss_q': sgt_loss_q,
            'sgt_loss_sigma': sgt_loss_sigma,
            'model_type': model_type,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'output_length': output_length,
            'test_split': test_split,
            'seed': seed,

            # ES params
            'es_generations': es_generations,
            'es_popsize': es_popsize,
            'es_sigma': es_sigma,
            'es_lr': es_lr,
            'es_lr_sigma': es_lr_sigma,
            'es_antithetic': es_antithetic,
            'es_rank_transform': es_rank_transform,
            'es_max_batches_per_eval': es_max_batches_per_eval,
            'es_resample_each_call': es_resample_each_call,
            'es_log_every': es_log_every,
            'es_save_top_k': es_save_top_k,
        })

        # ES config + trainer
        es_cfg = ESConfig(
            popsize=es_popsize,
            sigma=es_sigma,
            lr=es_lr,
            lr_sigma=es_lr_sigma,
            antithetic=es_antithetic,
            rank_transform=es_rank_transform,
            max_batches_per_eval=es_max_batches_per_eval,
            resample_each_call=es_resample_each_call,
            log_every=es_log_every,
            save_top_k=es_save_top_k,
            track_epochs=TRACK_EPOCHS,
        )

        # Use training mini-batches for the ES objective by default
        trainer = TrainerES(
            experiment_name=experiment_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            model_save_path=model_save_path,
            device=str(device),
            es_cfg=es_cfg,
            use_train_for_objective=True,
        )

        best_obj, hall = trainer.fit(generations=es_generations)
        mlflow.log_metric('best_objective', float(best_obj))

        # sample preds / images using your existing helper
        try:
            log_sample_images(model, val_loader, model_path=model_save_path)
        except Exception as e:
            logger.warning(f'log_sample_images failed: {e}')

        logger.success('Simple-ES optimization complete.')
        logger.info(f'Models saved in {output_dir}')


if __name__ == '__main__':
    app()
