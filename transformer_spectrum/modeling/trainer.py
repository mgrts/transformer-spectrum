import mlflow
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from transformer_spectrum.metrics.spectral_metrics import get_spectral_metrics
from transformer_spectrum.metrics.training_metrics import compute_training_metrics
from transformer_spectrum.modeling.utils import EarlyStopping
from transformer_spectrum.config import TRACKING_URI


mlflow.set_tracking_uri(TRACKING_URI)


class Trainer:
    def __init__(self, experiment_name, model, train_loader, val_loader, criterion, optimizer,
                 model_save_path, num_epochs, early_stopping_patience, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_save_path = model_save_path
        self.output_dir = model_save_path.parent
        self.num_epochs = num_epochs
        self.early_stopper = EarlyStopping(patience=early_stopping_patience)
        self.device = device

        self.best_val_loss = float('inf')
        self.best_train_metrics = {}
        self.best_val_metrics = {}
        self.best_spectral_metrics = {}
        self.best_weights = None

        mlflow.set_experiment(experiment_name)

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._evaluate()

            train_metrics = self._collect_metrics(self.train_loader, train_loss)
            val_metrics = self._collect_metrics(self.val_loader, val_loss)

            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mape': train_metrics['mape'],
                'train_smape': train_metrics['smape'],
                'val_mape': val_metrics['mape'],
                'val_smape': val_metrics['smape'],
            }, step=epoch)

            logger.info(
                f'Epoch {epoch}/{self.num_epochs} | '
                f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
                f'Train sMAPE: {train_metrics["smape"]:.2f} | Val sMAPE: {val_metrics["smape"]:.2f}'
            )

            param_name = 'transformer.encoder.layers.0.self_attn.out_proj.weight'
            param = self.model.state_dict().get(param_name)
            weights = param.detach().cpu().numpy()
            spec_metrics = get_spectral_metrics(weights)
            mlflow.log_metrics(spec_metrics, step=epoch)

            if val_loss < self.best_val_loss - 1e-6:
                self.best_val_loss = val_loss
                self.best_train_metrics = train_metrics
                self.best_val_metrics = val_metrics
                self.best_spectral_metrics = spec_metrics
                self.best_weights = weights

                torch.save(self.model.state_dict(), self.model_save_path)
                mlflow.log_artifact(self.model_save_path)
                logger.info(f'Saved best model to {self.model_save_path} (val_loss={val_loss:.4f})')

            if self.early_stopper.step(val_loss):
                logger.warning(f'Early stopping triggered after {epoch} epochs.')
                break

        return self.best_val_loss, self.best_train_metrics, self.best_val_metrics, self.best_spectral_metrics, self.best_weights

    def _train_epoch(self) -> float:
        return self._run_epoch(
            self.model, self.train_loader, self.criterion, self.optimizer, torch.device(self.device), train=True
        )

    def _evaluate(self) -> float:
        return self._run_epoch(
            self.model, self.val_loader, self.criterion, device=torch.device(self.device), train=False
        )

    def _run_epoch(self, model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                   optimizer: torch.optim.Optimizer = None, device: torch.device = 'cuda', train: bool = True) -> float:
        model.train() if train else model.eval()
        losses = []

        iterator = dataloader

        with torch.no_grad() if not train else torch.enable_grad():
            for src, tgt in iterator:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt)
                loss = criterion(output, tgt)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())

        return float(np.mean(losses))

    def _collect_metrics(self, dataloader, loss_value):
        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for src, tgt in dataloader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                pred = self.model(src, tgt)
                preds.append(pred)
                targets.append(tgt)

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        metrics = compute_training_metrics(preds, targets)
        metrics['loss'] = loss_value
        return metrics
