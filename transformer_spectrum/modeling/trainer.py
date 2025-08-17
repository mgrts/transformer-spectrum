import mlflow
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from transformer_spectrum.metrics.spectral_metrics import get_spectral_metrics
from transformer_spectrum.metrics.training_metrics import compute_training_metrics
from transformer_spectrum.modeling.utils import EarlyStopping
from transformer_spectrum.modeling.visualize import plot_singular_values
from transformer_spectrum.config import TRACKING_URI

mlflow.set_tracking_uri(TRACKING_URI)


def _singular_values(W: np.ndarray) -> np.ndarray:
    return np.linalg.svd(W, compute_uv=False)


def _esd(W: np.ndarray) -> np.ndarray:
    s = _singular_values(W)
    return s * s


def _split_in_proj_qkv(in_proj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if in_proj.shape[0] % 3 != 0:
        raise ValueError(f"in_proj has unexpected shape {in_proj.shape}")
    return np.split(in_proj, 3, axis=0)


def _plot_esd_hist(W: np.ndarray, title: str, bins: int | None = None) -> list[plt.Figure]:
    lam = _esd(W)
    lam = lam[np.isfinite(lam) & (lam >= 0)]
    if lam.size == 0:
        return []
    if bins is None:
        bins = int(np.clip(np.sqrt(lam.size), 10, 60))

    figs: list[plt.Figure] = []

    fig1 = plt.figure()
    plt.hist(lam, bins=bins)
    plt.title(f"ESD - {title}")
    plt.xlabel("eigenvalue")
    plt.ylabel("count")
    figs.append(fig1)

    lam_pos = lam[lam > 0]
    if lam_pos.size > 0:
        lo, hi = lam_pos.min(), lam_pos.max()
        if hi > lo:
            log_bins = np.logspace(np.log10(lo), np.log10(hi), bins)
            fig2 = plt.figure()
            plt.hist(lam_pos, bins=log_bins)
            plt.xscale("log")
            plt.yscale("log")
            plt.title(f"ESD (log-log) - {title}")
            plt.xlabel("eigenvalue (log)")
            plt.ylabel("count (log)")
            figs.append(fig2)

    return figs


def _plot_svals_inline(W: np.ndarray, title: str) -> plt.Figure:
    s = _singular_values(W)
    s = s[np.isfinite(s) & (s > 0)]
    fig = plt.figure()
    if s.size == 0:
        plt.title(f"Singular values - {title} (empty)")
        return fig
    s = np.sort(s)[::-1]
    x = np.arange(1, s.size + 1)
    plt.loglog(x, s, marker=".")
    plt.title(f"Singular values - {title}")
    plt.xlabel("index")
    plt.ylabel("singular value")
    return fig


def _iter_layer_idxs(sd: dict, prefix: str, tail: str) -> list[int]:
    out, i = [], 0
    while f"{prefix}.{i}.{tail}" in sd:
        out.append(i)
        i += 1
    return out


def _get_in_proj(sd: dict, prefix: str, tail: str, i: int) -> np.ndarray | None:
    t = sd.get(f"{prefix}.{i}.{tail}")
    return None if t is None else t.detach().cpu().numpy()


class Trainer:
    def __init__(
        self,
        experiment_name,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        model_save_path,
        num_epochs,
        early_stopping_patience,
        device: str = "cuda",
        esd_plot_epochs: tuple | list = (1, 10, 50),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_save_path = model_save_path
        self.out_dir = model_save_path.parent
        self.num_epochs = num_epochs
        self.early_stopper = EarlyStopping(patience=early_stopping_patience)
        self.device = device

        self.esd_plot_epochs = set(int(e) for e in esd_plot_epochs)

        self.best_val_loss = float("inf")
        self.best_train_metrics: dict = {}
        self.best_val_metrics: dict = {}
        self.best_spectral_metrics: dict = {}

        mlflow.set_experiment(experiment_name)

        self._enc = ("transformer.encoder.layers", "self_attn.in_proj_weight", "encoder")
        self._dec = ("transformer.decoder.layers", "multihead_attn.in_proj_weight", "decoder")

    def train(self):
        self.model.eval()
        with torch.no_grad():
            self._log_qkv_figs(*self._enc, epoch=0)
            self._log_qkv_figs(*self._dec, epoch=0)
            base_enc = self._log_qkv_metrics(*self._enc, epoch=0)
            base_dec = self._log_qkv_metrics(*self._dec, epoch=0)
            logger.info(f"Baseline metrics logged: encoder={len(base_enc)}, decoder={len(base_dec)}")

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)

            train_m = self._collect_metrics(self.train_loader, train_loss)
            val_m = self._collect_metrics(self.val_loader, val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mape": train_m["mape"],
                "train_smape": train_m["smape"],
                "val_mape": val_m["mape"],
                "val_smape": val_m["smape"],
            }, step=epoch)

            logger.info(
                f"Epoch {epoch}/{self.num_epochs} "
                f"| Train {train_loss:.4f} | Val {val_loss:.4f} "
                f"| sMAPE train {train_m['smape']:.2f} / val {val_m['smape']:.2f}"
            )

            enc_metrics = self._log_qkv_metrics(*self._enc, epoch=epoch)
            dec_metrics = self._log_qkv_metrics(*self._dec, epoch=epoch)
            spectral_snapshot = {**enc_metrics, **dec_metrics}

            if epoch in self.esd_plot_epochs:
                self._log_qkv_figs(*self._enc, epoch=epoch)
                self._log_qkv_figs(*self._dec, epoch=epoch)

            if val_loss < self.best_val_loss - 1e-6:
                self.best_val_loss = val_loss
                self.best_train_metrics = train_m
                self.best_val_metrics = val_m
                self.best_spectral_metrics = spectral_snapshot

                torch.save(self.model.state_dict(), self.model_save_path)
                mlflow.log_artifact(self.model_save_path)
                logger.info(f"Saved best model to {self.model_save_path} (val={val_loss:.4f})")

                # plot singular values for the best model weights (per-layer Q/K/V)
                self._plot_best_model_svals()

            if self.early_stopper.step(val_loss):
                logger.warning(f"Early stopping at epoch {epoch}")
                break

        for k, v in self.best_spectral_metrics.items():
            if np.isfinite(v):
                mlflow.log_metric(f"best_{k}", float(v))

        return self.best_val_loss, self.best_train_metrics, self.best_val_metrics, self.best_spectral_metrics

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        losses = []
        with torch.enable_grad() if train else torch.no_grad():
            for src, tgt in loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                out = self.model(src, tgt)
                loss = self.criterion(out, tgt)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    def _collect_metrics(self, loader: DataLoader, loss_value: float) -> dict:
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for src, tgt in loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                preds.append(self.model(src, tgt))
                targets.append(tgt)
        preds = torch.cat(preds) if preds else torch.empty(0)
        targets = torch.cat(targets) if targets else torch.empty(0)
        m = compute_training_metrics(preds, targets)
        m["loss"] = loss_value
        return m

    def _metrics_with_powerlaws(self, W: np.ndarray) -> dict[str, float]:
        # get_spectral_metrics now already includes Hill and KS on the ESD
        m = get_spectral_metrics(W)
        return {k: (float(v) if np.isfinite(v) else float("nan")) for k, v in m.items()}

    def _log_qkv_metrics(self, prefix: str, tail: str, group: str, epoch: int) -> dict[str, float]:
        sd = self.model.state_dict()
        out: dict[str, float] = {}

        for i in _iter_layer_idxs(sd, prefix, tail):
            W = _get_in_proj(sd, prefix, tail, i)
            if W is None or W.ndim != 2:
                continue
            try:
                Q, K, V = _split_in_proj_qkv(W)
            except Exception as e:
                logger.warning(f"QKV split failed for {group}.L{i}: {e}")
                continue

            for tag, M in (("q", Q), ("k", K), ("v", V)):
                metrics = self._metrics_with_powerlaws(M)
                payload = {f"{group}.L{i}.{tag}.{k}": v for k, v in metrics.items() if np.isfinite(v)}
                if payload:
                    mlflow.log_metrics(payload, step=epoch)
                    out.update(payload)

        return out

    def _log_qkv_figs(self, prefix: str, tail: str, group: str, epoch: int) -> None:
        sd = self.model.state_dict()

        for i in _iter_layer_idxs(sd, prefix, tail):
            W = _get_in_proj(sd, prefix, tail, i)
            if W is None or W.ndim != 2:
                continue

            try:
                Q, K, V = _split_in_proj_qkv(W)
            except Exception as e:
                logger.warning(f"QKV split failed for {group}.L{i}: {e}")
                continue

            for tag, M in (("q", Q), ("k", K), ("v", V)):
                figs = _plot_esd_hist(M, title=f"{group} L{i} {tag}", bins=None)
                for j, fig in enumerate(figs):
                    name = "linear" if j == 0 else "loglog"
                    mlflow.log_figure(fig, f"esd_{group}_L{i}_{tag}_epoch{epoch}_{name}.png")
                    plt.close(fig)

                fig_sv = _plot_svals_inline(M, title=f"{group} L{i} {tag}")
                mlflow.log_figure(fig_sv, f"sv_{group}_L{i}_{tag}_epoch{epoch}.png")
                plt.close(fig_sv)

    def _plot_best_model_svals(self) -> None:
        sd = self.model.state_dict()

        def handle_group(prefix: str, tail: str, group: str):
            for i in _iter_layer_idxs(sd, prefix, tail):
                W = _get_in_proj(sd, prefix, tail, i)
                if W is None or W.ndim != 2:
                    continue
                try:
                    Q, K, V = _split_in_proj_qkv(W)
                except Exception as e:
                    logger.warning(f"QKV split failed for {group}.L{i} during best-plot: {e}")
                    continue

                for tag, M in (("q", Q), ("k", K), ("v", V)):
                    # create a stable subdir per matrix
                    save_dir = self.out_dir / "best_svals" / f"{group}_L{i}_{tag}"
                    plot_singular_values(M, str(save_dir))

        handle_group(*self._enc)
        handle_group(*self._dec)
