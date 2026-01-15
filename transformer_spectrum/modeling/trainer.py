import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import GeneticAlgorithm
from evotorch.logging import StdOutLogger
from evotorch.operators import SimulatedBinaryCrossOver, GaussianMutation
from loguru import logger
from torch.utils.data import DataLoader

from transformer_spectrum.settings import TRACKING_URI, TRACK_EPOCHS
from transformer_spectrum.metrics.spectral_metrics import get_spectral_metrics
from transformer_spectrum.metrics.training_metrics import compute_training_metrics
from transformer_spectrum.modeling.utils import EarlyStopping
from transformer_spectrum.modeling.utils_ga import flatten_model, set_model_from_flat, make_objective_fn
from transformer_spectrum.modeling.visualize import log_sample_images
from transformer_spectrum.modeling.visualize import plot_singular_values

mlflow.set_tracking_uri(TRACKING_URI)


def _singular_values(weights: np.ndarray) -> np.ndarray:
    return np.linalg.svd(weights, compute_uv=False)


def _esd(weights: np.ndarray) -> np.ndarray:
    s = _singular_values(weights)
    return s * s


def _split_in_proj_qkv(in_proj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if in_proj.shape[0] % 3 != 0:
        raise ValueError(f'in_proj has unexpected shape {in_proj.shape}')
    return np.split(in_proj, 3, axis=0)


def _plot_esd_hist(weights: np.ndarray, title: str, bins: int | None = None) -> List[plt.Figure]:
    lam = _esd(weights)
    lam = lam[np.isfinite(lam) & (lam >= 0)]
    if lam.size == 0:
        return []
    if bins is None:
        bins = int(np.clip(np.sqrt(lam.size), 10, 60))

    figs: list[plt.Figure] = []

    fig1 = plt.figure()
    plt.hist(lam, bins=bins)
    plt.title(f'ESD - {title}')
    plt.xlabel('eigenvalue')
    plt.ylabel('count')
    figs.append(fig1)

    lam_pos = lam[lam > 0]
    if lam_pos.size > 0:
        lo, hi = lam_pos.min(), lam_pos.max()
        if hi > lo:
            log_bins = np.logspace(np.log10(lo), np.log10(hi), bins)
            fig2 = plt.figure()
            plt.hist(lam_pos, bins=log_bins)
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f'ESD (log-log) - {title}')
            plt.xlabel('eigenvalue (log)')
            plt.ylabel('count (log)')
            figs.append(fig2)

    return figs


def _plot_svals_inline(weights: np.ndarray, title: str) -> plt.Figure:
    s = _singular_values(weights)
    s = s[np.isfinite(s) & (s > 0)]
    fig = plt.figure()
    if s.size == 0:
        plt.title(f'Singular values - {title} (empty)')
        return fig
    s = np.sort(s)[::-1]
    x = np.arange(1, s.size + 1)
    plt.loglog(x, s, marker='.')
    plt.title(f'Singular values - {title}')
    plt.xlabel('index')
    plt.ylabel('singular value')
    return fig


def _iter_layer_idxs(sd: dict, prefix: str, tail: str) -> list[int]:
    out, i = [], 0
    while f'{prefix}.{i}.{tail}' in sd:
        out.append(i)
        i += 1
    return out


def _get_in_proj(sd: dict, prefix: str, tail: str, i: int) -> np.ndarray | None:
    t = sd.get(f'{prefix}.{i}.{tail}')
    return None if t is None else t.detach().cpu().numpy()


class SpectralLoggingMixin:
    def log_qkv_metrics(self, step: int) -> Dict[str, float]:
        """
        Log spectral metrics (Q/K/V) for encoder & decoder at a given step.
        """
        sd = self.model.state_dict()
        out: dict[str, float] = {}

        for prefix, tail, group in (self._enc, self._dec):
            for i in _iter_layer_idxs(sd, prefix, tail):
                W = _get_in_proj(sd, prefix, tail, i)

                if W is None or W.ndim != 2:
                    continue
                try:
                    Q, K, V = _split_in_proj_qkv(W)
                except Exception as e:
                    logger.warning(f'QKV split failed for {group}.L{i}: {e}')
                    continue

                for tag, M in (('q', Q), ('k', K), ('v', V)):
                    metrics = get_spectral_metrics(M)
                    payload = {
                        f'{group}.L{i}.{tag}.{k}': float(v)
                        for k, v in metrics.items()
                        if np.isfinite(v)
                    }
                    if payload:
                        mlflow.log_metrics(payload, step=step)
                        out.update(payload)
        return out

    def log_qkv_figs(self, step: int) -> None:
        """
        Log ESD histograms (linear & log-log) and SV curves for Q/K/V.
        Dump JSON spectra on steps that are in self.track_epochs.
        """
        sd = self.model.state_dict()
        dump_json_arrays = step in self.track_epochs

        for prefix, tail, group in (self._enc, self._dec):
            for i in _iter_layer_idxs(sd, prefix, tail):
                W = _get_in_proj(sd, prefix, tail, i)

                if W is None or W.ndim != 2:
                    continue

                try:
                    Q, K, V = _split_in_proj_qkv(W)
                except Exception as e:
                    logger.warning(f'QKV split failed for {group}.L{i}: {e}')
                    continue

                for tag, M in (('q', Q), ('k', K), ('v', V)):
                    title = f'epoch {step} | {group} L{i} {tag}'

                    # figures
                    figs = _plot_esd_hist(M, title=title, bins=None)
                    for j, fig in enumerate(figs):
                        name = 'linear' if j == 0 else 'loglog'
                        mlflow.log_figure(fig, f'esd_{group}_L{i}_{tag}_epoch{step}_{name}.png')
                        plt.close(fig)

                    fig_sv = _plot_svals_inline(M, title=title)
                    mlflow.log_figure(fig_sv, f'sv_{group}_L{i}_{tag}_epoch{step}.png')
                    plt.close(fig_sv)

                    # optional JSON spectra
                    if dump_json_arrays:
                        singular_vals = _singular_values(M)
                        singular_vals = singular_vals[np.isfinite(singular_vals) & (singular_vals >= 0)]
                        singular_vals = np.sort(singular_vals)[::-1]
                        lam = singular_vals * singular_vals

                        spectra_dir = self.out_dir / 'spectra'
                        spectra_dir.mkdir(parents=True, exist_ok=True)

                        sv_path = spectra_dir / f'singular_vals_{group}_L{i}_{tag}_epoch{step}.json'
                        esd_path = spectra_dir / f'esd_vals_{group}_L{i}_{tag}_epoch{step}.json'

                        with open(sv_path, 'w') as f:
                            json.dump(singular_vals.tolist(), f)
                        with open(esd_path, 'w') as f:
                            json.dump(lam.tolist(), f)

                        mlflow.log_artifact(str(sv_path))
                        mlflow.log_artifact(str(esd_path))

    def eval_metrics(self, loader: DataLoader, loss_fn) -> dict:
        self.model.eval()

        preds, targets = [], []
        total_loss = 0.0

        with torch.no_grad():
            for src, tgt in loader:
                src, tgt = src.to(self.device), tgt.to(self.device)
                out = self.model(src, tgt)
                preds.append(out)
                targets.append(tgt)
                total_loss += loss_fn(out, tgt).item()

        if preds:
            preds = torch.cat(preds)
            targets = torch.cat(targets)
            m = compute_training_metrics(preds, targets)
        else:
            m = {'mape': float('nan'), 'smape': float('nan')}

        m['loss'] = (total_loss / len(loader)) if len(loader) > 0 else float('nan')

        return m

    def plot_best_model_svals(self) -> None:
        sd = self.model.state_dict()

        def handle_group(prefix: str, tail: str, group: str):
            for i in _iter_layer_idxs(sd, prefix, tail):

                W = _get_in_proj(sd, prefix, tail, i)

                if W is None or W.ndim != 2:
                    continue
                try:
                    Q, K, V = _split_in_proj_qkv(W)
                except Exception as e:
                    logger.warning(f'QKV split failed for {group}.L{i} during best-plot: {e}')
                    continue
                for tag, M in (('q', Q), ('k', K), ('v', V)):
                    save_dir = self.out_dir / 'best_svals' / f'{group}_L{i}_{tag}'
                    plot_singular_values(M, str(save_dir))

        handle_group(*self._enc)
        handle_group(*self._dec)


class Trainer(SpectralLoggingMixin):
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
            device: str = 'cuda',
            track_epochs: tuple | list = TRACK_EPOCHS,
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

        self.track_epochs = set(
            int(e) for e in (track_epochs if isinstance(track_epochs, (tuple, list)) else [track_epochs])
        )
        self.esd_plot_epochs = set(int(e) for e in track_epochs)

        self.best_val_loss = float('inf')
        self.best_train_metrics: dict = {}
        self.best_val_metrics: dict = {}
        self.best_spectral_metrics: dict = {}

        mlflow.set_experiment(experiment_name)

        self._enc = ('transformer.encoder.layers', 'self_attn.in_proj_weight', 'encoder')
        self._dec = ('transformer.decoder.layers', 'multihead_attn.in_proj_weight', 'decoder')

    def train(self):
        # Baseline at epoch 0
        if 0 in self.track_epochs:
            self.model.eval()
            with torch.no_grad():
                self.log_qkv_figs(step=0)
                base_spec_enc_dec = self.log_qkv_metrics(step=0)
                logger.info(f'Baseline spectral metrics logged: count={len(base_spec_enc_dec)}')

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True)
            val_loss = self._run_epoch(self.val_loader, train=False)

            train_m = self.eval_metrics(self.train_loader, self.criterion)
            val_m = self.eval_metrics(self.val_loader, self.criterion)

            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mape': train_m['mape'],
                'train_smape': train_m['smape'],
                'val_mape': val_m['mape'],
                'val_smape': val_m['smape'],
            }, step=epoch)

            logger.info(
                f'Epoch {epoch}/{self.num_epochs} '
                f'| Train {train_loss:.4f} | Val {val_loss:.4f} '
                f'| sMAPE train {train_m["smape"]:.2f} / val {val_m["smape"]:.2f}'
            )

            # spectral logging
            spectral_snapshot = self.log_qkv_metrics(step=epoch)
            if epoch in self.esd_plot_epochs:
                self.log_qkv_figs(step=epoch)

            # track the best by val loss
            if val_loss < self.best_val_loss - 1e-6:
                self.best_val_loss = val_loss
                self.best_train_metrics = train_m
                self.best_val_metrics = val_m
                self.best_spectral_metrics = spectral_snapshot

                torch.save(self.model.state_dict(), self.model_save_path)
                mlflow.log_artifact(self.model_save_path)
                logger.info(f'Saved best model to {self.model_save_path} (val={val_loss:.4f})')

                self.plot_best_model_svals()

            if self.early_stopper.step(val_loss):
                logger.warning(f'Early stopping at epoch {epoch}')
                break

        for k, v in self.best_spectral_metrics.items():
            if np.isfinite(v):
                mlflow.log_metric(f'best_{k}', float(v))

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
        return float(np.mean(losses)) if losses else float('nan')


class TrainerGA(SpectralLoggingMixin):
    def __init__(
            self,
            experiment_name: str,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion,
            model_save_path: Path,
            device: str,
            ga_generations: int,
            ga_popsize: int,
            ga_mutation_stdev: float,
            ga_crossover_eta: float,
            ga_eval_max_batches: int = 32,
            track_epochs: tuple | list = TRACK_EPOCHS,
            cross_over_rate: float = 0.9,
            tournament_size: int = 3,
            init_bounds: tuple[float, float] | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.model_save_path = model_save_path
        self.out_dir = model_save_path.parent
        self.device = device

        self.ga_generations = int(ga_generations)
        self.ga_popsize = int(ga_popsize)
        self.ga_mutation_stdev = float(ga_mutation_stdev)
        self.ga_crossover_eta = float(ga_crossover_eta)
        self.cross_over_rate = float(cross_over_rate)
        self.tournament_size = int(tournament_size)
        self.ga_eval_max_batches = int(ga_eval_max_batches)
        self.init_bounds = init_bounds

        self.track_epochs = set(
            int(e) for e in (track_epochs if isinstance(track_epochs, (tuple, list)) else [track_epochs]))

        mlflow.set_experiment(experiment_name)

        self._enc = ('transformer.encoder.layers', 'self_attn.in_proj_weight', 'encoder')
        self._dec = ('transformer.decoder.layers', 'multihead_attn.in_proj_weight', 'decoder')

    def _make_objective(self):
        return make_objective_fn(
            self.model,
            self.train_loader,
            self.criterion,
            self.device,
            max_batches=self.ga_eval_max_batches,
            resample_each_call=True,
        )

    def fit(self):
        init_vec = flatten_model(self.model).to(self.device)
        dim = init_vec.numel()
        objective_fn = self._make_objective()

        problem = Problem(
            objective_sense='max',
            objective_func=objective_fn,
            solution_length=dim,
            dtype=torch.float32,
            device=self.device,
            initial_bounds=self.init_bounds,
        )

        algo = GeneticAlgorithm(
            problem,
            popsize=self.ga_popsize,
            operators=[
                SimulatedBinaryCrossOver(
                    problem,
                    cross_over_rate=self.cross_over_rate,
                    eta=self.ga_crossover_eta,
                    tournament_size=self.tournament_size,
                ),
                GaussianMutation(problem, stdev=self.ga_mutation_stdev),
            ],
            elitist=True,
        )

        StdOutLogger(algo)

        # Baseline logs at 'gen 0'
        self.log_qkv_figs(step=0)
        base_spec = self.log_qkv_metrics(step=0)

        logger.info(f'Baseline spectral metrics logged: count={len(base_spec)}')

        base_train = self.eval_metrics(self.train_loader, self.criterion)
        base_val = self.eval_metrics(self.val_loader, self.criterion)

        mlflow.log_metrics({
            'train_loss': base_train['loss'],
            'val_loss': base_val['loss'],
            'train_mape': base_train['mape'],
            'train_smape': base_train['smape'],
            'val_mape': base_val['mape'],
            'val_smape': base_val['smape'],
        }, step=0)

        best_obj_ever = -float('inf')
        best_sol_ever: torch.Tensor | None = None
        hall: list[tuple[float, torch.Tensor]] = []

        for gen in range(1, self.ga_generations + 1):
            algo.step()

            population = [
                (float(sol.evaluation), sol.values.detach().clone()) for sol in algo.population
            ]
            hall.extend(population)
            hall = sorted(hall, key=lambda x: x[0], reverse=True)[:5]

            best_obj, best_sol = hall[0]
            mlflow.log_metric('ga_best_objective', best_obj, step=gen)

            # materialize best
            set_model_from_flat(self.model, best_sol)

            # metrics for current best model
            tr = self.eval_metrics(self.train_loader, self.criterion)
            vl = self.eval_metrics(self.val_loader, self.criterion)

            mlflow.log_metrics({
                'train_loss': tr['loss'],
                'val_loss': vl['loss'],
                'train_mape': tr['mape'],
                'train_smape': tr['smape'],
                'val_mape': vl['mape'],
                'val_smape': vl['smape'],
            }, step=gen)

            self.log_qkv_metrics(step=gen)

            if gen in self.track_epochs:
                self.log_qkv_figs(step=gen)

            if best_obj > best_obj_ever + 1e-12:
                best_obj_ever = best_obj
                best_sol_ever = best_sol.clone()

        # Save best model and final plots
        if best_sol_ever is not None:
            set_model_from_flat(self.model, best_sol_ever)

        torch.save(self.model.state_dict(), self.model_save_path)

        mlflow.log_metric('best_objective', best_obj_ever)

        logger.info(f'Saved best model to {self.model_save_path} (objective={best_obj_ever:.6f})')

        self.plot_best_model_svals()

        try:
            log_sample_images(self.model, self.val_loader, model_path=self.model_save_path)
        except Exception as e:
            logger.warning(f'log_sample_images failed: {e}')

        return best_obj_ever


@dataclass
class ESConfig:
    popsize: int = 128
    sigma: float = 0.15
    lr: float = 0.05
    lr_sigma: float = 0.0
    antithetic: bool = True
    rank_transform: bool = True
    max_batches_per_eval: int = 16
    resample_each_call: bool = False
    log_every: int = 5
    save_top_k: int = 5
    track_epochs: Tuple[int, ...] = (0, 10, 25, 50, 100)


class TrainerES(SpectralLoggingMixin):
    def __init__(
            self,
            experiment_name: str,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion,
            model_save_path: Path,
            device: str | torch.device,
            es_cfg: ESConfig = ESConfig(),
            use_train_for_objective: bool = True,
            eval_budget_per_gen: int | None = None,
            heartbeat_every: int = 32,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.model_save_path = model_save_path
        self.out_dir = model_save_path.parent
        self.device = device
        self.cfg = es_cfg
        self.use_train_for_objective = use_train_for_objective
        self.eval_budget_per_gen = eval_budget_per_gen
        self.heartbeat_every = int(max(1, heartbeat_every))

        mlflow.set_experiment(experiment_name)

        # encoder/decoder weight paths for QKV logging
        self._enc = ('transformer.encoder.layers', 'self_attn.in_proj_weight', 'encoder')
        self._dec = ('transformer.decoder.layers', 'multihead_attn.in_proj_weight', 'decoder')

        # which generations also dump JSON spectra
        self.track_epochs = set(
            int(e) for e in (
                self.cfg.track_epochs if isinstance(self.cfg.track_epochs, (tuple, list))
                else [self.cfg.track_epochs]
            )
        )

        # flatten current weights as initial mean
        with torch.no_grad():
            self.mu = flatten_model(self.model).detach().to(device).float()

        self.dim = self.mu.numel()
        if self.cfg.antithetic and (self.cfg.popsize % 2 != 0):
            raise ValueError('ESConfig.popsize must be even when antithetic=True')

        self.sigma = torch.tensor(float(self.cfg.sigma), device=device)
        self.best_objective = -float('inf')
        self.best_solution = self.mu.clone()

        # build a small cache of device-resident batches to speed up evaluation
        loader_for_obj = self.train_loader if self.use_train_for_objective else self.val_loader
        self._cached_batches = self._make_cached_batches(loader_for_obj, self.cfg.max_batches_per_eval)

    def _make_cached_batches(self, loader: DataLoader, max_batches: int):
        cached = []
        for i, (src, tgt) in enumerate(loader):
            if i >= max_batches:
                break
            cached.append((
                src.to(self.device, non_blocking=True),
                tgt.to(self.device, non_blocking=True)
            ))
        return cached

    def _objective_value(self, solution: torch.Tensor) -> float:
        set_model_from_flat(self.model, solution)
        self.model.eval()

        # obey per-generation budget if set
        batches = self._cached_batches
        if self._per_gen_budget_remaining is not None:
            limit = min(len(batches), self._per_gen_budget_remaining)
            if limit <= 0:
                return -float('inf')  # out of budget
            batches = batches[:limit]
            self._per_gen_budget_remaining -= limit

        use_amp = (
                          isinstance(self.device, torch.device) and self.device.type == 'cuda'
                  ) or (
                          not isinstance(self.device, torch.device) and str(self.device) == 'cuda'
                  )

        ctx = torch.amp.autocast('cuda', enabled=use_amp) if use_amp else contextlib.nullcontext()

        total = 0.0
        with torch.inference_mode(), ctx:
            for (src, tgt) in batches:
                pred = self.model(src, tgt)
                total += float(self.criterion(pred, tgt).item())

        mean_loss = total / max(1, len(batches))

        return -mean_loss

    @torch.no_grad()
    def _eval_population(self, candidates: torch.Tensor) -> torch.Tensor:
        vals = torch.empty(candidates.size(0), device=self.device, dtype=torch.float32)
        for i in range(candidates.size(0)):
            if (i % self.heartbeat_every) == 0 and i > 0:
                logger.info(f'[ES] evaluating candidate {i}/{candidates.size(0)} in current generation...')
            vals[i] = self._objective_value(candidates[i])
        return vals

    def _rank_transform(self, scores: torch.Tensor) -> torch.Tensor:
        r = scores.argsort().argsort().float()
        return (r - r.mean()) / (r.std() + 1e-8)

    def fit(self, generations: int) -> tuple[float, List[tuple[float, torch.Tensor]]]:
        popsize = self.cfg.popsize
        sigma = self.sigma

        # Baseline logs at gen 0
        set_model_from_flat(self.model, self.mu)
        self.log_qkv_metrics(step=0)

        if 0 in self.track_epochs:
            self.log_qkv_figs(step=0)

        base_train = self.eval_metrics(self.train_loader, self.criterion)
        base_val = self.eval_metrics(self.val_loader, self.criterion)

        mlflow.log_metrics({
            'train_loss': base_train['loss'],
            'val_loss': base_val['loss'],
            'train_mape': base_train['mape'],
            'train_smape': base_train['smape'],
            'val_mape': base_val['mape'],
            'val_smape': base_val['smape'],
        }, step=0)

        hall: List[tuple[float, torch.Tensor]] = []

        for gen in range(1, generations + 1):
            logger.info(f'[ES] Starting generation {gen}/{generations}...')

            # refresh per-gen eval budget
            self._per_gen_budget_remaining = None if self.eval_budget_per_gen is None else int(self.eval_budget_per_gen)

            # sample perturbations (mirrored for antithetic)
            half = popsize // 2 if self.cfg.antithetic else popsize
            Z = torch.randn(half, self.dim, device=self.device, dtype=torch.float32)

            if self.cfg.antithetic:
                Z = torch.cat([Z, -Z], dim=0)

            # form candidate population around mean
            C = self.mu.unsqueeze(0) + sigma * Z

            # evaluate (maximize)
            scores = self._eval_population(C)

            # Hall of fame, best-so-far
            best_idx = int(torch.argmax(scores).item())
            best_score = float(scores[best_idx].item())
            hall.append((best_score, C[best_idx].detach().clone()))
            hall = sorted(hall, key=lambda x: x[0], reverse=True)[: self.cfg.save_top_k]

            if best_score > self.best_objective:
                self.best_objective = best_score
                self.best_solution = hall[0][1].detach().clone()
                set_model_from_flat(self.model, self.best_solution)
                torch.save(self.model.state_dict(), self.model_save_path)
                mlflow.log_artifact(self.model_save_path)

            # Fitness shaping
            w = scores.clone()
            if self.cfg.rank_transform:
                w = self._rank_transform(w)
            else:
                w = (w - w.mean()) / (w.std() + 1e-8)

            # Gradient estimate wrt mu (OpenAI-ES)
            grad_mu = (w.unsqueeze(1) * Z).mean(dim=0) / (sigma + 1e-8)
            self.mu = self.mu + self.cfg.lr * grad_mu

            # Optional isotropic sigma adaptation
            if self.cfg.lr_sigma > 0:
                z2 = (Z ** 2).sum(dim=1)
                grad_log_sigma = ((w * (z2 - self.dim))).mean() / 2.0
                sigma = sigma * torch.exp(self.cfg.lr_sigma * grad_log_sigma)
                sigma = torch.clamp(sigma, 1e-4, 2.0)
                self.sigma = sigma

            # Population progress metrics
            mlflow.log_metrics({
                'es_sigma': float(sigma.item()),
                'es_population_mean_obj': float(scores.mean().item()),
                'es_population_median_obj': float(scores.median().item()),
                'es_population_best_obj': best_score,
                'es_population_worst_obj': float(scores.min().item()),
                'es_best_objective': float(self.best_objective),
            }, step=gen)

            # Log train/val metrics for current *best* solution
            set_model_from_flat(self.model, self.best_solution)
            tr = self.eval_metrics(self.train_loader, self.criterion)
            vl = self.eval_metrics(self.val_loader, self.criterion)

            mlflow.log_metrics({
                'train_loss': tr['loss'],
                'val_loss': vl['loss'],
                'train_mape': tr['mape'],
                'train_smape': tr['smape'],
                'val_mape': vl['mape'],
                'val_smape': vl['smape'],
            }, step=gen)

            # Spectral metrics every gen; heavy figs only on selected gens
            self.log_qkv_metrics(step=gen)
            if gen in self.track_epochs:
                self.log_qkv_figs(step=gen)

            if (gen % self.cfg.log_every) == 0:
                logger.info(
                    f'[ES] gen {gen}/{generations} | '
                    f'mean={float(scores.mean()):.4f} | best={best_score:.4f} | '
                    f'sigma={float(sigma):.4f} | val_smape={vl["smape"]:.3f}'
                )

        # Finalize with best solution
        set_model_from_flat(self.model, self.best_solution)
        self.plot_best_model_svals()

        mlflow.log_metric('best_objective', float(self.best_objective))

        try:
            log_sample_images(self.model, self.val_loader, model_path=self.model_save_path)
        except Exception as e:
            logger.warning(f'log_sample_images failed: {e}')

        return float(self.best_objective), hall
