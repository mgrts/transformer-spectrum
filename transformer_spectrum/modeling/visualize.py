import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.linalg import svd
import torch
import mlflow


def plot_time_series(time_series):
    """
    Plots the given time series.

    Parameters:
        time_series (np.ndarray): The time series to plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, label='Non-Stationary Time Series with Varying Variance')
    plt.title('Non-Stationary Time Series Generation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_prediction(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pred: torch.Tensor,
    pred_infer: torch.Tensor,
    idx=0,
):
    """
    Visualizes a given sample including predictions.

    Args:
        src: source sequence [bs, src_seq_len, num_features]
        tgt: target sequence [bs, tgt_seq_len, num_features]
        pred: prediction of the model [bs, tgt_seq_len, num_features]
        pred_infer: prediction obtained by running inference
            [bs, tgt_seq_len, num_features]
        idx: batch index to visualize
    """
    x = np.arange(src.shape[1] + tgt.shape[1])
    src_len = src.shape[1]

    fig = plt.figure(figsize=(20, 10))

    plt.plot(x[:src_len], src[idx].cpu().detach(), 'bo-', label='src')
    plt.plot(x[src_len:], tgt[idx].cpu().detach(), 'go-', label='tgt')
    plt.plot(x[src_len:], pred[idx].cpu().detach(), 'ro-', label='pred')
    plt.plot(x[src_len:], pred_infer[idx].cpu().detach(), 'yo-', label='pred_infer')

    plt.legend()

    return fig


def plot_singular_values(weights, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    singular_values = svd(weights, compute_uv=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(singular_values, marker='o')
    ax[0].set_title(f'Singular Values')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Value')

    ax[1].plot(np.log1p(singular_values), marker='o')
    ax[1].set_title(f'Log Singular Values')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('log(1 + value)')

    fig.tight_layout()
    filename = os.path.join(save_dir, 'singular_values.png')
    fig.savefig(filename)
    mlflow.log_artifact(filename)
    plt.close(fig)

    return filename


def log_sample_images(model, val_loader, model_path, num_vis_examples: int = 5):
    """
    Log visualizations of predictions for num_vis_examples from the validation set.
    """
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        vis_count = 0
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            pred = model(src, tgt)
            pred_infer = model.infer(src, tgt_len=tgt.shape[1])

            batch_size = src.shape[0]
            for i in range(batch_size):
                if vis_count >= num_vis_examples:
                    break

                fig = visualize_prediction(src, tgt, pred, pred_infer, idx=i)
                mlflow.log_figure(fig, f'prediction_{vis_count}.png')
                plt.close(fig)

                vis_count += 1

            if vis_count >= num_vis_examples:
                break