import torch


def compute_training_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> dict:
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()

    abs_diff = torch.abs(y_pred - y_true)
    denominator = torch.clamp(torch.abs(y_true), min=eps)

    mape = (abs_diff / denominator).mean().item() * 100

    smape = (abs_diff / torch.clamp((torch.abs(y_true) + torch.abs(y_pred)) / 2, min=eps)).mean().item() * 100

    return {
        'mape': round(mape, 4),
        'smape': round(smape, 4),
    }
