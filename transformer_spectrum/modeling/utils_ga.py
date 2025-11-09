import torch


def flatten_model(model: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_model_from_flat(model: torch.nn.Module, flat: torch.Tensor):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            p.copy_(flat[offset:offset + numel].view_as(p))
            offset += numel
    return model


def make_objective_fn(model, loader, loss_fn, device, max_batches: int = 16, resample_each_call: bool = True):
    cached = []

    def _sample_batches():
        # draw fresh mini-batches each call
        cached.clear()
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            cached.append(batch)

    @torch.no_grad()
    def objective(solution: torch.Tensor) -> float:
        if resample_each_call or not cached:
            _sample_batches()

        set_model_from_flat(model, solution)
        model.eval()

        total = 0.0
        for src, tgt in cached:
            src, tgt = src.to(device), tgt.to(device)
            pred = model(src, tgt)
            total += loss_fn(pred, tgt).item()
        return - (total / len(cached))
    return objective
