import torch
import torch.nn.functional as F


@torch.no_grad()
def recall_at_k(query, target, ks=(1, 5, 10)):
    """
    Compute retrieval recall@K.

    query:  Tensor [N, D]
    target: Tensor [N, D]

    Assumes the i-th query corresponds to the i-th target.
    """
    query = F.normalize(query, dim=-1)
    target = F.normalize(target, dim=-1)

    sim = query @ target.t()
    ranks = torch.argsort(sim, dim=-1, descending=True)

    labels = torch.arange(query.shape[0], device=query.device).unsqueeze(1)

    results = {}
    for k in ks:
        hit = (ranks[:, :k] == labels).any(dim=1).float().mean()
        results[f"R@{k}"] = hit.item() * 100.0

    return results


@torch.no_grad()
def bidirectional_retrieval_metrics(a, b, prefix_ab="A-B", prefix_ba="B-A", ks=(1, 5, 10)):
    out = {}

    ab = recall_at_k(a, b, ks)
    ba = recall_at_k(b, a, ks)

    for k, v in ab.items():
        out[f"{prefix_ab}/{k}"] = v
    for k, v in ba.items():
        out[f"{prefix_ba}/{k}"] = v

    return out
