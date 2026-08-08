from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_normalize(x, dim=-1, eps=1e-8):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def slerp(v0, v1, t, eps=1e-7):
    """
    Spherical linear interpolation.

    v0, v1: Tensor [B, D]
    t: Tensor [B, 1] or scalar-like tensor
    """
    v0 = safe_normalize(v0, dim=-1)
    v1 = safe_normalize(v1, dim=-1)

    dot = (v0 * v1).sum(dim=-1, keepdim=True)
    dot = dot.clamp(-1.0 + eps, 1.0 - eps)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)

    out = (
        torch.sin((1.0 - t) * omega) / sin_omega * v0
        + torch.sin(t * omega) / sin_omega * v1
    )

    return safe_normalize(out, dim=-1)


class ConfidenceAwareRetrieval(nn.Module):
    """
    Confidence-Aware Retrieval.

    It performs:
    1. top-K retrieval from image-text external memory
    2. attention-based prototype construction
    3. confidence-gated fusion
    4. SLERP between v_sem and v_proto

    Retrieval score:
        eta * dot(v_sem, k_img) + (1 - eta) * dot(v_sem, k_txt)

    Prototype:
        alpha_i = softmax(v_sem dot v_ret_i / tau)
        v_proto = sum alpha_i v_ret_i

    Confidence:
        lambda_base = sigmoid(mu * (sim_best - delta))
        lambda_final = lambda_base * sigmoid(gamma * (S_cons - beta))
    """

    def __init__(
        self,
        clip_dim: int = 768,
        top_k: int = 16,
        eta: float = 0.7,
        tau: float = 0.07,
        mu_init: float = 12.0,
        delta_init: float = 0.25,
        gamma_init: float = 10.0,
        beta_init: float = 0.3,
        learnable_gates: bool = True,
    ):
        super().__init__()

        self.clip_dim = clip_dim
        self.top_k = top_k
        self.eta = eta
        self.tau = tau

        param = nn.Parameter if learnable_gates else lambda x: x

        self.mu = param(torch.tensor(float(mu_init)))
        self.delta = param(torch.tensor(float(delta_init)))
        self.gamma = param(torch.tensor(float(gamma_init)))
        self.beta = param(torch.tensor(float(beta_init)))

    def compute_scores(self, query, mem_img, mem_txt):
        query = safe_normalize(query, dim=-1)
        mem_img = safe_normalize(mem_img, dim=-1)
        mem_txt = safe_normalize(mem_txt, dim=-1)

        sim_img = query @ mem_img.t()
        sim_txt = query @ mem_txt.t()

        scores = self.eta * sim_img + (1.0 - self.eta) * sim_txt
        return scores, sim_img, sim_txt

    def retrieve_topk(self, query, mem_img, mem_txt):
        scores, sim_img, sim_txt = self.compute_scores(query, mem_img, mem_txt)

        top_scores, top_indices = torch.topk(
            scores,
            k=min(self.top_k, scores.shape[-1]),
            dim=-1,
            largest=True,
            sorted=True,
        )

        top_img = mem_img[top_indices]
        top_txt = mem_txt[top_indices]

        return {
            "top_scores": top_scores,
            "top_indices": top_indices,
            "top_img": top_img,
            "top_txt": top_txt,
            "sim_img": sim_img,
            "sim_txt": sim_txt,
        }

    def build_prototype(self, query, top_img):
        query = safe_normalize(query, dim=-1)
        top_img = safe_normalize(top_img, dim=-1)

        logits = torch.einsum("bd,bkd->bk", query, top_img) / self.tau
        alpha = F.softmax(logits, dim=-1)

        proto = torch.einsum("bk,bkd->bd", alpha, top_img)
        proto = safe_normalize(proto, dim=-1)

        return proto, alpha

    def neighborhood_consistency(self, top_img):
        """
        Average pairwise cosine similarity within top-K neighborhood.
        """
        top_img = safe_normalize(top_img, dim=-1)

        b, k, d = top_img.shape
        sim = torch.bmm(top_img, top_img.transpose(1, 2))

        triu = torch.triu(
            torch.ones(k, k, device=top_img.device, dtype=torch.bool),
            diagonal=1,
        )

        pairwise = sim[:, triu]

        if pairwise.numel() == 0:
            return torch.ones(b, device=top_img.device)

        return pairwise.mean(dim=-1)

    def confidence_gate(self, query, top_img, proto):
        query = safe_normalize(query, dim=-1)
        top_img = safe_normalize(top_img, dim=-1)

        best = top_img[:, 0, :]
        align_strength = (query * best).sum(dim=-1)

        s_cons = self.neighborhood_consistency(top_img)

        lambda_base = torch.sigmoid(self.mu * (align_strength - self.delta))
        lambda_cons = torch.sigmoid(self.gamma * (s_cons - self.beta))
        lambda_final = lambda_base * lambda_cons

        return lambda_final.clamp(0.0, 1.0), {
            "align_strength": align_strength,
            "s_cons": s_cons,
            "lambda_base": lambda_base,
            "lambda_cons": lambda_cons,
        }

    def forward(self, query, mem_img, mem_txt):
        ret = self.retrieve_topk(query, mem_img, mem_txt)

        proto, alpha = self.build_prototype(
            query=query,
            top_img=ret["top_img"],
        )

        lamb, gate_info = self.confidence_gate(
            query=query,
            top_img=ret["top_img"],
            proto=proto,
        )

        v_final = slerp(query, proto, lamb.unsqueeze(-1))

        return {
            "v_final": v_final,
            "v_proto": proto,
            "lambda_final": lamb,
            "alpha": alpha,
            **ret,
            **gate_info,
        }
