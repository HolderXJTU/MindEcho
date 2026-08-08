import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(query, target, temperature: float = 0.07):
    """
    Generic InfoNCE loss.

    Paper:
        L_con(v_sem, v_m)
        =
        - 1/B sum_i log exp(v_i dot m_i / temp)
        / sum_j exp(v_i dot m_j / temp)
    """
    query = F.normalize(query, dim=-1)
    target = F.normalize(target, dim=-1)

    logits = query @ target.t()
    logits = logits / temperature

    labels = torch.arange(query.shape[0], device=query.device)
    loss = F.cross_entropy(logits, labels)

    return loss


def bidirectional_contrastive_loss(a, b, temperature: float = 0.07):
    loss_ab = contrastive_loss(a, b, temperature)
    loss_ba = contrastive_loss(b, a, temperature)
    return 0.5 * (loss_ab + loss_ba)


def causal_consistency_loss(v_clean, v_perturbed):
    """
    L_causal = || f_theta(Z_i) - f_theta(Z_tilde_i) ||_2^2
    """
    return F.mse_loss(v_clean, v_perturbed)


def structural_loss(pred_struct, gt_struct):
    """
    L_struct = || m_struct - M_gt ||_2^2
    """
    if pred_struct.shape != gt_struct.shape:
        gt_struct = F.interpolate(
            gt_struct,
            size=pred_struct.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return F.mse_loss(pred_struct, gt_struct)


class Stage1MindEchoLoss(nn.Module):
    """
    Full Stage-1 objective:

    L_stage1 =
        lambda_sem * L_sem
        + lambda_struct * L_struct
        + lambda_causal * L_causal

    where:
        L_sem = lambda_img * L_con(v_sem, v_img)
              + lambda_txt * L_con(v_sem, v_txt)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_img: float = 1.0,
        lambda_txt: float = 0.5,
        lambda_sem: float = 1.0,
        lambda_struct: float = 10.0,
        lambda_causal: float = 1.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.temperature = temperature
        self.lambda_img = lambda_img
        self.lambda_txt = lambda_txt
        self.lambda_sem = lambda_sem
        self.lambda_struct = lambda_struct
        self.lambda_causal = lambda_causal
        self.bidirectional = bidirectional

    def _con(self, a, b):
        if self.bidirectional:
            return bidirectional_contrastive_loss(a, b, self.temperature)
        return contrastive_loss(a, b, self.temperature)

    def forward(self, outputs, batch):
        v_sem = outputs["v_sem"]
        v_clean = outputs["v_sem_clean"]
        v_perturbed = outputs["v_sem_perturbed"]
        m_struct = outputs["m_struct"]

        v_img = batch["clip_img"]
        v_txt = batch["clip_txt"]
        edge = batch["edge"]

        loss_img = self._con(v_sem, v_img)
        loss_txt = self._con(v_sem, v_txt)

        loss_sem = self.lambda_img * loss_img + self.lambda_txt * loss_txt
        loss_struct = structural_loss(m_struct, edge)
        loss_causal = causal_consistency_loss(v_clean, v_perturbed)

        total = (
            self.lambda_sem * loss_sem
            + self.lambda_struct * loss_struct
            + self.lambda_causal * loss_causal
        )

        return {
            "loss": total,
            "loss_sem": loss_sem.detach(),
            "loss_img": loss_img.detach(),
            "loss_txt": loss_txt.detach(),
            "loss_struct": loss_struct.detach(),
            "loss_causal": loss_causal.detach(),
        }
