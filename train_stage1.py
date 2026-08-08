import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from mindecho.utils.config import load_config
from mindecho.utils.seed import seed_everything
from mindecho.utils.checkpoint import save_checkpoint
from mindecho.data.fmri_dataset import FMRIDataset, fmri_collate_fn
from mindecho.models.mindecho import MindEchoEncoder
from mindecho.losses.objectives import Stage1MindEchoLoss


def build_model(cfg):
    subject_voxel_dims = {
        int(k): int(v)
        for k, v in cfg.subjects.voxel_dims.items()
    }

    model = MindEchoEncoder(
        subject_voxel_dims=subject_voxel_dims,
        shared_dim=cfg.model.shared_dim,
        latent_channels=cfg.model.latent_channels,
        latent_hw=cfg.model.latent_hw,
        clip_dim=cfg.model.clip_dim,
        semantic_hidden=cfg.model.semantic_hidden,
        structural_channels=cfg.model.structural_channels,
        structural_out_channels=cfg.model.structural_out_channels,
        dropout=cfg.model.dropout,
        cpd_kwargs={
            "perturb_prob": cfg.cpd.perturb_prob,
            "omega_min": cfg.cpd.omega_min,
            "omega_max": cfg.cpd.omega_max,
            "use_batch_shuffle": cfg.cpd.use_batch_shuffle,
        },
    )

    return model


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch,
    cfg,
):
    model.train()

    meters = {
        "loss": 0.0,
        "loss_sem": 0.0,
        "loss_img": 0.0,
        "loss_txt": 0.0,
        "loss_struct": 0.0,
        "loss_causal": 0.0,
    }

    pbar = tqdm(loader, desc=f"Stage1 Epoch {epoch}", dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        batch = move_batch_to_device(batch, device)

        fmri = batch["fmri"]
        subject_ids = batch["subject_id"]

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.stage1.amp):
            outputs = model(
                fmri=fmri,
                subject_ids=subject_ids,
                force_perturb=False,
            )
            loss_dict = criterion(outputs, batch)
            loss = loss_dict["loss"]

        if cfg.stage1.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k in meters:
            meters[k] += float(loss_dict[k].detach().cpu())

        if step % cfg.stage1.log_interval == 0:
            avg = {k: meters[k] / max(1, step + 1) for k in meters}
            pbar.set_postfix({k: f"{v:.4f}" for k, v in avg.items()})

    scheduler.step()

    return {k: v / max(1, len(loader)) for k, v in meters.items()}


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()

    meters = {
        "loss": 0.0,
        "loss_sem": 0.0,
        "loss_img": 0.0,
        "loss_txt": 0.0,
        "loss_struct": 0.0,
        "loss_causal": 0.0,
    }

    for batch in tqdm(loader, desc="Validate", dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)

        with autocast(enabled=cfg.stage1.amp):
            outputs = model(
                fmri=batch["fmri"],
                subject_ids=batch["subject_id"],
                force_perturb=True,
            )
            loss_dict = criterion(outputs, batch)

        for k in meters:
            meters[k] += float(loss_dict[k].detach().cpu())

    return {k: v / max(1, len(loader)) for k, v in meters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mindecho_nsd.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_set = FMRIDataset(
        cfg.data.train_file,
        struct_size=cfg.data.struct_size,
        normalize_clip=True,
    )

    val_set = FMRIDataset(
        cfg.data.val_file,
        struct_size=cfg.data.struct_size,
        normalize_clip=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.stage1.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=fmri_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.stage1.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=fmri_collate_fn,
    )

    model = build_model(cfg).to(device)

    criterion = Stage1MindEchoLoss(
        temperature=cfg.stage1.temperature,
        lambda_img=cfg.stage1.lambda_img,
        lambda_txt=cfg.stage1.lambda_txt,
        lambda_sem=cfg.stage1.lambda_sem,
        lambda_struct=cfg.stage1.lambda_struct,
        lambda_causal=cfg.stage1.lambda_causal,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.stage1.lr,
        weight_decay=cfg.stage1.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.stage1.epochs,
        eta_min=cfg.stage1.lr * 0.01,
    )

    scaler = GradScaler(enabled=cfg.stage1.amp)

    ckpt_dir = Path(cfg.project.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, cfg.stage1.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            cfg=cfg,
        )

        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        print(f"[Epoch {epoch}] train={train_stats}")
        print(f"[Epoch {epoch}] val={val_stats}")

        latest_path = ckpt_dir / "stage1_latest.pt"
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            extra={"train": train_stats, "val": val_stats},
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_path = ckpt_dir / "stage1_best.pt"
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                extra={"train": train_stats, "val": val_stats},
            )
            print(f"Saved best checkpoint to {best_path}")

        if epoch % cfg.stage1.save_interval == 0:
            save_checkpoint(
                ckpt_dir / f"stage1_epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                extra={"train": train_stats, "val": val_stats},
            )


if __name__ == "__main__":
    main()
