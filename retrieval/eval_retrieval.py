import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mindecho.utils.config import load_config
from mindecho.utils.checkpoint import load_checkpoint
from mindecho.data.fmri_dataset import FMRIDataset, fmri_collate_fn
from mindecho.train_stage1 import build_model, move_batch_to_device
from mindecho.metrics.retrieval_metrics import bidirectional_retrieval_metrics


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()

    brain_feats = []
    image_feats = []
    text_feats = []

    for batch in tqdm(loader, desc="Extract features", dynamic_ncols=True):
        batch = move_batch_to_device(batch, device)

        out = model.infer(
            fmri=batch["fmri"],
            subject_ids=batch["subject_id"],
        )

        brain_feats.append(out["v_sem"].cpu())
        image_feats.append(batch["clip_img"].cpu())
        text_feats.append(batch["clip_txt"].cpu())

    brain_feats = torch.cat(brain_feats, dim=0)
    image_feats = torch.cat(image_feats, dim=0)
    text_feats = torch.cat(text_feats, dim=0)

    return brain_feats, image_feats, text_feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mindecho_nsd.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage1_best.pt")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split_file = args.split_file or cfg.data.val_file

    dataset = FMRIDataset(
        split_file,
        struct_size=cfg.data.struct_size,
        normalize_clip=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.stage1.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=fmri_collate_fn,
    )

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, strict=True)

    brain, image, text = extract_features(model, loader, device)

    ks = tuple(cfg.eval.retrieval_topk)

    metrics = {}
    metrics.update(
        bidirectional_retrieval_metrics(
            brain,
            image,
            prefix_ab="Brain-Image",
            prefix_ba="Image-Brain",
            ks=ks,
        )
    )
    metrics.update(
        bidirectional_retrieval_metrics(
            brain,
            text,
            prefix_ab="Brain-Text",
            prefix_ba="Text-Brain",
            ks=ks,
        )
    )

    print("Retrieval metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")


if __name__ == "__main__":
    main()
