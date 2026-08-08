from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class FMRIDataset(Dataset):
    """
    Generic fMRI dataset for MindEcho.

    Expected serialized file format:

    A torch list of dictionaries:

    [
        {
            "fmri": Tensor [Q_i],
            "subject_id": int,
            "image": Tensor [3, H, W],
            "clip_img": Tensor [D],
            "clip_txt": Tensor [D],
            "edge": Tensor [1, Hs, Ws],
            "image_id": optional
        },
        ...
    ]

    This class intentionally keeps preprocessing light. For NSD/GOD, it is better
    to preprocess heavy objects such as CLIP features and Canny edges offline.
    """

    def __init__(
        self,
        file_path: str,
        struct_size: int = 64,
        normalize_clip: bool = True,
    ):
        self.file_path = Path(file_path)
        self.struct_size = struct_size
        self.normalize_clip = normalize_clip

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.file_path}. "
                f"Please preprocess NSD/GOD into the expected .pt format."
            )

        self.samples = torch.load(self.file_path, map_location="cpu")

        if not isinstance(self.samples, list):
            raise TypeError("Dataset file should contain a list of dictionaries.")

    def __len__(self):
        return len(self.samples)

    def _norm(self, x):
        return F.normalize(x.float(), dim=-1)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]

        fmri = item["fmri"].float()
        subject_id = int(item["subject_id"])
        image = item.get("image", torch.empty(0)).float()

        clip_img = item["clip_img"].float()
        clip_txt = item["clip_txt"].float()

        if self.normalize_clip:
            clip_img = self._norm(clip_img)
            clip_txt = self._norm(clip_txt)

        edge = item.get("edge", None)
        if edge is None:
            edge = torch.zeros(1, self.struct_size, self.struct_size)
        else:
            edge = edge.float()
            if edge.ndim == 2:
                edge = edge.unsqueeze(0)
            if edge.shape[-1] != self.struct_size or edge.shape[-2] != self.struct_size:
                edge = F.interpolate(
                    edge.unsqueeze(0),
                    size=(self.struct_size, self.struct_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        return {
            "fmri": fmri,
            "subject_id": subject_id,
            "image": image,
            "clip_img": clip_img,
            "clip_txt": clip_txt,
            "edge": edge,
            "index": index,
            "image_id": item.get("image_id", index),
        }


def fmri_collate_fn(batch):
    """
    Collate function for a multi-subject setting.

    Important:
    Different subjects may have different voxel dimensions. For simplicity,
    this engineering version assumes each batch has already been padded or
    contains same-dimensional fMRI vectors per dataset file.

    If you mix raw NSD subjects with different Q_i, preprocess them into:
        - padded vectors
        - or subject-specific files
        - or store them as list and route individually
    """
    output = {}

    for key in batch[0].keys():
        values = [b[key] for b in batch]

        if torch.is_tensor(values[0]):
            try:
                output[key] = torch.stack(values, dim=0)
            except RuntimeError:
                output[key] = values
        else:
            output[key] = values

    return output
