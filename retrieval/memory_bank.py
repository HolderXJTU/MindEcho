from pathlib import Path
import torch
import torch.nn.functional as F


class MemoryBank:
    """
    External visual-textual semantic memory.

    Expected file:
        {
            "image_features": Tensor [N, D],
            "text_features": Tensor [N, D],
            "meta": optional list
        }

    The memory should be built offline from COCO / ImageNet after leakage removal.
    """

    def __init__(self, memory_file: str, device="cpu", normalize=True):
        self.memory_file = Path(memory_file)
        self.device = device
        self.normalize = normalize

        if not self.memory_file.exists():
            raise FileNotFoundError(
                f"Memory file not found: {self.memory_file}. "
                f"Please run tools/build_external_memory.py first."
            )

        data = torch.load(self.memory_file, map_location="cpu")

        self.image_features = data["image_features"].float()
        self.text_features = data["text_features"].float()
        self.meta = data.get("meta", None)

        if normalize:
            self.image_features = F.normalize(self.image_features, dim=-1)
            self.text_features = F.normalize(self.text_features, dim=-1)

        self.to(device)

    def to(self, device):
        self.device = device
        self.image_features = self.image_features.to(device)
        self.text_features = self.text_features.to(device)
        return self

    def __len__(self):
        return self.image_features.shape[0]

    @property
    def dim(self):
        return self.image_features.shape[-1]

    def get(self):
        return self.image_features, self.text_features
