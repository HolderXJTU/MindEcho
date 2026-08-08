import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigNode(dict):
    """
    A tiny config wrapper that supports both dict-style and dot-style access.

    Example:
        cfg.model.clip_dim
        cfg["model"]["clip_dim"]
    """

    def __init__(self, data: Dict[str, Any]):
        super().__init__()
        for k, v in data.items():
            if isinstance(v, dict):
                v = ConfigNode(v)
            self[k] = v

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, ConfigNode):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out


def load_config(path: str) -> ConfigNode:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ConfigNode(data)


def save_config(cfg: ConfigNode, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)
