import os
import platform
import subprocess
from typing import Optional

import torch


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device(prefer_cuda: bool = True, index: int = 0) -> torch.device:
    """
    Return CUDA device if available, otherwise CPU.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def _run_command(command):
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return result.stderr.strip()
    except Exception as exc:
        return str(exc)


def collect_env_info() -> str:
    """
    Collect useful environment information for debugging.
    """
    lines = []

    lines.append("Environment Information")
    lines.append("=" * 80)

    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Python: {platform.python_version()}")
    lines.append(f"PyTorch: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    lines.append(f"CUDA available: {cuda_available}")

    if cuda_available:
        lines.append(f"CUDA version used by PyTorch: {torch.version.cuda}")
        lines.append(f"cuDNN version: {torch.backends.cudnn.version()}")
        lines.append(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            lines.append(
                f"GPU {i}: {props.name}, "
                f"capability={props.major}.{props.minor}, "
                f"memory={mem_gb:.2f} GB"
            )

    lines.append(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    git_hash = _run_command("git rev-parse HEAD")
    if git_hash:
        lines.append(f"Git commit: {git_hash}")

    nvidia_smi = _run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if nvidia_smi:
        lines.append("nvidia-smi:")
        lines.append(nvidia_smi)

    lines.append("=" * 80)

    return "\n".join(lines)
