import logging
import sys
from pathlib import Path
from typing import Optional


_LOGGERS = {}


def setup_logger(
    name: str = "mindecho",
    save_dir: Optional[str] = None,
    filename: str = "log.txt",
    level: int = logging.INFO,
    use_time: bool = True,
):
    """
    Create a console/file logger.

    Args:
        name:
            Logger name.
        save_dir:
            Directory for log file. If None, file logging is disabled.
        filename:
            Log file name.
        level:
            Logging level.
        use_time:
            Whether to include timestamp in formatter.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if use_time:
        fmt = "[%(asctime)s] [%(levelname)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
    else:
        fmt = "[%(levelname)s] %(message)s"
        datefmt = None

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(save_dir / filename, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = "mindecho"):
    """
    Get an existing logger or create a default console logger.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    return setup_logger(name=name)


def log_config(logger, cfg):
    """
    Pretty-print config into logger.
    """
    if hasattr(cfg, "pretty_text"):
        text = cfg.pretty_text()
    elif hasattr(cfg, "to_dict"):
        import yaml

        text = yaml.safe_dump(
            cfg.to_dict(),
            sort_keys=False,
            allow_unicode=True,
        )
    else:
        text = str(cfg)

    logger.info("Configuration:")
    for line in text.splitlines():
        logger.info(f"  {line}")
