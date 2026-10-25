"""Structured experiment directories, reproducibility metadata and JSONL metric logging."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PACKAGES = ("numpy", "pymunk", "gymnasium", "pettingzoo", "pyyaml", "torch", "pygame-ce")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def environment_metadata() -> dict[str, Any]:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": versions,
        "git_commit": commit,
        "git_dirty": dirty,
        "argv": sys.argv,
    }


class RunDirectory:
    """runs/<group>/<timestamp>_<name>/ with config.yaml, metadata.json, metrics.jsonl, run.log."""

    def __init__(self, base: str | Path, group: str, name: str, config: dict[str, Any] | None = None,
                 existing: str | Path | None = None):
        """Create a new run directory, or reopen ``existing`` (resume) and append to its logs."""
        if existing is not None:
            self.path = Path(existing)
            if not (self.path / "config.yaml").exists():
                raise FileNotFoundError(f"{self.path} is not a run directory (no config.yaml)")
            meta = self.path / "metadata.json"
            history = json.loads(meta.read_text()) if meta.exists() else {}
            history.setdefault("resumed", []).append(_jsonable(environment_metadata()))
            meta.write_text(json.dumps(history, indent=2))
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = Path(base) / group / f"{stamp}_{name}"
            self.path.mkdir(parents=True, exist_ok=False)
            if config is not None:
                (self.path / "config.yaml").write_text(yaml.safe_dump(_jsonable(config), sort_keys=False))
            (self.path / "metadata.json").write_text(json.dumps(_jsonable(environment_metadata()), indent=2))
        self._metrics = (self.path / "metrics.jsonl").open("a")
        self.logger = setup_logging(self.path / "run.log")

    def log_metrics(self, record: dict[str, Any]) -> None:
        self._metrics.write(json.dumps(_jsonable(record)) + "\n")
        self._metrics.flush()

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_jsonable(obj), indent=2))
        return p

    def close(self) -> None:
        self._metrics.close()
        logger = logging.getLogger("softbody_marl")
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == (self.path / "run.log").resolve():
                logger.removeHandler(h)
                h.close()


def setup_logging(logfile: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("softbody_marl")
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
