"""Artifact layout, JSON/JSONL/CSV IO, stable seeds, and a tiny .env loader."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def stable_seed(key: str) -> int:
    """Deterministic 32-bit seed from a string key (sha256, NOT Python hash())."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader; sets vars that aren't already in the environment."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def experiment_paths(experiment_dir: str | Path) -> dict[str, Path]:
    base = Path(experiment_dir)
    paths = {
        "base": base,
        "logits": base / "logits",
        "results": base / "results",
        "figures": base / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _to_full_dict(pred: Any) -> dict:
    if is_dataclass(pred):
        return asdict(pred)
    return dict(pred)


def save_predictions(predictions: list[Any], paths: dict[str, Path]):
    """Write full predictions (incl. probs) to results/predictions.jsonl and a flat CSV."""
    import pandas as pd

    res = paths["results"]
    with open(res / "predictions.jsonl", "w") as f:
        for p in predictions:
            f.write(json.dumps(_to_full_dict(p)) + "\n")

    rows = []
    for p in predictions:
        if hasattr(p, "to_row"):
            rows.append(p.to_row())
        else:
            d = dict(p)
            d.pop("probs", None)
            d.pop("metadata", None)
            rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(res / "results.csv", index=False)
    return df


def load_predictions(results_dir: str | Path) -> list[dict]:
    path = Path(results_dir)
    jsonl = path / "predictions.jsonl" if path.is_dir() else path
    preds = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)
