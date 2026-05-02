"""Core inference logic: loads all models, exposes analyze()."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.ensemble import EnsembleClassifier, EnsembleResult  # noqa: E402

log = logging.getLogger(__name__)

_ensemble: EnsembleClassifier | None = None


def get_ensemble() -> EnsembleClassifier:
    global _ensemble
    if _ensemble is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    return _ensemble


def load_models() -> None:
    global _ensemble
    log.info("Loading all models …")
    _ensemble = EnsembleClassifier()
    _ensemble.load()
    log.info("All models loaded.")


def analyze(text: str) -> EnsembleResult:
    return get_ensemble().predict(text)


def analyze_batch(texts: list[str]) -> list[EnsembleResult]:
    ens = get_ensemble()
    return [ens.predict(t) for t in texts]


def load_metrics() -> dict:
    metrics_path = ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open() as f:
            return json.load(f)
    return {}


def load_dataset(
    split: str = "test",
    page: int = 1,
    page_size: int = 50,
    label: int | None = None,
    attack_type: str | None = None,
    source: str | None = None,
) -> dict:
    import pandas as pd

    path = ROOT / "data" / "final" / f"{split}.csv"
    if not path.exists():
        return {"rows": [], "total": 0, "page": page, "page_size": page_size,
                "attack_count": 0, "benign_count": 0}

    df = pd.read_csv(path)

    if label is not None:
        df = df[df["label"] == label]
    if attack_type:
        df = df[df["attack_type"] == attack_type]
    if source:
        df = df[df["source"] == source]

    total = len(df)
    attack_count = int((df["label"] == 1).sum())
    benign_count = int((df["label"] == 0).sum())

    start = (page - 1) * page_size
    page_df = df.iloc[start: start + page_size]

    rows = page_df.fillna("").to_dict(orient="records")
    return {
        "rows": rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "attack_count": attack_count,
        "benign_count": benign_count,
    }


def load_gallery(n: int = 20) -> list[dict]:
    """Return a curated selection of interesting examples with pre-computed scores."""
    import pandas as pd
    import torch
    torch.set_num_threads(1)

    path = ROOT / "data" / "final" / "test.csv"
    if not path.exists():
        return []

    df = pd.read_csv(path)
    # Sample across attack subtypes
    attack_df = df[df["label"] == 1].groupby("attack_subtype").head(3)
    benign_df = df[df["label"] == 0].head(5)
    sample = pd.concat([attack_df, benign_df]).head(n).reset_index(drop=True)

    results = []
    ens = get_ensemble()
    for _, row in sample.iterrows():
        try:
            res = ens.predict(str(row["prompt"]))
            results.append({
                "prompt": str(row["prompt"])[:300],
                "label": int(row["label"]),
                "risk_level": res.risk_level.value,
                "ensemble_score": round(res.ensemble_score, 4),
                "attack_subtype": str(row.get("attack_subtype", "unknown")),
                "source": str(row.get("source", "unknown")),
            })
        except Exception as exc:  # noqa: BLE001
            log.warning("Gallery inference error: %s", exc)
    return results
