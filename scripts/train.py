"""Unified training entry point for all models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)

ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = ROOT / "data" / "final"
REPORTS_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _save_confusion_matrix(y_true: list, y_pred: list, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["benign", "attack"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    out = REPORTS_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved confusion matrix → %s", out)


def _save_roc_curve(y_true: list, y_score: list, name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {name}")
    ax.legend()
    fig.tight_layout()
    out = REPORTS_DIR / f"roc_curve_{name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved ROC curve → %s", out)


def train_embedding() -> dict[str, float]:
    sys.path.insert(0, str(ROOT))
    from models.embedding_classifier import EmbeddingClassifier

    clf = EmbeddingClassifier()
    clf.train(FINAL_DIR / "train.csv", FINAL_DIR / "val.csv")
    metrics = clf.evaluate(FINAL_DIR / "test.csv")

    # Collect predictions for plots
    import pandas as pd_inner
    import numpy as np_inner
    test_df = pd_inner.read_csv(FINAL_DIR / "test.csv")
    y_true = test_df["label"].tolist()
    y_score = [clf.predict(t).score for t in test_df["prompt"].tolist()]
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]

    _save_confusion_matrix(y_true, y_pred, "embedding")
    _save_roc_curve(y_true, y_score, "embedding")
    return metrics


def train_bert() -> dict[str, float]:
    sys.path.insert(0, str(ROOT))
    from models.bert_classifier import BertClassifier

    clf = BertClassifier()
    best_f1 = clf.train(FINAL_DIR / "train.csv", FINAL_DIR / "val.csv")
    log.info("Best val F1: %.4f", best_f1)

    clf.load()
    metrics = clf.evaluate(FINAL_DIR / "test.csv")

    # SHAP report
    clf.generate_shap_report(FINAL_DIR / "val.csv", n=100)

    # Plots using best-model predictions
    import pandas as pd_inner
    import torch
    from torch.utils.data import DataLoader
    from models.bert_classifier import PromptDataset, BATCH_SIZE

    test_df = pd_inner.read_csv(FINAL_DIR / "test.csv")
    test_ds = PromptDataset(test_df["prompt"].tolist(), test_df["label"].tolist(), clf.tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    clf.model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(clf.device)
            attention_mask = batch["attention_mask"].to(clf.device)
            logits = clf.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            y_score.extend(probs.tolist())
            y_true.extend(batch["labels"].numpy().tolist())

    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    _save_confusion_matrix(y_true, y_pred, "bert")
    _save_roc_curve(y_true, y_score, "bert")
    return metrics


def print_metrics_table(all_metrics: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>7}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(
            f"{name:<20} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>7.4f} {m['f1']:>7.4f}"
        )
    print("=" * 60)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train prompt-injection classifier models")
    parser.add_argument(
        "--model",
        choices=["all", "embedding", "bert"],
        default="all",
        help="Which model to train (default: all)",
    )
    args = parser.parse_args(argv)

    all_metrics: dict[str, dict[str, float]] = {}

    if args.model in ("all", "embedding"):
        log.info("=== Training Embedding Classifier ===")
        all_metrics["embedding"] = train_embedding()

    if args.model in ("all", "bert"):
        log.info("=== Training BERT Classifier ===")
        all_metrics["bert"] = train_bert()

    print_metrics_table(all_metrics)

    out = REPORTS_DIR / "metrics.json"
    with out.open("w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved to %s", out)
    print(f"\n✅ Training complete. Reports saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
