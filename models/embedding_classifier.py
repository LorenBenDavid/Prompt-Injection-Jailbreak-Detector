"""Layer 2: Sentence-embedding + LogisticRegression classifier."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SAVED_DIR = ROOT / "models" / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ST_LOCAL_PATH = SAVED_DIR / "sentence_transformer"
CLF_PATH = SAVED_DIR / "embedding_clf.pkl"
EMBEDDER_PATH = SAVED_DIR / "embedder.pkl"


@dataclass
class EmbeddingResult:
    """Result from the embedding classifier."""

    is_attack: bool
    score: float
    nearest_attacks: list[dict]
    latency_ms: float


class EmbeddingClassifier:
    """Logistic regression on top of sentence-transformer embeddings."""

    def __init__(self) -> None:
        self.embedder: SentenceTransformer | None = None
        self.clf: LogisticRegression | None = None
        self._attack_embeddings: np.ndarray | None = None

    def _load_embedder(self) -> SentenceTransformer:
        if self.embedder is None:
            if ST_LOCAL_PATH.exists():
                log.info("Loading sentence-transformer from local path …")
                self.embedder = SentenceTransformer(str(ST_LOCAL_PATH))
            else:
                log.info("Downloading sentence-transformer model …")
                self.embedder = SentenceTransformer(EMBEDDING_MODEL)
                self.embedder.save(str(ST_LOCAL_PATH))
                log.info("Saved sentence-transformer to %s", ST_LOCAL_PATH)
        return self.embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        embedder = self._load_embedder()
        return embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def train(self, train_path: Path, val_path: Path) -> float:
        """Train on train split, evaluate on val split. Returns val F1."""
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        log.info("Embedding training data (%d rows) …", len(train_df))
        X_train = self._embed(train_df["prompt"].tolist())
        y_train = train_df["label"].values

        log.info("Embedding val data (%d rows) …", len(val_df))
        X_val = self._embed(val_df["prompt"].tolist())
        y_val = val_df["label"].values

        self.clf = LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0, random_state=42
        )
        log.info("Training LogisticRegression …")
        self.clf.fit(X_train, y_train)

        y_pred = self.clf.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        log.info("Val F1: %.4f", f1)
        print(classification_report(y_val, y_pred, target_names=["benign", "attack"]))

        # Cache attack embeddings for nearest-neighbor lookup
        attack_mask = y_train == 1
        self._attack_embeddings = X_train[attack_mask]
        self._attack_texts = train_df["prompt"].values[attack_mask]

        # Save
        joblib.dump(self.clf, CLF_PATH)
        joblib.dump(
            {
                "attack_embeddings": self._attack_embeddings,
                "attack_texts": self._attack_texts,
            },
            EMBEDDER_PATH,
        )
        log.info("Saved classifier to %s", CLF_PATH)
        return f1

    def load(self) -> None:
        """Load saved model from disk."""
        self.clf = joblib.load(CLF_PATH)
        data = joblib.load(EMBEDDER_PATH)
        self._attack_embeddings = data["attack_embeddings"]
        self._attack_texts = data["attack_texts"]
        self._load_embedder()

    def predict(self, text: str) -> EmbeddingResult:
        """Classify a single prompt."""
        t0 = time.perf_counter()

        emb = self._embed([text])

        proba = self.clf.predict_proba(emb)[0]
        score = float(proba[1])
        is_attack = score >= 0.5

        # Top-3 nearest attacks by cosine similarity (embeddings already normalized)
        sims = (self._attack_embeddings @ emb.T).flatten()
        top3_idx = np.argsort(sims)[-3:][::-1]
        nearest = [
            {"text": self._attack_texts[i][:120], "similarity": float(sims[i])}
            for i in top3_idx
        ]

        latency_ms = (time.perf_counter() - t0) * 1000
        return EmbeddingResult(
            is_attack=is_attack,
            score=score,
            nearest_attacks=nearest,
            latency_ms=latency_ms,
        )

    def evaluate(self, test_path: Path) -> dict[str, float]:
        """Evaluate on test set and return metrics dict."""
        test_df = pd.read_csv(test_path)
        X_test = self._embed(test_df["prompt"].tolist())
        y_test = test_df["label"].values
        y_pred = self.clf.predict(X_test)
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the embedding classifier")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate on test set")
    args = parser.parse_args()

    clf = EmbeddingClassifier()
    if args.train:
        train_path = ROOT / "data" / "final" / "train.csv"
        val_path = ROOT / "data" / "final" / "val.csv"
        f1 = clf.train(train_path, val_path)
        print(f"Val F1: {f1:.4f}")
    if args.eval:
        clf.load()
        metrics = clf.evaluate(ROOT / "data" / "final" / "test.csv")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
