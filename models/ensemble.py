"""Weighted ensemble combining all 3 layers with short-circuit logic."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from models.heuristic_classifier import classify as heuristic_classify, HeuristicResult
from models.embedding_classifier import EmbeddingClassifier, EmbeddingResult
from models.bert_classifier import BertClassifier, BertResult

HEURISTIC_WEIGHT = 0.20
EMBEDDING_WEIGHT = 0.30
BERT_WEIGHT = 0.50

SHORT_CIRCUIT_THRESHOLD = 0.95


class RiskLevel(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def _score_to_risk(score: float) -> RiskLevel:
    if score >= 0.95:
        return RiskLevel.CRITICAL
    if score >= 0.75:
        return RiskLevel.HIGH
    if score >= 0.50:
        return RiskLevel.MEDIUM
    if score >= 0.25:
        return RiskLevel.LOW
    return RiskLevel.SAFE


@dataclass
class EnsembleResult:
    """Full ensemble classification result."""

    is_attack: bool
    risk_level: RiskLevel
    ensemble_score: float
    heuristic: HeuristicResult
    embedding: EmbeddingResult | None
    bert: BertResult | None
    short_circuited: bool
    latency_ms: float
    explanation: str = field(default="")

    def to_dict(self) -> dict:
        return {
            "is_attack": self.is_attack,
            "risk_level": self.risk_level.value,
            "ensemble_score": round(self.ensemble_score, 4),
            "short_circuited": self.short_circuited,
            "latency_ms": round(self.latency_ms, 2),
            "explanation": self.explanation,
            "layers": {
                "heuristic": {
                    "is_attack": self.heuristic.is_attack,
                    "score": self.heuristic.score,
                    "triggered_rules": self.heuristic.triggered_rules,
                    "latency_ms": round(self.heuristic.latency_ms, 3),
                },
                "embedding": {
                    "is_attack": self.embedding.is_attack if self.embedding else None,
                    "score": round(self.embedding.score, 4) if self.embedding else None,
                    "nearest_attacks": self.embedding.nearest_attacks if self.embedding else [],
                    "latency_ms": round(self.embedding.latency_ms, 2) if self.embedding else None,
                },
                "bert": {
                    "is_attack": self.bert.is_attack if self.bert else None,
                    "score": round(self.bert.score, 4) if self.bert else None,
                    "shap_tokens": self.bert.shap_tokens[:10] if self.bert else [],
                    "latency_ms": round(self.bert.latency_ms, 2) if self.bert else None,
                },
            },
        }


class EnsembleClassifier:
    """Three-layer ensemble: heuristic → embedding → BERT."""

    def __init__(self) -> None:
        self._embedding_clf = EmbeddingClassifier()
        self._bert_clf = BertClassifier()
        self._loaded = False

    def load(self) -> None:
        """Load embedding + BERT models from disk."""
        self._embedding_clf.load()
        self._bert_clf.load()
        self._loaded = True

    def predict(self, text: str) -> EnsembleResult:
        """Classify text using the full 3-layer ensemble."""
        t0 = time.perf_counter()

        # Layer 1 — always runs
        heuristic = heuristic_classify(text)

        if heuristic.score >= SHORT_CIRCUIT_THRESHOLD:
            latency_ms = (time.perf_counter() - t0) * 1000
            return EnsembleResult(
                is_attack=True,
                risk_level=RiskLevel.CRITICAL,
                ensemble_score=1.0,
                heuristic=heuristic,
                embedding=None,
                bert=None,
                short_circuited=True,
                latency_ms=latency_ms,
                explanation=f"Short-circuited by heuristic rules: {heuristic.triggered_rules}",
            )

        # Layer 2
        embedding = self._embedding_clf.predict(text)

        # Layer 3
        bert = self._bert_clf.predict(text)

        ensemble_score = (
            HEURISTIC_WEIGHT * heuristic.score
            + EMBEDDING_WEIGHT * embedding.score
            + BERT_WEIGHT * bert.score
        )

        is_attack = ensemble_score >= 0.5
        risk_level = _score_to_risk(ensemble_score)

        parts = []
        if heuristic.triggered_rules:
            parts.append(f"Rules triggered: {heuristic.triggered_rules}")
        if embedding.score > 0.5:
            parts.append(f"Embedding score {embedding.score:.2f}")
        if bert.score > 0.5:
            parts.append(f"BERT score {bert.score:.2f}")
        explanation = "; ".join(parts) if parts else "No attack signals detected"

        latency_ms = (time.perf_counter() - t0) * 1000
        return EnsembleResult(
            is_attack=is_attack,
            risk_level=risk_level,
            ensemble_score=ensemble_score,
            heuristic=heuristic,
            embedding=embedding,
            bert=bert,
            short_circuited=False,
            latency_ms=latency_ms,
            explanation=explanation,
        )
