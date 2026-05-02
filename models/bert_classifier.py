"""Layer 3: Fine-tuned DistilBERT sequence classifier with SHAP explainability."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SAVED_DIR = ROOT / "models" / "saved"
REPORTS_DIR = ROOT / "reports"
SAVED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)

BASE_MODEL = "distilbert-base-uncased"
BERT_BEST_PATH = SAVED_DIR / "bert_best"
MAX_LEN = 256
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
PATIENCE = 2


@dataclass
class BertResult:
    """Result from the BERT classifier."""

    is_attack: bool
    score: float
    shap_tokens: list[dict]
    latency_ms: float


class PromptDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: DistilBertTokenizerFast) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class BertClassifier:
    """Fine-tuned DistilBERT binary classifier."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: DistilBertForSequenceClassification | None = None
        self.tokenizer: DistilBertTokenizerFast | None = None

    def _load_tokenizer(self) -> DistilBertTokenizerFast:
        if self.tokenizer is None:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)
        return self.tokenizer

    def train(self, train_path: Path, val_path: Path) -> float:
        """Fine-tune DistilBERT. Returns best val F1."""
        tokenizer = self._load_tokenizer()

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        train_ds = PromptDataset(train_df["prompt"].tolist(), train_df["label"].tolist(), tokenizer)
        val_ds = PromptDataset(val_df["prompt"].tolist(), val_df["label"].tolist(), tokenizer)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

        self.model = DistilBertForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        total_steps = len(train_loader) * EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_f1 = 0.0
        no_improve = 0

        for epoch in range(1, EPOCHS + 1):
            self.model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_f1 = self._evaluate_loader(val_loader)
            log.info("Epoch %d — loss: %.4f  val_F1: %.4f", epoch, avg_loss, val_f1)
            print(f"Epoch {epoch}: loss={avg_loss:.4f}  val_F1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                no_improve = 0
                self.model.save_pretrained(BERT_BEST_PATH)
                tokenizer.save_pretrained(BERT_BEST_PATH)
                log.info("New best model saved (F1=%.4f)", best_f1)
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    log.info("Early stopping triggered after epoch %d", epoch)
                    break

        return best_f1

    def _evaluate_loader(self, loader: DataLoader) -> float:
        assert self.model is not None
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())
        return float(f1_score(all_labels, all_preds))

    def load(self) -> None:
        """Load best saved model from disk."""
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_BEST_PATH)
        self.model = DistilBertForSequenceClassification.from_pretrained(BERT_BEST_PATH).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> BertResult:
        """Classify a single prompt and return SHAP token scores."""
        assert self.model is not None and self.tokenizer is not None
        t0 = time.perf_counter()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        score = float(probs[1])
        is_attack = score >= 0.5

        shap_tokens = self._token_importance(text, input_ids, attention_mask)
        latency_ms = (time.perf_counter() - t0) * 1000

        return BertResult(
            is_attack=is_attack,
            score=score,
            shap_tokens=shap_tokens,
            latency_ms=latency_ms,
        )

    def _token_importance(
        self,
        text: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[dict]:
        """Approximate token importance via input × gradient."""
        assert self.model is not None and self.tokenizer is not None

        self.model.eval()
        embeds = self.model.distilbert.embeddings(input_ids)
        embeds.retain_grad()

        logits = self.model(inputs_embeds=embeds, attention_mask=attention_mask).logits
        attack_logit = logits[0, 1]
        attack_logit.backward()

        if embeds.grad is None:
            return []

        grad = embeds.grad[0].cpu().numpy()          # (seq_len, hidden)
        emb_val = embeds[0].detach().cpu().numpy()   # (seq_len, hidden)
        importance = (grad * emb_val).sum(axis=-1)   # integrated gradient approx

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        result = []
        for tok, imp in zip(tokens, importance):
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            result.append({"token": tok, "importance": float(imp)})
        return result

    def generate_shap_report(self, val_path: Path, n: int = 100) -> None:
        """Run token importance on n val samples and save to reports/shap_values.json."""
        val_df = pd.read_csv(val_path).head(n)
        records = []
        for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="SHAP report"):
            result = self.predict(str(row["prompt"]))
            records.append({
                "prompt": str(row["prompt"])[:200],
                "label": int(row["label"]),
                "score": result.score,
                "tokens": result.shap_tokens[:20],
            })
        out = REPORTS_DIR / "shap_values.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        log.info("SHAP report saved to %s", out)

    def evaluate(self, test_path: Path) -> dict[str, float]:
        """Evaluate on test set."""
        assert self.model is not None and self.tokenizer is not None
        test_df = pd.read_csv(test_path)
        test_ds = PromptDataset(test_df["prompt"].tolist(), test_df["label"].tolist(), self._load_tokenizer())
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        from sklearn.metrics import accuracy_score, precision_score, recall_score
        print(classification_report(all_labels, all_preds, target_names=["benign", "attack"]))
        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds),
            "recall": recall_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds),
        }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Train or evaluate the BERT classifier")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--shap", action="store_true")
    args = parser.parse_args()

    clf = BertClassifier()
    if args.train:
        f1 = clf.train(
            ROOT / "data" / "final" / "train.csv",
            ROOT / "data" / "final" / "val.csv",
        )
        print(f"Best val F1: {f1:.4f}")
    if args.eval:
        clf.load()
        metrics = clf.evaluate(ROOT / "data" / "final" / "test.csv")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    if args.shap:
        clf.load()
        clf.generate_shap_report(ROOT / "data" / "final" / "val.csv")
