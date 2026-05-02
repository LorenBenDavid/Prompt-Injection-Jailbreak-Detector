"""Merge, clean, and split all data sources into train/val/test splits."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FINAL_DIR = ROOT / "data" / "final"
LOG_DIR = ROOT / "logs"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
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


def _infer_subtype(text: str, attack_type: str) -> str:
    """Heuristically assign an attack subtype based on text content."""
    if attack_type == "benign":
        return "none"
    t = text.lower()
    if any(kw in t for kw in ["pretend", "act as", "roleplay", "you are now", "dan"]):
        return "roleplay"
    if any(kw in t for kw in ["ignore", "override", "forget", "disregard", "new directive"]):
        return "override"
    if any(kw in t for kw in ["<!-- ", "[inst]", "system:", "\n\nhuman:"]):
        return "indirect"
    if any(kw in t for kw in ["base64", "decode", "rot13", "leet", "1337"]):
        return "obfuscated"
    if any(ord(c) > 127 for c in text[:100]):
        return "multilingual"
    return "other"


def _infer_severity(label: int, attack_type: str) -> int:
    if label == 0:
        return 0
    if attack_type == "injection":
        return 3
    if attack_type == "jailbreak":
        return 2
    return 1


def _infer_language(text: str) -> str:
    non_ascii = sum(1 for c in text[:200] if ord(c) > 127)
    ratio = non_ascii / max(len(text[:200]), 1)
    if ratio > 0.3:
        return "mixed" if ratio < 0.7 else "other"
    return "en"


def load_csv_source(path: Path) -> pd.DataFrame:
    """Load a downloaded CSV source file."""
    try:
        df = pd.read_csv(path)
        required = {"text", "label", "attack_type", "source"}
        if not required.issubset(df.columns):
            log.warning("Skipping %s — missing required columns", path.name)
            return pd.DataFrame()
        return df[list(required)]
    except Exception as exc:
        log.error("Error loading %s: %s", path.name, exc)
        return pd.DataFrame()


def load_jsonl_source(path: Path) -> pd.DataFrame:
    """Load synthetic JSONL file."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ["text", "label", "attack_type", "source"]:
        if col not in df.columns:
            df[col] = "unknown" if col != "label" else 0
    return df[["text", "label", "attack_type", "source"]]


def build_dataframe(raw_dir: Path) -> pd.DataFrame:
    """Load and merge all source files."""
    frames: list[pd.DataFrame] = []

    for csv_path in tqdm(sorted(raw_dir.glob("*.csv")), desc="Loading CSV sources"):
        df = load_csv_source(csv_path)
        if not df.empty:
            frames.append(df)
            log.info("Loaded %d rows from %s", len(df), csv_path.name)

    jsonl_path = raw_dir / "synthetic.jsonl"
    if jsonl_path.exists():
        df = load_jsonl_source(jsonl_path)
        if not df.empty:
            frames.append(df)
            log.info("Loaded %d synthetic rows", len(df))

    if not frames:
        log.error("No data sources found in %s", raw_dir)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Total rows before cleaning: %d", len(combined))
    return combined


def clean_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, deduplicate, and enrich the dataframe with required columns."""
    # Strip whitespace
    df["text"] = df["text"].astype(str).str.strip()

    # Remove short prompts
    df = df[df["text"].str.len() >= 10].copy()

    # Deduplicate on lowercase text
    df["_lower"] = df["text"].str.lower()
    df = df.drop_duplicates(subset=["_lower"]).drop(columns=["_lower"])

    # Ensure label is int
    df["label"] = df["label"].astype(int)

    # Add enriched columns
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Enriching rows"):
        subtype = _infer_subtype(row["text"], row["attack_type"])
        rows.append({
            "id": str(uuid.uuid4()),
            "prompt": row["text"],
            "label": int(row["label"]),
            "attack_type": row["attack_type"],
            "attack_subtype": subtype,
            "source": row["source"],
            "severity": _infer_severity(int(row["label"]), row["attack_type"]),
            "language": _infer_language(row["text"]),
            "created_at": now,
        })

    return pd.DataFrame(rows)


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Cap majority class at 2x minority class."""
    counts = df["label"].value_counts()
    minority_count = counts.min()
    max_count = minority_count * 2

    parts = []
    for label_val in counts.index:
        subset = df[df["label"] == label_val]
        if len(subset) > max_count:
            subset = subset.sample(n=max_count, random_state=42)
        parts.append(subset)

    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Preprocess and split datasets")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--output-dir", default=str(FINAL_DIR))
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe(raw_dir)
    df = clean_and_enrich(df)
    log.info("Rows after cleaning: %d", len(df))

    df = balance_classes(df)
    log.info("Rows after balancing: %d", len(df))

    # Stratified split 70 / 15 / 15
    strat_col = df["label"].astype(str) + "_" + df["attack_type"]
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=strat_col, random_state=42)
    strat_temp = temp_df["label"].astype(str) + "_" + temp_df["attack_type"]
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=strat_temp, random_state=42)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    log.info("Train: %d  Val: %d  Test: %d", len(train_df), len(val_df), len(test_df))

    # Print stats
    print("\n=== Final Dataset Statistics ===")
    print(f"Total rows: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\nSource breakdown:")
    print(df["source"].value_counts().to_string())
    print("\nAttack type breakdown:")
    print(df["attack_type"].value_counts().to_string())
    print(f"\nSaved splits to {out_dir}")
    print("✅ Step 4 complete: data preprocessed and split")


if __name__ == "__main__":
    main()
