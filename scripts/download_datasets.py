"""Download all 6 datasets from HuggingFace and save raw files to data/raw/."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
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


DATASETS: list[dict] = [
    {
        "id": "JailbreakBench/JBB-Behaviors",
        "name": "jailbreakbench",
        "label": 1,
        "attack_type": "jailbreak",
        "text_col": "Goal",
        "split": "harmful",
        "sample": None,
        "config": "behaviors",
    },
    {
        "id": "rubend18/ChatGPT-Jailbreak-Prompts",
        "name": "chatgpt_jailbreaks",
        "label": 1,
        "attack_type": "jailbreak",
        "text_col": "Prompt",
        "split": "train",
        "sample": None,
        "config": None,
    },
    {
        "id": "deepset/prompt-injections",
        "name": "deepset_injections",
        "label": None,  # has its own label col
        "attack_type": "injection",
        "text_col": None,
        "split": "train",
        "sample": None,
        "config": None,
    },
    {
        "id": "fka/awesome-chatgpt-prompts",
        "name": "awesome_prompts",
        "label": 0,
        "attack_type": "benign",
        "text_col": None,
        "split": "train",
        "sample": None,
        "config": None,
    },
    {
        "id": "allenai/real-toxicity-prompts",
        "name": "real_toxicity",
        "label": 0,
        "attack_type": "benign",
        "text_col": None,
        "split": "train",
        "sample": 2000,
        "config": None,
    },
    {
        "id": "Abirate/english_quotes",
        "name": "english_quotes",
        "label": 0,
        "attack_type": "benign",
        "text_col": None,
        "split": "train",
        "sample": 2000,
        "config": None,
    },
]


def _extract_text(row: dict, dataset_name: str, text_col: str | None = None) -> str | None:
    """Extract prompt text from a dataset row using heuristics per dataset."""
    if text_col and text_col in row and row[text_col]:
        return str(row[text_col])[:2000]
    candidates = ["text", "prompt", "Prompt", "instruction", "input", "goal", "Goal",
                  "question", "content", "conversations", "passage", "quote"]
    for col in candidates:
        if col in row and row[col]:
            val = row[col]
            if isinstance(val, list):
                # ShareGPT style: list of conversation turns
                for turn in val:
                    if isinstance(turn, dict):
                        msg = turn.get("value") or turn.get("content") or ""
                        if msg:
                            return str(msg)[:2000]
            if isinstance(val, dict):
                # real-toxicity-prompts: {"text": ..., "toxicity": ...}
                return str(val.get("text", "") or "")[:2000]
            return str(val)[:2000]
    return None


def download_dataset(cfg: dict) -> pd.DataFrame:
    """Download one dataset and return a DataFrame with columns: text, label, attack_type, source."""
    log.info("Downloading %s …", cfg["id"])
    rows: list[dict] = []

    try:
        config_name = cfg.get("config")
        if config_name:
            ds = load_dataset(cfg["id"], config_name, split=cfg["split"])
        else:
            ds = load_dataset(cfg["id"], split=cfg["split"])
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load %s: %s — skipping", cfg["id"], exc)
        return pd.DataFrame()

    items = list(ds)
    if cfg["sample"] and len(items) > cfg["sample"]:
        import random
        random.seed(42)
        items = random.sample(items, cfg["sample"])

    label_col = None
    if items:
        first = items[0]
        for possible in ["label", "Label", "injection", "is_injection"]:
            if possible in first:
                label_col = possible
                break

    for item in tqdm(items, desc=cfg["name"], leave=False):
        text = _extract_text(dict(item), cfg["name"], cfg.get("text_col"))
        if not text or len(text.strip()) < 10:
            continue

        if cfg["label"] is not None:
            label = cfg["label"]
        elif label_col:
            raw_label = item.get(label_col, 0)
            label = int(bool(raw_label))
        else:
            label = 0

        rows.append({
            "text": text.strip(),
            "label": label,
            "attack_type": cfg["attack_type"] if label == 1 else "benign",
            "source": cfg["name"],
        })

    df = pd.DataFrame(rows)
    log.info("  → %d rows from %s", len(df), cfg["id"])
    return df


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Download datasets for prompt-injection-detector")
    parser.add_argument("--output-dir", default=str(RAW_DIR), help="Directory to save raw CSV files")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    attack_count = 0
    benign_count = 0

    for cfg in tqdm(DATASETS, desc="Datasets"):
        df = download_dataset(cfg)
        if df.empty:
            continue
        out_path = out / f"{cfg['name']}.csv"
        df.to_csv(out_path, index=False)
        log.info("Saved %s → %s", cfg["name"], out_path)
        all_frames.append(df)
        attack_count += int((df["label"] == 1).sum())
        benign_count += int((df["label"] == 0).sum())

    print(f"\nDownloaded {attack_count} attack rows, {benign_count} benign rows")
    log.info("Download complete. Attack: %d  Benign: %d", attack_count, benign_count)
    print("✅ Step 2 complete: datasets downloaded")


if __name__ == "__main__":
    main()
