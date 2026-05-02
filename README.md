# Prompt Injection & Jailbreak Detector

A production-grade, portfolio-quality ML system for detecting prompt injection and jailbreak attacks in real time.

## Architecture

Three-layer ensemble classifier with weighted voting:

| Layer | Model | Weight | Latency |
|-------|-------|--------|---------|
| 1 — Heuristic | Rule-based regex patterns | 0.20 | < 1ms |
| 2 — Embedding | `all-MiniLM-L6-v2` + LogisticRegression | 0.30 | ~30ms |
| 3 — BERT | Fine-tuned `distilbert-base-uncased` | 0.50 | ~60ms |

**Short-circuit**: if heuristic score ≥ 0.95 → return CRITICAL immediately, skip ML layers.

**Ensemble score** = `0.20 × heuristic + 0.30 × embedding + 0.50 × BERT`

**Risk levels**: SAFE (< 0.25) / LOW / MEDIUM / HIGH / CRITICAL (≥ 0.95)

## Quickstart

### 1. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

### 3. Run the pipeline

```bash
# Download datasets from HuggingFace
python scripts/download_datasets.py

# Generate synthetic training data (requires ANTHROPIC_API_KEY)
python scripts/generate_synthetic.py

# Preprocess and split data
python scripts/preprocess.py

# Train all models
python scripts/train.py --model all

# Start the API
uvicorn backend.main:app --reload

# Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

### 4. Docker (full stack)

```bash
docker compose up --build
```

Frontend: http://localhost:3000  
API: http://localhost:8000  
API docs: http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze` | Classify a single prompt |
| POST | `/api/analyze/batch` | Classify up to 100 prompts |
| GET | `/api/stats` | Runtime statistics |
| GET | `/api/dataset` | Browse dataset (paginated) |
| GET | `/api/model/info` | Model metadata |
| GET | `/api/metrics` | Test-set evaluation metrics |
| GET | `/api/gallery` | Curated example gallery |
| GET | `/api/health` | Health check |

## Frontend Pages

- **Analyzer** — Interactive prompt classification with layer-by-layer scores, token importance, and nearest-attack lookup
- **Dataset Explorer** — Browse train/val/test splits with filtering and pagination
- **Performance** — Bar charts, radar charts, and live runtime statistics
- **Gallery** — Curated examples from the test set with pre-computed scores

## Data Sources

| Dataset | Type | Count |
|---------|------|-------|
| JailbreakBench/JBB-Behaviors | Attack (jailbreak) | varies |
| rubend18/ChatGPT-Jailbreak-Prompts | Attack (jailbreak) | varies |
| deepset/prompt-injections | Attack (injection) | varies |
| markush1/LLM-Injection-Dataset | Attack (injection) | varies |
| allenai/real-toxicity-prompts | Benign | 2,000 sampled |
| anon8231489123/ShareGPT_Vicuna_unfiltered | Benign | 2,000 sampled |
| Synthetic (Anthropic claude-sonnet-4-6) | Mixed | 1,500 |

## Project Structure

```
prompt-injection-detector/
├── data/
│   ├── raw/          # Downloaded CSVs + synthetic.jsonl
│   └── final/        # train.csv, val.csv, test.csv
├── models/
│   ├── heuristic_classifier.py
│   ├── embedding_classifier.py
│   ├── bert_classifier.py
│   ├── ensemble.py
│   └── saved/        # Trained model artifacts
├── scripts/
│   ├── download_datasets.py
│   ├── generate_synthetic.py
│   ├── preprocess.py
│   └── train.py
├── backend/
│   ├── main.py       # FastAPI app
│   ├── analyzer.py   # Inference logic
│   ├── schemas.py    # Pydantic v2 models
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── pages/    # Analyzer, Dataset, Performance, Gallery
│   │   └── api/      # API client
│   └── Dockerfile
├── reports/          # metrics.json, confusion matrices, ROC curves, shap_values.json
├── docker-compose.yml
└── .env.example
```
