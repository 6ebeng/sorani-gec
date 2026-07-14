# Web Interface

Lives in the sibling folder `Implementation/web/` (same repo level as `sorani-gec/`). Two servers: the Gradio demo and a lightweight annotation server used for the human study.

## Gradio demo (`web/app.py` + `web/ui.py`)

```bash
cd Implementation
python run_dev.py          # dev launcher
# or
docker compose up web
```

Six tabs:

1. **Correct** — single-sentence correction with model choice (baseline / morphology-aware), diff highlighting
2. **Batch** — file upload, batch correction
3. **Analysis** — per-word morphological features + agreement-graph visualization (`web/analysis.py`)
4. **Evaluation** — score a correction against a reference (F₀.₅, agreement checks) (`web/evaluation.py`)
5. **Models** — checkpoint status / loading (`web/models.py`)
6. **About** — project info

## REST API (`web/app.py`)

| Endpoint   | Method | Purpose                 |
| ---------- | ------ | ----------------------- |
| `/health`  | GET    | Liveness                |
| `/status`  | GET    | Model/checkpoint status |
| `/correct` | POST   | Correct one sentence    |
| `/batch`   | POST   | Correct many sentences  |
| `/models`  | GET    | Available model list    |
| `/metrics` | GET    | Service metrics         |

Configuration in `web/config.py` (checkpoint paths, device, generation params).

## Annotation server (`web/annotation_server.py`)

Dependency-light HTTP server built for the 37-rater human evaluation: serves blind pairs from `results/human_eval/evaluation_pairs.jsonl`, records per-rater `ratings_<id>.jsonl`, Arabic-Indic digit UI, mobile-friendly. Launch with `web/host_annotation.ps1`. The rating scale and manifest format are documented in [[Results]].

Keep this server if further rating rounds are planned; the analysis script is `scripts/analyze_human_eval.py`.

## Docker services (`Implementation/docker-compose.yml`)

Seven services: `web`, `train-baseline`, `train-morphaware`, `train`, `evaluate`, `pipeline`, `test`.

Next: [[Testing]]
