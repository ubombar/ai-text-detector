# AI Text Detector

A small local web app that checks whether a piece of text looks AI-generated. It runs four open-source methods side by side and shows why each one decided what it did.

| Method | Type | Signal | AI if |
|---|---|---|---|
| [Binoculars](https://github.com/ahans30/Binoculars) | zero-shot | log-perplexity ÷ cross-perplexity (base vs. instruct model) | lower |
| [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) | zero-shot | analytic probability-curvature z-score | higher |
| [desklib DeBERTa-v3 classifier](https://huggingface.co/desklib/ai-text-detector-v1.01) | trained | P(AI), averaged over 512-token chunks | higher |
| [GLTR](https://github.com/HendrikStrobelt/detecting-fake-text) | visual | share of tokens in GPT-2's top-10 | higher (display only) |

Binoculars and Fast-DetectGPT share one model pair (`Qwen2.5-3B` + `Qwen2.5-3B-Instruct`, ~12 GB), so both come from the same two forward passes. **They are disabled by default** (`ENABLE_LLM_PAIR=false`) while their calibration with the Qwen pair is being fixed, and the verdict then comes from the classifier alone. With `ENABLE_LLM_PAIR=true`, the verdict needs a majority (2 of 3) of Binoculars, Fast-DetectGPT and the classifier. GLTR is only shown for context.

> **Scores are statistical evidence, not proof.** These detectors produce false positives, especially on short, formal or non-native-English writing, and paraphrasing fools them.

## Requirements
- Apple Silicon Mac (MPS) with 32 GB RAM recommended (~14 GB of models in memory). A CUDA GPU or CPU also works, but more slowly.
- [uv](https://docs.astral.sh/uv/) and about 14 GB of disk for the Hugging Face model downloads (fetched on first use).

## Run
```bash
uv sync
uv run python -m app.setup_env      # choose username + password, writes .env
uv run uvicorn app.main:app --port 8765
```
Open http://localhost:8765. The first analysis loads the models (~30 s). After that each analysis takes a few seconds.

## Calibrate (strongly recommended)
The default thresholds are rough starting points for the Qwen-3B pair. Put 20+ `.txt` files that you **know** were written by people, ideally in the same style you'll be checking, into a folder, then run:
```bash
uv run python -m app.calibrate path/to/human_texts/
```
This writes `thresholds.json`, with each threshold set so that only ~1% of those human texts would be flagged. It uses 5% if you have fewer than 100 files.

## Configuration
See `.env.example` to swap models (e.g. `tiiuae/falcon-7b` + `falcon-7b-instruct`, the original Binoculars pair, if you have more memory) or change `MAX_TOKENS`.

## Evaluate on RAID
```bash
uv run --with pandas --with pyarrow python scripts/build_dataset.py   # ~2.4 GB download
uv run python scripts/evaluate.py --write-thresholds
```
Reports AUROC and TPR@1%FPR on held-out human and AI texts from [RAID](https://github.com/liamdugan/raid), and fits thresholds on a separate human split.

## Tests
```bash
uv run pytest
```

## License
MIT. The models keep their own licenses.
