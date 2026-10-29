"""Sample a labeled eval set from RAID (https://github.com/liamdugan/raid).

    uv run --with pandas --with pyarrow python scripts/build_dataset.py

Uses the auto-converted parquet shards of RAID's train split, which cover the
abstracts, books, news and poetry domains (poetry is skipped: no GPT-4/ChatGPT).
Writes data/raid_sample.jsonl with fields text, label (0 human / 1 AI), split,
model, domain, attack.
"""
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

OUT = Path(__file__).resolve().parent.parent / "data" / "raid_sample.jsonl"
DOMAINS = ["abstracts", "books", "news"]
CHAT_MODELS = ["chatgpt", "gpt4", "cohere-chat", "llama-chat", "mistral-chat", "mpt-chat"]
MIN_WORDS, SEED = 150, 0
N_HUMAN_PER_DOMAIN, N_AI_PER_MODEL, N_PARA_PER_MODEL = 134, 50, 15

frames = []
for i in range(10):
    path = hf_hub_download("liamdugan/raid", f"raid/partial-train/{i:04d}.parquet",
                           repo_type="dataset", revision="refs/convert/parquet")
    df = pd.read_parquet(path, columns=["model", "attack", "domain", "generation"])
    frames.append(df[df.domain.isin(DOMAINS) & df.attack.isin(["none", "paraphrase"])])
df = pd.concat(frames)
df = df[df.generation.str.split().str.len() >= MIN_WORDS]

human = df[(df.model == "human") & (df.attack == "none")]
human = human.groupby("domain").sample(N_HUMAN_PER_DOMAIN, random_state=SEED).sample(frac=1, random_state=SEED)
half = len(human) // 2
human = human.assign(label=0, split=["calib"] * half + ["eval"] * (len(human) - half))

ai = df[df.model.isin(CHAT_MODELS) & (df.attack == "none")]
ai = ai.groupby("model").sample(N_AI_PER_MODEL, random_state=SEED).assign(label=1, split="eval")
para = df[df.model.isin(CHAT_MODELS) & (df.attack == "paraphrase")]
para = para.groupby("model").sample(N_PARA_PER_MODEL, random_state=SEED).assign(label=1, split="eval_paraphrase")

out = pd.concat([human, ai, para])
OUT.parent.mkdir(exist_ok=True)
with OUT.open("w") as f:
    for r in out.itertuples():
        f.write(json.dumps({"text": r.generation, "label": r.label, "split": r.split,
                            "model": r.model, "domain": r.domain, "attack": r.attack}) + "\n")
print(out.groupby(["split", "model"]).size())
print(f"Wrote {len(out)} rows to {OUT}")
