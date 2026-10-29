"""Fit thresholds from a folder of known human-written .txt files.

    uv run python -m app.calibrate path/to/human_texts/

Each threshold is set so only ~1% of the human texts would be flagged as AI
(5% if there are fewer than 100 files). Writes thresholds.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

from . import analysis, config


def main(folder: str) -> None:
    files = sorted(Path(folder).glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files in {folder}")
    fpr = 0.01 if len(files) >= 100 else 0.05
    if fpr == 0.05:
        print(f"Only {len(files)} files; calibrating at 5% FPR. Use 100+ for 1%.")

    scores = {m: [] for m in analysis.METHOD_INFO}
    for f in files:
        try:
            report = analysis.analyze(f.read_text())
        except analysis.TextTooShort:
            print(f"skip (too short): {f.name}")
            continue
        for m in report["methods"]:
            scores[m["id"]].append(m["score"])
        print(f"scored {f.name}")

    thresholds = {}
    for mid, (_, direction) in analysis.METHOD_INFO.items():
        vals = np.array(scores[mid])
        if len(vals) == 0:
            continue
        q = fpr if direction == "lower_is_ai" else 1 - fpr
        thresholds[mid] = float(np.quantile(vals, q))
    config.THRESHOLDS_FILE.write_text(json.dumps(thresholds, indent=2))
    print(json.dumps(thresholds, indent=2))
    print(f"Wrote {config.THRESHOLDS_FILE}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
