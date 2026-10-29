"""Score a labeled set, calibrate on its human 'calib' split, report metrics.

    uv run python scripts/evaluate.py [data/raid_sample.jsonl] [--write-thresholds]

Scores are cached in data/scores.jsonl, so re-runs only score new rows.
Metrics on the held-out 'eval' split:
  AUROC        threshold-free separability
  TPR@1%FPR    share of AI caught when the threshold lets 1% of eval humans through
  calibrated   FPR/TPR using thresholds fit on the separate 'calib' humans
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import analysis, config  # noqa: E402

FPR = 0.01
METHODS = analysis.enabled_methods()


def oriented(mid, x):
    """Flip so that higher always means more AI-like."""
    return -np.asarray(x) if analysis.METHOD_INFO[mid][1] == "lower_is_ai" else np.asarray(x)


def auroc(h, a):
    # Mann-Whitney U: probability a random AI text outranks a random human one.
    h, a = np.asarray(h), np.asarray(a)
    return float(((a[:, None] > h[None]).sum() + 0.5 * (a[:, None] == h[None]).sum()) / (len(a) * len(h)))


def tpr_at_fpr(h, a, fpr=FPR):
    t = np.quantile(h, 1 - fpr)
    return float((np.asarray(a) > t).mean())


def score_all(rows, cache_path):
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[r["i"]] = r["scores"]
    with cache_path.open("a") as f:
        for i, row in enumerate(rows):
            if i in cache:
                continue
            report = analysis.analyze(row["text"])
            cache[i] = {m["id"]: m["score"] for m in report["methods"]}
            if not set(METHODS) <= cache[i].keys():
                raise SystemExit("cached scores missing methods; delete data/scores.jsonl")
            f.write(json.dumps({"i": i, "scores": cache[i]}) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"scored {i}/{len(rows)}", flush=True)
    return [cache[i] for i in range(len(rows))]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data = Path(args[0] if args else config.ROOT / "data" / "raid_sample.jsonl")
    rows = [json.loads(l) for l in data.read_text().splitlines()]
    scores = score_all(rows, data.with_name("scores.jsonl"))

    def pick(mid, pred):
        return oriented(mid, [s[mid] for s, r in zip(scores, rows) if pred(r)])

    is_calib_h = lambda r: r["split"] == "calib"
    is_eval_h = lambda r: r["split"] == "eval" and r["label"] == 0
    is_eval_ai = lambda r: r["split"] == "eval" and r["label"] == 1
    is_para = lambda r: r["split"] == "eval_paraphrase"

    n = {k: sum(map(f, rows)) for k, f in
         [("calib_h", is_calib_h), ("eval_h", is_eval_h), ("eval_ai", is_eval_ai), ("para", is_para)]}
    print(f"\ncalib humans {n['calib_h']} | eval humans {n['eval_h']} | eval AI {n['eval_ai']} | paraphrased AI {n['para']}\n")

    thresholds, flags = {}, {}
    print(f"{'method':<16}{'AUROC':>7}{'TPR@1%':>8}{'AUROC(para)':>13}{'TPR@1%(para)':>14}{'thr':>9}{'FPR':>7}{'TPR':>7}{'TPR(para)':>11}")
    for mid in METHODS:
        ch, eh, ea, pa = (pick(mid, f) for f in (is_calib_h, is_eval_h, is_eval_ai, is_para))
        t = float(np.quantile(ch, 1 - FPR))
        thresholds[mid] = -t if analysis.METHOD_INFO[mid][1] == "lower_is_ai" else t
        flags[mid] = {r: v > t for r, v in (("h", eh), ("a", ea), ("p", pa))}
        print(f"{mid:<16}{auroc(eh, ea):>7.3f}{tpr_at_fpr(eh, ea):>8.1%}{auroc(eh, pa):>13.3f}{tpr_at_fpr(eh, pa):>14.1%}"
              f"{thresholds[mid]:>9.3f}{flags[mid]['h'].mean():>7.1%}{flags[mid]['a'].mean():>7.1%}{flags[mid]['p'].mean():>11.1%}")

    voters = analysis.voting()
    votes = {k: sum(flags[m][k].astype(int) for m in voters) >= len(voters) // 2 + 1 for k in ("h", "a", "p")}
    print(f"{'majority vote':<16}{'':>7}{'':>8}{'':>13}{'':>14}{'':>9}{votes['h'].mean():>7.1%}{votes['a'].mean():>7.1%}{votes['p'].mean():>11.1%}")

    print("\nTPR per generator at calibrated thresholds (eval split):")
    by_model = defaultdict(list)
    for s, r in zip(scores, rows):
        if is_eval_ai(r):
            by_model[r["model"]].append(s)
    print(f"{'model':<14}" + "".join(f"{m:>16}" for m in METHODS))
    for model, ss in sorted(by_model.items()):
        cells = []
        for mid in METHODS:
            v = oriented(mid, [s[mid] for s in ss])
            t = thresholds[mid] * (-1 if analysis.METHOD_INFO[mid][1] == "lower_is_ai" else 1)
            cells.append(f"{(v > t).mean():>16.0%}")
        print(f"{model:<14}" + "".join(cells))

    print("\nFPR per domain at calibrated thresholds (eval humans):")
    for dom in sorted({r["domain"] for r in rows}):
        cells = []
        for mid in METHODS:
            v = pick(mid, lambda r: is_eval_h(r) and r["domain"] == dom)
            t = thresholds[mid] * (-1 if analysis.METHOD_INFO[mid][1] == "lower_is_ai" else 1)
            cells.append(f"{(v > t).mean():>16.1%}")
        print(f"{dom:<14}" + "".join(cells))

    print("\nthresholds @1% FPR on calib humans:", json.dumps({k: round(v, 4) for k, v in thresholds.items()}))
    if "--write-thresholds" in sys.argv:
        config.THRESHOLDS_FILE.write_text(json.dumps(thresholds, indent=2))
        print(f"Wrote {config.THRESHOLDS_FILE}")


if __name__ == "__main__":
    main()
