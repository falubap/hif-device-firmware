"""Python baseline: every model on every feature set, on the fixed split.

    .venv/bin/python python/baseline.py [--draws 50]

Feature sets
    all               all 7056 thesis features
    thesis_train      top 15 from the thesis's chain re-run on the training split only
    thesis_published  the thesis's published top 15 (its selection saw the test soils)
    random            uniformly random 15-feature subsets, --draws of them (the control)

Output: results/baseline.csv (one row per model x set x draw) and results/baseline.md.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import models
import paths


def evaluate(kind: str, Xtr, ytr, Xte, yte) -> dict:
    t = time.perf_counter()
    model = models.make_model(kind).fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "acc": accuracy_score(yte, pred),
        "f1_macro": f1_score(yte, pred, average="macro"),
        "bal_acc": balanced_accuracy_score(yte, pred),
        "fit_s": time.perf_counter() - t,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=50)
    args = parser.parse_args()

    names, Xtr, ytr, Xte, yte = models.load_split()
    col = {n: i for i, n in enumerate(names)}

    sets: list[tuple[str, int, list[int]]] = [("all", 0, list(range(len(names))))]
    train15 = models.thesis_train_top15()
    if train15 is None:
        print("results/selection_train.json missing: thesis_train skipped (run select_train.py)")
    else:
        sets.append(("thesis_train", 0, [col[f] for f in train15]))
    sets.append(("thesis_published", 0, [col[f] for f in models.thesis_published_top15()]))
    rng = np.random.default_rng(models.SEED)
    for d in range(args.draws):
        sets.append(("random", d, sorted(rng.choice(len(names), 15, replace=False).tolist())))

    rows = []
    for set_name, draw, cols in sets:
        for kind in models.MODELS:
            r = evaluate(kind, Xtr[:, cols], ytr, Xte[:, cols], yte)
            rows.append({"feature_set": set_name, "draw": draw, "k": len(cols), "model": kind, **r})
            if set_name != "random":
                print(f"{set_name:17s} {kind:7s} acc {r['acc']:.3f}  f1 {r['f1_macro']:.3f}")
    df = pd.DataFrame(rows)
    paths.RESULTS.mkdir(exist_ok=True)
    df.to_csv(paths.RESULTS / "baseline.csv", index=False)

    summary = df.groupby(["feature_set", "model"]).agg(
        k=("k", "first"), n=("draw", "size"),
        acc=("acc", "mean"), f1_macro=("f1_macro", "mean"), f1_sd=("f1_macro", "std"),
        f1_p95=("f1_macro", lambda s: s.quantile(0.95)),
    ).reset_index()
    order = {"all": 0, "thesis_train": 1, "thesis_published": 2, "random": 3}
    summary = (summary.assign(set_order=summary.feature_set.map(order))
               .sort_values(["model", "set_order"]).drop(columns="set_order"))
    lines = [
        f"# Python baseline (SNR {paths.SNR} dB, fixed soil-grouped split)", "",
        "Test set: 1645 files from 7 held-out soils. Random rows: mean over draws; "
        "f1_p95 is the 95th percentile of the random draws.", "",
        "| model | feature set | k | acc | macro-F1 | F1 sd | F1 p95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in summary.iterrows():
        rand = r.feature_set == "random"
        lines.append(
            f"| {r.model} | {r.feature_set} | {r.k} | {r.acc:.3f} | {r.f1_macro:.3f} | "
            f"{r.f1_sd:.3f} | {r.f1_p95:.3f} |" if rand else
            f"| {r.model} | {r.feature_set} | {r.k} | {r.acc:.3f} | {r.f1_macro:.3f} | | |"
        )
    (paths.RESULTS / "baseline.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
