"""Run the thesis's selection (Run B: stability selection + capped ranking) on the TRAINING split.

    .venv/bin/python python/select_train.py [--pairs 50] [--jobs 8]

The thesis published its top 15 from Run B over all 7990 files, so that ranking has seen the
test soils. This re-runs the identical chain -- same code, same configs/selection.yaml -- on the
training soils only, so the test split stays unseen by selection. The thesis's correlation
clusters are reused: they are fitted on X alone (no labels), and reusing them keeps cluster ids
comparable with the published ranking.

Each of the 2 x pairs runs is checkpointed under data/selection_runs/, so an interrupted run
resumes. Output: results/selection_train.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict

import numpy as np

import paths

paths.import_thesis()
from hif.config import load_selection  # noqa: E402
from hif.selection.clustering import load as load_clusters  # noqa: E402
from hif.selection.ranking import max_pairwise_correlation, rank_clusters  # noqa: E402
from hif.validation.stability import run_b  # noqa: E402

log = logging.getLogger("select_train")


def signed_log1p(X: np.ndarray) -> np.ndarray:
    """The thesis's input transform (scripts/run_selection.py:load_matrix)."""
    return np.sign(X) * np.log1p(np.abs(X))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_selection()
    pairs = args.pairs or cfg["validation"]["run_b"]["n_pairs"]

    data = np.load(paths.FEATURES, allow_pickle=True)
    names = data["names"].tolist()
    X = signed_log1p(data["X_train"])
    y, soil = data["y_train"], data["soil_train"]
    clusters = load_clusters(paths.THESIS_CLUSTERS.as_posix().format(snr=paths.SNR))
    log.info("%d train rows, %d features, %d clusters, %d soils; %d pairs on %d workers",
             X.shape[0], X.shape[1], clusters.n_clusters, len(np.unique(soil)), pairs, args.jobs)

    started = time.perf_counter()
    result = run_b(
        X, y, soil, names, clusters,
        n_pairs=pairs,
        n_jobs=args.jobs,
        checkpoint=paths.DATA / "selection_runs",
        n_features=cfg["target"]["n_features"],
        n_filter=cfg["filter"]["n_keep"],
        n_embedded=cfg["embedded"]["n_keep"],
        mi_seeds=cfg["filter"]["mutual_information"]["n_seeds"],
        c_grid=tuple(cfg["embedded"]["lasso_l1"]["C_grid"]),
        l1_max_iter=cfg["embedded"]["lasso_l1"]["max_iter"],
        rf_estimators=cfg["embedded"]["random_forest"]["n_estimators"],
        permutation_repeats=cfg["embedded"]["random_forest"]["n_repeats"],
        mi_neighbors=cfg["filter"]["mutual_information"]["n_neighbors"],
        inner_splits=cfg["validation"]["run_a"]["inner_splits"],
        scoring=cfg["wrapper"]["scoring"],
    )
    log.info("%d usable runs in %.0f s; phi = %.3f", len(result.selections),
             time.perf_counter() - started, result.phi)

    caps = cfg["ranking"]["diversity_caps"]
    top, candidates = rank_clusters(
        result, clusters, caps=dict(caps), n_features=cfg["target"]["n_features"],
        X=X, feature_names=names, max_rho=cfg["ranking"]["max_pairwise_abs_rho"],
    )
    uncapped, _ = rank_clusters(result, clusters, caps=None, n_features=cfg["target"]["n_features"])
    columns = [names.index(f.feature_id) for f in top]

    payload = {
        "snr": paths.SNR,
        "rows": "train split only (data/split.json)",
        "n_runs": len(result.selections),
        "phi": result.phi,
        "phi_features": result.phi_features,
        "max_pairwise_rho": max_pairwise_correlation(X, columns),
        "top_capped": [asdict(f) for f in top],
        "top_uncapped": [asdict(f) for f in uncapped],
        "all_clusters": candidates[:200],
        "feature_frequency": result.feature_frequency,
    }
    paths.RESULTS.mkdir(exist_ok=True)
    out = paths.RESULTS / "selection_train.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    for f in top:
        log.info("  %2d. pi=%.2f  %s", f.rank, f.selection_frequency, f.feature_id)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
