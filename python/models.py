"""Downstream models and the feature sets they are trained on.

Linear and neural models see the thesis's input transform: signed log1p, then standardisation.
Trees and forests take raw features: they are invariant to monotone transforms, so the transform
would change nothing but would cost one log1p per feature on the device.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import paths

SEED = 0


def signed_log1p(X: np.ndarray) -> np.ndarray:
    return np.sign(X) * np.log1p(np.abs(X))


PREPROCESSED = ("logreg", "mlp")


def make_model(kind: str, *, n_trees: int = 30, depth: int = 8, hidden: int = 16):
    """A fresh, unfitted pipeline: [signed log1p -> StandardScaler ->] estimator."""
    if kind == "logreg":
        # The thesis's wrapper/evaluation estimator (tcc/src/hif/selection/pipeline.py).
        est = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)
    elif kind == "tree":
        est = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=SEED)
    elif kind == "forest":
        est = RandomForestClassifier(n_estimators=n_trees, max_depth=depth,
                                     class_weight="balanced", random_state=SEED, n_jobs=2)
    elif kind == "mlp":
        est = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=2000, early_stopping=True,
                            random_state=SEED)
    else:
        raise KeyError(kind)
    if kind in PREPROCESSED:
        return make_pipeline(FunctionTransformer(signed_log1p), StandardScaler(), est)
    return make_pipeline(est)


MODELS = ("logreg", "tree", "forest", "mlp")


def estimator_input(pipe, X: np.ndarray) -> np.ndarray:
    """What the final estimator of a fitted pipeline sees for raw features X."""
    return pipe[:-1].transform(X) if len(pipe) > 1 else X


def load_split():
    d = np.load(paths.FEATURES, allow_pickle=True)
    return d["names"].tolist(), d["X_train"], d["y_train"], d["X_test"], d["y_test"]


def thesis_published_top15() -> list[str]:
    """The thesis's published top 15 at 40 dB (Run B on ALL files, test soils included)."""
    path = paths.THESIS_RANKING.as_posix().format(snr=paths.SNR)
    with open(path) as fh:
        return [f["feature_id"] for f in json.load(fh)["top_capped"]]


def thesis_train_top15() -> list[str] | None:
    """The same chain re-run on the training split only (python/select_train.py)."""
    path = paths.RESULTS / "selection_train.json"
    if not path.exists():
        return None
    return [f["feature_id"] for f in json.loads(path.read_text())["top_capped"]]
