"""Train one configuration and convert it to C: firmware/include/model.h and feature_config.h.

    .venv/bin/python python/convert.py --model forest [--features thesis_train]
                                       [--trees 30] [--depth 8] [--hidden 16] [--method inline]

model.h exposes one entry point, `int hif_model_predict(const float *features)`, that returns the
region (1..4) for the features in HIF_FEATURE_IDS order. It contains, in order:
  - the preprocessing (signed log1p + standardisation) for logreg and mlp;
  - the estimator: emlearn's code for tree / forest; generated weights with a hand-written forward
    pass for logreg (emlearn has no converter) and mlp (see mlp_code).

Gate 3: the exact header is compiled for the host (firmware/test/host_model.c) and its predictions
are compared with scikit-learn on the whole test split. Also writes data/model.json (the config,
feature ids and Python predictions) for export_vectors.py and the UART checks.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import tempfile

import emlearn
import emlearn.cgen
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import feature_config
import models
import paths

# Importing emlearn runs numpy.seterr(all="raise") (emlearn/bayes.py), which turns a harmless
# underflow inside scikit-learn's logistic-regression fit (7056 features) into an exception.
# Back to numpy's defaults.
np.seterr(divide="warn", over="warn", under="ignore", invalid="warn")

HEADER = paths.FIRMWARE_INCLUDE / "model.h"
CONFIG = paths.WORK / "model.json"
HOST_SRC = paths.FIRMWARE / "test" / "host_model.c"
HOST_BIN = paths.FIRMWARE / "build" / "host" / "host_model"


def c_float(value) -> str:
    """A float32 literal that round-trips exactly (9 significant digits) and is always valid C."""
    return f"{float(value):.9e}f"


# emlearn writes floats with "%.6f": thresholds of small-magnitude features would collapse to 0.
_emlearn_constant = emlearn.cgen.constant
emlearn.cgen.constant = lambda val, dtype="float": (
    c_float(val) if dtype == "float" else _emlearn_constant(val, dtype)
)


def c_array(name: str, values, ctype: str = "float") -> str:
    values = np.asarray(values).ravel()
    fmt = c_float if ctype == "float" else (lambda v: str(int(v)))
    body = ",\n    ".join(", ".join(fmt(v) for v in values[i : i + 6]) for i in range(0, len(values), 6))
    return f"static const {ctype} {name}[{len(values)}] = {{\n    {body}\n}};\n"


RANDOM_SIZE = 15


def feature_set(name: str, names: list[str]) -> list[str]:
    """thesis_train | thesis_published | all | top:<k> (first k of thesis_train's ranking) |
    random:<d> (draw d of RANDOM_SIZE features, uniformly over all 7056, fixed seed per draw)."""
    if name in ("thesis_train",) or name.startswith("top:"):
        ids = models.thesis_train_top15()
        if ids is None:
            raise SystemExit("results/selection_train.json missing; run python/select_train.py")
        return ids if name == "thesis_train" else ids[: int(name.split(":")[1])]
    if name == "thesis_published":
        return models.thesis_published_top15()
    if name == "all":
        return list(names)
    if name.startswith("random:"):
        rng = np.random.default_rng(1000 + int(name.split(":")[1]))
        return [names[i] for i in sorted(rng.choice(len(names), RANDOM_SIZE, replace=False))]
    raise SystemExit(f"unknown feature set {name!r}")


def estimator_code(kind: str, est, method: str) -> str:
    """C for the estimator alone, as `int32_t hif_est_predict(const float *x, int32_t n)`."""
    if kind == "logreg":
        n_classes, n_features = est.coef_.shape
        return (
            c_array("hif_logreg_coef", est.coef_) + c_array("hif_logreg_intercept", est.intercept_)
            + f"""
/* Multinomial logistic regression: argmax of intercept + coef . x (the first maximum wins,
 * as numpy's argmax). */
static inline int32_t hif_est_predict(const float *x, int32_t n)
{{
    (void)n;
    int32_t best = 0;
    float best_score = 0.0f;
    for (int32_t c = 0; c < {n_classes}; c++) {{
        float s = hif_logreg_intercept[c];
        for (int32_t i = 0; i < {n_features}; i++)
            s += hif_logreg_coef[c * {n_features} + i] * x[i];
        if (c == 0 || s > best_score) {{
            best = c;
            best_score = s;
        }}
    }}
    return best;
}}
"""
        )
    if kind == "mlp":
        return mlp_code(est)
    # emlearn test-compiles some models into ./tmp; keep that out of the repo.
    with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
        return emlearn.convert(est, method=method, dtype="float", leaf_bits=8).save(name="hif_est")


def mlp_code(est) -> str:
    """Dense forward pass for an MLPClassifier: ReLU hidden layers, argmax of the output logits
    (softmax is monotone, so it picks what scikit-learn's predict picks).

    Hand-written rather than emlearn's: eml_net_infer requires both activation buffers to be as
    wide as the widest layer INCLUDING the input, i.e. 2 x 7056 floats (55 KB) for the
    all-features set, which does not fit next to the extractor in 128 KB. Here the buffers are as
    wide as the widest hidden/output layer.
    """
    if est.activation != "relu":
        raise SystemExit(f"MLP activation {est.activation!r} not supported")
    sizes = [est.coefs_[0].shape[0]] + [w.shape[1] for w in est.coefs_]
    width = max(sizes[1:])
    parts = []
    for i, (w, b) in enumerate(zip(est.coefs_, est.intercepts_)):
        # Stored output-major, w_t[o][i], so each output is one contiguous dot product.
        parts.append(c_array(f"hif_mlp_w{i}", w.T) + c_array(f"hif_mlp_b{i}", b))
    layers = ",\n    ".join(
        f"{{{sizes[i]}, {sizes[i + 1]}, hif_mlp_w{i}, hif_mlp_b{i}}}" for i in range(len(est.coefs_)))
    return "".join(parts) + f"""
typedef struct {{
    int32_t n_in, n_out;
    const float *w;   /* [n_out][n_in] */
    const float *b;
}} hif_mlp_layer_t;

static const hif_mlp_layer_t hif_mlp_layers[{len(est.coefs_)}] = {{
    {layers}
}};

/* Forward pass: ReLU on hidden layers, identity on the output; returns the argmax class index
 * (the first maximum wins, as numpy's argmax). */
static inline int32_t hif_est_predict(const float *x, int32_t n)
{{
    static float act[2][{width}];
    (void)n;
    const float *in = x;
    int32_t n_layers = {len(est.coefs_)}, buf = 0;
    for (int32_t l = 0; l < n_layers; l++) {{
        const hif_mlp_layer_t *L = &hif_mlp_layers[l];
        float *out = act[buf];
        for (int32_t o = 0; o < L->n_out; o++) {{
            const float *w = L->w + o * L->n_in;
            float s = L->b[o];
            for (int32_t i = 0; i < L->n_in; i++)
                s += w[i] * in[i];
            out[o] = (l < n_layers - 1 && s < 0.0f) ? 0.0f : s;
        }}
        in = out;
        buf ^= 1;
    }}
    int32_t best = 0;
    for (int32_t c = 1; c < hif_mlp_layers[n_layers - 1].n_out; c++)
        if (in[c] > in[best])
            best = c;
    return best;
}}
"""


def write_header(kind: str, pipe, ids: list[str], description: str, method: str) -> None:
    est = pipe[-1]
    preprocess = kind in models.PREPROCESSED
    lines = [
        "/* Generated by python/convert.py -- do not edit.",
        f" * {description}",
        " */",
        "#ifndef HIF_MODEL_H",
        "#define HIF_MODEL_H",
        "",
        "#include <stdint.h>",
        "#include <math.h>",
        "",
        f"#define HIF_N_FEATURES {len(ids)}",
        f"#define HIF_N_CLASSES {len(est.classes_)}",
        f'#define HIF_MODEL_KIND "{kind}"',
        "",
        "/* Model input order:",
        *(f" *  {i:2d} {fid}" for i, fid in enumerate(ids)),
        " */",
        c_array("hif_class_labels", est.classes_, "int32_t"),
    ]
    if preprocess:
        scaler = pipe[-2]
        lines += [c_array("hif_scaler_mean", scaler.mean_), c_array("hif_scaler_scale", scaler.scale_)]
    lines += [estimator_code(kind, est, method), ""]

    if preprocess:
        lines.append("""/* Signed log1p, then standardisation (python/models.py), into a static buffer in the CCM. */
static inline int hif_model_predict(const float *features)
{
    static float x[HIF_N_FEATURES] __attribute__((section(".ccmram")));
    for (int i = 0; i < HIF_N_FEATURES; i++) {
        float v = log1pf(fabsf(features[i]));
        v = features[i] < 0.0f ? -v : v;
        x[i] = (v - hif_scaler_mean[i]) / hif_scaler_scale[i];
    }
    return hif_class_labels[hif_est_predict(x, HIF_N_FEATURES)];
}
""")
    else:
        lines.append("""/* Trees take raw features. */
static inline int hif_model_predict(const float *features)
{
    return hif_class_labels[hif_est_predict(features, HIF_N_FEATURES)];
}
""")
    lines.append("#endif")
    HEADER.parent.mkdir(parents=True, exist_ok=True)
    HEADER.write_text("\n".join(lines) + "\n")


def host_predict(X: np.ndarray) -> np.ndarray:
    """Compile model.h for the host and run it on X (float32)."""
    HOST_BIN.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gcc", "-std=c11", "-O2", "-Wall", "-Wno-unused-parameter", "-Wno-unused-function",
         f"-I{HEADER.parent}", f"-I{emlearn.includedir}", str(HOST_SRC), "-o", str(HOST_BIN), "-lm"],
        check=True,
    )
    out = subprocess.run([str(HOST_BIN)], input=np.ascontiguousarray(X, np.float32).tobytes(),
                         capture_output=True, check=True)
    return np.array(out.stdout.split(), dtype=int)


def convert(model: str, features: str = "thesis_train", trees: int = 30, depth: int = 8,
            hidden: int = 16, method: str = "inline") -> dict:
    """Train, write model.h + feature_config.h + model.json/joblib, check C vs scikit-learn on the
    host (Gate 3). Returns Python's and the host C model's scores on the whole test split."""
    names, Xtr, ytr, Xte, yte = models.load_split()
    ids = feature_set(features, names)
    cols = [names.index(f) for f in ids]
    pipe = models.make_model(model, n_trees=trees, depth=depth, hidden=hidden).fit(Xtr[:, cols], ytr)

    hyper = {"tree": f"depth {depth}, {method}",
             "forest": f"{trees} trees, depth {depth}, {method}, soft voting",
             "mlp": f"{hidden} hidden units", "logreg": "C=1, balanced"}[model]
    description = f"{model} ({hyper}) on {features} ({len(ids)} features), SNR {paths.SNR} dB"
    write_header(model, pipe, ids, description, method)
    feature_config.write(ids, description)

    X = Xte[:, cols].astype(np.float32)
    py = pipe.predict(X)
    c = host_predict(X)
    agree = float((c == py).mean())
    result = {
        "description": description,
        "py_acc": accuracy_score(yte, py), "py_f1": f1_score(yte, py, average="macro"),
        "host_acc": accuracy_score(yte, c), "host_f1": f1_score(yte, c, average="macro"),
        "agreement_host": agree,
    }
    print(description)
    print(f"  python  acc {result['py_acc']:.4f}  macro-F1 {result['py_f1']:.4f}")
    print(f"  C/host  acc {result['host_acc']:.4f}  macro-F1 {result['host_f1']:.4f}")
    print(f"  agreement C vs python: {agree:.4f} ({int((c != py).sum())} of {len(c)} differ)"
          f"  -> Gate 3 {'PASS' if agree > 0.99 else 'FAIL'}")

    paths.WORK.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({
        "model": model, "features": features, "feature_ids": ids,
        "trees": trees, "depth": depth, "hidden": hidden, "method": method,
        "description": description, "classes": pipe[-1].classes_.tolist(),
        "agreement_host": agree,
    }, indent=1))
    joblib.dump(pipe, paths.WORK / "model.joblib")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=models.MODELS, required=True)
    parser.add_argument("--features", default="thesis_train",
                        help="thesis_train | thesis_published | all | top:<k> | random:<d>")
    parser.add_argument("--trees", type=int, default=30)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--method", choices=("inline", "loadable"), default="inline")
    args = parser.parse_args()
    convert(args.model, args.features, args.trees, args.depth, args.hidden, args.method)


if __name__ == "__main__":
    main()
