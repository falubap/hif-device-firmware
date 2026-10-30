"""Save the raw test windows, and check features.py against the thesis on every test row.

    .venv/bin/python python/export_windows.py [--jobs 2]

For each test file: rebuild the float64 window through the thesis's code (windows.py), check that
features.extract reproduces the thesis's 7056 features, then store the window as float32 -- the
device's input format -- together with the features computed from that float32 window, which is
the exact reference the C code will be compared against.

Output: data/test_windows_snr40.npz
    x            (n, 2, 7, 2560) float32   raw windows, RAW_CHANNELS order
    phase        (n,) int8                 faulted phase 0/1/2
    file_id, y   (n,)
    feat_f32in   (n, 7056) float64         features.extract on the float32 window
    parity_rel   (n,) float64              max relative error vs the thesis matrix (float64 window)
"""

from __future__ import annotations

import argparse

import numpy as np
from joblib import Parallel, delayed

import features
import paths
import windows


def one(file_id: str, names: list[str], ref: np.ndarray):
    x, phase = windows.window(file_id)
    got = features.extract(x, phase, names)
    parity = float((np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)).max())
    x32 = x.astype(np.float32)
    return x32, phase, features.extract(x32, phase, names), parity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()

    d = np.load(paths.FEATURES, allow_pickle=True)
    names = d["names"].tolist()
    ids, X = d["file_id_test"], d["X_test"].astype(np.float64)

    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(one)(fid, names, X[k]) for k, fid in enumerate(ids)
    )
    x, phase, feat, parity = map(np.array, zip(*rows))
    np.savez(
        paths.TEST_WINDOWS, x=x, phase=phase.astype(np.int8), file_id=ids, y=d["y_test"],
        feat_f32in=feat, parity_rel=parity, names=np.array(names),
    )

    print(f"{len(ids)} windows -> {paths.TEST_WINDOWS}")
    print(f"features.py vs thesis matrix: max rel error {parity.max():.2e} "
          f"(float32 storage rounding is ~6e-8)")
    rel = np.abs(feat - X) / np.maximum(np.abs(X), 1e-30)
    worst = rel.max(axis=0)
    print(f"float32 input vs float64 input: median {np.median(rel):.1e}, "
          f"features with max rel > 1e-3: {(worst > 1e-3).sum()} of {len(names)}")


if __name__ == "__main__":
    main()
