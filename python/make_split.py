"""Fix the train/test split once, and save the feature matrices for both sides.

    .venv/bin/python python/make_split.py

The split holds out whole soils (HIF signal models), as the thesis's Run A does: files sharing a
soil share the same arc waveform, so splitting by row would put that waveform on both sides.
Every soil carries the same scenario grid, so the region mix is identical on both sides.

Outputs:
    data/split.json          the definition: test soils and file ids (reproducible from SEED)
    data/features_snr40.npz  X_train, X_test (raw thesis features, float32), labels, metadata
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import paths

N_TEST_SOILS = 7  # of 34, ~20%
SEED = 0


def main() -> None:
    frame = pd.read_parquet(paths.THESIS_FEATURES.as_posix().format(snr=paths.SNR))
    names = [c for c in frame.columns if c.count(".") == 4]

    soils = np.sort(frame["soil"].unique())
    rng = np.random.default_rng(SEED)
    test_soils = np.sort(rng.choice(soils, size=N_TEST_SOILS, replace=False))
    is_test = frame["soil"].isin(test_soils).to_numpy()

    split = {
        "snr_db": paths.SNR,
        "group": "soil",
        "seed": SEED,
        "test_soils": test_soils.tolist(),
        "n_train": int((~is_test).sum()),
        "n_test": int(is_test.sum()),
        "test_file_ids": frame.loc[is_test, "file_id"].tolist(),
    }
    paths.DATA.mkdir(exist_ok=True)
    paths.SPLIT.write_text(json.dumps(split, indent=1))

    X = frame[names].to_numpy(np.float32)
    meta = {c: frame[c].to_numpy() for c in ("file_id", "region", "soil", "bus", "faulted_phase")}
    np.savez(
        paths.FEATURES,
        names=np.array(names),
        X_train=X[~is_test], X_test=X[is_test],
        y_train=meta["region"][~is_test], y_test=meta["region"][is_test],
        **{f"{k}_train": v[~is_test] for k, v in meta.items() if k != "region"},
        **{f"{k}_test": v[is_test] for k, v in meta.items() if k != "region"},
    )

    print(f"test soils {test_soils.tolist()}: {split['n_train']} train / {split['n_test']} test")
    for side, mask in (("train", ~is_test), ("test", is_test)):
        counts = pd.Series(meta["region"][mask]).value_counts(normalize=True).sort_index()
        print(f"  {side:5s} region mix", counts.round(3).to_dict())
    print(f"  non-finite features: {int((~np.isfinite(X)).sum())}")


if __name__ == "__main__":
    main()
