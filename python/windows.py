"""Raw analysis windows for a simulation, produced by the thesis's own code path.

read Msub -> add noise (seeded per file and SNR) -> cubic resample to 256 samples/cycle -> cut
the pre-fault and post-fault windows. Using the thesis's functions, not a re-implementation,
guarantees the windows are exactly those its feature matrix was computed from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

import features
import paths

paths.import_thesis()
from hif.config import load_extraction  # noqa: E402
from hif.features.extract import _windows_for_raw_channels  # noqa: E402
from hif.io.matreader import CHANNEL_PATHS, RawSignals, read_substation  # noqa: E402
from hif.io.metadata import build_manifest  # noqa: E402
from hif.preprocess import noise  # noqa: E402

assert tuple(CHANNEL_PATHS) == features.RAW_CHANNELS


@lru_cache(maxsize=1)
def manifest() -> dict[str, object]:
    """file_id -> SampleMeta, for the corpus the thesis's matrix was built from."""
    metas = build_manifest(paths.THESIS_RAW, corpus="dadosIC", include=("FAI_fase", "FaseB"))
    return {m.file_id: m for m in metas}


def window(file_id: str, snr: str = paths.SNR) -> tuple[np.ndarray, int]:
    """(x, phase): x[w, c, n] float64 with w = pre/post, c in RAW_CHANNELS order; phase 0/1/2."""
    meta = manifest()[file_id]
    cfg = load_extraction()
    signals = read_substation(meta.path)
    signals = RawSignals(
        time=signals.time,
        channels=noise.apply(signals.channels, snr, file_id=Path(meta.path).stem),
    )
    pairs = _windows_for_raw_channels(signals, cfg)
    x = np.stack([
        np.stack([pairs[c].pre for c in CHANNEL_PATHS]),
        np.stack([pairs[c].post for c in CHANNEL_PATHS]),
    ])
    return x, "ABC".index(meta.phase.upper())
