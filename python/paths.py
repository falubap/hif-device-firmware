"""Locations shared by every script. The thesis repo is read, never written."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
RESULTS = REPO / "results"
#: Firmware tree and per-configuration work files (model.json, model.joblib, Renode logs).
#: The sweep gives each parallel worker its own copy through these two variables.
FIRMWARE = Path(os.environ.get("HIF_FIRMWARE", REPO / "firmware"))
FIRMWARE_INCLUDE = FIRMWARE / "include"
WORK = Path(os.environ.get("HIF_WORK", DATA))

#: The thesis repository (code, feature matrices, raw simulations). Override with HIF_THESIS.
THESIS = Path(os.environ.get("HIF_THESIS", "/home/pedrofaluba/tcc"))
THESIS_FEATURES = THESIS / "data" / "processed" / "features_snr_{snr}.parquet"
THESIS_CLUSTERS = THESIS / "data" / "processed" / "clusters_snr_{snr}.npz"
THESIS_RAW = THESIS / "data" / "raw" / "distilled"
THESIS_RANKING = THESIS / "reports" / "snr_{snr}" / "run_b.json"

SNR = "40"

SPLIT = DATA / "split.json"
FEATURES = DATA / f"features_snr{SNR}.npz"
TEST_WINDOWS = DATA / f"test_windows_snr{SNR}.npz"


def import_thesis() -> None:
    """Make the thesis package (`hif`) and its scripts importable."""
    for p in (THESIS / "src", THESIS / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
