"""Build the firmware for the configuration in data/model.json, run it in Renode, check its output.

    .venv/bin/python python/run_device.py model     # Step 3: model on all test rows' features

Output is the UART log parsed into predictions and compared with Python's: agreement with
scikit-learn, and accuracy / macro-F1 on the device. Renode's own files go to data/renode/.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import export_vectors
import paths

FIRMWARE = paths.REPO / "firmware"
ELF = FIRMWARE / "build" / "hif.elf"
RUN_DIR = paths.DATA / "renode"


def build(mode: str, opt: str = "-Os") -> None:
    subprocess.run(["make", "-s", "-C", str(FIRMWARE), f"MODE={mode}", f"OPT={opt}"], check=True)


def run_renode(log_name: str) -> str:
    """Run the current ELF until DONE; return what USART2 printed."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log = RUN_DIR / log_name
    log.unlink(missing_ok=True)
    result = subprocess.run(
        ["renode-test", str(paths.REPO / "renode" / "run.robot"),
         "--variable", f"ELF:{ELF}", "--variable", f"UART_LOG:{log}"],
        cwd=RUN_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0 or not log.exists():
        raise SystemExit(f"Renode run failed:\n{result.stdout[-3000:]}")
    return log.read_text(errors="replace")


def parse_predictions(text: str) -> dict[int, int]:
    return {int(r): int(p) for r, p in re.findall(r"^P,(\d+),(\d+)\s*$", text, re.M)}


def check_model() -> None:
    ref = export_vectors.export_features_only()
    build("model")
    text = run_renode("model.log")
    got = parse_predictions(text)
    if "DONE" not in text or sorted(got) != ref["rows"].tolist():
        raise SystemExit(f"expected {len(ref['rows'])} predictions, got {len(got)}")
    pred = np.array([got[r] for r in ref["rows"]])
    agree = (pred == ref["preds"]).mean()
    print(f"device predictions: {len(pred)}")
    print(f"  agreement with scikit-learn: {agree:.4f} ({int((pred != ref['preds']).sum())} differ)")
    acc = accuracy_score(ref["labels"], pred)
    f1 = f1_score(ref["labels"], pred, average="macro")
    print(f"  device acc {acc:.4f}  macro-F1 {f1:.4f}")
    record(paths.RESULTS / "model_check.csv", {
        "description": json.loads((paths.DATA / "model.json").read_text())["description"],
        "n": len(pred), "agreement": agree, "acc": acc, "f1_macro": f1,
        "flash_bytes_incl_features": elf_flash_bytes(),
    })


def elf_flash_bytes() -> int:
    """text + data of the current ELF (what occupies flash)."""
    out = subprocess.run(["arm-none-eabi-size", str(ELF)], capture_output=True, text=True, check=True)
    text, data = out.stdout.splitlines()[1].split()[:2]
    return int(text) + int(data)


def record(path, row: dict) -> None:
    """Upsert one result row, keyed by description, into a CSV."""
    import pandas as pd
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df):
        df = df[df["description"] != row["description"]]
    pd.concat([df, pd.DataFrame([row])]).to_csv(path, index=False, float_format="%.4f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["model"])
    args = parser.parse_args()
    if args.what == "model":
        check_model()


if __name__ == "__main__":
    main()
