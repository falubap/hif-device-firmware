"""Build the firmware for the configuration in data/model.json, run it in Renode, check its output.

    .venv/bin/python python/run_device.py model     # Step 3: model on all test rows' features
    .venv/bin/python python/run_device.py host      # Step 4: C features on the PC, all test windows
    .venv/bin/python python/run_device.py features  # Step 4: C features on the device, 120 windows

Predictions are compared with scikit-learn (agreement, device accuracy / macro-F1). Features are
compared with features.extract on the same float32 windows (data/test_windows), using the
tolerance in CLAUDE.md: |c - py| <= 1e-3 |py| + 1e-3 spread, angles by wrapped difference.
Renode's own files go to data/renode/.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import export_vectors
import feature_config
import features
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


def parse_features(text: str) -> dict[int, np.ndarray]:
    """F,<row>,<hex bits>,... -> {row: float32 features}."""
    out = {}
    for line in re.findall(r"^F,(\d+),([0-9a-f,]+)\s*$", text, re.M):
        bits = np.array([int(h, 16) for h in line[1].split(",")], dtype=np.uint32)
        out[int(line[0])] = bits.view(np.float32)
    return out


def feature_reference():
    """Feature ids, Python reference on every test window (float32 input) and per-feature spread."""
    cfg = json.loads((paths.DATA / "model.json").read_text())
    w = np.load(paths.TEST_WINDOWS, allow_pickle=True)
    col = {n: i for i, n in enumerate(w["names"].tolist())}
    ref = w["feat_f32in"][:, [col[f] for f in cfg["feature_ids"]]]
    return cfg["feature_ids"], ref, ref.std(axis=0), w


def report_parity(ids, rows, got, ref, spread, summary_only: bool = False) -> bool:
    """Print per-feature parity; return True if every value of every feature is within tolerance.

    summary_only: instead of one line per feature, group by transform / parameter / statistic and
    list only the failing or closest features."""
    diff = got - ref
    angle = np.array([features.parse(f)[3] in features.CIRCULAR for f in ids])
    diff[:, angle] = np.angle(np.exp(1j * diff[:, angle]))
    tol = 1e-3 * np.abs(ref) + 1e-3 * spread
    score = np.abs(diff) / tol                      # <= 1 passes
    ok = score <= 1
    rel = np.abs(diff) / np.maximum(np.abs(ref), 1e-30)
    print(f"{len(rows)} windows x {len(ids)} features: {int(ok.sum())}/{ok.size} values within tolerance")
    worst = score.max(axis=0)
    if summary_only:
        parts = np.array([fid.split(".") for fid in ids])
        for axis, name in ((0, "transform"), (3, "parameter"), (4, "statistic")):
            groups = {g: worst[parts[:, axis] == g] for g in np.unique(parts[:, axis])}
            print(f"  by {name}: " + ", ".join(f"{g} {int((v <= 1).sum())}/{len(v)} (worst {v.max():.2f})"
                                               for g, v in groups.items()))
        show = np.argsort(-worst)[:10]
        print("  closest to / over tolerance:")
    else:
        show = range(len(ids))
    print(f"  {'pass':>9} {'worst/tol':>9} {'max rel':>8}  feature")
    for j in show:
        fid = ids[j]
        print(f"  {int(ok[:, j].sum()):4d}/{len(rows):<4d} {worst[j]:9.2e} {rel[:, j].max():8.1e}  {fid}")
    passed = bool(ok.all())
    print(f"Gate 4: {'PASS' if passed else 'FAIL'}")
    report_parity.last = {"windows": len(rows), "features": len(ids),
                          "values_ok": f"{int(ok.sum())}/{ok.size}",
                          "worst_over_tol": float(worst.max()), "gate4": "PASS" if passed else "FAIL"}
    return passed


def check_host(all_features: bool = False) -> None:
    """C extractor compiled for the PC, on every test window.

    all_features: configure the extractor with all 7056 thesis features instead of the model's,
    to exercise every transform, channel, parameter and statistic (restored afterwards).
    """
    ids, ref, spread, w = feature_reference()
    if all_features:
        ids = w["names"].tolist()
        ref = w["feat_f32in"]
        spread = ref.std(axis=0)
        feature_config.write(ids, "all thesis features (host parity check)")
    head = np.stack([np.arange(len(w["phase"])), w["phase"]], axis=1).astype(np.int32)
    payload = b"".join(head[k].tobytes() + w["x"][k].astype(np.float32).tobytes()
                       for k in range(len(head)))
    try:
        subprocess.run(["make", "-s", "-C", str(FIRMWARE), "host"], check=True)
        out = subprocess.run([str(FIRMWARE / "build" / "host" / "host_features")], input=payload,
                             capture_output=True, check=True).stdout.decode()
    finally:
        if all_features:
            cfg = json.loads((paths.DATA / "model.json").read_text())
            feature_config.write(cfg["feature_ids"], cfg["description"])
    got = parse_features(out)
    rows = np.arange(len(head))
    got = np.array([got[r] for r in rows], dtype=np.float64)
    if all_features:
        np.save(paths.DATA / "host_features_all.npy", got.astype(np.float32))
    report_parity(ids, rows, got, ref[rows], spread, summary_only=all_features)
    cfg = json.loads((paths.DATA / "model.json").read_text())
    record(paths.RESULTS / "feature_check.csv", {
        "description": "host, all 7056 features" if all_features else f"host, {cfg['description']}",
        **report_parity.last})


def check_features(batch_list: list[int]) -> None:
    """C extractor on the device, over the device set, one firmware build per batch."""
    ids, ref, spread, _ = feature_reference()
    got: dict[int, np.ndarray] = {}
    for b in batch_list:
        out = export_vectors.export_batch(b)
        build("features")
        text = run_renode(f"features_{b:02d}.log")
        batch = parse_features(text)
        if "DONE" not in text or sorted(batch) != sorted(out["rows"].tolist()):
            raise SystemExit(f"batch {b}: expected rows {out['rows'].tolist()}, got {sorted(batch)}")
        got.update(batch)
        print(f"batch {b:2d}: {len(batch)} windows, flash {elf_flash_bytes() / 1024:.0f} KB", flush=True)
    rows = np.array(sorted(got))
    report_parity(ids, rows, np.array([got[r] for r in rows], dtype=np.float64), ref[rows], spread)
    cfg = json.loads((paths.DATA / "model.json").read_text())
    record(paths.RESULTS / "feature_check.csv", {
        "description": f"device, {cfg['description']}", **report_parity.last})


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
    parser.add_argument("what", choices=["model", "host", "features"])
    parser.add_argument("--all-features", action="store_true",
                        help="host: check all 7056 features, not just the model's")
    parser.add_argument("--batches", default="all", help="device batches, e.g. 0 or 0-4 (default all)")
    args = parser.parse_args()
    if args.what == "model":
        check_model()
    elif args.what == "host":
        check_host(all_features=args.all_features)
    else:
        n = export_vectors.batches()
        if args.batches == "all":
            batch_list = list(range(n))
        elif "-" in args.batches:
            lo, hi = map(int, args.batches.split("-"))
            batch_list = list(range(lo, hi + 1))
        else:
            batch_list = [int(args.batches)]
        check_features(batch_list)


if __name__ == "__main__":
    main()
