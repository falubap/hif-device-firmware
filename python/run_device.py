"""Build the firmware for the configuration in data/model.json, run it in Renode, check its output.

    .venv/bin/python python/run_device.py model     # Step 3: model on all test rows' features
    .venv/bin/python python/run_device.py host      # Step 4: C features on the PC, all test windows
    .venv/bin/python python/run_device.py features  # Step 4: C features on the device, 120 windows
    .venv/bin/python python/run_device.py full      # Step 5: raw window -> features -> model

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


FLASH_BYTES = 1024 * 1024
#: Flash one more raw window costs: 2 x 7 x 2560 float32, plus its features and small metadata.
WINDOW_BYTES = 2 * 7 * 2560 * 4 + 64 * 4 + 8


def fit_per_batch(mode: str) -> int:
    """How many raw windows fit next to this configuration's code: build with one, then fill."""
    export_vectors.export_batch(0, per_batch=1)
    build(mode)
    spare = FLASH_BYTES - elf_flash_bytes() - 1024          # keep 1 KB of headroom
    return max(1, 1 + spare // WINDOW_BYTES)


def batch_list(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    if "-" in spec:
        lo, hi = map(int, spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(spec)]


def run_batches(mode: str, spec: str = "all") -> tuple[dict, dict, dict]:
    """Run the device set through firmware `mode`, one build per batch of raw windows.

    Returns (features by row, device predictions by row, Python references by row)."""
    per_batch = fit_per_batch(mode)
    n = export_vectors.batches(per_batch=per_batch)
    print(f"{per_batch} windows per build, {n} builds", flush=True)
    feats, preds, refs = {}, {}, {}
    for b in batch_list(spec, n):
        out = export_vectors.export_batch(b, per_batch)
        build(mode)
        text = run_renode(f"{mode}_{b:02d}.log")
        f = parse_features(text)
        rows = out["rows"].tolist()
        if "DONE" not in text or sorted(f) != sorted(rows):
            raise SystemExit(f"batch {b}: expected rows {rows}, got {sorted(f)}")
        feats.update(f)
        preds.update(parse_predictions(text))
        for k, r in enumerate(rows):
            refs[r] = {key: out[key][k] for key in ("labels", "preds", "preds_thesis")}
        print(f"batch {b:2d}: {len(f)} windows, flash {elf_flash_bytes() / 1024:.0f} KB", flush=True)
    return feats, preds, refs


def check_features(spec: str) -> None:
    """C extractor on the device, over the device set."""
    ids, ref, spread, _ = feature_reference()
    got, _, _ = run_batches("features", spec)
    rows = np.array(sorted(got))
    report_parity(ids, rows, np.array([got[r] for r in rows], dtype=np.float64), ref[rows], spread)
    cfg = json.loads((paths.DATA / "model.json").read_text())
    record(paths.RESULTS / "feature_check.csv", {
        "description": f"device, {cfg['description']}", **report_parity.last})


def check_full(spec: str) -> None:
    """Step 5: the whole pipeline on the device. Gate 5: >= 99% agreement with scikit-learn."""
    ids, ref, spread, _ = feature_reference()
    feats, preds, refs = run_batches("full", spec)
    rows = np.array(sorted(preds))
    dev = np.array([preds[r] for r in rows])
    py = np.array([refs[r]["preds"] for r in rows])
    thesis = np.array([refs[r]["preds_thesis"] for r in rows])
    y = np.array([refs[r]["labels"] for r in rows])
    got = np.array([feats[r] for r in rows], dtype=np.float64)

    report_parity(ids, rows, got, ref[rows], spread)
    agree = float((dev == py).mean())
    print(f"\npredictions on {len(rows)} raw windows:")
    print(f"  agreement with scikit-learn on the same float32 windows: {agree:.4f} "
          f"({int((dev != py).sum())} differ)")
    print(f"  agreement with scikit-learn on the thesis's features:     {(dev == thesis).mean():.4f}")
    acc, f1 = accuracy_score(y, dev), f1_score(y, dev, average="macro")
    print(f"  device acc {acc:.4f}  macro-F1 {f1:.4f}   "
          f"(python on the same windows: acc {accuracy_score(y, py):.4f}  "
          f"macro-F1 {f1_score(y, py, average='macro'):.4f})")
    for k in np.flatnonzero(dev != py):
        # Which features moved most, in units of their tolerance.
        d = np.abs(got[k] - ref[rows[k]]) / (1e-3 * np.abs(ref[rows[k]]) + 1e-3 * spread)
        top = np.argsort(-d)[:3]
        print(f"  row {rows[k]}: device {dev[k]}, python {py[k]}; largest feature errors: "
              + ", ".join(f"{ids[j]} ({d[j]:.2f} tol)" for j in top))
    passed = agree >= 0.99
    print(f"Gate 5: {'PASS' if passed else 'FAIL'}")
    cfg = json.loads((paths.DATA / "model.json").read_text())
    record(paths.RESULTS / "pipeline_check.csv", {
        "description": cfg["description"], "windows": len(rows), "agreement": agree,
        "agreement_thesis_features": float((dev == thesis).mean()),
        "device_acc": acc, "device_f1_macro": f1, "features_ok": report_parity.last["values_ok"],
        "gate5": "PASS" if passed else "FAIL",
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
    parser.add_argument("what", choices=["model", "host", "features", "full"])
    parser.add_argument("--all-features", action="store_true",
                        help="host: check all 7056 features, not just the model's")
    parser.add_argument("--batches", default="all", help="device batches, e.g. 0 or 0-4 (default all)")
    args = parser.parse_args()
    if args.what == "model":
        check_model()
    elif args.what == "host":
        check_host(all_features=args.all_features)
    elif args.what == "features":
        check_features(args.batches)
    else:
        check_full(args.batches)


if __name__ == "__main__":
    main()
