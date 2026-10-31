"""Step 7: the automated sweep. For each configuration: train -> convert (Gate 3) -> device
pipeline on the 120-window set (Gate 5) -> measurement -> one row of results/sweep.csv.

    .venv/bin/python python/sweep.py [--workers 4] [--only A] [--dry-run]

Groups (one variable at a time, then a combination):
  A  feature set x model, defaults, -Os: all 7056, thesis_train, thesis_published, top-3/5/10 of
     thesis_train's ranking, 10 random 15-feature sets
  B  model complexity on thesis_train: forest trees x depth, tree depth, MLP width
  C  -O2 vs -Os, thesis_train x each model
Window length / FFT size is not swept: the thesis fixes 256 samples/cycle and 10-cycle windows,
and changing them would change the features themselves.

Resumable: configurations already in results/sweep.csv are skipped, failures are logged to
data/sweep/failures.log and retried on the next run. Parallel workers each get their own copy of
the firmware tree and work directory (HIF_FIRMWARE / HIF_WORK), so builds never collide; this
process is the only writer of the CSV. Per-configuration logs: data/sweep/logs/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import paths

SWEEP_DIR = paths.DATA / "sweep"
CSV = paths.RESULTS / "sweep.csv"
DEFAULTS = {"trees": 30, "depth": 8, "hidden": 16, "opt": "-Os"}
MODELS = ("logreg", "tree", "forest", "mlp")
RANDOM_DRAWS = 10


def configs() -> list[dict]:
    out = []
    feature_sets = (["all", "thesis_train", "thesis_published", "top:3", "top:5", "top:10"]
                    + [f"random:{d}" for d in range(RANDOM_DRAWS)])
    for fs in feature_sets:
        for m in MODELS:
            out.append({"group": "A", "features": fs, "model": m, **DEFAULTS})
    for trees in (5, 10, 30):
        for depth in (4, 8, 12):
            if (trees, depth) != (30, 8):
                out.append({"group": "B", "features": "thesis_train", "model": "forest",
                            **DEFAULTS, "trees": trees, "depth": depth})
    for depth in (4, 12):
        out.append({"group": "B", "features": "thesis_train", "model": "tree", **DEFAULTS,
                    "depth": depth})
    for hidden in (8, 32):
        out.append({"group": "B", "features": "thesis_train", "model": "mlp", **DEFAULTS,
                    "hidden": hidden})
    for m in MODELS:
        out.append({"group": "C", "features": "thesis_train", "model": m, **DEFAULTS, "opt": "-O2"})
    return out


def key(c: dict) -> str:
    """Identity of a configuration (group is a label only)."""
    t = c["trees"] if c["model"] == "forest" else "-"
    d = c["depth"] if c["model"] in ("tree", "forest") else "-"
    h = c["hidden"] if c["model"] == "mlp" else "-"
    return f"{c['features']}|{c['model']}|trees={t}|depth={d}|hidden={h}|{c['opt']}"


# ------------------------------------------------------------------ one configuration (child)

def run_job(c: dict) -> dict:
    """Runs in a child process whose HIF_FIRMWARE / HIF_WORK point at this worker's copies."""
    import convert
    import measure
    import run_device

    started = time.time()
    conv = convert.convert(c["model"], c["features"], c["trees"], c["depth"], c["hidden"])
    gate = run_device.check_full("all", opt=c["opt"], save=False)
    meas = measure.measure(c["opt"], save=False)
    cfg = json.loads((paths.WORK / "model.json").read_text())
    return {
        "key": key(c), "group": c["group"], "feature_set": c["features"],
        "k": len(cfg["feature_ids"]), "model": c["model"],
        "n_trees": meas["trees"], "depth": meas["depth"], "hidden": meas["hidden"],
        "fft_n": 256, "opt": c["opt"],
        # Python on the whole 1645-row test split; equals the device when agreement is 100%.
        "acc": conv["py_acc"], "f1": conv["py_f1"],
        "agreement_host": conv["agreement_host"],
        "agreement": gate["agreement"], "gate5": gate["gate5"],
        "device_acc_120": gate["device_acc"], "device_f1_120": gate["device_f1_macro"],
        "features_ok": gate["features_ok"], "features_worst_over_tol": gate["features_worst_over_tol"],
        **{k: v for k, v in meas.items()
           if k not in ("description", "opt", "model", "features", "k", "trees", "depth", "hidden")},
        "description": conv["description"], "minutes": (time.time() - started) / 60,
    }


# ------------------------------------------------------------------ orchestration (parent)

def prepare_slot(i: int) -> dict[str, str]:
    """Worker i's private firmware tree (sources copied, CMSIS linked) and work directory."""
    root = SWEEP_DIR / f"w{i}"
    fw = root / "firmware"
    src = paths.REPO / "firmware"
    for sub in ("src", "test"):
        shutil.rmtree(fw / sub, ignore_errors=True)
        shutil.copytree(src / sub, fw / sub)
    for f in ("Makefile", "linker.ld"):
        shutil.copy2(src / f, fw / f)
    (fw / "include").mkdir(parents=True, exist_ok=True)
    if not (fw / "third_party").exists():
        (fw / "third_party").symlink_to(src / "third_party")
    shutil.rmtree(fw / "build", ignore_errors=True)
    (root / "work").mkdir(parents=True, exist_ok=True)
    return {**os.environ, "HIF_FIRMWARE": str(fw), "HIF_WORK": str(root / "work")}


def run_in_slot(c: dict, env: dict) -> dict:
    log = SWEEP_DIR / "logs" / (key(c).replace("|", "_").replace(":", "-") + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    result_file = SWEEP_DIR / f"result_{threading.get_ident()}.json"
    result_file.unlink(missing_ok=True)
    with open(log, "w") as fh:
        proc = subprocess.run(
            [sys.executable, __file__, "--job", json.dumps(c), "--result", str(result_file)],
            cwd=paths.REPO / "python", env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not result_file.exists():
        raise RuntimeError(f"exit {proc.returncode}; see {log}")
    row = json.loads(result_file.read_text())
    result_file.unlink()
    return row


def append_row(row: dict) -> None:
    df = pd.read_csv(CSV, keep_default_na=False) if CSV.exists() else pd.DataFrame()
    df = df[df["key"] != row["key"]] if len(df) else df
    pd.concat([df, pd.DataFrame([row])]).to_csv(CSV, index=False, float_format="%.6g")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", choices=["A", "B", "C"], help="run one group")
    parser.add_argument("--match", help="run only configurations whose key contains this")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--job", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.job:                                  # child: one configuration
        row = run_job(json.loads(args.job))
        with open(args.result, "w") as fh:
            json.dump(row, fh, default=float)
        return

    done = set(pd.read_csv(CSV)["key"]) if CSV.exists() else set()
    todo = [c for c in configs()
            if key(c) not in done and (not args.only or c["group"] == args.only)
            and (not args.match or args.match in key(c))]
    # Slowest first (all-features builds take ~4x longer), so they do not finish last.
    todo.sort(key=lambda c: c["features"] != "all")
    print(f"{len(configs())} configurations, {len(done)} done, {len(todo)} to run "
          f"on {args.workers} workers", flush=True)
    if args.dry_run:
        for c in todo:
            print("  ", c["group"], key(c))
        return

    slots = [prepare_slot(i) for i in range(args.workers)]
    free = list(range(args.workers))
    lock = threading.Lock()

    def task(c: dict) -> dict:
        with lock:
            slot = free.pop()
        try:
            return run_in_slot(c, slots[slot])
        finally:
            with lock:
                free.append(slot)

    failures = SWEEP_DIR / "failures.log"
    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, c): c for c in todo}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001 - one failed configuration must not stop the sweep
                with open(failures, "a") as fh:
                    fh.write(f"{time.strftime('%F %T')} {key(c)}: {exc}\n")
                print(f"FAILED {key(c)}: {exc}", flush=True)
                continue
            append_row(row)
            n_ok += 1
            print(f"[{n_ok}/{len(todo)}] {key(c)}: f1 {row['f1']:.3f} agreement "
                  f"{row['agreement']:.3f} flash {row['flash_kb']:.0f} KB "
                  f"instr {row['instr_feat'] / 1e6:.1f}M+{row['instr_inf']:,} "
                  f"({row['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
