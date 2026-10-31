"""Step 6: measure the configuration in data/model.json on the simulated device.

    .venv/bin/python python/measure.py [--opt -Os]

One MODE=bench build (renode/bench.robot) gives:
  instructions   executed per feature extraction (mean over 2 passes of the build's raw windows)
                 and per inference (mean over all 1645 test rows), plus one window split by
                 extraction stage (DFT / Stockwell / wavelet / statistics). Read from Renode's
                 counter by a CPU hook on bench_mark(): exact to the instruction.
  flash, RAM     per module, from the linker map; test data (hif_tv_*, hif_bench_*) excluded
  stack          peak, by stack painting; plus the largest per-function frames (-fstack-usage)
  input buffer   RAM a real device needs for the raw window (here it sits in flash as test
                 data): pre + post x 2560 samples x the raw signals the features can read
The run also re-checks correctness: inference must agree with Python on every row, and the
benchmarked windows' predictions with scikit-learn.

Renode is a functional simulator: instruction counts compare configurations, they are not
cycles. `est_ms_168mhz` assumes 1 instruction per cycle at the STM32F407's 168 MHz -- a rough
estimate only (see report/measurement.md).

Output: one row per (configuration, optimisation level) in results/measurements.csv.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np

import export_vectors
import paths
from run_device import (FIRMWARE, FLASH_BYTES, RUN_DIR, build, elf_flash_bytes, n_features,
                        parse_predictions, record, run_renode, window_bytes)

BUILD = FIRMWARE / "build"
CLOCK_HZ = 168e6
TEST_DATA = re.compile(r"hif_(tv|bench)_")
FLASH_SECTIONS = (".isr_vector", ".text", ".ARM.exidx", ".data")
RAM_SECTIONS = (".data", ".bss", ".ccmram")


def category(section: str, origin: str) -> str:
    """Which part of the firmware an input section belongs to."""
    if TEST_DATA.search(section):
        return "test_data"
    if "/cmsis/" in origin:
        return "cmsis"
    if origin.endswith(("features.o", "dsp2560.o")):
        return "extraction"
    if origin.endswith("model.o"):
        return "model"
    if ".a(" in origin or origin.endswith(".a"):
        return "libc"                      # newlib libc/libm, libgcc
    return "app"                           # startup, uart, main


def map_breakdown(map_path) -> dict[str, dict[str, int]]:
    """{"flash"|"ram": {category: bytes}} from a GNU ld map file."""
    lines = map_path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Linker script and memory map"))
    out = {"flash": defaultdict(int), "ram": defaultdict(int), "ccm": 0}
    output_section = None
    pending = None                          # input section name whose address is on the next line
    entry = re.compile(r"^\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S.*)$")
    for line in lines[start:]:
        m = re.match(r"^(\.[\w.]+|/DISCARD/)\b", line)
        if m:
            output_section = m.group(1)
            continue
        m = re.match(r"^ (\.[\w.$]+|COMMON)\s*(.*)$", line)
        if m:
            name, rest = m.groups()
            if not rest:
                pending = name
                continue
            m2 = entry.match(" " + rest)
            if m2:
                pending = None
                _add(out, output_section, name, *m2.groups())
            continue
        if pending:
            m2 = entry.match(line)
            if m2:
                _add(out, output_section, pending, *m2.groups())
            pending = None
    return {"flash": dict(out["flash"]), "ram": dict(out["ram"]), "ccm": out["ccm"]}


def _add(out, output_section, name, addr, size, origin) -> None:
    size = int(size, 16)
    if not size or int(addr, 16) == 0 or output_section is None:
        return
    cat = category(name, origin.strip())
    if output_section in FLASH_SECTIONS:
        out["flash"][cat] += size
    if output_section in RAM_SECTIONS:
        out["ram"][cat] += size
    if output_section == ".ccmram":
        out["ccm"] += size


def ram_symbols(map_path, top: int = 6) -> list[tuple[int, str]]:
    """Largest .bss/.data input sections (one per variable with -fdata-sections)."""
    found = re.findall(r"^ \.(?:bss|data)\.(\S+)\s*\n?\s+0x[0-9a-f]+\s+0x([0-9a-f]+)",
                       map_path.read_text(), re.M)
    return sorted(((int(size, 16), name) for name, size in found), reverse=True)[:top]


def input_buffer_kb(feature_ids: list[str]) -> float:
    """Raw signals a device must buffer for these features (the faulted phase is only known at
    run time, so a phase quantity needs all three phases)."""
    channels = set()
    for fid in feature_ids:
        for ch in fid.split(".")[1].split("__"):
            if ch in ("fault_voltage", "voltage_sum", "negative_sequence_voltage"):
                channels |= {"Va", "Vb", "Vc"}
            elif ch in ("fault_current", "negative_sequence_current"):
                channels |= {"Ia", "Ib", "Ic"}
            elif ch == "neutral_current":
                channels.add("In")
    return 2 * 2560 * 4 * len(channels) / 1024


def stack_frames(top: int = 5) -> list[tuple[int, str]]:
    """Largest per-function stack frames from the -fstack-usage (.su) files."""
    frames = []
    for su in BUILD.rglob("*.su"):
        for line in su.read_text().splitlines():
            where, size, _kind = line.split("\t")
            frames.append((int(size), where.split(":")[-1]))
    return sorted(frames, reverse=True)[:top]


def fit_bench(opt: str) -> tuple[int, int]:
    """(raw windows, inference rows) that fit next to this configuration's code.

    Build with one of each, then give the spare flash to inference rows first (up to every test
    row: trees cost different amounts per input, so more rows is a better mean) and the rest to
    windows (extraction is nearly data-independent, so few windows suffice)."""
    export_vectors.export_bench(1, n_rows=1)
    build("bench", opt)
    spare = FLASH_BYTES - elf_flash_bytes() - 1024
    nf = n_features()
    rows = int(min(export_vectors.N_TEST, 1 + max(0, spare - window_bytes(nf)) // (4 * nf + 1)))
    spare -= (rows - 1) * (4 * nf + 1)
    return max(1, 1 + spare // window_bytes(nf)), rows


def measure(opt: str, save: bool = True) -> dict:
    """Measure the configuration in model.json at `opt`; save=False leaves the CSV alone."""
    cfg = json.loads((paths.WORK / "model.json").read_text())
    k, n_rows = fit_bench(opt)
    ref = export_vectors.export_bench(k, n_rows=n_rows)
    build("bench", opt)
    marks_file = RUN_DIR / "bench_marks.txt"
    marks_file.unlink(missing_ok=True)
    uart = run_renode("bench.log", robot="bench.robot", extra={"MARKS": marks_file})

    # bench_mark ids (firmware/src/main.c): 0 extraction, 1 inference, 2 end,
    # 3 profile start, 4 after DFT, 5 after Stockwell, 6 after wavelet, 7 after statistics.
    marks = dict(tuple(map(int, m.split(","))) for m in marks_file.read_text().strip(";").split(";"))
    c0, c1, c2 = marks[0], marks[1], marks[2]
    stages = {name: marks[i + 1] - marks[i]
              for i, name in zip(range(3, 7), ("dft", "stockwell", "wavelet", "stats"))}
    n_extract, n_infer, wrong = map(int, re.search(r"BENCH,(\d+),(\d+),(\d+)", uart).groups())
    stack = int(re.search(r"STACK,(\d+)", uart).group(1))
    preds = parse_predictions(uart)
    agree = np.mean([preds[r] == p for r, p in zip(ref["rows"], ref["preds"])])
    # Same threshold as Gates 3 and 5: the C model may legitimately differ from scikit-learn on
    # a near-tie (emlearn quantises forest leaf proportions to 8 bits); a broken build does not
    # get within 1%.
    bench_agree = 1 - wrong / n_infer
    if bench_agree < 0.99 or agree < 0.99:
        raise SystemExit(f"bench build disagrees with Python: {wrong} of {n_infer} inferences, "
                         f"{agree:.3f} window agreement")

    br = map_breakdown(BUILD / "hif.map")
    flash = {c: v for c, v in br["flash"].items() if c != "test_data"}
    ram = br["ram"]
    instr_feat = (c1 - c0) / n_extract
    instr_inf = (c2 - c1) / n_infer
    row = {
        "description": cfg["description"], "opt": opt,
        "model": cfg["model"], "features": cfg["features"], "k": len(cfg["feature_ids"]),
        "trees": cfg["trees"] if cfg["model"] == "forest" else "",
        "depth": cfg["depth"] if cfg["model"] in ("tree", "forest") else "",
        "hidden": cfg["hidden"] if cfg["model"] == "mlp" else "",
        "flash_kb": sum(flash.values()) / 1024,
        **{f"flash_{c}_kb": flash.get(c, 0) / 1024 for c in ("extraction", "cmsis", "model", "libc", "app")},
        "ram_kb": sum(ram.values()) / 1024,
        **{f"ram_{c}_kb": ram.get(c, 0) / 1024 for c in ("extraction", "cmsis", "model", "libc", "app")},
        "ram_ccm_kb": br["ccm"] / 1024,
        "stack_b": stack,
        "input_kb": input_buffer_kb(cfg["feature_ids"]),
        "instr_feat": round(instr_feat), "instr_inf": round(instr_inf),
        **{f"instr_{k}": v for k, v in stages.items()},
        "est_ms_168mhz": (instr_feat + instr_inf) / CLOCK_HZ * 1e3,
        "bench_windows": n_extract, "bench_rows": n_infer, "bench_agreement": bench_agree,
    }
    print(f"{cfg['description']}  [{opt}]")
    print(f"  flash {row['flash_kb']:.1f} KB  (extraction {row['flash_extraction_kb']:.1f}, "
          f"cmsis {row['flash_cmsis_kb']:.1f}, model {row['flash_model_kb']:.1f}, "
          f"libc {row['flash_libc_kb']:.1f}, app {row['flash_app_kb']:.1f})"
          f"   test data excluded: {br['flash'].get('test_data', 0) / 1024:.0f} KB")
    print(f"  static RAM {row['ram_kb']:.1f} KB, of which CCM {row['ram_ccm_kb']:.1f}  (extraction {row['ram_extraction_kb']:.1f}, "
          f"cmsis {row['ram_cmsis_kb']:.1f}, model {row['ram_model_kb']:.1f}, "
          f"libc {row['ram_libc_kb']:.1f}, app {row['ram_app_kb']:.1f})")
    print("  largest RAM variables: "
          + ", ".join(f"{name} {size / 1024:.1f}" for size, name in ram_symbols(BUILD / "hif.map")))
    print(f"  raw input buffer a real device would need: {row['input_kb']:.0f} KB")
    print(f"  stack peak {stack} B; largest frames: "
          + ", ".join(f"{name} {size}" for size, name in stack_frames()))
    print(f"  instructions: extraction {instr_feat:,.0f} / window, inference {instr_inf:,.0f}"
          f"   (rough: {row['est_ms_168mhz']:.1f} ms at 168 MHz, 1 instr/cycle)")
    total = sum(stages.values())
    print("  extraction by stage (one window): "
          + ", ".join(f"{k} {v:,} ({100 * v / total:.0f}%)" for k, v in stages.items()))
    print(f"  correctness: {n_infer - wrong}/{n_infer} inferences agree with Python; "
          f"{agree:.0%} of {len(ref['rows'])} windows")
    if save:
        record(paths.RESULTS / "measurements.csv", row, key=("description", "opt"))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", default="-Os", choices=["-Os", "-O2", "-O3"])
    measure(parser.parse_args().opt)


if __name__ == "__main__":
    main()
