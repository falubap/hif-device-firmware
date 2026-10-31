"""Step 7 figures and summary table from results/sweep.csv.

    .venv/bin/python python/plots.py

Only configurations whose device run passed Gate 5 (>= 99% agreement with scikit-learn on the
120 raw windows) are plotted (Gate 7). Writes results/plots/*.png and results/sweep.md.

Colour encodes the feature-set family (3 slots of the reference categorical palette, validated
all-pairs for scatter: CVD dE 9.2, normal-vision dE 24), marker shape the model, so identity
never rests on colour alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402

OUT = paths.RESULTS / "plots"
SURFACE, INK, INK_2, GRID, FRONT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#8a8984"
FAMILY_COLOR = {"thesis selection": "#2a78d6", "all 7056 features": "#eb6834", "random 15": "#1baf7a"}
MODEL_MARKER = {"logreg": "o", "tree": "s", "forest": "^", "mlp": "D"}
STAGE_COLOR = {"DFT": "#2a78d6", "Stockwell": "#eb6834", "wavelet": "#1baf7a",
               "statistics": "#eda100", "inference": "#e87ba4"}


def family(fs: str) -> str:
    if fs == "all":
        return "all 7056 features"
    if fs.startswith("random"):
        return "random 15"
    return "thesis selection"


def style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def figure(width: float = 7.2, height: float = 4.6):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def legends(ax) -> None:
    """One row under the axes: feature-set colours, then model shapes (never over the data)."""
    fam = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=7, label=f)
           for f, c in FAMILY_COLOR.items()]
    mod = [plt.Line2D([], [], marker=m, ls="", markerfacecolor="none", markeredgecolor=INK_2,
                      markersize=7, label=k) for k, m in MODEL_MARKER.items()]
    ax.legend(handles=fam + mod, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=7,
              fontsize=8, frameon=False, labelcolor=INK_2, handletextpad=0.3, columnspacing=1.1)


def log_ticks(axis, values, fmt) -> None:
    """Fixed, plainly labelled ticks on a log axis (matplotlib's default writes 2x10^1)."""
    axis.set_major_locator(matplotlib.ticker.FixedLocator(values))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt(v)))
    axis.set_minor_locator(matplotlib.ticker.NullLocator())


def scatter(ax, df: pd.DataFrame, x: str) -> None:
    for (fam_name, model), g in df.groupby(["family", "model"]):
        ax.scatter(g[x], g["f1"], s=46, marker=MODEL_MARKER[model], color=FAMILY_COLOR[fam_name],
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)


def pareto(df: pd.DataFrame, x: str) -> pd.DataFrame:
    """Configurations no other beats on both cost (lower x) and macro-F1 (higher)."""
    best, keep = -np.inf, []
    for i, r in df.sort_values([x, "f1"], ascending=[True, False]).iterrows():
        if r["f1"] > best:
            keep.append(i)
            best = r["f1"]
    return df.loc[keep]


def steps(front: pd.DataFrame, min_gain: float = 0.03) -> pd.DataFrame:
    """The front's points worth naming: each that gains >= min_gain macro-F1 over the last named
    one, plus the best. The others keep their ring, so the front stays visible."""
    keep, last = [], -np.inf
    for i, r in front.iterrows():
        if r["f1"] - last >= min_gain:
            keep.append(i)
            last = r["f1"]
    if front.index[-1] not in keep:
        keep[-1] = front.index[-1]
    return front.loc[keep]


def label(ax, r, text: str, dx: int = 10, dy: int = 4):
    return ax.annotate(text, (r.x, r.f1), xytext=(dx, dy), textcoords="offset points",
                       fontsize=7.5, color=INK, va="center",
                       arrowprops={"arrowstyle": "-", "color": FRONT, "linewidth": 0.6,
                                   "shrinkA": 0, "shrinkB": 6})


def separate(fig, labels, step: float = 2.0, rounds: int = 30, pad: float = 4.0) -> None:
    """Nudge overlapping annotations apart vertically, keeping `pad` pixels between them (each
    keeps a leader line to its own mark). Bounded: at most rounds x step points of travel."""
    renderer = fig.canvas.get_renderer()
    for _ in range(rounds):
        moved = False
        boxes = [a.get_window_extent(renderer).expanded(1.0, 1.0).padded(pad) for a in labels]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if boxes[i].overlaps(boxes[j]):
                    lower, upper = (i, j) if boxes[i].y0 <= boxes[j].y0 else (j, i)
                    for k, d in ((lower, -step), (upper, step)):
                        x, y = labels[k].xyann
                        labels[k].xyann = (x, y + d)
                    moved = True
        if not moved:
            return


def short(r) -> str:
    fs = {"thesis_train": "thesis 15", "thesis_published": "published 15", "all": "all 7056"}.get(
        r.feature_set, r.feature_set.replace("top:", "top-").replace("random:", "random #"))
    extra = ""
    if r.model == "forest":
        extra = f" {int(r.n_trees)}x{int(r.depth)}"
    elif r.model == "tree":
        extra = f" d{int(r.depth)}"
    elif r.model == "mlp":
        extra = f" {int(r.hidden)}"
    return f"{fs}, {r.model}{extra}{' -O2' if r.opt == '-O2' else ''}"


def tradeoff(df: pd.DataFrame, x: str, xlabel: str, name: str, title: str, ticks, fmt) -> None:
    fig, ax = figure(7.2, 5.0)
    style(ax, xlabel, "macro-F1 on the test split (1645 files)")
    ax.set_xscale("log")
    log_ticks(ax.xaxis, ticks, fmt)
    d = df.assign(x=df[x])
    front = pareto(d, "x")
    ax.step(front["x"], front["f1"], where="post", color=FRONT, linewidth=1.2, zorder=2)
    scatter(ax, d, "x")
    ax.scatter(front["x"], front["f1"], s=120, facecolors="none", edgecolors=INK_2,
               linewidths=0.8, zorder=4)
    legends(ax)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    separate(fig, [label(ax, r, short(r)) for _, r in steps(front).iterrows()])
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)


def cost_split(df: pd.DataFrame) -> None:
    picks = [("thesis_train", m) for m in ("tree", "logreg", "forest", "mlp")]
    picks += [("top:3", "tree"), ("random:0", "forest"), ("all", "forest"), ("all", "mlp")]
    rows = []
    for fs, m in picks:
        g = df[(df.feature_set == fs) & (df.model == m) & (df.opt == "-Os") & (df.group == "A")]
        if len(g):
            rows.append(g.iloc[0])
    if not rows:
        return
    fig, ax = figure(7.2, 0.45 * len(rows) + 1.4)
    style(ax, "million instructions per window", "")
    stages = [("DFT", "instr_dft"), ("Stockwell", "instr_stockwell"), ("wavelet", "instr_wavelet"),
              ("statistics", "instr_stats"), ("inference", "instr_inf")]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for name, col in stages:
        v = np.array([r[col] for r in rows]) / 1e6
        ax.barh(y, v, left=left, height=0.62, color=STAGE_COLOR[name], edgecolor=SURFACE,
                linewidth=2, label=name)
        left += v
    for yi, r, total in zip(y, rows, left):
        share = 100 * r["instr_inf"] / (r["instr_feat"] + r["instr_inf"])
        ax.text(total + left.max() * 0.01, yi, f"inference {r['instr_inf']:,.0f} ({share:.2g}%)",
                va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, [short(r) for r in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, left.max() * 1.32)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK_2, ncol=5,
              bbox_to_anchor=(1, 1.0))
    ax.set_title("Where the instructions go: feature extraction by stage vs inference",
                 color=INK, fontsize=11, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(OUT / "cost_split.png", facecolor=SURFACE)
    plt.close(fig)


BOARDS = [  # (label, RAM KB, flash KB)
    ("Cortex-M0+ class: 32 KB RAM, 256 KB flash", 32, 256),
    ("STM32F407 (this target): 128 KB SRAM + 64 KB CCM, 1 MB flash", 192, 1024),
    ("Cortex-M4 class: 256 KB RAM, 1 MB flash", 256, 1024),
]


def boards(df: pd.DataFrame) -> None:
    fig, ax = figure(7.2, 5.0)
    style(ax, "flash (KB, test data excluded)", "RAM: static (incl. CCM) + peak stack (KB)")
    d = df.assign(x=df["flash_kb"], ram=df["ram_kb"] + df["stack_b"] / 1024)
    for (fam_name, model), g in d.groupby(["family", "model"]):
        ax.scatter(g["x"], g["ram"], s=46, marker=MODEL_MARKER[model],
                   color=FAMILY_COLOR[fam_name], edgecolors=SURFACE, linewidths=1.5, zorder=3)
    for i, (name, ram, flash) in enumerate(BOARDS):
        ax.plot([0, flash, flash], [ram, ram, 0], color=FRONT, linewidth=1, ls=(0, (4, 3)), zorder=2)
        ax.text(12, ram + 3, name, ha="left", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 290)
    legends(ax)
    ax.set_title("Which boards the firmware fits on", color=INK, fontsize=11, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.02, 0.012, "Not shown: a real device also buffers the raw window "
             f"({d['input_kb'].min():.0f}–{d['input_kb'].max():.0f} KB here; it sits in flash "
             "as test data on the simulator).", fontsize=7.5, color=INK_2)
    fig.savefig(OUT / "boards.png", facecolor=SURFACE)
    plt.close(fig)


def table(df: pd.DataFrame, excluded: pd.DataFrame) -> None:
    cols = ["group", "feature_set", "k", "model", "n_trees", "depth", "hidden", "opt", "f1",
            "agreement", "flash_kb", "flash_model_kb", "ram_kb", "stack_b", "input_kb",
            "instr_feat", "instr_inf", "est_ms_168mhz"]
    d = df.sort_values(["group", "feature_set", "model", "n_trees", "depth", "hidden", "opt"])
    lines = [
        "# Sweep results", "",
        f"{len(df)} configurations passed Gate 5 (>= 99% agreement with scikit-learn on 120 raw "
        f"windows on the simulated device); {len(excluded)} excluded. macro-F1 is scikit-learn's on "
        "the whole 1645-file test split; the C model behind it agrees with scikit-learn on at "
        f"least {df['agreement_host'].min():.2%} of those rows (host build) and the device on "
        f"at least {df['agreement'].min():.2%} of the 120 device windows. Flash excludes test "
        "data; RAM is static (incl. CCM); est_ms assumes 1 instruction/cycle at 168 MHz (rough).",
        "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
    ]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and c in ("n_trees", "depth", "hidden"):
                v = "" if np.isnan(v) else f"{v:.0f}"
            elif isinstance(v, float):
                v = "" if np.isnan(v) else (f"{v:.3f}" if c in ("f1", "agreement") else
                                            f"{v:,.0f}" if abs(v) >= 100 else f"{v:.1f}")
            elif isinstance(v, (int, np.integer)) and c.startswith("instr"):
                v = f"{v:,}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    if len(excluded):
        lines += ["", "Excluded (Gate 5 failed): " + ", ".join(excluded["key"])]
    (paths.RESULTS / "sweep.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(paths.RESULTS / "sweep.csv")
    passed = df["agreement"] >= 0.99
    excluded, df = df[~passed], df[passed].copy()
    df["family"] = df["feature_set"].map(family)
    df["instr_total"] = df["instr_feat"] + df["instr_inf"]
    a = df[df.opt == "-Os"]
    tradeoff(a, "flash_kb", "flash (KB, log scale; test data excluded)", "f1_vs_flash.png",
             "Accuracy vs flash: does the thesis selection buy accuracy per KB?",
             [16, 32, 64, 128, 256, 512], lambda v: f"{v:.0f}")
    tradeoff(a, "instr_total", "instructions per window: extraction + inference (log scale)",
             "f1_vs_instructions.png", "Accuracy vs computational cost",
             [2e6, 5e6, 1e7, 2e7, 5e7, 1e8], lambda v: f"{v / 1e6:.0f}M")
    cost_split(df)
    boards(a)
    table(df, excluded)
    print(f"{len(df)} configurations plotted, {len(excluded)} excluded -> {OUT}, results/sweep.md")


if __name__ == "__main__":
    main()
