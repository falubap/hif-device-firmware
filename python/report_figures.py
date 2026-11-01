"""Figures for the reports, in the thesis's figure style: Portuguese (report/relatorio.tex,
fig-*.pdf) and English (report/report.tex, fig-*-en.pdf).

    .venv/bin/python python/report_figures.py

Follows tcc/scripts/build_figures.py: DejaVu Sans 9 pt, 13 cm wide, decimal comma in Portuguese,
recessive axes, no title inside the figure (the LaTeX caption carries it), vector PDF. Data:
results/sweep.csv, Gate-5-passing configurations only (Gate 7). Colour encodes the feature-set
family (3 slots validated all-pairs for scatter), marker shape the model.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

import paths  # noqa: E402

OUT = paths.REPO / "report"
CM = 1 / 2.54
INK, INK2, GRID, FRONT = "#0b0b0b", "#52514e", "#d8d8d4", "#8a8984"
FAMILY = {"selecao": "#2a78d6", "todos": "#eb6834", "aleatorio": "#1baf7a"}
MODEL = {"logreg": "o", "tree": "s", "forest": "^", "mlp": "D"}
STAGES = [("instr_dft", "#2a78d6"), ("instr_stockwell", "#eb6834"), ("instr_wavelet", "#1baf7a"),
          ("instr_stats", "#eda100"), ("instr_inf", "#e87ba4")]
TEXT = {
    "pt": {
        "suffix": "", "point": ",", "sep": ".",
        "family": {"selecao": "seleção da monografia", "todos": "todos os 7056 atributos",
                   "aleatorio": "15 atributos aleatórios"},
        "model": {"logreg": "regressão logística", "tree": "AD", "forest": "FA", "mlp": "RNA"},
        "stages": ["TDF", "TS", "TW", "métricas estatísticas", "inferência"],
        "sets": {"thesis_train": "15 da monografia", "thesis_published": "15 publicados",
                 "all": "7056 atributos", "top:": "primeiros ", "random:": "aleatório nº "},
        "depth": "prof.",
        "f1": "macro-F1 no conjunto de teste",
        "instr": "instruções executadas por janela, extração e inferência (escala logarítmica)",
        "flash": "memória flash ocupada, em KB (escala logarítmica)",
        "per_window": "milhões de instruções executadas por janela",
    },
    "en": {
        "suffix": "-en", "point": ".", "sep": ",",
        "family": {"selecao": "thesis selection", "todos": "all 7056 features",
                   "aleatorio": "15 random features"},
        "model": {"logreg": "logistic regression", "tree": "DT", "forest": "RF", "mlp": "ANN"},
        "stages": ["DFT", "ST", "WT", "statistical metrics", "inference"],
        "sets": {"thesis_train": "thesis 15", "thesis_published": "published 15",
                 "all": "7056 features", "top:": "top ", "random:": "random #"},
        "depth": "depth",
        "f1": "macro-F1 on the test set",
        "instr": "instructions executed per window, extraction and inference (log scale)",
        "flash": "flash used, in KB (log scale)",
        "per_window": "million instructions executed per window",
    },
}
T = TEXT["pt"]


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def br(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", T["point"])


def br_int(n: float) -> str:
    """Integer with the language's thousands separator: 1.328.863 / 1,328,863."""
    return f"{n:,.0f}".replace(",", T["sep"])


def br_sig(x: float, sig: int = 2) -> str:
    """Significant digits with the language's decimal mark: 0,0013 / 0.0013."""
    return f"{x:.{sig}g}".replace(".", T["point"])


def recessive(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def family(fs: str) -> str:
    return "todos" if fs == "all" else "aleatorio" if fs.startswith("random") else "selecao"


def name(r) -> str:
    sets = T["sets"]
    fs = sets.get(r.feature_set, r.feature_set.replace("top:", sets["top:"])
                  .replace("random:", sets["random:"]))
    model = T["model"][r.model]
    if r.model == "forest":
        model += f" {int(r.n_trees)}×{int(r.depth)}"
    elif r.model == "tree":
        model += f" {T['depth']} {int(r.depth)}"
    elif r.model == "mlp":
        model += f" {int(r.hidden)}"
    return f"{fs}, {model}"


def legend(ax) -> None:
    fam = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=6, label=T["family"][f])
           for f, c in FAMILY.items()]
    mod = [plt.Line2D([], [], marker=m, ls="", markerfacecolor="none", markeredgecolor=INK2,
                      markersize=6, label=T["model"][k]) for k, m in MODEL.items()]
    ax.legend(handles=fam + mod, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4,
              frameon=False, labelcolor=INK2, handletextpad=0.3, columnspacing=1.0,
              fontsize=8)


def pareto(d: pd.DataFrame) -> pd.DataFrame:
    best, keep = -np.inf, []
    for i, r in d.sort_values(["x", "f1"], ascending=[True, False]).iterrows():
        if r["f1"] > best:
            keep.append(i)
            best = r["f1"]
    return d.loc[keep]


def steps(front: pd.DataFrame, min_gain: float = 0.03) -> pd.DataFrame:
    keep, last = [], -np.inf
    for i, r in front.iterrows():
        if r["f1"] - last >= min_gain:
            keep.append(i)
            last = r["f1"]
    if front.index[-1] not in keep:
        keep[-1] = front.index[-1]
    return front.loc[keep]


def separate(fig, labels, step: float = 2.0, rounds: int = 30, pad: float = 3.0) -> None:
    renderer = fig.canvas.get_renderer()
    for _ in range(rounds):
        moved = False
        boxes = [a.get_window_extent(renderer).padded(pad) for a in labels]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if boxes[i].overlaps(boxes[j]):
                    lo, hi = (i, j) if boxes[i].y0 <= boxes[j].y0 else (j, i)
                    for k, d in ((lo, -step), (hi, step)):
                        x, y = labels[k].xyann
                        labels[k].xyann = (x, y + d)
                    moved = True
        if not moved:
            return


def tradeoff(df: pd.DataFrame, x: str, xlabel: str, ticks, fmt, out: str) -> None:
    fig, ax = plt.subplots(figsize=(15 * CM, 9.5 * CM))
    d = df.assign(x=df[x])
    front = pareto(d)
    ax.step(front["x"], front["f1"], where="post", color=FRONT, lw=1.1, zorder=2)
    for (fam, model), g in d.groupby(["fam", "model"]):
        ax.scatter(g["x"], g["f1"], s=34, marker=MODEL[model], color=FAMILY[fam],
                   edgecolors="white", linewidths=1.2, zorder=3)
    ax.scatter(front["x"], front["f1"], s=95, facecolors="none", edgecolors=INK2, lw=0.8,
               zorder=4)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(T["f1"])
    recessive(ax)
    legend(ax)
    fig.tight_layout()
    labels = [ax.annotate(name(r), (r.x, r.f1), xytext=(10, 4), textcoords="offset points",
                          fontsize=7.5, color=INK, va="center",
                          arrowprops={"arrowstyle": "-", "color": FRONT, "lw": 0.6,
                                      "shrinkA": 0, "shrinkB": 5})
              for _, r in steps(front).iterrows()]
    separate(fig, labels)
    fig.savefig(OUT / f"{out}{T['suffix']}.pdf")
    plt.close(fig)


def cost_split(df: pd.DataFrame) -> None:
    picks = [("thesis_train", m) for m in ("tree", "logreg", "forest", "mlp")]
    picks += [("top:3", "tree"), ("all", "forest"), ("all", "mlp")]
    rows = [df[(df.feature_set == fs) & (df.model == m) & (df.group == "A")].iloc[0]
            for fs, m in picks]
    fig, ax = plt.subplots(figsize=(15 * CM, 7.6 * CM))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for label, (col, color) in zip(T["stages"], STAGES):
        v = np.array([r[col] for r in rows]) / 1e6
        ax.barh(y, v, left=left, height=0.62, color=color, edgecolor="white", lw=1.5, label=label)
        left += v
    for yi, r, total in zip(y, rows, left):
        share = 100 * r["instr_inf"] / (r["instr_feat"] + r["instr_inf"])
        ax.text(total + 0.8, yi,
                f"{T['stages'][-1]}: {br_int(r['instr_inf'])} ({br_sig(share)}%)",
                va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [name(r) for r in rows], fontsize=8)
    ax.set_xlim(0, left.max() * 1.4)
    ax.set_xlabel(T["per_window"])
    ax.grid(axis="y", visible=False)
    recessive(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=5, frameon=False,
              labelcolor=INK2, fontsize=8, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / f"fig-custo-etapas{T['suffix']}.pdf")
    plt.close(fig)


def main() -> None:
    global T
    style()
    df = pd.read_csv(paths.RESULTS / "sweep.csv")
    df = df[df["agreement"] >= 0.99].copy()
    df["fam"] = df["feature_set"].map(family)
    df["instr_total"] = df["instr_feat"] + df["instr_inf"]
    a = df[df.opt == "-Os"]
    for lang in TEXT:
        T = TEXT[lang]
        tradeoff(a, "instr_total", T["instr"], [2e6, 5e6, 1e7, 2e7, 5e7, 1e8],
                 lambda v: f"{v / 1e6:.0f} M", "fig-f1-instrucoes")
        tradeoff(a, "flash_kb", T["flash"], [16, 32, 64, 128, 256, 512],
                 lambda v: f"{v:.0f}", "fig-f1-flash")
        cost_split(df)
    print(f"{len(df)} configurations -> {OUT}/fig-*.pdf")


if __name__ == "__main__":
    main()
