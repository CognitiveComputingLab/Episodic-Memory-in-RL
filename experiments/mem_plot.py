"""
mem_plot.py -- draw ONE figure from results saved by mem_eval.py.

    python mem_plot.py FILE.npz [FILE.npz ...] --graphs GRAPH [GRAPH ...] [--out fig.png]

Grouping
--------
The last number in a file name is the seed. Files whose names differ only there are one
group and are averaged ('ttt_0.npz' + 'ttt_1.npz' -> group 'ttt'). Checkpoints carry a step
before the seed, so 'ttt_100_0.npz' belongs to group 'ttt_100', never to 'ttt'.
Each group is one model / setting in the figure. Wildcards work even if your shell does not
expand them ('results/ttt_*.npz').

Graphs (one row each, in the order given)
------------------------------------------
    heatmap     loss of every chunk vs. chunks written so far          (one panel per group)
    snapshots   loss vs. chunk position at a few write steps           (one panel per group,
                                                                        all on the same y-scale)
    forgetting  loss / held-out loss vs. age of the chunk              (groups overlaid)
    time        written / not-yet-written / held-out loss over time    (groups overlaid)
    training    outer loss during W_0 meta-training                    (groups overlaid)

Examples
--------
    python mem_plot.py results/ttt_*.npz results/titans_*.npz --graphs heatmap forgetting time
    python mem_plot.py results/ttt_0_0.npz results/ttt_100_0.npz results/ttt_0.npz \\
                       --graphs snapshots forgetting --labels init step-100 final
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

PER_GROUP = ("heatmap", "snapshots")              # one panel per group
OVERLAY = ("forgetting", "time", "training")      # one panel, groups overlaid
GRAPHS = PER_GROUP + OVERLAY

# config keys that may legitimately differ between files of one group
CFG_IGNORE = {"seed", "name", "out_dir", "device", "save_w0", "ckpt_every"}


# --------------------------------------------------------------------------- #
# Loading and grouping
# --------------------------------------------------------------------------- #
@dataclass
class Group:
    key: str
    label: str
    files: list[Path]
    L: np.ndarray               # [n+1, N, n]   N = files x sequences per file
    H: np.ndarray               # [n+1, N]
    minibatch: int
    histories: list[np.ndarray]

    @property
    def Lbar(self) -> np.ndarray:       # [n+1, n]   mean over files and sequences
        return self.L.mean(1)

    @property
    def Hbar(self) -> np.ndarray:       # [n+1]
        return self.H.mean(1)

    @property
    def title(self) -> str:
        return self.label + (f"  (mean of {len(self.files)} runs)" if len(self.files) > 1 else "")


def split_name(path: Path) -> tuple[str, int | None]:
    m = re.fullmatch(r"(.+)_(\d+)", path.stem)
    return (m.group(1), int(m.group(2))) if m else (path.stem, None)


def expand(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for a in args:
        hits = sorted(glob.glob(a)) if any(c in a for c in "*?[") else [a]
        if not hits:
            sys.exit(f"error: no files match '{a}'")
        out += [Path(h) for h in hits]
    seen, uniq = set(), []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def check_configs(key: str, files: list[Path]) -> None:
    """Warn if files being averaged were produced with different settings."""
    cfgs = []
    for p in files:
        j = p.with_suffix(".json")
        if j.exists():
            cfgs.append((p.name, {k: v for k, v in json.load(open(j))["cfg"].items() if k not in CFG_IGNORE}))
    if len(cfgs) < 2:
        return
    ref_name, ref = cfgs[0]
    for name, c in cfgs[1:]:
        diff = sorted(k for k in set(ref) | set(c) if ref.get(k) != c.get(k))
        if diff:
            print(f"warning: group '{key}': {name} differs from {ref_name} in {diff} -- "
                  f"averaging runs with different settings", file=sys.stderr)


def load_groups(paths: list[Path], labels: list[str] | None) -> list[Group]:
    buckets: dict[str, list[Path]] = {}
    for p in paths:
        if not p.exists():
            sys.exit(f"error: {p} does not exist")
        buckets.setdefault(split_name(p)[0], []).append(p)
    if labels is not None and len(labels) != len(buckets):
        sys.exit(f"error: {len(labels)} labels given but there are {len(buckets)} groups: {list(buckets)}")

    groups = []
    for i, (key, files) in enumerate(buckets.items()):
        data = [np.load(f) for f in files]
        for f, d in zip(files, data):
            if "L" not in d.files or d["L"].ndim != 3:
                sys.exit(f"error: {f} is not in the mem_eval.py format (expected L with shape [steps, seqs, chunks])")
        ref = data[0]
        for f, d in zip(files, data):
            if d["L"].shape[0] != ref["L"].shape[0] or d["L"].shape[2] != ref["L"].shape[2] \
                    or int(d["minibatch"]) != int(ref["minibatch"]):
                sys.exit(f"error: group '{key}': {f.name} has a different sequence length / chunk size "
                         f"than {files[0].name}; cannot average them")
        check_configs(key, files)
        groups.append(Group(
            key=key,
            label=labels[i] if labels else key,
            files=files,
            L=np.concatenate([d["L"] for d in data], axis=1),
            H=np.concatenate([d["H"] for d in data], axis=1),
            minibatch=int(ref["minibatch"]),
            histories=[d["history"] for d in data if "history" in d.files and d["history"].size],
        ))
    return groups


# --------------------------------------------------------------------------- #
# Derived quantities
# --------------------------------------------------------------------------- #
def seen_unseen(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L [S, N, n] -> mean loss of written / not-yet-written chunks, each [S, N] (NaN if empty)."""
    S, N, n = L.shape
    seen = np.full((S, N), np.nan)
    unseen = np.full((S, N), np.nan)
    for s in range(S):
        if s > 0:
            seen[s] = L[s, :, :s].mean(-1)
        if s < n:
            unseen[s] = L[s, :, s:].mean(-1)
    return seen, unseen


def forgetting_curve(Lbar: np.ndarray, Hbar: np.ndarray, mb: int):
    """Memorisation ratio L[s, c] / H[s] as a function of age.

    Dividing by the held-out loss at the same step removes the "memory has learned the
    distribution" trend: 1 = recalled no better than a never-written chunk, < 1 = memorised.
    age = 0: most recent write; > 0: older; < 0: not written yet.
    """
    R = Lbar / Hbar[:, None]
    S, n = R.shape
    age = np.arange(S)[:, None] - (np.arange(n)[None, :] + 1)
    ages = np.arange(-n, n)
    mean, lo, hi = [], [], []
    for a in ages:
        vals = R[age == a]
        mean.append(vals.mean())
        lo.append(np.percentile(vals, 10))
        hi.append(np.percentile(vals, 90))
    return ages * mb, np.array(mean), np.array(lo), np.array(hi)


def snapshot_rows(n: int) -> list[int]:
    return sorted({0, n // 4, n // 2, 3 * n // 4, n})


def finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a).ravel()
    return a[np.isfinite(a) & (a > 0)]


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def draw_heatmap(ax, g: Group, norm):
    L = g.Lbar
    S, n = L.shape
    im = ax.imshow(L, aspect="auto", norm=norm, cmap="magma", interpolation="nearest")
    ax.plot(np.arange(1, n + 1) - 0.5, np.arange(1, n + 1), color="w", lw=0.9, ls="--")  # write frontier
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(S - 0.5, -0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"{g.title}\nloss per chunk (log scale)", fontsize=10)
    ax.set_xlabel("chunk position in sequence")
    ax.set_ylabel("chunks written so far")
    return im


def draw_snapshots(ax, g: Group, ylim):
    L = g.Lbar
    n = L.shape[1]
    picks = snapshot_rows(n)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(picks)))
    for s, c in zip(picks, cols):
        ax.plot(np.arange(n), L[s], color=c, label=f"{s} chunks written")
        if 0 < s < n:
            ax.axvline(s - 0.5, color=c, lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)                       # identical for every group
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"{g.title}\nsnapshots (dotted = write frontier)", fontsize=10)
    ax.set_xlabel("chunk position in sequence")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=7, loc="best", framealpha=0.8)
    ax.grid(alpha=0.3)


def group_handles(groups, colors):
    return [Line2D([], [], color=colors[i], lw=2, label=g.label) for i, g in enumerate(groups)]


def draw_forgetting(ax, groups, colors):
    for i, g in enumerate(groups):
        ages, m, lo, hi = forgetting_curve(g.Lbar, g.Hbar, g.minibatch)
        ax.plot(ages, m, color=colors[i])
        ax.fill_between(ages, lo, hi, color=colors[i], alpha=0.15)
    ax.axvline(-0.5 * groups[0].minibatch, color="k", lw=0.8, ls=":")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("age in tokens   (0 = just written, < 0 = not written yet)")
    ax.set_ylabel("MSE / held-out MSE   (1 = no trace)")
    ax.set_title("Forgetting curve (mean; band = 10-90% over write steps)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(handles=group_handles(groups, colors), fontsize=8, framealpha=0.8)


def draw_time(ax, groups, colors, bands: bool):
    for i, g in enumerate(groups):
        seen, unseen = seen_unseen(g.L)
        N = g.L.shape[1]
        for arr, ls in ((seen, "-"), (unseen, "--"), (g.H, ":")):
            m = arr.mean(1)
            ax.plot(m, color=colors[i], ls=ls)
            if bands:
                sem = arr.std(1) / np.sqrt(N)
                ax.fill_between(np.arange(len(m)), m - sem, m + sem, color=colors[i], alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("chunks written")
    ax.set_ylabel("MSE")
    ax.set_title("Written vs not-yet-written vs held-out" + (" (band = s.e.m. over sequences)" if bands else ""),
                 fontsize=10)
    ax.grid(alpha=0.3)
    kinds = [Line2D([], [], color="gray", ls="-", label="written chunks"),
             Line2D([], [], color="gray", ls="--", label="not-yet-written chunks"),
             Line2D([], [], color="gray", ls=":", label="held-out sequence")]
    leg = ax.legend(handles=group_handles(groups, colors), fontsize=8, loc="upper left", framealpha=0.8)
    ax.add_artist(leg)
    ax.legend(handles=kinds, fontsize=8, loc="lower right", framealpha=0.8)


def draw_training(ax, groups, colors):
    for i, g in enumerate(groups):
        if not g.histories:
            continue
        m = min(len(h) for h in g.histories)
        H = np.stack([h[:m] for h in g.histories])
        for h in H:
            ax.plot(h, color=colors[i], alpha=0.2, lw=0.7)
        mean = H.mean(0)
        k = max(1, m // 20)
        smooth = np.convolve(mean, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(smooth)) + (k - 1) / 2, smooth, color=colors[i], lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("outer iteration (new x each time)")
    ax.set_ylabel("outer loss")
    ax.set_title("W_0 meta-training (faint = single runs, bold = smoothed mean)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(handles=[h for h, g in zip(group_handles(groups, colors), groups) if g.histories],
              fontsize=8, framealpha=0.8)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(groups: list[Group], graphs: list[str], bands: bool, title: str | None):
    R, nrows = len(groups), len(graphs)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(R)]

    for g in groups:
        if not np.isfinite(g.L).all():
            print(f"warning: group '{g.key}' contains non-finite losses (that run diverged); "
                  f"plots ignore them", file=sys.stderr)
    if "training" in graphs and not any(g.histories for g in groups):
        sys.exit("error: 'training' requested but none of the files contains a training history "
                 "(they were produced without --train-w0)")

    # shared scales across groups
    allL = finite(np.concatenate([g.Lbar.ravel() for g in groups]))
    if allL.size == 0:
        sys.exit("error: no finite positive losses to plot")
    norm = LogNorm(vmin=allL.min(), vmax=allL.max())
    snap_vals = finite(np.concatenate([g.Lbar[snapshot_rows(g.Lbar.shape[1])].ravel() for g in groups]))
    snap_ylim = (snap_vals.min() / 1.15, snap_vals.max() * 1.15)

    fig = plt.figure(figsize=(max(7.5, 4.2 * R), 3.9 * nrows + 0.4), layout="constrained")
    gs = fig.add_gridspec(nrows, R, height_ratios=[1.15 if g == "heatmap" else 1.0 for g in graphs])
    for row, graph in enumerate(graphs):
        if graph in PER_GROUP:
            axes = [fig.add_subplot(gs[row, c]) for c in range(R)]
            if graph == "heatmap":
                ims = [draw_heatmap(ax, g, norm) for ax, g in zip(axes, groups)]
                fig.colorbar(ims[0], ax=axes, label="MSE", shrink=0.9)
            else:
                for ax, g in zip(axes, groups):
                    draw_snapshots(ax, g, snap_ylim)
        else:
            ax = fig.add_subplot(gs[row, :])
            if graph == "forgetting":
                draw_forgetting(ax, groups, colors)
            elif graph == "time":
                draw_time(ax, groups, colors, bands)
            else:
                draw_training(ax, groups, colors)
    if title:
        fig.suptitle(title)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="npz files written by mem_eval.py")
    ap.add_argument("--graphs", nargs="+", required=True, choices=GRAPHS, metavar="GRAPH",
                    help=f"one or more of: {', '.join(GRAPHS)} (one row each, in this order)")
    ap.add_argument("--out", default="mem_plot.png", help="output image (default: %(default)s)")
    ap.add_argument("--labels", nargs="+", help="legend names, one per group, in order of first appearance")
    ap.add_argument("--title", help="figure title")
    ap.add_argument("--bands", action="store_true", help="'time' graph: shade +-1 s.e.m. over sequences")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)

    graphs = list(dict.fromkeys(args.graphs))                    # drop duplicates, keep order
    groups = load_groups(expand(args.files), args.labels)
    for g in groups:
        print(f"group '{g.label}': {len(g.files)} file(s) "
              f"[{', '.join(f.stem for f in g.files)}], {g.L.shape[1]} sequences")
    fig = make_figure(groups, graphs, args.bands, args.title)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
