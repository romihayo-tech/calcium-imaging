import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


def write_notebook(path, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))


comparison_cells = [
    md("""
# Thesis comparison results — clean reproducible notebook

This notebook reproduces the comparison outputs for the WT / HET / MUT calcium-imaging thesis analysis from the pooled CSV outputs.

Run order:
1. Edit `DATA_ROOT` if needed.
2. Run all cells.
3. Figures and summary tables are saved to `comparison_clean_outputs/`.

Expected input folders:
- `pooling_WT/pooled_results`
- `pooling_HET/pooled_results`
- `pooling_MUT/pooled_results`

This notebook does not depend on hidden variables from older notebooks.
"""),
    code("""
from pathlib import Path
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu, gaussian_kde
except Exception as exc:
    raise ImportError("This notebook needs scipy. Install scipy or run in the original analysis environment.") from exc

%config InlineBackend.figure_format = "retina"
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ------------------------------
# CONFIG
# ------------------------------
import os
import getpass
_default_user = os.environ.get("JUPYTERHUB_USER") or os.environ.get("USER") or getpass.getuser()
DATA_ROOT = Path(f"/data/{_default_user}") if os.path.exists(f"/data/{_default_user}") else Path("/data")

GENOTYPE_DIRS = {
    "WT":  DATA_ROOT / "pooling_WT"  / "pooled_results",
    "HET": DATA_ROOT / "pooling_HET" / "pooled_results",
    "MUT": DATA_ROOT / "pooling_MUT" / "pooled_results",
}

CLUSTER_FILES = {
    "WT":  "WT_gmm_clusters_global.csv",
    "HET": "HET_cluster_assignments.csv",
    "MUT": "MUT_cluster_assignments.csv",
}

ROI_PHASE_FILES = {
    "WT":  "WT_roi_metrics_per_phase.csv",
    "HET": "HET_roi_metrics_per_phase.csv",
    "MUT": "MUT_roi_metrics_per_phase.csv",
}

EVENT_FILES = {
    "WT":  "WT_all_Events_acinar_clean_yfiltered.csv",
    "HET": "HET_all_Events_acinar_clean_yfiltered.csv",
    "MUT": "MUT_all_Events_acinar_clean_yfiltered.csv",
}

MUT_EXCLUDE = ["278a_F", "267a_M"]
GENOTYPE_ORDER = ["WT", "HET", "MUT"]
CLUSTER_ORDER = ["Phasic", "Sustained"]
SCOPE_ORDER = ["lif", "nd2"]

PHASIC_COLOR = "#2196F3"
SUSTAINED_COLOR = "#FF5722"
GENO_COLORS = {"WT": "#1b5e20", "HET": "#e65100", "MUT": "#b71c1c"}
CT_COLORS = {"Phasic": PHASIC_COLOR, "Sustained": SUSTAINED_COLOR}

OUTPUT_DIR = DATA_ROOT / "comparison_clean_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("DATA_ROOT:", DATA_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)
"""),
    md("## 1. Load and harmonize cluster assignments"),
    code("""
def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def add_or_harmonize_cell_type(df):
    df = df.copy()
    if "cell_type" not in df.columns:
        if not {"cluster", "halfwidth_ACh"}.issubset(df.columns):
            raise ValueError("Need either cell_type or both cluster + halfwidth_ACh.")
        hw = df.groupby("cluster")["halfwidth_ACh"].median()
        df["cell_type"] = df["cluster"].map({
            hw.idxmin(): "Phasic",
            hw.idxmax(): "Sustained",
        })
    return df


def add_scope_if_missing(df, genotype):
    df = df.copy()
    if "scope" in df.columns:
        return df
    rp_path = GENOTYPE_DIRS[genotype] / ROI_PHASE_FILES[genotype]
    if rp_path.exists():
        rp = pd.read_csv(rp_path)
        if "scope" in rp.columns:
            scope_map = rp.drop_duplicates("exp_name").set_index("exp_name")["scope"].to_dict()
            df["scope"] = df["exp_name"].map(scope_map)
            return df
    df["scope"] = "unknown"
    return df


def load_cluster_table(genotype):
    path = require_file(GENOTYPE_DIRS[genotype] / CLUSTER_FILES[genotype])
    df = pd.read_csv(path)
    if genotype == "MUT":
        df = df[~df["exp_name"].isin(MUT_EXCLUDE)].copy()
    df = add_or_harmonize_cell_type(df)
    df = add_scope_if_missing(df, genotype)
    df["genotype"] = genotype
    return df


tables = {geno: load_cluster_table(geno) for geno in GENOTYPE_ORDER}
all_data = pd.concat([tables[g] for g in GENOTYPE_ORDER], ignore_index=True)

for geno in GENOTYPE_ORDER:
    df = tables[geno]
    scopes = df["scope"].value_counts(dropna=False).to_dict()
    print(f"{geno:4s}: {len(df):,} ROIs | {df['exp_name'].nunique()} slices | scopes: {scopes}")

print("\\nColumns:")
print(all_data.columns.tolist())
"""),
    md("## 2. Core summary tables"),
    code("""
FEATURES = [
    "halfwidth_ACh",
    "event_rate_ACh",
    "latency_ACh_s",
    "cv_dt_ACh",
    "event_rate_8mM",
]
FEATURES = [f for f in FEATURES if f in all_data.columns]

population = (
    all_data.groupby(["genotype", "cell_type"])
    .size()
    .rename("n_rois")
    .reset_index()
)
population["pct"] = population.groupby("genotype")["n_rois"].transform(lambda s: s / s.sum() * 100)
population_pivot = population.pivot(index="genotype", columns="cell_type", values="pct").reindex(GENOTYPE_ORDER)

counts = (
    all_data.groupby(["cell_type", "genotype"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=CLUSTER_ORDER, columns=GENOTYPE_ORDER)
)

medians = (
    all_data.groupby(["cell_type", "genotype"])[FEATURES]
    .median()
    .round(3)
    .reindex(pd.MultiIndex.from_product([CLUSTER_ORDER, GENOTYPE_ORDER], names=["cell_type", "genotype"]))
)

scope_medians = (
    all_data.groupby(["scope", "cell_type", "genotype"])[FEATURES]
    .median()
    .round(3)
)

population.to_csv(OUTPUT_DIR / "population_proportions.csv", index=False)
counts.to_csv(OUTPUT_DIR / "roi_counts_by_genotype_cluster.csv")
medians.to_csv(OUTPUT_DIR / "feature_medians_by_genotype_cluster.csv")
scope_medians.to_csv(OUTPUT_DIR / "scope_stratified_feature_medians.csv")

print("Population proportions (%):")
display(population_pivot.round(1))
print("\\nROI counts:")
display(counts)
print("\\nFeature medians:")
display(medians)
print("\\nSaved summary CSVs to:", OUTPUT_DIR)
"""),
    md("## 3. Figure — population structure"),
    code("""
def per_slice_pct_phasic(df):
    g = df.groupby(["exp_name", "cell_type"]).size().unstack(fill_value=0)
    if "Phasic" not in g.columns:
        g["Phasic"] = 0
    return (100 * g["Phasic"] / g.sum(axis=1)).values


phasic_vals = []
sustained_vals = []
slice_pcts = []
n_info_labels = []

for geno in GENOTYPE_ORDER:
    sub = all_data[all_data["genotype"] == geno]
    counts_norm = sub["cell_type"].value_counts(normalize=True) * 100
    phasic_vals.append(counts_norm.get("Phasic", 0))
    sustained_vals.append(counts_norm.get("Sustained", 0))
    slice_pcts.append(per_slice_pct_phasic(sub))
    n_info_labels.append(f"n = {sub['exp_name'].nunique()} slices\\n({len(sub):,} ROIs)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})

ax = axes[0]
ax.bar(GENOTYPE_ORDER, phasic_vals, color=PHASIC_COLOR, label="Phasic", edgecolor="white")
ax.bar(GENOTYPE_ORDER, sustained_vals, bottom=phasic_vals, color=SUSTAINED_COLOR, label="Sustained", edgecolor="white")
for i, (p, s) in enumerate(zip(phasic_vals, sustained_vals)):
    ax.text(i, p / 2, f"{p:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.text(i, p + s / 2, f"{s:.1f}%", ha="center", va="center", color="white", fontweight="bold")
for i, txt in enumerate(n_info_labels):
    ax.text(i, -8, txt, ha="center", va="top", fontsize=8, color="dimgray")
ax.set_ylabel("% of ROIs")
ax.set_ylim(0, 100)
ax.set_title("Overall population structure")
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

ax = axes[1]
positions = np.arange(len(GENOTYPE_ORDER))
bp = ax.boxplot(slice_pcts, positions=positions, widths=0.45, showfliers=False, patch_artist=True, medianprops=dict(color="black", linewidth=2))
for patch in bp["boxes"]:
    patch.set_facecolor("#cfd8dc")
    patch.set_alpha(0.5)

rng = np.random.default_rng(0)
for pos, vals in zip(positions, slice_pcts):
    jitter = rng.uniform(-0.1, 0.1, size=len(vals))
    ax.scatter(np.full(len(vals), pos) + jitter, vals, color=PHASIC_COLOR, edgecolor="black", s=40, zorder=3, alpha=0.8)
    if len(vals) > 1:
        m, sd = np.mean(vals), np.std(vals, ddof=1)
        ax.errorbar(pos + 0.30, m, yerr=sd, fmt="o", color="black", markersize=5, capsize=4, zorder=4)
        ax.text(pos + 0.38, m, f"  {m:.1f}±{sd:.1f}%", va="center", fontsize=8)

ax.set_xticks(positions)
ax.set_xticklabels([f"{g}\\n(n={len(s)} slices)" for g, s in zip(GENOTYPE_ORDER, slice_pcts)])
ax.set_ylabel("% Phasic per slice")
ax.set_ylim(0, 100)
ax.set_title("Per-slice variability")

fig.suptitle("Population structure — WT vs HET vs MUT", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "WT_HET_MUT_population_structure.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("## 4. Figure — halfwidth comparison"),
    code("""
def plot_violin_trio(ax, data, title):
    vals_by_geno = [data.loc[data["genotype"] == geno, "halfwidth_ACh"].dropna().values for geno in GENOTYPE_ORDER]
    vals_by_geno = [v[v > 0] for v in vals_by_geno]
    colors = [GENO_COLORS[g] for g in GENOTYPE_ORDER]

    wt_median = np.median(vals_by_geno[0])
    ax.axhline(np.log10(wt_median), color=GENO_COLORS["WT"], linestyle="--", linewidth=1, alpha=0.4, zorder=1)

    parts = ax.violinplot([np.log10(v) for v in vals_by_geno], positions=[0, 1, 2], widths=0.70, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.50)
        body.set_edgecolor("black")
        body.set_linewidth(1)

    for pos, vals in zip([0, 1, 2], vals_by_geno):
        med = np.median(vals)
        ax.hlines(np.log10(med), pos - 0.28, pos + 0.28, colors="black", linewidth=2.5, zorder=5)
        ax.text(pos + 0.40, np.log10(med), f"{med:.2f}s", ha="left", va="center", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

    ax.set_xlim(-0.6, 2.8)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["0.1", "1", "10"])
    ax.set_ylabel("halfwidth (s, log scale)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f"{g}\\n(n={len(v):,})" for g, v in zip(GENOTYPE_ORDER, vals_by_geno)], fontsize=9)
    ax.set_title(title, fontweight="bold")


both = all_data[all_data["halfwidth_ACh"] > 0].copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
plot_violin_trio(axes[0], both[both["cell_type"] == "Phasic"], "Phasic cluster")
plot_violin_trio(axes[1], both[both["cell_type"] == "Sustained"], "Sustained cluster")
fig.suptitle("Event halfwidth at ACh — WT vs HET vs MUT", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "WT_HET_MUT_halfwidth_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("## 5. Scope-stratified confirmation and gene-dose figure"),
    code("""
print("Scope-stratified median halfwidth_ACh")
for scope in SCOPE_ORDER:
    print(f"\\n--- {scope} only ---")
    sub = all_data[all_data["scope"] == scope]
    display(sub.groupby(["cell_type", "genotype"])["halfwidth_ACh"].agg(["median", "count"]).round(3))

gene_dose_rows = []
for scope_label, sub in [("pooled", all_data)] + [(s, all_data[all_data["scope"] == s]) for s in SCOPE_ORDER]:
    sus = sub[sub["cell_type"] == "Sustained"]
    row = {"scope": scope_label}
    for geno in GENOTYPE_ORDER:
        row[geno] = sus.loc[sus["genotype"] == geno, "halfwidth_ACh"].median()
    row["HET_delta_pct_from_WT"] = 100 * (row["HET"] - row["WT"]) / row["WT"]
    row["MUT_delta_pct_from_WT"] = 100 * (row["MUT"] - row["WT"]) / row["WT"]
    gene_dose_rows.append(row)

gene_dose = pd.DataFrame(gene_dose_rows)
gene_dose.to_csv(OUTPUT_DIR / "gene_dose_sustained_halfwidth.csv", index=False)
display(gene_dose.round(3))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, scope_label in zip(axes, ["pooled", "lif", "nd2"]):
    sub = all_data if scope_label == "pooled" else all_data[all_data["scope"] == scope_label]
    sus = sub[(sub["cell_type"] == "Sustained") & (sub["halfwidth_ACh"] > 0)]
    medians = [sus.loc[sus["genotype"] == geno, "halfwidth_ACh"].median() for geno in GENOTYPE_ORDER]
    bars = ax.bar(GENOTYPE_ORDER, medians, color=[GENO_COLORS[g] for g in GENOTYPE_ORDER], edgecolor="black", alpha=0.7)
    for i, med in enumerate(medians):
        n = len(sus[sus["genotype"] == GENOTYPE_ORDER[i]])
        label = f"{med:.2f}s" if scope_label == "pooled" else f"{med:.2f}s\\n(n={n:,})"
        ax.text(i, med + 0.15, label, ha="center", fontsize=9, fontweight="bold" if scope_label == "pooled" else None)
    ax.set_title("Pooled" if scope_label == "pooled" else f"{scope_label} only")
    ax.set_ylim(0, max(medians) * 1.35)
    if ax is axes[0]:
        ax.set_ylabel("Median halfwidth (s)")

fig.suptitle("Sustained halfwidth — gene-dose effect", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "WT_HET_MUT_gene_dose_effect.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("## 6. Scope-stratified feature comparison"),
    code("""
features_config = [
    ("halfwidth_ACh",  "Halfwidth (s)", True),
    ("event_rate_ACh", "Event rate (events/min)", False),
    ("latency_ACh_s",  "Latency (s)", False),
    ("cv_dt_ACh",      "CV of inter-event interval", False),
]
features_config = [x for x in features_config if x[0] in all_data.columns]

fig, axes = plt.subplots(len(features_config), 2, figsize=(13, 4 * len(features_config)), sharey="row")
if len(features_config) == 1:
    axes = np.array([axes])

for row, (feat, ylabel, use_log) in enumerate(features_config):
    for col, ct in enumerate(CLUSTER_ORDER):
        ax = axes[row, col]
        x_positions = np.array([0, 1, 2])
        bar_width = 0.35
        for scope, offset, hatch in [("lif", -bar_width/2, ""), ("nd2", bar_width/2, "///")]:
            medians = []
            counts = []
            for geno in GENOTYPE_ORDER:
                sub = all_data[(all_data["genotype"] == geno) & (all_data["cell_type"] == ct) & (all_data["scope"] == scope)]
                medians.append(sub[feat].median() if len(sub) else np.nan)
                counts.append(len(sub))
            bars = ax.bar(x_positions + offset, medians, bar_width, color=[GENO_COLORS[g] for g in GENOTYPE_ORDER],
                          edgecolor="black", linewidth=0.8, alpha=0.85 if hatch == "" else 0.5, hatch=hatch,
                          label=scope if col == 0 and row == 0 else None)
            for x, med in zip(x_positions + offset, medians):
                if np.isfinite(med) and med > 0:
                    ax.text(x, med * 1.05, f"{med:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(GENOTYPE_ORDER)
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(ct, fontweight="bold")
        if use_log:
            ax.set_yscale("log")

axes[0, 1].legend(fontsize=9, loc="upper right")
fig.suptitle("Feature comparison — solid = lif, hatched = nd2", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "WT_HET_MUT_scope_stratified_features.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("## 7. Dose-response and bimodality from ROI-phase tables"),
    code("""
def load_roi_phase(genotype):
    path = GENOTYPE_DIRS[genotype] / ROI_PHASE_FILES[genotype]
    if not path.exists():
        warnings.warn(f"Skipping {genotype}: missing {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if genotype == "MUT":
        df = df[~df["exp_name"].isin(MUT_EXCLUDE)].copy()
    df["genotype"] = genotype
    return df


def halfwidth_col(df):
    for col in ["halfwidth_phase", "mean_halfwidth", "halfwidth_ACh", "median_halfwidth"]:
        if col in df.columns:
            return col
    raise ValueError("No halfwidth column found in roi_phase table.")


rp_all = pd.concat([load_roi_phase(g) for g in GENOTYPE_ORDER], ignore_index=True)
if rp_all.empty:
    print("No roi_phase files found. Skipping dose-response and bimodality.")
else:
    cluster_key = all_data[["exp_name", "roi", "cell_type", "genotype"]].drop_duplicates()
    rp_all = rp_all.merge(cluster_key, on=["exp_name", "roi", "genotype"], how="inner")
    ach_phases = [p for p in ["1nM ACh", "10nM ACh", "100nM ACh"] if p in set(rp_all["phase"])]
    dose_table = (
        rp_all[rp_all["phase"].isin(ach_phases)]
        .groupby(["phase", "genotype", "cell_type"])["event_rate"]
        .median()
        .reset_index()
    )
    dose_table.to_csv(OUTPUT_DIR / "dose_response_event_rate_medians.csv", index=False)
    display(dose_table.pivot_table(index=["phase", "cell_type"], columns="genotype", values="event_rate").round(3))

    from scipy.signal import find_peaks

    hw_col = halfwidth_col(rp_all)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    bimodality_rows = []

    for col_i, geno in enumerate(GENOTYPE_ORDER):
        for row_i, phase in enumerate(["10nM ACh", "100nM ACh"]):
            ax = axes[row_i][col_i]
            sub = rp_all[
                (rp_all["genotype"] == geno) &
                (rp_all["phase"] == phase) &
                (rp_all[hw_col] > 0)
            ][hw_col].dropna()

            if len(sub) < 50:
                ax.text(0.5, 0.5, f"n={len(sub)}\\n(insufficient)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="gray")
                ax.set_title(f"{geno} — {phase}")
                continue

            log_hw = np.log10(sub)
            x = np.linspace(-1.5, 2.0, 400)
            kde = gaussian_kde(log_hw, bw_method=0.2)
            kde_vals = kde(x)
            peaks, _ = find_peaks(kde_vals, distance=30, prominence=0.1)

            color = GENO_COLORS[geno]
            ax.plot(x, kde_vals, color=color, linewidth=2.5)
            ax.fill_between(x, kde_vals, alpha=0.15, color=color)
            ax.scatter(x[peaks], kde_vals[peaks], s=35, color="black", zorder=5)
            for peak in peaks:
                ax.text(x[peak], kde_vals[peak], f"{10**x[peak]:.2f}s",
                        ha="center", va="bottom", fontsize=8)

            ax.axvline(np.log10(0.185), color=PHASIC_COLOR, linestyle="--",
                       alpha=0.7, linewidth=1.5, label="WT Phasic (0.19s)")
            ax.axvline(np.log10(4.23),  color=SUSTAINED_COLOR, linestyle="--",
                       alpha=0.7, linewidth=1.5, label="WT Sustained (4.23s)")
            ax.set_title(f"{geno} — {phase}\\n"
                         f"n={len(sub):,} ROIs | {len(peaks)} peak(s) detected",
                         fontsize=10, fontweight="bold",
                         color="darkgreen" if len(peaks) >= 2 else "darkred")
            ax.set_xlabel("log10(halfwidth, s)")
            ax.set_ylabel("Density")
            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels(["0.1s", "1s", "10s"])

            bimodality_rows.append({
                "genotype": geno,
                "phase": phase,
                "n_rois": len(sub),
                "n_peaks": len(peaks),
                "peak_halfwidth_s": ", ".join(f"{10**x[p]:.3f}" for p in peaks),
                "pct_lt_0_5s": (sub < 0.5).mean() * 100,
                "pct_0_5_to_1s": ((sub >= 0.5) & (sub <= 1.0)).mean() * 100,
                "pct_gt_1s": (sub > 1.0).mean() * 100,
            })

    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Bimodality check: halfwidth KDE at 10nM vs 100nM ACh\\n"
                 "2 peaks at 10nM → dichotomy is concentration-independent",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "WT_HET_MUT_bimodality_10nM_vs_100nM.png", dpi=300, bbox_inches="tight")
    plt.show()

    bimodality_summary = pd.DataFrame(bimodality_rows)
    bimodality_summary.to_csv(OUTPUT_DIR / "bimodality_summary.csv", index=False)
    display(bimodality_summary.round(2))
"""),
    md("## 8. WT vs MUT effect sizes and WT-like ranges"),
    code("""
def cliffs_delta(x, y):
    x = pd.Series(x).dropna().to_numpy()
    y = pd.Series(y).dropna().to_numpy()
    if len(x) == 0 or len(y) == 0:
        return np.nan
    all_vals = np.concatenate([x, y])
    ranks = pd.Series(all_vals).rank(method="average").to_numpy()
    rx = ranks[:len(x)].sum()
    nx, ny = len(x), len(y)
    u = rx - nx * (nx + 1) / 2
    return (2 * u) / (nx * ny) - 1


def effect_label(delta):
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


effect_features = [f for f in ["halfwidth_ACh", "event_rate_ACh", "latency_ACh_s", "cv_dt_ACh"] if f in all_data.columns]
rows = []
for ct in CLUSTER_ORDER:
    for feat in effect_features:
        wt_vals = all_data[(all_data["genotype"] == "WT") & (all_data["cell_type"] == ct)][feat].dropna()
        mut_vals = all_data[(all_data["genotype"] == "MUT") & (all_data["cell_type"] == ct)][feat].dropna()
        if len(wt_vals) < 10 or len(mut_vals) < 10:
            continue
        p = mannwhitneyu(wt_vals, mut_vals, alternative="two-sided").pvalue
        d = cliffs_delta(wt_vals, mut_vals)
        wt_med = wt_vals.median()
        mut_med = mut_vals.median()
        rows.append({
            "cell_type": ct,
            "feature": feat,
            "WT_median": wt_med,
            "MUT_median": mut_med,
            "delta_pct_MUT_from_WT": 100 * (mut_med - wt_med) / wt_med if wt_med != 0 else np.nan,
            "cliffs_delta_WT_vs_MUT": d,
            "effect_size": effect_label(d),
            "p_value": p,
        })

effect_df = pd.DataFrame(rows)
effect_df.to_csv(OUTPUT_DIR / "WT_vs_MUT_effect_sizes.csv", index=False)
display(effect_df.round(4))

fig, ax = plt.subplots(figsize=(8, 5))
plot_effect = effect_df[effect_df["feature"].isin(["halfwidth_ACh", "event_rate_ACh"])].copy()
labels = plot_effect["cell_type"] + "\\n" + plot_effect["feature"].str.replace("_ACh", "", regex=False)
bars = ax.bar(labels, plot_effect["cliffs_delta_WT_vs_MUT"], color=[CT_COLORS[c] for c in plot_effect["cell_type"]], edgecolor="black", alpha=0.8)
for y in [0, 0.147, -0.147, 0.474, -0.474]:
    ax.axhline(y, color="black" if y == 0 else "gray", linestyle="-" if y == 0 else ":", linewidth=1)
for bar, val in zip(bars, plot_effect["cliffs_delta_WT_vs_MUT"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + (0.03 if val >= 0 else -0.05), f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontweight="bold")
ax.set_ylabel("Cliff's δ: WT vs MUT")
ax.set_title("Effect size by cluster and feature", fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "F1_effect_size_WT_vs_MUT.png", dpi=300, bbox_inches="tight")
plt.show()

WT_PHASIC_PEAK = 0.19
WT_SUSTAINED_PEAK = 4.23
f2_rows = []
for geno in ["WT", "MUT"]:
    ph = all_data[(all_data["genotype"] == geno) & (all_data["cell_type"] == "Phasic")]["halfwidth_ACh"].dropna()
    sus = all_data[(all_data["genotype"] == geno) & (all_data["cell_type"] == "Sustained")]["halfwidth_ACh"].dropna()
    f2_rows += [
        {"genotype": geno, "cluster": "Phasic", "category": f"< {WT_PHASIC_PEAK}s WT-like fast", "pct": (ph < WT_PHASIC_PEAK).mean() * 100},
        {"genotype": geno, "cluster": "Phasic", "category": f">= {WT_PHASIC_PEAK}s right-shifted", "pct": (ph >= WT_PHASIC_PEAK).mean() * 100},
        {"genotype": geno, "cluster": "Sustained", "category": f">= {WT_SUSTAINED_PEAK}s WT-like long", "pct": (sus >= WT_SUSTAINED_PEAK).mean() * 100},
        {"genotype": geno, "cluster": "Sustained", "category": f"< {WT_SUSTAINED_PEAK}s shortened", "pct": (sus < WT_SUSTAINED_PEAK).mean() * 100},
    ]
f2_df = pd.DataFrame(f2_rows)
f2_df.to_csv(OUTPUT_DIR / "WT_like_range_summary.csv", index=False)
display(f2_df.round(2))
"""),
    md("## 9. Mixed ROI analysis from event-level data"),
    code("""
def load_events(genotype):
    path = GENOTYPE_DIRS[genotype] / EVENT_FILES[genotype]
    if not path.exists():
        warnings.warn(f"Skipping {genotype}: missing {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if genotype == "MUT":
        df = df[~df["exp_name"].isin(MUT_EXCLUDE)].copy()
    df["genotype"] = genotype
    return df


def get_ach_starts_from_roi_phase(rp):
    if "phase_start" in rp.columns:
        starts = rp[rp["phase"].eq("100nM ACh")].drop_duplicates("exp_name").set_index("exp_name")["phase_start"].to_dict()
        if starts:
            return starts
    return {}


all_ev = pd.concat([load_events(g) for g in GENOTYPE_ORDER], ignore_index=True)
ach_starts = get_ach_starts_from_roi_phase(rp_all) if "rp_all" in globals() and not rp_all.empty else {}

if all_ev.empty:
    print("No event files found. Skipping mixed ROI analysis.")
elif not ach_starts:
    print("Could not infer 100nM ACh starts from roi_phase. If needed, add phase_start to roi_phase or run the raw pipeline notebook.")
else:
    all_ev = all_ev[all_ev["exp_name"].isin(ach_starts)].copy()
    all_ev["ach_start"] = all_ev["exp_name"].map(ach_starts)
    all_ev["t_rel"] = all_ev["peakpoint"] - all_ev["ach_start"]
    ach_ev = all_ev[(all_ev["t_rel"] >= 200) & (all_ev["t_rel"] < 300)].copy()
    cluster_key = all_data[["exp_name", "roi", "cell_type", "genotype"]].drop_duplicates()
    ach_ev = ach_ev.merge(cluster_key, on=["exp_name", "roi", "genotype"], how="inner")
    ach_ev["ev_type"] = np.where(ach_ev["halfwidth"] < 1.0, "phasic_like", "sustained_like")

    roi_counts = (
        ach_ev.groupby(["genotype", "exp_name", "roi", "cell_type", "ev_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["phasic_like", "sustained_like"]:
        if col not in roi_counts.columns:
            roi_counts[col] = 0
    roi_counts["total_events"] = roi_counts["phasic_like"] + roi_counts["sustained_like"]
    roi_counts["roi_category"] = np.select(
        [roi_counts["sustained_like"].eq(0), roi_counts["phasic_like"].eq(0)],
        ["Pure Phasic", "Pure Sustained"],
        default="Mixed",
    )
    mixed_summary = (
        roi_counts.groupby(["genotype", "cell_type", "roi_category"])
        .size()
        .rename("n_rois")
        .reset_index()
    )
    mixed_summary["pct"] = mixed_summary.groupby(["genotype", "cell_type"])["n_rois"].transform(lambda s: s / s.sum() * 100)
    mixed_summary.to_csv(OUTPUT_DIR / "mixed_roi_summary.csv", index=False)
    display(mixed_summary)
"""),
    md("## 10. Spatial cluster maps"),
    code("""
def load_coords(genotype):
    path = GENOTYPE_DIRS[genotype] / f"{genotype}_roi_coordinates.csv"
    if not path.exists():
        warnings.warn(f"Skipping {genotype}: missing {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["genotype"] = genotype
    return df


coords_all = pd.concat([load_coords(g) for g in GENOTYPE_ORDER], ignore_index=True)
if coords_all.empty:
    print("No coordinate files found. Skipping spatial maps.")
else:
    spatial = all_data[["exp_name", "roi", "cell_type", "max_prob", "genotype"]].merge(
        coords_all[["exp_name", "roi", "x", "y"]],
        on=["exp_name", "roi"], how="inner"
    )
    if spatial.empty:
        print("No overlapping clustered ROIs and coordinates.")
    else:
        for geno in GENOTYPE_ORDER:
            sub_spatial = spatial[spatial["genotype"] == geno].copy()
            if sub_spatial.empty:
                continue
            experiments = sorted(sub_spatial["exp_name"].unique())
            ncols = 4
            nrows = math.ceil(len(experiments) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
            axes = np.atleast_1d(axes).flat

            for ax, exp_name in zip(axes, experiments):
                sub = sub_spatial[sub_spatial["exp_name"] == exp_name]
                for cell_type, color in [("Sustained", SUSTAINED_COLOR), ("Phasic", PHASIC_COLOR)]:
                    s = sub[sub["cell_type"] == cell_type]
                    ax.scatter(s["x"], s["y"], s=12, alpha=0.7, color=color,
                               label=cell_type if exp_name == experiments[0] else "")
                lc = sub[sub["max_prob"] < 0.7] if "max_prob" in sub.columns else pd.DataFrame()
                if len(lc):
                    ax.scatter(lc["x"], lc["y"], s=12, color="lightgray", alpha=0.5,
                               label="Low confidence" if exp_name == experiments[0] else "")
                ax.set_title(exp_name, fontsize=8)
                ax.set_aspect("equal")
                ax.axis("off")

            for ax in list(axes)[len(experiments):]:
                ax.set_visible(False)

            fig.legend(["Sustained", "Phasic", "Low confidence"],
                       loc="lower center", ncol=3, fontsize=9,
                       markerscale=2, bbox_to_anchor=(0.5, 0))
            fig.suptitle(f"Spatial cluster maps — all {geno} slices", fontsize=12)
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            fig.savefig(OUTPUT_DIR / f"{geno}_spatial_cluster_maps.png", dpi=300, bbox_inches="tight")
            plt.show()
"""),
    md("## 11. Final file list"),
    code("""
print("Saved files:")
for p in sorted(OUTPUT_DIR.glob("*")):
    print(" -", p.name)
"""),
]


pipeline_cells = [
    md("""
# Thesis raw-data pipeline — from `pathToRois` to comparison results

This notebook is the clean pipeline version intended for reproducibility.

The main input is a list of experiments with `pathToRois` pointing to each `5.6_rois.pkl`.
For each experiment, the notebook derives the events path as:

```python
pathToEvents = pathToRois.split("_rois")[0] + "_auto_events.csv"
```

Then it:
1. Loads raw event tables and protocol from `islets.Regions.load_regions`.
2. Applies exclusions and nd2 Y-axis filtering when coordinates are available.
3. Assigns stimulation phases and excludes the first 200 s of non-first phases.
4. Computes ROI × phase metrics.
5. Builds the GMM feature vector.
6. Runs genotype-specific 2-component GMM.
7. Saves pooled CSVs in the same shape expected by the comparison notebook.

After this notebook runs, open `THESIS_comparison_results_clean.ipynb` and set `DATA_ROOT` to the same `DATA_ROOT` below.
"""),
    code("""
from pathlib import Path
import re
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

try:
    from islets.Regions import load_regions
except Exception as exc:
    raise ImportError("This notebook must be run in the lab environment where the islets package is installed.") from exc

%config InlineBackend.figure_format = "retina"
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ------------------------------
# CONFIG — edit this cell
# ------------------------------
import os
import getpass
_default_user = os.environ.get("JUPYTERHUB_USER") or os.environ.get("USER") or getpass.getuser()
DATA_ROOT = Path(f"/data/{_default_user}") if os.path.exists(f"/data/{_default_user}") else Path("/data")

# Add one dictionary per experiment.
# exp_name should match the thesis naming convention, for example "279a_F".
# pathToRois may be /local_data/... or /data/...; `normalize_path` below fixes /local_ -> /.
EXPERIMENTS = [
    # Example:
    # {
    #     "genotype": "WT",
    #     "exp_name": "example_a_F",
    #     "sex": "F",
    #     "pathToRois": "/local_data/Xiao/Isradipine/2026-04-08/exp072a.nd2_analysis/all/5.6_rois.pkl",
    # },
]

GENOTYPE_ORDER = ["WT", "HET", "MUT"]
ACH_PRIORITY = ["100nM ACh", "10nM ACh", "1nM ACh"]
Y_THRESHOLD = 150
TRANSITION_SEC = 200
RANDOM_STATE = 42

EXCLUSIONS = {
    "WT": ["275e_F"],
    "HET": ["271b_M", "271f_M"],
    "MUT": ["278a_F", "267a_M"],
}

# tiff is kept in clean event/phase outputs but excluded from GMM clustering.
GMM_ALLOWED_SCOPES = ["nd2", "lif"]

print("DATA_ROOT:", DATA_ROOT)
print("Experiments configured:", len(EXPERIMENTS))
"""),
    md("## 1. Helpers"),
    code("""
def normalize_path(path):
    path = str(path)
    path = path.replace("/local_", "/")
    return Path(path)


def derive_events_path(path_to_rois):
    s = str(path_to_rois)
    if "_rois" not in s:
        raise ValueError(f"Cannot derive events path because '_rois' is missing: {s}")
    return Path(s.split("_rois")[0] + "_auto_events.csv")


def infer_scope(path):
    match = re.search(r"\\.(lif|nd2|tiff|czi)_analysis", str(path))
    return match.group(1) if match else "unknown"


def infer_experiment_number(exp_name):
    match = re.search(r"(\\d+)", str(exp_name))
    return int(match.group(1)) if match else np.nan


def load_protocol(path_to_rois, exp_name):
    regions = load_regions(str(path_to_rois))
    if hasattr(regions, "detrend_traces"):
        try:
            regions.detrend_traces(method="debleach")
        except Exception as exc:
            warnings.warn(f"{exp_name}: detrend_traces failed, continuing. {exc}")
    proto = getattr(regions, "protocol", None)
    if proto is None or len(proto) == 0:
        warnings.warn(f"{exp_name}: empty protocol")
        return pd.DataFrame()
    proto = (
        proto[["compound", "concentration", "t_begin", "t_end"]]
        .dropna(subset=["compound", "concentration", "t_begin"])
        .sort_values("t_begin")
        .reset_index(drop=True)
    )
    rows = []
    for _, p in proto.iterrows():
        rows.append({
            "exp_name": exp_name,
            "compound": p["compound"],
            "concentration": str(p["concentration"]).replace(".0", ""),
            "t_begin_s": p["t_begin"],
            "t_end_s": p["t_end"],
        })
    return pd.DataFrame(rows)


def extract_roi_coordinates_from_regions(path_to_rois, exp_name):
    \"\"\"Best-effort coordinate extraction. If unavailable, returns an empty table and nd2 Y-filter is skipped.\"\"\"
    try:
        regions = load_regions(str(path_to_rois))
    except Exception as exc:
        warnings.warn(f"{exp_name}: could not reload regions for coordinates: {exc}")
        return pd.DataFrame()

    candidates = ["roi_coordinates", "coordinates", "centroids", "centers"]
    for attr in candidates:
        obj = getattr(regions, attr, None)
        if obj is None:
            continue
        try:
            df = pd.DataFrame(obj)
            if {"roi", "x", "y"}.issubset(df.columns):
                out = df[["roi", "x", "y"]].copy()
                out["exp_name"] = exp_name
                return out
        except Exception:
            pass

    warnings.warn(f"{exp_name}: ROI coordinates not found in regions object. Add coordinates manually if nd2 Y-filter is required.")
    return pd.DataFrame()


def build_phase_windows(protocol_df):
    rows = []
    for exp_name, group in protocol_df.groupby("exp_name"):
        group = group.sort_values("t_begin_s").reset_index(drop=True)
        for i, row in group.iterrows():
            phase_start = row["t_begin_s"]
            phase_end = group.loc[i + 1, "t_begin_s"] if i < len(group) - 1 else np.inf
            rows.append({
                "exp_name": exp_name,
                "phase": f"{row['concentration']} {row['compound']}",
                "compound": row["compound"],
                "concentration": row["concentration"],
                "phase_start": phase_start,
                "phase_end": phase_end,
                "is_first_phase": i == 0,
            })
    return pd.DataFrame(rows)


def assign_phases(events_df, phase_windows, transition_sec=200):
    result = []
    for exp_name, exp_events in events_df.groupby("exp_name"):
        exp_phases = phase_windows[phase_windows["exp_name"] == exp_name]
        exp_events = exp_events.copy()
        exp_events["phase"] = np.nan
        exp_events["phase_start"] = np.nan
        exp_events["transition"] = False
        for _, ph in exp_phases.iterrows():
            in_phase = (exp_events["peakpoint"] >= ph["phase_start"]) & (exp_events["peakpoint"] < ph["phase_end"])
            exp_events.loc[in_phase, "phase"] = ph["phase"]
            exp_events.loc[in_phase, "phase_start"] = ph["phase_start"]
            if not ph["is_first_phase"]:
                exp_events.loc[in_phase & (exp_events["peakpoint"] < ph["phase_start"] + transition_sec), "transition"] = True
        result.append(exp_events)
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame()


def compute_roi_metrics_per_phase(events_df, phase_windows, scope_map, transition_sec=200):
    rows = []
    for (exp_name, phase), group in events_df.dropna(subset=["phase"]).groupby(["exp_name", "phase"]):
        ph = phase_windows[(phase_windows["exp_name"] == exp_name) & (phase_windows["phase"] == phase)]
        if ph.empty:
            continue
        phase_start = ph["phase_start"].iloc[0]
        phase_end = ph["phase_end"].iloc[0]
        if np.isinf(phase_end):
            phase_end = group["peakpoint"].max()
        T_phase_sec = phase_end - phase_start
        if T_phase_sec <= 0:
            continue
        T_stable_sec = max(T_phase_sec - transition_sec, 0)
        T_stable_min = T_stable_sec / 60
        stable = group[group["transition"] == False]
        if stable.empty:
            continue

        agg_cols = {"n_events": ("roi", "size"), "mean_halfwidth": ("halfwidth", "mean")}
        for col in ["z_max", "auc", "height"]:
            if col in stable.columns:
                agg_cols[f"mean_{col}"] = (col, "mean")
        roi_metrics = stable.groupby("roi").agg(**agg_cols).reset_index()
        roi_metrics["event_rate"] = roi_metrics["n_events"] / T_stable_min if T_stable_min > 0 else np.nan
        roi_metrics["cell_activation_rate"] = roi_metrics["mean_halfwidth"] * roi_metrics["event_rate"]
        roi_metrics["exp_name"] = exp_name
        roi_metrics["phase"] = phase
        roi_metrics["phase_start"] = phase_start
        roi_metrics["T_minutes"] = round(T_phase_sec / 60, 2)
        roi_metrics["T_stable_min"] = round(T_stable_min, 2)
        meta_cols = [c for c in ["exp_name", "experiment", "letter", "sex", "genotype"] if c in events_df.columns]
        meta = events_df[events_df["exp_name"] == exp_name][meta_cols].drop_duplicates()
        roi_metrics = roi_metrics.merge(meta, on="exp_name", how="left")
        rows.append(roi_metrics)

    roi_phase = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not roi_phase.empty:
        roi_phase["scope"] = roi_phase["exp_name"].map(scope_map)
    return roi_phase
"""),
    md("## 2. Load raw experiments and save clean pooled tables"),
    code("""
if not EXPERIMENTS:
    raise ValueError("Fill EXPERIMENTS in the CONFIG cell before running the pipeline.")

nb_rows = []
events_rows = []
coords_rows = []
protocol_rows = []

for item in EXPERIMENTS:
    genotype = item["genotype"]
    exp_name = item["exp_name"]
    sex = item.get("sex", exp_name.split("_")[-1] if "_" in exp_name else "")
    path_to_rois = normalize_path(item["pathToRois"])
    path_to_events = normalize_path(item.get("pathToEvents", derive_events_path(path_to_rois)))
    scope = item.get("scope", infer_scope(path_to_rois))

    if not path_to_rois.exists():
        raise FileNotFoundError(f"{exp_name}: missing rois file: {path_to_rois}")
    if not path_to_events.exists():
        raise FileNotFoundError(f"{exp_name}: missing events file: {path_to_events}")

    ev = pd.read_csv(path_to_events)
    required = {"roi", "peakpoint", "halfwidth"}
    missing = required - set(ev.columns)
    if missing:
        raise ValueError(f"{exp_name}: events file missing required columns: {missing}")

    ev["genotype"] = genotype
    ev["exp_name"] = exp_name
    ev["experiment"] = infer_experiment_number(exp_name)
    ev["letter"] = item.get("letter", re.sub(r"\\d+", "", exp_name.split("_")[0]))
    ev["sex"] = sex
    ev["scope"] = scope
    events_rows.append(ev)

    coords = extract_roi_coordinates_from_regions(path_to_rois, exp_name)
    if not coords.empty:
        coords["genotype"] = genotype
        coords["scope"] = scope
        coords_rows.append(coords)

    proto = load_protocol(path_to_rois, exp_name)
    if not proto.empty:
        protocol_rows.append(proto)

    nb_rows.append({
        "genotype": genotype,
        "exp_name": exp_name,
        "sex": sex,
        "scope": scope,
        "pathToRois": str(path_to_rois),
        "pathToEvents": str(path_to_events),
        "n_events": len(ev),
    })

events = pd.concat(events_rows, ignore_index=True)
coords = pd.concat(coords_rows, ignore_index=True) if coords_rows else pd.DataFrame()
protocol_df = pd.concat(protocol_rows, ignore_index=True) if protocol_rows else pd.DataFrame()
nb_info = pd.DataFrame(nb_rows)

print(f"Loaded events: {len(events):,} rows from {events['exp_name'].nunique()} experiments")
display(nb_info)
"""),
    md("## 3. Coordinates and nd2 Y-filter"),
    code("""
def prepare_coordinates(coords):
    if coords.empty:
        return coords
    out = coords.copy()
    out["x_centered"] = out.groupby("exp_name")["x"].transform(lambda s: s - s.median())
    out["y_centered"] = out.groupby("exp_name")["y"].transform(lambda s: s - s.median())
    out["radial_dist"] = np.sqrt(out["x_centered"] ** 2 + out["y_centered"] ** 2)
    return out


def apply_y_filter(df, coords_df, scope_map, y_threshold):
    if coords_df.empty or "y_centered" not in coords_df.columns:
        warnings.warn("Coordinates unavailable; skipping nd2 Y-filter.")
        return df.copy()
    valid_nd2 = (
        coords_df[coords_df["exp_name"].map(scope_map) == "nd2"]
        .loc[lambda d: d["y_centered"].abs() <= y_threshold, ["roi", "exp_name"]]
    )
    non_nd2 = df[df["exp_name"].map(scope_map) != "nd2"]
    nd2 = df[df["exp_name"].map(scope_map) == "nd2"]
    nd2_filt = nd2.merge(valid_nd2, on=["roi", "exp_name"], how="inner")
    return pd.concat([non_nd2, nd2_filt], ignore_index=True)


scope_map = nb_info.set_index("exp_name")["scope"].to_dict()
coords = prepare_coordinates(coords)

events_clean = events.copy()
for genotype, excluded in EXCLUSIONS.items():
    events_clean = events_clean[~((events_clean["genotype"] == genotype) & (events_clean["exp_name"].isin(excluded)))].copy()

events_filt = apply_y_filter(events_clean, coords, scope_map, Y_THRESHOLD)

roi_summary = (
    events_clean.groupby(["genotype", "exp_name", "experiment", "letter", "sex", "scope", "roi"])
    .agg(n_events=("roi", "size"), mean_halfwidth=("halfwidth", "mean"))
    .reset_index()
)

print(f"Events before exclusions/filter: {len(events):,}")
print(f"Events after exclusions: {len(events_clean):,}")
print(f"Events after nd2 Y-filter: {len(events_filt):,}")
"""),
    md("## 4. Phase assignment and ROI metrics"),
    code("""
if protocol_df.empty:
    raise ValueError("No protocol rows loaded. Check that regions.protocol exists in the rois files.")

phase_windows = build_phase_windows(protocol_df)
events_filt = assign_phases(events_filt, phase_windows, transition_sec=TRANSITION_SEC)
roi_phase = compute_roi_metrics_per_phase(events_filt, phase_windows, scope_map, transition_sec=TRANSITION_SEC)

print("Phase distribution:")
display(events_filt["phase"].value_counts(dropna=False).to_frame("n_events"))
print(f"ROI-phase rows: {len(roi_phase):,}")
display(roi_phase.head())
"""),
    md("## 5. Save pooled clean tables per genotype"),
    code("""
def genotype_output_dir(genotype):
    out = DATA_ROOT / f"pooling_{genotype}" / "pooled_results"
    out.mkdir(parents=True, exist_ok=True)
    return out


for genotype in GENOTYPE_ORDER:
    out = genotype_output_dir(genotype)
    ev_g = events_clean[events_clean["genotype"] == genotype].copy()
    evf_g = events_filt[events_filt["genotype"] == genotype].copy()
    rp_g = roi_phase[roi_phase["genotype"] == genotype].copy()
    rs_g = roi_summary[roi_summary["genotype"] == genotype].copy()
    nb_g = nb_info[nb_info["genotype"] == genotype].copy()
    co_g = coords[coords["genotype"] == genotype].copy() if not coords.empty else pd.DataFrame()

    ev_g.to_csv(out / f"{genotype}_all_Events_acinar_clean.csv", index=False)
    evf_g.to_csv(out / f"{genotype}_all_Events_acinar_clean_yfiltered.csv", index=False)
    rp_g.to_csv(out / f"{genotype}_roi_metrics_per_phase.csv", index=False)
    rs_g.to_csv(out / f"{genotype}_all_roi_summary_acinar_clean.csv", index=False)
    nb_g.to_csv(out / "notebook_paths_summary.csv", index=False)
    if not co_g.empty:
        co_g.to_csv(out / f"{genotype}_roi_coordinates.csv", index=False)
    print(genotype, "saved to", out)
"""),
    md("## 6. Build feature vectors"),
    code("""
def pick_best_ach(df, value_cols, rename_map):
    out = []
    for exp_name, exp_df in df.groupby("exp_name"):
        for candidate in ACH_PRIORITY:
            sub = exp_df[exp_df["phase"] == candidate]
            if not sub.empty:
                cols = ["exp_name", "roi"] + value_cols
                out.append(sub[cols].copy().assign(ach_phase_used=candidate))
                break
    if not out:
        return pd.DataFrame(columns=["exp_name", "roi"] + list(rename_map.values()))
    return pd.concat(out, ignore_index=True).rename(columns=rename_map)


def compute_latency(events_df):
    rows = []
    for (exp_name, roi), grp in events_df.dropna(subset=["phase"]).groupby(["exp_name", "roi"]):
        ach = grp[grp["phase"].isin(ACH_PRIORITY)].copy()
        if ach.empty:
            continue
        for candidate in ACH_PRIORITY:
            sub = ach[(ach["phase"] == candidate) & (ach["transition"] == False)]
            if not sub.empty:
                rows.append({
                    "exp_name": exp_name,
                    "roi": roi,
                    "latency_ACh_s": sub["peakpoint"].min() - sub["phase_start"].iloc[0],
                })
                break
    return pd.DataFrame(rows)


def compute_cv(events_df):
    rows = []
    stable_ach = events_df[(events_df["phase"].isin(ACH_PRIORITY)) & (events_df["transition"] == False)].copy()
    for (exp_name, roi), grp in stable_ach.groupby(["exp_name", "roi"]):
        for candidate in ACH_PRIORITY:
            sub = grp[grp["phase"] == candidate]
            if not sub.empty:
                peaks = sub["peakpoint"].sort_values().to_numpy()
                if len(peaks) >= 3:
                    dt = np.diff(peaks)
                    if dt.mean() != 0:
                        rows.append({"exp_name": exp_name, "roi": roi, "cv_dt_ACh": dt.std() / dt.mean()})
                break
    return pd.DataFrame(rows)


feature_tables = {}
for genotype in GENOTYPE_ORDER:
    rp = roi_phase[roi_phase["genotype"] == genotype].copy()
    ev = events_filt[events_filt["genotype"] == genotype].copy()
    if genotype == "HET":
        rp = rp[rp["scope"].isin(GMM_ALLOWED_SCOPES)].copy()
        ev = ev[ev["scope"].isin(GMM_ALLOWED_SCOPES)].copy()

    base = rp[rp["phase"] == "8mM Glucose"][["exp_name", "roi", "event_rate"]].rename(columns={"event_rate": "event_rate_8mM"})
    ach_roi = pick_best_ach(
        rp,
        value_cols=["event_rate", "mean_halfwidth"],
        rename_map={"event_rate": "event_rate_ACh", "mean_halfwidth": "halfwidth_ACh"},
    ).drop(columns="ach_phase_used", errors="ignore")
    lat = compute_latency(ev)
    cv = compute_cv(ev)

    fv = (
        base
        .merge(ach_roi, on=["exp_name", "roi"], how="outer")
        .merge(lat, on=["exp_name", "roi"], how="left")
        .merge(cv, on=["exp_name", "roi"], how="left")
    )
    meta = rp[["exp_name", "experiment", "letter", "sex", "scope", "genotype"]].drop_duplicates()
    fv = fv.merge(meta, on="exp_name", how="left")
    feature_tables[genotype] = fv
    out = genotype_output_dir(genotype)
    fv.to_csv(out / f"{genotype}_feature_vector.csv", index=False)
    print(genotype, f"feature vector: {len(fv):,} rows")
    display(fv.isna().sum().to_frame("missing").T)
"""),
    md("## 7. Run genotype-specific 2-component GMM"),
    code("""
FEATURE_COLS = [
    "event_rate_8mM_log",
    "event_rate_ACh_log",
    "halfwidth_ACh_log",
    "latency_ACh_s_log",
    "cv_dt_ACh",
]


def run_gmm_for_genotype(genotype, fv):
    fv_log = fv.copy()
    for col in ["event_rate_8mM", "event_rate_ACh", "halfwidth_ACh", "latency_ACh_s"]:
        fv_log[f"{col}_log"] = np.log1p(fv_log[col])
    fv_clean = fv_log.dropna(subset=FEATURE_COLS).copy()
    if len(fv_clean) < 20:
        warnings.warn(f"{genotype}: too few complete ROIs for GMM ({len(fv_clean)})")
        return fv_clean

    X = StandardScaler().fit_transform(fv_clean[FEATURE_COLS].to_numpy())
    gm = GaussianMixture(n_components=2, covariance_type="full", n_init=20, random_state=RANDOM_STATE)
    gm.fit(X)
    probs = gm.predict_proba(X)
    labels = gm.predict(X)
    fv_clean["cluster"] = labels
    fv_clean["max_prob"] = probs.max(axis=1).round(3)
    fv_clean["low_conf"] = fv_clean["max_prob"] < 0.7

    hw_medians = fv_clean.groupby("cluster")["halfwidth_ACh"].median()
    sustained_cluster = hw_medians.idxmax()
    phasic_cluster = hw_medians.idxmin()
    fv_clean["cell_type"] = fv_clean["cluster"].map({sustained_cluster: "Sustained", phasic_cluster: "Phasic"})
    return fv_clean


cluster_tables = {}
for genotype, fv in feature_tables.items():
    if genotype in EXCLUSIONS:
        fv = fv[~fv["exp_name"].isin(EXCLUSIONS[genotype])].copy()
    if genotype == "HET":
        fv = fv[fv["scope"].isin(GMM_ALLOWED_SCOPES)].copy()
    clustered = run_gmm_for_genotype(genotype, fv)
    cluster_tables[genotype] = clustered
    out = genotype_output_dir(genotype)
    filename = "WT_gmm_clusters_global.csv" if genotype == "WT" else f"{genotype}_cluster_assignments.csv"
    clustered.to_csv(out / filename, index=False)
    print(f"\\n{genotype}: saved {len(clustered):,} clustered ROIs to {filename}")
    if not clustered.empty and "cell_type" in clustered.columns:
        display(clustered["cell_type"].value_counts().to_frame("n"))
"""),
    md("## 8. Quick validation"),
    code("""
all_clusters = pd.concat(cluster_tables.values(), ignore_index=True)
if all_clusters.empty:
    print("No clustered data.")
else:
    print("Clustered ROI overview:")
    overview = all_clusters.groupby(["genotype", "cell_type"]).size().rename("n_rois").reset_index()
    overview["pct"] = overview.groupby("genotype")["n_rois"].transform(lambda s: s / s.sum() * 100)
    display(overview)
    display(all_clusters.groupby(["genotype", "cell_type"])["halfwidth_ACh"].median().unstack().round(3))

print("\\nNext: run THESIS_comparison_results_clean.ipynb with DATA_ROOT =", DATA_ROOT)
"""),
    md("## 9. Optional spatial cluster maps"),
    code("""
if coords.empty:
    print("No coordinate table available. Spatial maps cannot be drawn.")
elif all_clusters.empty:
    print("No clustered data. Spatial maps cannot be drawn.")
else:
    spatial = all_clusters[["exp_name", "roi", "cell_type", "max_prob", "genotype"]].merge(
        coords[["exp_name", "roi", "x", "y"]],
        on=["exp_name", "roi"], how="inner"
    )
    if spatial.empty:
        print("No overlapping clustered ROIs and coordinates.")
    else:
        for genotype in sorted(spatial["genotype"].dropna().unique()):
            sub_spatial = spatial[spatial["genotype"] == genotype].copy()
            experiments = sorted(sub_spatial["exp_name"].unique())
            ncols = 4
            nrows = math.ceil(len(experiments) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
            axes = np.atleast_1d(axes).flat

            for ax, exp_name in zip(axes, experiments):
                sub = sub_spatial[sub_spatial["exp_name"] == exp_name]
                for cell_type, color in [("Sustained", "#FF5722"), ("Phasic", "#2196F3")]:
                    s = sub[sub["cell_type"] == cell_type]
                    ax.scatter(s["x"], s["y"], s=12, alpha=0.7, color=color,
                               label=cell_type if exp_name == experiments[0] else "")
                lc = sub[sub["max_prob"] < 0.7]
                if len(lc):
                    ax.scatter(lc["x"], lc["y"], s=12, color="lightgray", alpha=0.5,
                               label="Low confidence" if exp_name == experiments[0] else "")
                ax.set_title(exp_name, fontsize=8)
                ax.set_aspect("equal")
                ax.axis("off")

            for ax in list(axes)[len(experiments):]:
                ax.set_visible(False)

            fig.legend(["Sustained", "Phasic", "Low confidence"],
                       loc="lower center", ncol=3, fontsize=9,
                       markerscale=2, bbox_to_anchor=(0.5, 0))
            fig.suptitle(f"Spatial cluster maps — all {genotype} slices", fontsize=12)
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            out = genotype_output_dir(genotype)
            fig.savefig(out / f"{genotype}_spatial_cluster_maps.png", dpi=300, bbox_inches="tight")
            plt.show()
"""),
    md("## 10. Optional bimodality peak check"),
    code("""
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

if roi_phase.empty:
    print("No ROI-phase table available. Bimodality check cannot be drawn.")
else:
    hw_col = "mean_halfwidth"
    fig_genotypes = sorted(roi_phase["genotype"].dropna().unique())
    fig, axes = plt.subplots(2, len(fig_genotypes), figsize=(5 * len(fig_genotypes), 9), squeeze=False)
    rows = []

    for col_i, genotype in enumerate(fig_genotypes):
        for row_i, phase in enumerate(["10nM ACh", "100nM ACh"]):
            ax = axes[row_i][col_i]
            sub = roi_phase[
                (roi_phase["genotype"] == genotype) &
                (roi_phase["phase"] == phase) &
                (roi_phase[hw_col] > 0)
            ][hw_col].dropna()

            if len(sub) < 50:
                ax.text(0.5, 0.5, f"n={len(sub)}\\n(insufficient)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="gray")
                ax.set_title(f"{genotype} — {phase}")
                continue

            log_hw = np.log10(sub)
            x = np.linspace(-1.5, 2.0, 400)
            kde = gaussian_kde(log_hw, bw_method=0.2)
            kde_vals = kde(x)
            peaks, _ = find_peaks(kde_vals, distance=30, prominence=0.1)

            ax.plot(x, kde_vals, linewidth=2.5)
            ax.fill_between(x, kde_vals, alpha=0.15)
            ax.scatter(x[peaks], kde_vals[peaks], s=35, color="black", zorder=5)
            for peak in peaks:
                ax.text(x[peak], kde_vals[peak], f"{10**x[peak]:.2f}s",
                        ha="center", va="bottom", fontsize=8)

            ax.axvline(np.log10(0.185), color="#2196F3", linestyle="--",
                       alpha=0.7, linewidth=1.5, label="WT Phasic (0.19s)")
            ax.axvline(np.log10(4.23), color="#FF5722", linestyle="--",
                       alpha=0.7, linewidth=1.5, label="WT Sustained (4.23s)")
            ax.set_title(f"{genotype} — {phase}\\n"
                         f"n={len(sub):,} ROIs | {len(peaks)} peak(s) detected",
                         fontsize=10, fontweight="bold",
                         color="darkgreen" if len(peaks) >= 2 else "darkred")
            ax.set_xlabel("log10(halfwidth, s)")
            ax.set_ylabel("Density")
            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels(["0.1s", "1s", "10s"])

            rows.append({
                "genotype": genotype,
                "phase": phase,
                "n_rois": len(sub),
                "n_peaks": len(peaks),
                "peak_halfwidth_s": ", ".join(f"{10**x[p]:.3f}" for p in peaks),
                "pct_lt_0_5s": (sub < 0.5).mean() * 100,
                "pct_0_5_to_1s": ((sub >= 0.5) & (sub <= 1.0)).mean() * 100,
                "pct_gt_1s": (sub > 1.0).mean() * 100,
            })

    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Bimodality check: halfwidth KDE at 10nM vs 100nM ACh",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

    bimodality_summary = pd.DataFrame(rows)
    display(bimodality_summary.round(2))
"""),
]


write_notebook(ROOT / "THESIS_comparison_results_clean.ipynb", comparison_cells)
write_notebook(ROOT / "THESIS_reproducible_pipeline_from_raw_rois.ipynb", pipeline_cells)

print("Wrote:")
print(ROOT / "THESIS_comparison_results_clean.ipynb")
print(ROOT / "THESIS_reproducible_pipeline_from_raw_rois.ipynb")
