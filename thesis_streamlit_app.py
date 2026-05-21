from pathlib import Path
import importlib.util
import io
import math
import re
import sys
import traceback
import types
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This app needs Streamlit. Install it with: pip install streamlit"
    ) from exc


GENOTYPE_ORDER = ["WT", "HET", "MUT"]
CLUSTER_ORDER = ["Phasic", "Sustained"]
SCOPE_ORDER = ["lif", "nd2"]
ACH_PRIORITY = ["100nM ACh", "10nM ACh", "1nM ACh"]

MUT_EXCLUDE = ["278a_F", "267a_M"]
DEFAULT_EXCLUSIONS = {
    "WT": ["275e_F"],
    "HET": ["271b_M", "271f_M"],
    "MUT": MUT_EXCLUDE,
}

CLUSTER_FILES = {
    "WT": "WT_gmm_clusters_global.csv",
    "HET": "HET_cluster_assignments.csv",
    "MUT": "MUT_cluster_assignments.csv",
}

ROI_PHASE_FILES = {
    "WT": "WT_roi_metrics_per_phase.csv",
    "HET": "HET_roi_metrics_per_phase.csv",
    "MUT": "MUT_roi_metrics_per_phase.csv",
}

EVENT_FILES = {
    "WT": "WT_all_Events_acinar_clean_yfiltered.csv",
    "HET": "HET_all_Events_acinar_clean_yfiltered.csv",
    "MUT": "MUT_all_Events_acinar_clean_yfiltered.csv",
}

COORD_FILES = {
    "WT": "WT_roi_coordinates.csv",
    "HET": "HET_roi_coordinates.csv",
    "MUT": "MUT_roi_coordinates.csv",
}

PHASIC_COLOR = "#2196F3"
SUSTAINED_COLOR = "#FF5722"
GENO_COLORS = {"WT": "#1b5e20", "HET": "#e65100", "MUT": "#b71c1c"}
TYPE_COLORS = {"Phasic": PHASIC_COLOR, "Sustained": SUSTAINED_COLOR}
APP_VERSION = "2026-05-13-ui-pooling-spatial-v5"



def environment_diagnostics():
    """Return the exact Python/package environment used by this Streamlit process."""
    rows = [
        {"item": "python executable", "value": sys.executable},
        {"item": "python version", "value": sys.version.replace("\n", " ")},
        {"item": "pandas version", "value": pd.__version__},
        {"item": "pandas file", "value": getattr(pd, "__file__", "")},
        {"item": "numpy version", "value": np.__version__},
        {"item": "streamlit version", "value": getattr(st, "__version__", "")},
    ]
    try:
        import sklearn
        rows.append({"item": "scikit-learn version", "value": sklearn.__version__})
    except Exception as exc:
        rows.append({"item": "scikit-learn", "value": f"not importable: {exc}"})
    try:
        import scipy
        rows.append({"item": "scipy version", "value": scipy.__version__})
    except Exception as exc:
        rows.append({"item": "scipy", "value": f"not importable: {exc}"})
    spec = importlib.util.find_spec("islets")
    rows.append({"item": "islets location", "value": spec.origin if spec else "not found"})
    return pd.DataFrame(rows)


def install_pandas_pickle_compat():
    """Allow older pandas pickles with slice placements to load under newer pandas."""
    numeric_module = sys.modules.get("pandas.core.indexes.numeric")
    if numeric_module is None:
        numeric_module = types.ModuleType("pandas.core.indexes.numeric")
        numeric_module.Index = pd.Index
        numeric_module.RangeIndex = pd.RangeIndex
        numeric_module.Int64Index = pd.Index
        numeric_module.UInt64Index = pd.Index
        numeric_module.Float64Index = pd.Index
        numeric_module.NumericIndex = pd.Index
        sys.modules["pandas.core.indexes.numeric"] = numeric_module

    try:
        import pandas.core.internals.blocks as blocks
        from pandas._libs.internals import BlockPlacement
    except Exception:
        return False, "pandas internals not available"

    if getattr(blocks.new_block, "_calcium_compat_patch", False):
        return True, "already installed"

    original_new_block = blocks.new_block

    def new_block_compat(values, placement, ndim, **kwargs):
        if isinstance(placement, slice):
            placement = BlockPlacement(placement)
        kwargs.pop("refs", None)  # pandas 1.3.5 does not accept refs
        return original_new_block(values, placement=placement, ndim=ndim, **kwargs)

    new_block_compat._calcium_compat_patch = True
    new_block_compat._calcium_original_new_block = original_new_block
    blocks.new_block = new_block_compat
    return True, "installed"


def test_rois_loading(path_to_rois):
    """Load one rois.pkl exactly as the app would and report what happened."""
    path = normalize_path(path_to_rois)
    result = {
        "normalized_path": str(path),
        "path_exists": path.exists(),
        "python_executable": sys.executable,
        "pandas_version": pd.__version__,
        "status": "not started",
        "message": "",
        "has_df": False,
        "df_shape": "",
        "df_columns": "",
        "has_peak": False,
        "n_peak_nonnull": 0,
    }
    try:
        from islets.Regions import load_regions
    except Exception as exc:
        result.update(status="failed", message=f"cannot import load_regions: {exc}")
        return result, pd.DataFrame()

    try:
        ok, patch_msg = install_pandas_pickle_compat()
        result["message"] = f"pandas pickle compatibility patch: {patch_msg}; "
        regions = load_regions(str(path))
        result["status"] = "loaded"
        result["message"] += f"loaded Regions object: {type(regions)}"
    except Exception as exc:
        result.update(status="failed", message=f"load_regions failed: {type(exc).__name__}: {exc}")
        return result, pd.DataFrame()

    try:
        regions.detrend_traces(method="debleach")
        result["message"] += "; detrend_traces ok"
    except Exception as exc:
        result["message"] += f"; detrend_traces skipped/failed: {type(exc).__name__}: {exc}"

    preview = pd.DataFrame()
    if hasattr(regions, "df"):
        try:
            df = regions.df.copy()
            result["has_df"] = True
            result["df_shape"] = str(df.shape)
            result["df_columns"] = ", ".join(map(str, df.columns[:20]))
            result["has_peak"] = "peak" in df.columns
            if "peak" in df.columns:
                result["n_peak_nonnull"] = int(df["peak"].notna().sum())
            preview = df.head(10)
        except Exception as exc:
            result["message"] += f"; could not inspect regions.df: {type(exc).__name__}: {exc}"
    return result, preview


def normalize_path(path):
    return Path(str(path).strip().replace("/local_", "/"))


def resolve_rois_path(path):
    path = normalize_path(path)
    if path.is_dir():
        candidates = sorted(path.rglob("5.6_rois.pkl"))
        if not candidates:
            candidates = sorted(path.rglob("*_rois.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No rois.pkl file found under directory: {path}")
        return candidates[0]
    return path


def derive_events_path(path_to_rois):
    try:
        path_to_rois = resolve_rois_path(path_to_rois)
    except Exception:
        pass
    text = str(path_to_rois)
    if "_rois" not in text:
        return ""
    return text.split("_rois")[0] + "_auto_events.csv"


def infer_scope(path):
    match = re.search(r"\.(lif|nd2|tiff|czi)_analysis", str(path))
    return match.group(1) if match else "unknown"


def infer_experiment_number(exp_name):
    match = re.search(r"(\d+)", str(exp_name))
    return int(match.group(1)) if match else np.nan


def normalize_protocol_table(protocol_df, exp_name):
    """Return protocol rows with exp_name, compound, concentration, t_begin_s, t_end_s."""
    df = protocol_df.copy()
    rename = {}
    lower = {str(c).lower(): c for c in df.columns}
    aliases = {
        "compound": ["compound", "drug", "stimulus"],
        "concentration": ["concentration", "conc", "dose"],
        "t_begin_s": ["t_begin_s", "t_begin", "start", "start_s", "phase_start"],
        "t_end_s": ["t_end_s", "t_end", "end", "end_s", "phase_end"],
    }
    for canonical, options in aliases.items():
        for option in options:
            if option in lower:
                rename[lower[option]] = canonical
                break
    df = df.rename(columns=rename)

    required = {"compound", "concentration", "t_begin_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Protocol CSV must contain columns for compound, concentration and start time. "
            f"Missing: {sorted(missing)}"
        )
    if "t_end_s" not in df.columns:
        df["t_end_s"] = np.nan
    if "exp_name" not in df.columns:
        df["exp_name"] = exp_name
    else:
        df["exp_name"] = df["exp_name"].fillna(exp_name)

    df = df[["exp_name", "compound", "concentration", "t_begin_s", "t_end_s"]].copy()
    df["concentration"] = df["concentration"].astype(str).str.replace(".0", "", regex=False)
    df["t_begin_s"] = pd.to_numeric(df["t_begin_s"], errors="coerce")
    df["t_end_s"] = pd.to_numeric(df["t_end_s"], errors="coerce")
    df = df.dropna(subset=["compound", "concentration", "t_begin_s"]).sort_values(["exp_name", "t_begin_s"])
    if df.empty:
        raise ValueError(f"{exp_name}: protocol table has no usable rows.")
    return df.reset_index(drop=True)


def infer_sex(exp_name):
    text = str(exp_name)
    match = re.search(r"(?:_|-)([MF])$", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else ""


def output_dir_for(data_root, genotype):
    path = Path(data_root) / f"pooling_{genotype}" / "pooled_results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def add_cell_type(df):
    df = df.copy()
    if "cell_type" in df.columns:
        return df
    if not {"cluster", "halfwidth_ACh"}.issubset(df.columns):
        raise ValueError("Cluster table needs either cell_type or cluster + halfwidth_ACh.")
    med = df.groupby("cluster")["halfwidth_ACh"].median()
    df["cell_type"] = df["cluster"].map({med.idxmin(): "Phasic", med.idxmax(): "Sustained"})
    return df


def add_scope_if_missing(df, genotype, pooled_dir):
    df = df.copy()
    if "scope" in df.columns:
        return df
    rp_path = Path(pooled_dir) / ROI_PHASE_FILES[genotype]
    if rp_path.exists():
        rp = pd.read_csv(rp_path)
        if "scope" in rp.columns:
            scope_map = rp.drop_duplicates("exp_name").set_index("exp_name")["scope"].to_dict()
            df["scope"] = df["exp_name"].map(scope_map)
            return df
    df["scope"] = "unknown"
    return df


def load_existing_cluster_tables(pooled_dirs):
    tables = {}
    for genotype, folder in pooled_dirs.items():
        folder = Path(folder)
        cluster_path = folder / CLUSTER_FILES[genotype]
        if not cluster_path.exists():
            continue
        df = pd.read_csv(cluster_path)
        if genotype == "MUT":
            df = df[~df["exp_name"].isin(MUT_EXCLUDE)].copy()
        df = add_cell_type(df)
        df = add_scope_if_missing(df, genotype, folder)
        df["genotype"] = genotype
        tables[genotype] = df
    if not tables:
        raise FileNotFoundError("No cluster assignment CSVs were found.")
    return pd.concat(tables.values(), ignore_index=True), tables


def load_roi_phase_tables(pooled_dirs):
    rows = []
    for genotype, folder in pooled_dirs.items():
        path = Path(folder) / ROI_PHASE_FILES[genotype]
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if genotype == "MUT":
            df = df[~df["exp_name"].isin(MUT_EXCLUDE)].copy()
        df["genotype"] = genotype
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_coordinate_tables(pooled_dirs):
    rows = []
    for genotype, folder in pooled_dirs.items():
        path = Path(folder) / COORD_FILES[genotype]
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["genotype"] = genotype
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fig_to_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _store_output(name, data):
    """Register bytes (PNG or CSV) in session_state for later server-side save."""
    st.session_state.setdefault("outputs", {})[name] = data


def _dl_buttons(items):
    """Show a row of download buttons. items: list of (filename, bytes, mime)."""
    cols = st.columns(len(items))
    for col, (fname, data, mime) in zip(cols, items):
        with col:
            st.download_button(f"⬇ {fname}", data, file_name=fname, mime=mime, use_container_width=True)
        _store_output(fname, data)


def compute_summary_tables(all_data):
    features = [
        "halfwidth_ACh",
        "event_rate_ACh",
        "latency_ACh_s",
        "cv_dt_ACh",
        "event_rate_8mM",
    ]
    features = [f for f in features if f in all_data.columns]

    population = (
        all_data.groupby(["genotype", "cell_type"])
        .size()
        .rename("n_rois")
        .reset_index()
    )
    population["pct"] = population.groupby("genotype")["n_rois"].transform(lambda s: s / s.sum() * 100)

    medians = (
        all_data.groupby(["genotype", "cell_type"])[features]
        .median()
        .round(3)
        .reset_index()
    )

    counts = (
        all_data.groupby(["genotype", "cell_type"])
        .size()
        .rename("n_rois")
        .reset_index()
    )

    if "scope" in all_data.columns:
        scope = (
            all_data.groupby(["scope", "genotype", "cell_type"])[features]
            .median()
            .round(3)
            .reset_index()
        )
    else:
        scope = pd.DataFrame()

    return {
        "population": population,
        "feature_medians": medians,
        "roi_counts": counts,
        "scope_medians": scope,
    }


def save_summary_tables(tables, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if not df.empty:
            df.to_csv(output_dir / f"{name}.csv", index=False)


def plot_population(all_data):
    genotypes = [g for g in GENOTYPE_ORDER if g in set(all_data["genotype"])]
    phasic_vals, sustained_vals = [], []
    for genotype in genotypes:
        sub = all_data[all_data["genotype"] == genotype]
        pct = sub["cell_type"].value_counts(normalize=True) * 100
        phasic_vals.append(pct.get("Phasic", 0))
        sustained_vals.append(pct.get("Sustained", 0))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(genotypes, phasic_vals, color=PHASIC_COLOR, edgecolor="white", label="Phasic")
    ax.bar(genotypes, sustained_vals, bottom=phasic_vals, color=SUSTAINED_COLOR, edgecolor="white", label="Sustained")
    for i, (p, s) in enumerate(zip(phasic_vals, sustained_vals)):
        ax.text(i, p / 2, f"{p:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        ax.text(i, p + s / 2, f"{s:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.set_ylabel("% of ROIs")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.set_title("Population structure")
    return fig


def plot_halfwidth(all_data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    genotypes = [g for g in GENOTYPE_ORDER if g in set(all_data["genotype"])]
    for ax, cell_type in zip(axes, CLUSTER_ORDER):
        sub = all_data[(all_data["cell_type"] == cell_type) & (all_data["halfwidth_ACh"] > 0)]
        vals = [sub.loc[sub["genotype"] == g, "halfwidth_ACh"].dropna().values for g in genotypes]
        vals = [v for v in vals if len(v)]
        labels = [g for g in genotypes if len(sub.loc[sub["genotype"] == g, "halfwidth_ACh"].dropna())]
        if not vals:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue
        parts = ax.violinplot([np.log10(v) for v in vals], showextrema=False, showmedians=False)
        for body, label in zip(parts["bodies"], labels):
            body.set_facecolor(GENO_COLORS.get(label, "gray"))
            body.set_alpha(0.5)
            body.set_edgecolor("black")
        for pos, v in enumerate(vals, start=1):
            med = np.median(v)
            ax.hlines(np.log10(med), pos - 0.25, pos + 0.25, color="black", linewidth=2)
            ax.text(pos + 0.25, np.log10(med), f"{med:.2f}s", va="center", fontsize=8)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["0.1s", "1s", "10s"])
        ax.set_title(cell_type)
    axes[0].set_ylabel("halfwidth (log scale)")
    fig.tight_layout()
    return fig


def plot_spatial_maps(all_data, coords, title="Spatial cluster maps", population_mode="Analyzed/acinar ROIs", zoom_pct=0):
    if coords.empty:
        raise ValueError("No coordinate table found.")
    cluster_cols = ["exp_name", "roi", "cell_type", "max_prob"]
    left = all_data[[c for c in cluster_cols if c in all_data.columns]].copy()
    show_background = population_mode != "All ROIs"
    plot_coords = filter_coords_by_population(coords, population_mode) if show_background else coords
    if plot_coords.empty:
        raise ValueError(f"No ROI coordinates available for: {population_mode}.")
    right = plot_coords[["exp_name", "roi", "x", "y"]].copy()
    left["roi_key"] = pd.to_numeric(left["roi"], errors="coerce")
    right["roi_key"] = pd.to_numeric(right["roi"], errors="coerce")
    if left["roi_key"].notna().any() and right["roi_key"].notna().any():
        spatial = right.merge(left.drop(columns=["roi"]), on=["exp_name", "roi_key"], how="left")
    else:
        left["roi_key"] = left["roi"].astype(str)
        right["roi_key"] = right["roi"].astype(str)
        spatial = right.merge(left.drop(columns=["roi"]), on=["exp_name", "roi_key"], how="left")
    if spatial.empty:
        raise ValueError("No overlapping ROI coordinates for clustered ROIs.")

    experiments = sorted(spatial["exp_name"].unique())
    ncols = min(4, max(1, len(experiments)))
    nrows = math.ceil(len(experiments) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows))
    axes = list(np.atleast_1d(axes).ravel())

    for ax, exp_name in zip(axes, experiments):
        sub = spatial[spatial["exp_name"] == exp_name]
        bg_coords = coords[coords["exp_name"] == exp_name] if not coords.empty else pd.DataFrame()
        # Full-field gray background for tissue context
        if show_background and not bg_coords.empty:
            ax.scatter(bg_coords["x"], bg_coords["y"], s=3, alpha=0.10, color="#b0b0b0", zorder=1)
        if sub.empty and show_background:
            # No ROIs in this population for this experiment — show a clear note
            ax.set_title(exp_name, fontsize=8)
            ax.set_aspect("equal")
            ax.axis("off")
            if not bg_coords.empty:
                apply_spatial_zoom(ax, bg_coords, zoom_pct)
            ax.text(0.5, 0.5, f"No {population_mode.lower()} ROIs\ndefined for this experiment",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="#888", style="italic")
            continue
        other = sub[sub["cell_type"].isna()] if "cell_type" in sub.columns else sub.iloc[0:0]
        if len(other):
            # When a population is selected, "other" = population members without GMM assignment
            # (e.g. Y-filtered ROIs). Use a distinct color so they don't blend into the background.
            other_color = "#7bafd4" if show_background else "#a3aab8"
            other_alpha = 0.55 if show_background else 0.35
            ax.scatter(
                other["x"],
                other["y"],
                s=6,
                alpha=other_alpha,
                color=other_color,
                label="Other/islet candidates" if exp_name == experiments[0] else "",
            )
        for cell_type in CLUSTER_ORDER:
            s = sub[sub["cell_type"] == cell_type]
            ax.scatter(
                s["x"],
                s["y"],
                s=6,
                alpha=0.7,
                color=TYPE_COLORS[cell_type],
                label=cell_type if exp_name == experiments[0] else "",
            )
        if "max_prob" in sub.columns:
            lc = sub[sub["max_prob"] < 0.7]
            if len(lc):
                ax.scatter(
                    lc["x"],
                    lc["y"],
                    s=6,
                    color="lightgray",
                    alpha=0.5,
                    label="Low confidence" if exp_name == experiments[0] else "",
                )
        ax.set_title(exp_name, fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
        # Axis range is always based on the full tissue; zoom crops from there
        range_ref = bg_coords if show_background and not bg_coords.empty else sub
        apply_spatial_zoom(ax, range_ref, zoom_pct)

    for ax in axes[len(experiments):]:
        ax.set_visible(False)

    fig.legend(loc="lower center", ncol=3, fontsize=8, markerscale=1.5,
               bbox_to_anchor=(0.5, 0), framealpha=0.9)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def roi_key_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.astype("Int64").astype(str)
    return series.astype(str)


def parse_roi_id_list(text):
    """Parse ROI IDs from various input formats.

    Handles:
      - plain: ``126,132,143``
      - bracketed: ``[126, 132, 143]``
      - variable assignment: ``islet = [126, 132, 143]``
    """
    if not text or not str(text).strip():
        return []
    s = str(text).strip()
    # Strip optional "varname =" or "varname=" prefix
    s = re.sub(r"^\w+\s*=\s*", "", s)
    # Strip surrounding brackets
    s = s.strip("[]")
    tokens = re.split(r"[\s,;]+", s.strip())
    result = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        try:
            result.append(int(t))
        except ValueError:
            result.append(t)
    return result


def try_extract_roi_populations_from_regions(regions):
    """Look for islet/acinar classification in regions.df.

    Returns {"islet": [id, ...], "acinar": [id, ...]} or None if nothing found.
    """
    if not hasattr(regions, "df"):
        return None
    df = regions.df
    type_cols = [
        c for c in df.columns
        if any(kw in str(c).lower() for kw in ("type", "kind", "population", "class", "islet", "acinar", "cell"))
    ]
    if not type_cols:
        return None
    col = type_cols[0]
    unique_vals = df[col].dropna().unique()
    islet_kws = ("islet", "beta", "island")
    acinar_kws = ("acinar",)
    islet_vals = [v for v in unique_vals if any(kw in str(v).lower() for kw in islet_kws)]
    acinar_vals = [v for v in unique_vals if any(kw in str(v).lower() for kw in acinar_kws)]
    result = {}
    if islet_vals:
        result["islet"] = list(df[df[col].isin(islet_vals)].index)
    if acinar_vals:
        result["acinar"] = list(df[df[col].isin(acinar_vals)].index)
    return result if result else None


def filter_rp_by_population(rp_all, coords, population_mode):
    """Return rp_all filtered to ROIs in the selected population.

    Uses the 'population' column in coords set by tag_coords_with_population.
    Returns rp_all unchanged when no population column exists or mode is "All ROIs".
    """
    if population_mode == "All ROIs" or coords is None or coords.empty or "population" not in coords.columns:
        return rp_all
    target_pops = {
        "Islet": ["islet"],
        "Acinar (excl. islet)": ["acinar"],
        "Islet + Acinar": ["islet", "acinar"],
    }.get(population_mode)
    if not target_pops:
        return rp_all
    pop_coords = coords[coords["population"].isin(target_pops)][["exp_name", "roi"]].copy()
    if pop_coords.empty:
        return rp_all.iloc[0:0].copy()
    pop_coords["_roi_key"] = pd.to_numeric(pop_coords["roi"], errors="coerce")
    rp = rp_all.copy()
    rp["_roi_key"] = pd.to_numeric(rp["roi"], errors="coerce")
    if pop_coords["_roi_key"].notna().any() and rp["_roi_key"].notna().any():
        result = rp.merge(pop_coords[["exp_name", "_roi_key"]].drop_duplicates(), on=["exp_name", "_roi_key"], how="inner")
    else:
        pop_coords["_roi_key"] = pop_coords["roi"].astype(str)
        rp["_roi_key"] = rp["roi"].astype(str)
        result = rp.merge(pop_coords[["exp_name", "_roi_key"]].drop_duplicates(), on=["exp_name", "_roi_key"], how="inner")
    return result.drop(columns=["_roi_key"], errors="ignore")


def tag_coords_with_population(coords, roi_lists_per_exp):
    """Add a 'population' column: 'islet', 'acinar' (excl. islet), or 'unknown'.

    roi_lists_per_exp: {exp_name: {"islet": [...], "acinar": [...]}}
    ROIs present in both lists are classified as islet (acinar_clean = acinar − islet).
    """
    coords = coords.copy()
    coords["population"] = "unknown"
    for exp_name, lists in roi_lists_per_exp.items():
        islet_ids = set(str(r) for r in lists.get("islet", []))
        acinar_ids = set(str(r) for r in lists.get("acinar", []))
        acinar_clean = acinar_ids - islet_ids
        mask = coords["exp_name"] == exp_name
        roi_str = coords.loc[mask, "roi"].astype(str)
        coords.loc[mask & roi_str.isin(islet_ids), "population"] = "islet"
        coords.loc[mask & roi_str.isin(acinar_clean), "population"] = "acinar"
    return coords


def filter_coords_by_population(coords, population_mode):
    """Filter coords using the 'population' column set by tag_coords_with_population.

    population_mode: "All ROIs" | "Islet" | "Acinar (excl. islet)" | "Islet + Acinar"
    Returns all coords unchanged when no population column is present.
    """
    if coords is None or coords.empty or population_mode == "All ROIs":
        return coords
    if "population" not in coords.columns:
        return coords
    if population_mode == "Islet":
        return coords[coords["population"] == "islet"].copy()
    if population_mode == "Acinar (excl. islet)":
        return coords[coords["population"] == "acinar"].copy()
    if population_mode == "Islet + Acinar":
        return coords[coords["population"].isin(["islet", "acinar"])].copy()
    return coords


def apply_spatial_zoom(ax, sub, zoom_pct):
    if zoom_pct <= 0 or sub.empty:
        return
    keep = max(1, min(99, 100 - int(zoom_pct)))
    trim = (100 - keep) / 2
    try:
        x0, x1 = np.nanpercentile(sub["x"], [trim, 100 - trim])
        y0, y1 = np.nanpercentile(sub["y"], [trim, 100 - trim])
        if np.isfinite([x0, x1, y0, y1]).all() and x0 < x1 and y0 < y1:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
    except Exception:
        pass


def plot_spatial_phase_maps(rp_all, coords, phase, value_col=None, population_mode="All ROIs", zoom_pct=0):
    if coords.empty:
        raise ValueError("No coordinate table found.")
    if rp_all.empty:
        raise ValueError("No ROI-phase table found.")
    value_col = value_col or find_halfwidth_column(rp_all)
    if value_col not in rp_all.columns:
        raise ValueError(f"{value_col} is not available in ROI-phase data.")

    phase_data = rp_all[(rp_all["phase"] == phase) & rp_all[value_col].notna()].copy()
    if phase_data.empty:
        raise ValueError(f"No ROI-phase data found for phase: {phase}")

    left = phase_data[["exp_name", "roi", value_col]].copy()
    # plot_coords is the (possibly filtered) population; coords is always the full field for background.
    plot_coords = filter_coords_by_population(coords, population_mode)
    if plot_coords.empty:
        raise ValueError(f"No ROI coordinates available for: {population_mode}.")

    right = plot_coords[["exp_name", "roi", "x", "y"]].copy()
    left["roi_key"] = pd.to_numeric(left["roi"], errors="coerce")
    right["roi_key"] = pd.to_numeric(right["roi"], errors="coerce")
    if left["roi_key"].notna().any() and right["roi_key"].notna().any():
        spatial = right.merge(left.drop(columns=["roi"]), on=["exp_name", "roi_key"], how="left")
    else:
        left["roi_key"] = left["roi"].astype(str)
        right["roi_key"] = right["roi"].astype(str)
        spatial = right.merge(left.drop(columns=["roi"]), on=["exp_name", "roi_key"], how="left")
    if spatial.empty:
        raise ValueError("No overlapping ROI coordinates for this phase.")

    show_background = (population_mode != "All ROIs") and not coords.empty

    experiments = sorted(spatial["exp_name"].unique())
    ncols = min(4, max(1, len(experiments)))
    nrows = math.ceil(len(experiments) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows))
    axes = list(np.atleast_1d(axes).ravel())

    metric_spatial = spatial[spatial[value_col].notna()].copy()
    vals = metric_spatial[value_col].clip(lower=1e-6)
    if vals.empty:
        raise ValueError("No metric values overlap with the selected ROI coordinates.")
    color_vals = np.log10(vals) if "halfwidth" in value_col.lower() else vals
    vmin, vmax = np.nanpercentile(color_vals, [2, 98]) if len(color_vals) else (0, 1)

    last = None
    for ax, exp_name in zip(axes, experiments):
        sub = spatial[spatial["exp_name"] == exp_name].copy()
        bg_coords = coords[coords["exp_name"] == exp_name] if not coords.empty else pd.DataFrame()
        # Full-field background: all ROIs in faint gray for spatial context.
        if show_background and not bg_coords.empty:
            ax.scatter(bg_coords["x"], bg_coords["y"], s=3, alpha=0.10, color="#b0b0b0", zorder=1)
        if sub.empty and show_background:
            ax.set_title(exp_name, fontsize=8)
            ax.set_aspect("equal")
            ax.axis("off")
            if not bg_coords.empty:
                apply_spatial_zoom(ax, bg_coords, zoom_pct)
            ax.text(0.5, 0.5, f"No {population_mode.lower()} ROIs\ndefined for this experiment",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="#888", style="italic")
            continue
        # ROIs in the plot_coords that have no metric value (selected pop but no data this phase).
        other = sub[sub[value_col].isna()]
        if len(other):
            no_data_color = "#7bafd4" if show_background else "#a3aab8"
            no_data_alpha = 0.55 if show_background else 0.3
            ax.scatter(other["x"], other["y"], s=5, alpha=no_data_alpha, color=no_data_color, zorder=2)
        # ROIs with metric values, colored by value.
        metric_sub = sub[sub[value_col].notna()].copy()
        if len(metric_sub):
            sub_vals = metric_sub[value_col].clip(lower=1e-6)
            c = np.log10(sub_vals) if "halfwidth" in value_col.lower() else sub_vals
            last = ax.scatter(
                metric_sub["x"],
                metric_sub["y"],
                c=c,
                s=6,
                alpha=0.85,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                zorder=3,
            )
        ax.set_title(exp_name, fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
        # Use the full field (all coords for this experiment) for axis range, not just the subset.
        range_ref = bg_coords if show_background and not bg_coords.empty else sub
        apply_spatial_zoom(ax, range_ref, zoom_pct)

    for ax in axes[len(experiments):]:
        ax.set_visible(False)

    if last is not None:
        cbar = fig.colorbar(last, ax=axes, fraction=0.025, pad=0.02)
        label = f"log10({value_col})" if "halfwidth" in value_col.lower() else value_col
        cbar.set_label(label)

    pop_label = f" · {population_mode}" if population_mode != "All ROIs" else ""
    fig.suptitle(f"Spatial map — {phase} ({value_col}){pop_label}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def _coords_from_dataframe_like(obj):
    try:
        df = pd.DataFrame(obj)
    except Exception:
        return pd.DataFrame()
    lower = {str(c).lower(): c for c in df.columns}
    roi_col = next((lower[c] for c in ["roi", "label", "id", "index"] if c in lower), None)
    x_col = next((lower[c] for c in ["x", "center_x", "centroid_x", "x_center", "x0"] if c in lower), None)
    y_col = next((lower[c] for c in ["y", "center_y", "centroid_y", "y_center", "y0"] if c in lower), None)
    if roi_col is None and len(df):
        df = df.reset_index().rename(columns={"index": "roi"})
        roi_col = "roi"
    if roi_col is None or x_col is None or y_col is None:
        return pd.DataFrame()
    out = df[[roi_col, x_col, y_col]].rename(columns={roi_col: "roi", x_col: "x", y_col: "y"})
    return out.dropna(subset=["roi", "x", "y"]).drop_duplicates()


def extract_coordinates_from_regions(regions, exp_name, genotype, scope):
    """Best-effort ROI coordinate extraction from a Regions object, if one is available."""
    if hasattr(regions, "df"):
        try:
            df = regions.df.copy()
            if "peak" in df.columns:
                coords = pd.DataFrame({
                    "roi": df.index,
                    "x": df["peak"].apply(lambda p: p[0] if isinstance(p, (list, tuple, np.ndarray)) and len(p) > 0 else np.nan),
                    "y": df["peak"].apply(lambda p: p[1] if isinstance(p, (list, tuple, np.ndarray)) and len(p) > 1 else np.nan),
                })
                coords = coords.dropna(subset=["roi", "x", "y"]).drop_duplicates()
                if not coords.empty:
                    coords["exp_name"] = exp_name
                    coords["genotype"] = genotype
                    coords["scope"] = scope
                    return coords
        except Exception:
            pass

    candidate_attrs = [
        "roi_coordinates",
        "coordinates",
        "coords",
        "centroids",
        "centers",
        "roi_centers",
        "roiCenters",
        "rois",
        "regions",
    ]
    for attr in candidate_attrs:
        if not hasattr(regions, attr):
            continue
        obj = getattr(regions, attr)
        coords = _coords_from_dataframe_like(obj)
        if not coords.empty:
            coords["exp_name"] = exp_name
            coords["genotype"] = genotype
            coords["scope"] = scope
            return coords

    for attr in ["masks", "mask", "roi_masks"]:
        if not hasattr(regions, attr):
            continue
        masks = getattr(regions, attr)
        try:
            arr = np.asarray(masks)
        except Exception:
            continue
        rows = []
        if arr.ndim == 3:
            for roi_idx in range(arr.shape[0]):
                y, x = np.where(arr[roi_idx] > 0)
                if len(x):
                    rows.append({"roi": roi_idx, "x": float(np.mean(x)), "y": float(np.mean(y))})
        elif arr.ndim == 2:
            labels = [v for v in np.unique(arr) if v != 0]
            for label in labels:
                y, x = np.where(arr == label)
                if len(x):
                    rows.append({"roi": int(label), "x": float(np.mean(x)), "y": float(np.mean(y))})
        if rows:
            coords = pd.DataFrame(rows)
            coords["exp_name"] = exp_name
            coords["genotype"] = genotype
            coords["scope"] = scope
            return coords

    return pd.DataFrame()


def load_coordinates_from_rois(path_to_rois, exp_name, genotype, scope, clean_roi_ids=None):
    status = {
        "exp_name": exp_name,
        "source": "rois.pkl",
        "status": "started",
        "n_before_filter": 0,
        "n_after_filter": 0,
        "python": sys.executable,
        "pandas": pd.__version__,
        "message": "",
    }
    try:
        from islets.Regions import load_regions
    except Exception as exc:
        status.update(status="failed", message=f"cannot import load_regions: {exc}")
        warnings.warn(f"{exp_name}: cannot import load_regions for coordinate extraction: {exc}")
        return pd.DataFrame(), status, {}

    try:
        ok, patch_msg = install_pandas_pickle_compat()
        status["message"] = f"pandas pickle compatibility patch: {patch_msg}; "
        regions = load_regions(str(path_to_rois))
    except Exception as exc:
        status.update(status="failed", message=f"{status.get('message', '')}could not open rois.pkl: {exc}")
        warnings.warn(
            f"{exp_name}: could not open rois.pkl to extract coordinates. "
            f"Spatial maps and nd2 Y-filter need coordinates. Original error: {exc}"
        )
        return pd.DataFrame(), status, {}

    try:
        regions.detrend_traces(method="debleach")
    except Exception:
        pass

    auto_populations = try_extract_roi_populations_from_regions(regions)

    coords = extract_coordinates_from_regions(regions, exp_name, genotype, scope)
    if coords.empty:
        status.update(
            status="failed",
            message=f"{status.get('message', '')}rois.pkl opened, but no coordinates were found inside regions",
        )
        warnings.warn(f"{exp_name}: rois.pkl opened, but no coordinates were found inside regions.")
        return coords, status, auto_populations

    status["n_before_filter"] = len(coords)
    if clean_roi_ids is not None:
        roi_numeric = pd.to_numeric(coords["roi"], errors="coerce")
        clean_numeric = pd.to_numeric(pd.Series(clean_roi_ids), errors="coerce").dropna()
        if roi_numeric.notna().any() and len(clean_numeric):
            coords = coords[roi_numeric.isin(set(clean_numeric))].copy()
        else:
            coords = coords[coords["roi"].astype(str).isin(set(pd.Series(clean_roi_ids).astype(str)))].copy()
    status["n_after_filter"] = len(coords)
    if coords.empty:
        status.update(
            status="failed",
            message=f"{status.get('message', '')}coordinates were found, but none matched the ROI ids in the events table",
        )
    else:
        pop_note = f"; auto-detected populations: {list(auto_populations.keys())}" if auto_populations else ""
        status.update(status="ok", message=f"{status.get('message', '')}coordinates calculated from regions.df/peak{pop_note}")
    print(f"{exp_name}: calculated coordinates for {len(coords)} ROIs from rois.pkl")
    return coords, status, auto_populations


def find_halfwidth_column(rp_all):
    for col in ["mean_halfwidth", "median_halfwidth", "halfwidth_phase", "halfwidth_ACh"]:
        if col in rp_all.columns:
            return col
    raise ValueError("No halfwidth column found in ROI-phase table.")


def ordered_phases(rp_all):
    if rp_all.empty or "phase" not in rp_all.columns:
        return []
    if "phase_start" in rp_all.columns:
        phase_order = (
            rp_all.groupby("phase")["phase_start"]
            .median()
            .sort_values()
            .index
            .tolist()
        )
        return phase_order
    return sorted(rp_all["phase"].dropna().unique())


def phase_colors(phases):
    cmap = plt.get_cmap("tab10")
    return {phase: cmap(i % 10) for i, phase in enumerate(phases)}


def bimodality_summary_text(summary_df):
    if summary_df.empty:
        return "Bimodality summary:\n  No sufficient data."
    lines = ["Bimodality summary:"]
    for phase, group in summary_df.groupby("phase", sort=False):
        lines.append(f"\n{phase}:")
        for _, row in group.iterrows():
            label = row.get("genotype", "all")
            lines.append(
                f"  {label}: n={int(row['n_rois']):,} | "
                f"{row['pct_lt_0_5s']:.1f}% < 0.5s (Phasic-like) | "
                f"{row['pct_0_5_to_1s']:.1f}% 0.5-1s (Intermediate) | "
                f"{row['pct_gt_1s']:.1f}% > 1s (Sustained-like)"
            )
    return "\n".join(lines)


def plot_bimodality(rp_all, selected_phases=None, min_n=20):
    from scipy.stats import gaussian_kde
    from scipy.signal import find_peaks

    if rp_all.empty:
        raise ValueError("No ROI-phase table loaded.")
    hw_col = find_halfwidth_column(rp_all)
    phases = selected_phases or ordered_phases(rp_all)
    phases = [p for p in phases if p in set(rp_all["phase"])]
    if not phases:
        raise ValueError("No phases selected for bimodality.")
    if "genotype" in rp_all.columns:
        genotypes = [g for g in GENOTYPE_ORDER if g in set(rp_all["genotype"])]
        extras = [g for g in sorted(rp_all["genotype"].dropna().unique()) if g not in genotypes]
        genotypes += extras
    else:
        genotypes = ["all"]

    fig, axes = plt.subplots(len(phases), len(genotypes), figsize=(4 * len(genotypes), 3.5 * len(phases)), squeeze=False)
    summary_rows = []

    colors = phase_colors(phases)
    for row_i, phase in enumerate(phases):
        for col_i, genotype in enumerate(genotypes):
            ax = axes[row_i][col_i]
            mask = (rp_all["phase"] == phase) & (rp_all[hw_col] > 0)
            if genotype != "all":
                mask &= (rp_all["genotype"] == genotype)
            sub = rp_all.loc[mask, hw_col].dropna()

            if len(sub) < min_n:
                ax.text(0.5, 0.5, f"n={len(sub)}\n(insufficient)", ha="center", va="center", transform=ax.transAxes, color="gray")
                ax.set_title(f"{genotype} — {phase}")
                summary_rows.append({
                    "phase": phase,
                    "genotype": genotype,
                    "n_rois": len(sub),
                    "n_peaks": np.nan,
                    "peak_halfwidth_s": "",
                    "pct_lt_0_5s": np.nan,
                    "pct_0_5_to_1s": np.nan,
                    "pct_gt_1s": np.nan,
                })
                continue

            log_hw = np.log10(sub)
            x = np.linspace(-1.5, 2.0, 400)
            kde = gaussian_kde(log_hw, bw_method=0.2)
            kde_vals = kde(x)
            peaks, _ = find_peaks(kde_vals, distance=30, prominence=0.1)

            color = GENO_COLORS.get(genotype, colors.get(phase, "gray"))
            ax.plot(x, kde_vals, color=color, linewidth=2.5)
            ax.fill_between(x, kde_vals, alpha=0.15, color=color)
            ax.scatter(x[peaks], kde_vals[peaks], s=35, color="black", zorder=5)
            for peak in peaks:
                ax.text(x[peak], kde_vals[peak], f"{10 ** x[peak]:.2f}s", ha="center", va="bottom", fontsize=8)
            ax.axvline(np.log10(0.185), color=PHASIC_COLOR, linestyle="--", alpha=0.7, linewidth=1.5, label="WT Phasic (0.19s)")
            ax.axvline(np.log10(4.23), color=SUSTAINED_COLOR, linestyle="--", alpha=0.7, linewidth=1.5, label="WT Sustained (4.23s)")
            ax.set_title(
                f"{genotype} — {phase}\nn={len(sub):,} ROIs | {len(peaks)} peak(s)",
                fontsize=10,
                fontweight="bold",
                color="darkgreen" if len(peaks) >= 2 else "darkred",
            )
            ax.set_xlabel("log10(halfwidth, s)")
            ax.set_ylabel("Density")
            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels(["0.1s", "1s", "10s"])

            summary_rows.append({
                "phase": phase,
                "genotype": genotype,
                "n_rois": len(sub),
                "n_peaks": len(peaks),
                "peak_halfwidth_s": ", ".join(f"{10 ** x[p]:.3f}" for p in peaks),
                "pct_lt_0_5s": (sub < 0.5).mean() * 100,
                "pct_0_5_to_1s": ((sub >= 0.5) & (sub <= 1.0)).mean() * 100,
                "pct_gt_1s": (sub > 1.0).mean() * 100,
            })

    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Bimodality check: halfwidth KDE by phase", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, pd.DataFrame(summary_rows)


def plot_halfwidth_overlay(rp_all, selected_phases=None, min_n=20):
    from scipy.stats import gaussian_kde

    if rp_all.empty:
        raise ValueError("No ROI-phase table loaded.")
    hw_col = find_halfwidth_column(rp_all)
    phases = selected_phases or ordered_phases(rp_all)
    phases = [p for p in phases if p in set(rp_all["phase"])]
    if not phases:
        raise ValueError("No phases selected for overlay.")

    colors = phase_colors(phases)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    overlay_rows = []
    for phase in phases:
        hw = rp_all[(rp_all["phase"] == phase) & (rp_all[hw_col] > 0)][hw_col].dropna()
        overlay_rows.append({
            "phase": phase,
            "n_rois": len(hw),
            "median_halfwidth": hw.median() if len(hw) else np.nan,
            "q25": hw.quantile(0.25) if len(hw) else np.nan,
            "q75": hw.quantile(0.75) if len(hw) else np.nan,
        })
        if len(hw) < min_n:
            continue
        log_hw = np.log10(hw)
        x = np.linspace(-1.5, 2.0, 400)
        kde = gaussian_kde(log_hw, bw_method=0.2)
        kde_vals = kde(x)
        color = colors.get(phase, "#333")
        ax.plot(x, kde_vals, color=color, linewidth=2.5, label=f"{phase} (n={len(hw):,})")
        ax.fill_between(x, kde_vals, alpha=0.08, color=color)

    ax.axvline(np.log10(0.185), color=PHASIC_COLOR, ls="--", alpha=0.5, lw=1.5, label="Phasic ref (0.185s)")
    ax.axvline(np.log10(4.23), color=SUSTAINED_COLOR, ls="--", alpha=0.5, lw=1.5, label="Sustained ref (4.23s)")
    ax.set_xlabel("log10(halfwidth, s)")
    ax.set_ylabel("Density")
    ax.set_title("Halfwidth distribution overlay by phase", fontweight="bold")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["0.1s", "1s", "10s"])
    ax.legend(fontsize=8)
    return fig, pd.DataFrame(overlay_rows)


def plot_halfwidth_experiment_overlay(rp_all, selected_phase, min_n=10):
    from scipy.stats import gaussian_kde
    import matplotlib.cm as cm

    if rp_all.empty:
        raise ValueError("No ROI-phase table loaded.")
    hw_col = find_halfwidth_column(rp_all)
    
    # Filter for the selected phase and active halfwidth values
    sub_df = rp_all[(rp_all["phase"] == selected_phase) & (rp_all[hw_col] > 0)].copy()
    if sub_df.empty:
        raise ValueError(f"No data available for the phase: {selected_phase}")
        
    experiments = sorted(sub_df["exp_name"].dropna().unique())
    if not experiments:
        raise ValueError("No experiments found for this phase.")
        
    cmap = cm.get_cmap("tab10") if len(experiments) <= 10 else cm.get_cmap("tab20")
    
    fig, ax = plt.subplots(figsize=(6.5, 4))
    overlay_rows = []
    
    for i, exp in enumerate(experiments):
        hw = sub_df[sub_df["exp_name"] == exp][hw_col].dropna()
        overlay_rows.append({
            "experiment": exp,
            "phase": selected_phase,
            "n_rois": len(hw),
            "median_halfwidth": hw.median() if len(hw) else np.nan,
            "q25": hw.quantile(0.25) if len(hw) else np.nan,
            "q75": hw.quantile(0.75) if len(hw) else np.nan,
        })
        if len(hw) < min_n:
            continue
            
        log_hw = np.log10(hw)
        x = np.linspace(-1.5, 2.0, 400)
        
        # Ensure log_hw has enough variance for KDE
        if log_hw.nunique() <= 1:
            continue
            
        try:
            kde = gaussian_kde(log_hw, bw_method=0.2)
            kde_vals = kde(x)
            color = cmap(i % 20)
            ax.plot(x, kde_vals, color=color, linewidth=2.5, label=f"{exp} (n={len(hw):,})")
            ax.fill_between(x, kde_vals, alpha=0.06, color=color)
        except Exception:
            pass

    ax.axvline(np.log10(0.185), color=PHASIC_COLOR, ls="--", alpha=0.5, lw=1.5, label="Phasic ref (0.185s)")
    ax.axvline(np.log10(4.23), color=SUSTAINED_COLOR, ls="--", alpha=0.5, lw=1.5, label="Sustained ref (4.23s)")
    ax.set_xlabel("log10(halfwidth, s)")
    ax.set_ylabel("Density")
    ax.set_title(f"Experiment overlay for {selected_phase}", fontweight="bold")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["0.1s", "1s", "10s"])
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    
    return fig, pd.DataFrame(overlay_rows)


def plot_experiment_comparison(rp_all, metric_col, phase=None):
    if rp_all is None or rp_all.empty:
        raise ValueError("No ROI-phase table loaded.")
    if metric_col not in rp_all.columns:
        raise ValueError(f"{metric_col} is not available.")
    data = rp_all.copy()
    if phase:
        data = data[data["phase"] == phase].copy()
    data = data[data[metric_col].notna()].copy()
    if data.empty:
        raise ValueError("No data available for this comparison.")

    experiments = sorted(data["exp_name"].dropna().unique())
    phases = ordered_phases(data) if not phase else [phase]
    colors = phase_colors(phases)
    ylabel = f"log10({metric_col})" if "halfwidth" in metric_col.lower() else metric_col
    title_phase = phase or "all selected phases"

    ncols = min(3, len(experiments))
    nrows = math.ceil(len(experiments) / ncols)
    subplot_w = max(3.5, 1.0 * len(phases) + 1.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_w * ncols, 3.5 * nrows),
                             sharey=True)
    axes = list(np.atleast_1d(axes).ravel())

    for ax, exp_name in zip(axes, experiments):
        exp_data = data[data["exp_name"] == exp_name]
        box_data, positions, labels, facecolors = [], [], [], []
        pos = 1
        for ph in phases:
            vals = exp_data.loc[exp_data["phase"] == ph, metric_col].dropna()
            if len(vals):
                box_data.append(vals.values)
                positions.append(pos)
                labels.append(ph)
                facecolors.append(colors.get(ph, "#777"))
                pos += 1
        if box_data:
            plot_vals = [np.log10(np.clip(v, 1e-6, None)) if "halfwidth" in metric_col.lower() else v
                         for v in box_data]
            bp = ax.boxplot(plot_vals, positions=positions, patch_artist=True, showfliers=False, widths=0.7)
            for patch, color in zip(bp["boxes"], facecolors):
                patch.set_facecolor(color)
                patch.set_alpha(0.35)
                patch.set_edgecolor("#3a3d45")
            for med in bp["medians"]:
                med.set_color("#111827")
                med.set_linewidth(2)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#aaa")
        ax.set_title(exp_name, fontweight="bold", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(experiments):]:
        ax.set_visible(False)

    fig.suptitle(f"Experiment pooling comparison — {metric_col} ({title_phase})", fontweight="bold")
    fig.tight_layout()

    summary = (
        data.groupby(["exp_name", "phase"])[metric_col]
        .agg(n="count", median="median", mean="mean")
        .reset_index()
    )
    return fig, summary


def plot_active_roi_fraction(rp_all, coords=None, threshold=0.0):
    """Bar chart: fraction of ROIs responding per phase.

    Denominator = total ROIs per experiment from coords (if available).
    If coords is None or empty, the denominator is the max ROI count seen
    across all phases for that experiment (a conservative estimate).

    X-axis = phases (biological reading: "at this stimulus, how many cells respond?").
    Bars grouped by experiment.
    """
    if rp_all.empty or "event_rate" not in rp_all.columns:
        raise ValueError("event_rate column not found in ROI-phase data.")
    phases = ordered_phases(rp_all)
    exp_names = sorted(rp_all["exp_name"].dropna().unique())

    # Build denominator: total ROIs per experiment
    roi_totals = {}
    if coords is not None and not coords.empty and "roi" in coords.columns:
        for exp in exp_names:
            n = int(coords[coords["exp_name"] == exp]["roi"].nunique())
            roi_totals[exp] = n if n > 0 else None
    # Fallback: max ROIs seen across all phases for that experiment
    for exp in exp_names:
        if roi_totals.get(exp) is None:
            max_seen = int(rp_all[rp_all["exp_name"] == exp].groupby("phase")["roi"].nunique().max())
            roi_totals[exp] = max_seen if max_seen > 0 else 1

    rows = []
    for phase in phases:
        sub = rp_all[rp_all["phase"] == phase]
        for exp in exp_names:
            ep = sub[sub["exp_name"] == exp]
            n_total = roi_totals.get(exp, 1)
            # ROIs present in rp for this phase with event_rate above threshold
            n_active = int((ep["event_rate"] > threshold).sum()) if not ep.empty else 0
            rows.append({
                "phase": phase, "exp_name": exp,
                "n_total": n_total, "n_active": n_active,
                "fraction": n_active / n_total,
            })

    if not rows:
        raise ValueError("No data to plot.")
    df = pd.DataFrame(rows)
    denom_note = "of all ROIs in field" if (coords is not None and not coords.empty) else "of max seen across phases"

    cmap_phases = phase_colors(phases)
    ncols = min(3, len(exp_names))
    nrows = math.ceil(len(exp_names) / ncols)
    subplot_w = max(3.5, len(phases) * 0.9 + 1.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_w * ncols, 3.2 * nrows),
                             sharey=True)
    axes = list(np.atleast_1d(axes).ravel())

    x = np.arange(len(phases))
    for ax, exp in zip(axes, exp_names):
        sub = df[df["exp_name"] == exp]
        fracs = [float(sub[sub["phase"] == p]["fraction"].values[0])
                 if not sub[sub["phase"] == p].empty else 0.0
                 for p in phases]
        bar_colors = [cmap_phases.get(p, "#999") for p in phases]
        ax.bar(x, fracs, width=0.7, color=bar_colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Fraction responding")
        ax.set_ylim(0, 1.08)
        ax.axhline(1.0, color="#ccc", linewidth=0.8, linestyle="--")
        ax.set_title(exp, fontweight="bold", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(exp_names):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Responding ROI Fraction per Phase\n(event_rate > {threshold}, denominator = {denom_note})",
        fontweight="bold", fontsize=10,
    )
    fig.tight_layout()

    summary = df.pivot_table(index="exp_name", columns="phase", values="fraction", aggfunc="first")
    summary = summary.reindex(columns=[p for p in phases if p in summary.columns])
    summary_counts = df.pivot_table(index="exp_name", columns="phase", values="n_active", aggfunc="first")
    summary_counts = summary_counts.reindex(columns=[p for p in phases if p in summary_counts.columns])
    summary_counts.columns = [f"{c}_n" for c in summary_counts.columns]
    out = pd.concat([summary, summary_counts], axis=1).reset_index()
    return fig, out


def plot_event_rate_heatmap(rp_all):
    """Heatmap of median event rate: experiments × phases."""
    if rp_all.empty or "event_rate" not in rp_all.columns:
        raise ValueError("event_rate column not found.")
    phases = ordered_phases(rp_all)
    pivot = rp_all.pivot_table(index="exp_name", columns="phase",
                                values="event_rate", aggfunc="median")
    pivot = pivot.reindex(columns=[p for p in phases if p in pivot.columns])
    if pivot.empty:
        raise ValueError("No data for heatmap.")
    data = pivot.values.astype(float)
    finite = data[np.isfinite(data)]
    vmax = float(np.nanpercentile(finite, 95)) if len(finite) else 1.0
    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 1.1),
                                    max(2.5, len(pivot) * 0.55 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                txt_color = "white" if val > vmax * 0.65 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6.5, color=txt_color)
    fig.colorbar(im, ax=ax, label="Median event rate")
    ax.set_title("Median Event Rate — Experiments × Phases", fontweight="bold")
    fig.tight_layout()
    return fig, pivot.reset_index()


def plot_phase_transitions(rp_all, metric_col="event_rate"):
    """Line plot: mean ± SEM of metric across ordered phases, one line per experiment."""
    if rp_all.empty or metric_col not in rp_all.columns:
        raise ValueError(f"{metric_col} not found in ROI-phase data.")
    phases = ordered_phases(rp_all)
    exp_names = sorted(rp_all["exp_name"].dropna().unique())
    cmap = plt.cm.tab10(np.linspace(0, 1, max(1, len(exp_names))))
    fig, ax = plt.subplots(figsize=(max(5, len(phases) * 1.0), 3.8))
    rows = []
    for i, exp in enumerate(exp_names):
        sub = rp_all[rp_all["exp_name"] == exp]
        xs, means, sems = [], [], []
        for xi, p in enumerate(phases):
            vals = sub[sub["phase"] == p][metric_col].dropna()
            if len(vals) >= 2:
                xs.append(xi)
                means.append(float(vals.mean()))
                sems.append(float(vals.sem()))
                rows.append({"exp_name": exp, "phase": p, "mean": vals.mean(),
                              "sem": vals.sem(), "n": len(vals)})
        if means:
            ax.plot(xs, means, "o-", color=cmap[i], label=exp,
                    linewidth=1.8, markersize=5, zorder=3)
            ax.fill_between(xs,
                             np.array(means) - np.array(sems),
                             np.array(means) + np.array(sems),
                             color=cmap[i], alpha=0.12, zorder=2)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, rotation=35, ha="right", fontsize=8)
    ylabel = f"log10({metric_col})" if "halfwidth" in metric_col.lower() else metric_col
    if "halfwidth" in metric_col.lower():
        ax.set_yscale("log")
        ylabel = metric_col
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc="best", ncol=max(1, len(exp_names) // 6))
    ax.set_title(f"Phase Transitions — mean ± SEM ({metric_col})", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    summary = pd.DataFrame(rows) if rows else pd.DataFrame()
    return fig, summary


def compute_roi_count_table(rp_all):
    """Pivot table: experiments × phases → ROI count."""
    if rp_all.empty:
        return pd.DataFrame()
    phases = ordered_phases(rp_all)
    pivot = (rp_all.groupby(["exp_name", "phase"])["roi"]
             .count()
             .unstack("phase"))
    pivot = pivot.reindex(columns=[p for p in phases if p in pivot.columns])
    pivot["Total"] = pivot.sum(axis=1)
    return pivot.reset_index()


def build_phase_windows(protocol_df):
    rows = []
    for exp_name, group in protocol_df.groupby("exp_name"):
        group = group.sort_values("t_begin_s").reset_index(drop=True)
        for i, row in group.iterrows():
            rows.append({
                "exp_name": exp_name,
                "phase": f"{row['concentration']} {row['compound']}",
                "compound": row["compound"],
                "concentration": row["concentration"],
                "phase_start": row["t_begin_s"],
                "phase_end": group.loc[i + 1, "t_begin_s"] if i < len(group) - 1 else np.inf,
                "is_first_phase": i == 0,
            })
    return pd.DataFrame(rows)


def assign_phases(events_df, phase_windows, transition_sec):
    result = []
    for exp_name, exp_events in events_df.groupby("exp_name"):
        exp_events = exp_events.copy()
        exp_events["phase"] = np.nan
        exp_events["phase_start"] = np.nan
        exp_events["transition"] = False
        for _, ph in phase_windows[phase_windows["exp_name"] == exp_name].iterrows():
            in_phase = (exp_events["peakpoint"] >= ph["phase_start"]) & (exp_events["peakpoint"] < ph["phase_end"])
            exp_events.loc[in_phase, "phase"] = ph["phase"]
            exp_events.loc[in_phase, "phase_start"] = ph["phase_start"]
            if not ph["is_first_phase"]:
                exp_events.loc[in_phase & (exp_events["peakpoint"] < ph["phase_start"] + transition_sec), "transition"] = True
        result.append(exp_events)
    return pd.concat(result, ignore_index=True) if result else pd.DataFrame()


def compute_roi_metrics_per_phase(events_df, phase_windows, scope_map, transition_sec):
    rows = []
    for (exp_name, phase), group in events_df.dropna(subset=["phase"]).groupby(["exp_name", "phase"]):
        ph = phase_windows[(phase_windows["exp_name"] == exp_name) & (phase_windows["phase"] == phase)]
        if ph.empty:
            continue
        phase_start = ph["phase_start"].iloc[0]
        phase_end = ph["phase_end"].iloc[0]
        if np.isinf(phase_end):
            phase_end = group["peakpoint"].max()
        stable_min = max(phase_end - phase_start - transition_sec, 0) / 60
        if stable_min <= 0:
            continue
        stable = group[group["transition"] == False]
        if stable.empty:
            continue
        agg = {
            "n_events": ("roi", "size"),
            "mean_halfwidth": ("halfwidth", "mean"),
            "median_halfwidth": ("halfwidth", "median"),
        }
        for col in ["z_max", "auc", "height"]:
            if col in stable.columns:
                agg[f"mean_{col}"] = (col, "mean")
        metrics = stable.groupby("roi").agg(**agg).reset_index()
        metrics["event_rate"] = metrics["n_events"] / stable_min
        metrics["cell_activation_rate"] = metrics["mean_halfwidth"] * metrics["event_rate"]
        metrics["exp_name"] = exp_name
        metrics["phase"] = phase
        metrics["phase_start"] = phase_start
        metrics["T_stable_min"] = round(stable_min, 2)
        meta_cols = [c for c in ["genotype", "experiment", "letter", "sex"] if c in events_df.columns]
        meta = events_df[events_df["exp_name"] == exp_name][["exp_name"] + meta_cols].drop_duplicates()
        metrics = metrics.merge(meta, on="exp_name", how="left")
        rows.append(metrics)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out["scope"] = out["exp_name"].map(scope_map)
    return out


def prepare_coords(coords):
    if coords.empty or not {"x", "y"}.issubset(coords.columns):
        return coords
    coords = coords.copy()
    coords["x_centered"] = coords.groupby("exp_name")["x"].transform(lambda s: s - s.median())
    coords["y_centered"] = coords.groupby("exp_name")["y"].transform(lambda s: s - s.median())
    coords["radial_dist"] = np.sqrt(coords["x_centered"] ** 2 + coords["y_centered"] ** 2)
    return coords


def apply_y_filter(df, coords, scope_map, y_threshold):
    if coords.empty or "y_centered" not in coords.columns:
        warnings.warn("No coordinates available; nd2-only Y-filter was skipped.")
        return df.copy()
    valid_nd2 = (
        coords[coords["exp_name"].map(scope_map) == "nd2"]
        .loc[lambda d: d["y_centered"].abs() <= y_threshold, ["exp_name", "roi"]]
    )
    non_nd2 = df[df["exp_name"].map(scope_map) != "nd2"]
    nd2 = df[df["exp_name"].map(scope_map) == "nd2"]
    nd2_filt = nd2.merge(valid_nd2, on=["exp_name", "roi"], how="inner")
    return pd.concat([non_nd2, nd2_filt], ignore_index=True)


def pick_best_ach(df, value_cols, rename_map, ach_priority=None):
    if ach_priority is None:
        ach_priority = ACH_PRIORITY
    rows = []
    for exp_name, exp_df in df.groupby("exp_name"):
        for phase in ach_priority:
            sub = exp_df[exp_df["phase"] == phase]
            if not sub.empty:
                rows.append(sub[["exp_name", "roi"] + value_cols].copy().assign(ach_phase_used=phase))
                break
    if not rows:
        return pd.DataFrame(columns=["exp_name", "roi"] + list(rename_map.values()))
    return pd.concat(rows, ignore_index=True).rename(columns=rename_map)


def compute_latency(events_df, ach_priority=None):
    if ach_priority is None:
        ach_priority = ACH_PRIORITY
    rows = []
    stable_ach = events_df[(events_df["phase"].isin(ach_priority)) & (events_df["transition"] == False)]
    for (exp_name, roi), grp in stable_ach.groupby(["exp_name", "roi"]):
        for phase in ach_priority:
            sub = grp[grp["phase"] == phase]
            if not sub.empty:
                rows.append({
                    "exp_name": exp_name,
                    "roi": roi,
                    "latency_ACh_s": sub["peakpoint"].min() - sub["phase_start"].iloc[0],
                })
                break
    return pd.DataFrame(rows)


def compute_cv(events_df, ach_priority=None):
    if ach_priority is None:
        ach_priority = ACH_PRIORITY
    rows = []
    stable_ach = events_df[(events_df["phase"].isin(ach_priority)) & (events_df["transition"] == False)]
    for (exp_name, roi), grp in stable_ach.groupby(["exp_name", "roi"]):
        for phase in ach_priority:
            sub = grp[grp["phase"] == phase]
            if sub.empty:
                continue
            peaks = sub["peakpoint"].sort_values().to_numpy()
            if len(peaks) >= 3:
                dt = np.diff(peaks)
                if dt.mean() != 0:
                    rows.append({"exp_name": exp_name, "roi": roi, "cv_dt_ACh": dt.std() / dt.mean()})
            break
    return pd.DataFrame(rows)


def run_raw_pipeline(experiments, protocol_df, data_root, transition_sec=200, y_threshold=150, run_gmm=True, gmm_pop_filter="Acinar cells only", gmm_ach_priority=None):
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.mixture import GaussianMixture
    except Exception as exc:
        raise ImportError("Raw pipeline needs scikit-learn in the lab Python environment.") from exc

    events_rows, coords_rows, nb_rows, coord_status_rows = [], [], [], []
    roi_lists_per_exp = {}
    protocol_df = normalize_protocol_table(protocol_df, "")
    protocol_exp_names = set(protocol_df["exp_name"].astype(str))

    for item in experiments:
        genotype = item["genotype"]
        exp_name = item["exp_name"]
        path_to_rois = resolve_rois_path(item["pathToRois"])
        path_to_events = normalize_path(item.get("pathToEvents") or derive_events_path(path_to_rois))
        scope = item.get("scope") or infer_scope(path_to_rois)
        sex = item.get("sex") or infer_sex(exp_name)

        require_file(path_to_events)
        if exp_name not in protocol_exp_names:
            raise ValueError(f"{exp_name}: protocol table has no rows for this exp_name.")

        ev = pd.read_csv(path_to_events)
        ev["genotype"] = genotype
        ev["exp_name"] = exp_name
        ev["experiment"] = infer_experiment_number(exp_name)
        ev["letter"] = item.get("letter", re.sub(r"\d+", "", exp_name.split("_")[0]))
        ev["sex"] = sex
        ev["scope"] = scope
        events_rows.append(ev)

        # Collect user-supplied islet/acinar ROI ID lists for this experiment.
        manual_islet = item.get("islet_rois", [])
        manual_acinar = item.get("acinar_rois", [])
        if manual_islet or manual_acinar:
            roi_lists_per_exp[exp_name] = {"islet": manual_islet, "acinar": manual_acinar}

        path_to_coords = item.get("pathToCoords", "")
        if path_to_coords:
            co = pd.read_csv(normalize_path(path_to_coords))
            if not {"roi", "x", "y"}.issubset(co.columns):
                raise ValueError(f"{exp_name}: coordinate file must contain roi, x, y columns.")
            co = co[["roi", "x", "y"]].drop_duplicates().copy()
            co["exp_name"] = exp_name
            co["genotype"] = genotype
            co["scope"] = scope
            coords_rows.append(co)
            coord_status_rows.append({
                "exp_name": exp_name,
                "source": "pathToCoords",
                "status": "ok",
                "n_before_filter": len(co),
                "n_after_filter": len(co),
                "message": "coordinates loaded from user-provided CSV",
            })
        elif {"x", "y", "roi"}.issubset(ev.columns):
            co = ev[["exp_name", "roi", "x", "y"]].drop_duplicates().copy()
            co["genotype"] = genotype
            co["scope"] = scope
            coords_rows.append(co)
            coord_status_rows.append({
                "exp_name": exp_name,
                "source": "events CSV",
                "status": "ok",
                "n_before_filter": len(co),
                "n_after_filter": len(co),
                "message": "coordinates loaded from x/y columns in events",
            })
        else:
            clean_roi_ids = ev["roi"].dropna().unique() if "roi" in ev.columns else None
            co, coord_status, auto_pops = load_coordinates_from_rois(path_to_rois, exp_name, genotype, scope, clean_roi_ids=clean_roi_ids)
            coord_status_rows.append(coord_status)
            if not co.empty:
                coords_rows.append(co)
            elif scope == "nd2":
                warnings.warn(
                    f"{exp_name}: coordinates could not be calculated automatically. "
                    "The nd2 Y-filter and spatial maps require ROI coordinates."
                )
            # Use auto-detected populations only when user did not provide lists.
            if auto_pops and exp_name not in roi_lists_per_exp:
                roi_lists_per_exp[exp_name] = auto_pops

        nb_rows.append({
            "genotype": genotype,
            "exp_name": exp_name,
            "sex": sex,
            "scope": scope,
            "pathToRois": str(path_to_rois),
            "pathToEvents": str(path_to_events),
        })

    events = pd.concat(events_rows, ignore_index=True)
    coords = prepare_coords(pd.concat(coords_rows, ignore_index=True) if coords_rows else pd.DataFrame())
    if roi_lists_per_exp and not coords.empty:
        coords = tag_coords_with_population(coords, roi_lists_per_exp)
    nb_info = pd.DataFrame(nb_rows)
    scope_map = nb_info.set_index("exp_name")["scope"].to_dict()

    events_clean = events.copy()
    for genotype, excluded in DEFAULT_EXCLUSIONS.items():
        events_clean = events_clean[~((events_clean["genotype"] == genotype) & (events_clean["exp_name"].isin(excluded)))].copy()
    events_filt = apply_y_filter(events_clean, coords, scope_map, y_threshold)
    phase_windows = build_phase_windows(protocol_df)
    events_filt = assign_phases(events_filt, phase_windows, transition_sec)
    roi_phase = compute_roi_metrics_per_phase(events_filt, phase_windows, scope_map, transition_sec)

    cluster_tables = {}
    gmm_diagnostics = {}  # genotype → dict with info about why GMM ran or was skipped
    for genotype in sorted(roi_phase["genotype"].dropna().unique()):
        rp = roi_phase[roi_phase["genotype"] == genotype].copy()
        ev = events_filt[events_filt["genotype"] == genotype].copy()
        if genotype == "HET":
            rp = rp[rp["scope"].isin(["nd2", "lif"])].copy()
            ev = ev[ev["scope"].isin(["nd2", "lif"])].copy()

        # Cell population filtering for GMM using roi_lists_per_exp
        if run_gmm and roi_lists_per_exp:
            if gmm_pop_filter == "Islet cells only":
                allowed_pops = ["islet"]
            elif gmm_pop_filter == "Both (Acinar + Islet)":
                allowed_pops = ["acinar", "islet"]
            else: # Default: Acinar cells only
                allowed_pops = ["acinar"]
            
            # Build valid keys
            valid_keys_str = set()
            valid_keys_num = set()
            for exp_name, pops in roi_lists_per_exp.items():
                for pop in allowed_pops:
                    for roi in pops.get(pop, []):
                        valid_keys_str.add((str(exp_name).strip(), str(roi).strip()))
                        try:
                            valid_keys_num.add((str(exp_name).strip(), float(roi)))
                        except:
                            pass
            
            def is_valid_roi(row):
                exp_key = str(row["exp_name"]).strip()
                roi_key = row["roi"]
                try:
                    return (exp_key, str(roi_key).strip()) in valid_keys_str or (exp_key, float(roi_key)) in valid_keys_num
                except:
                    return (exp_key, str(roi_key).strip()) in valid_keys_str
            
            if not rp.empty:
                rp = rp[rp.apply(is_valid_roi, axis=1)].copy()
            if not ev.empty:
                ev = ev[ev.apply(is_valid_roi, axis=1)].copy()

        phases_present = sorted(rp["phase"].dropna().unique().tolist())
        base = rp[rp["phase"] == "8mM Glucose"][["exp_name", "roi", "event_rate"]].rename(columns={"event_rate": "event_rate_8mM"})
        ach = pick_best_ach(
            rp,
            ["event_rate", "mean_halfwidth"],
            {"event_rate": "event_rate_ACh", "mean_halfwidth": "halfwidth_ACh"},
            ach_priority=gmm_ach_priority,
        ).drop(columns="ach_phase_used", errors="ignore")
        fv = (
            base.merge(ach, on=["exp_name", "roi"], how="outer")
            .merge(compute_latency(ev, ach_priority=gmm_ach_priority), on=["exp_name", "roi"], how="left")
            .merge(compute_cv(ev, ach_priority=gmm_ach_priority), on=["exp_name", "roi"], how="left")
        )
        meta = rp[["exp_name", "experiment", "letter", "sex", "scope", "genotype"]].drop_duplicates()
        fv = fv.merge(meta, on="exp_name", how="left")
        fv_clean = pd.DataFrame()
        if run_gmm:
            for col in ["event_rate_8mM", "event_rate_ACh", "halfwidth_ACh", "latency_ACh_s"]:
                fv[f"{col}_log"] = np.log1p(fv[col])
            # Core features are required; cv_dt_ACh is optional (needs ≥3 events per ROI).
            core_cols = ["event_rate_8mM_log", "event_rate_ACh_log", "halfwidth_ACh_log", "latency_ACh_s_log"]
            fv_clean = fv.dropna(subset=core_cols).copy()
            feature_cols = core_cols + (["cv_dt_ACh"] if "cv_dt_ACh" in fv_clean.columns and fv_clean["cv_dt_ACh"].notna().any() else [])
            fv_clean = fv_clean.dropna(subset=feature_cols).copy()
            n_qualifying = len(fv_clean)
            priority_list = gmm_ach_priority if gmm_ach_priority is not None else ACH_PRIORITY
            gmm_diagnostics[genotype] = {
                "phases_present": phases_present,
                "n_8mM_glucose_rois": len(base),
                "n_ach_rois": len(ach),
                "n_qualifying": n_qualifying,
                "has_8mM_glucose": "8mM Glucose" in phases_present,
                "has_ach": any(p in phases_present for p in priority_list),
                "cv_dt_used": "cv_dt_ACh" in feature_cols,
                "priority_list": priority_list,
            }
            if n_qualifying >= 20:
                X = StandardScaler().fit_transform(fv_clean[feature_cols])
                gm = GaussianMixture(n_components=2, covariance_type="full", n_init=20, random_state=42)
                gm.fit(X)
                probs = gm.predict_proba(X)
                fv_clean["cluster"] = gm.predict(X)
                fv_clean["max_prob"] = probs.max(axis=1).round(3)
                med = fv_clean.groupby("cluster")["halfwidth_ACh"].median()
                fv_clean["cell_type"] = fv_clean["cluster"].map({med.idxmin(): "Phasic", med.idxmax(): "Sustained"})
                gmm_diagnostics[genotype]["gmm_ran"] = True
            else:
                gmm_diagnostics[genotype]["gmm_ran"] = False
            if not fv_clean.empty and "cell_type" in fv_clean.columns:
                cluster_tables[genotype] = fv_clean

        out = output_dir_for(data_root, genotype)
        events_clean[events_clean["genotype"] == genotype].to_csv(out / f"{genotype}_all_Events_acinar_clean.csv", index=False)
        events_filt[events_filt["genotype"] == genotype].to_csv(out / f"{genotype}_all_Events_acinar_clean_yfiltered.csv", index=False)
        roi_phase[roi_phase["genotype"] == genotype].to_csv(out / f"{genotype}_roi_metrics_per_phase.csv", index=False)
        fv.to_csv(out / f"{genotype}_feature_vector.csv", index=False)
        filename = "WT_gmm_clusters_global.csv" if genotype == "WT" else f"{genotype}_cluster_assignments.csv"
        if not fv_clean.empty:
            fv_clean.to_csv(out / filename, index=False)
        if not coords.empty:
            coords[coords["genotype"] == genotype].to_csv(out / f"{genotype}_roi_coordinates.csv", index=False)
        nb_info[nb_info["genotype"] == genotype].to_csv(out / "notebook_paths_summary.csv", index=False)

    all_clusters = pd.concat(cluster_tables.values(), ignore_index=True) if cluster_tables else pd.DataFrame()
    coord_status = pd.DataFrame(coord_status_rows)
    return all_clusters, roi_phase, coords, coord_status, gmm_diagnostics


def parse_experiment_table(df):
    rows = []
    errors = []
    for _, row in df.iterrows():
        if not str(row.get("pathToRois", "")).strip():
            continue
        missing = [
            col for col in ["genotype", "exp_name", "pathToRois"]
            if not str(row.get(col, "")).strip()
        ]
        if missing:
            errors.append(f"Row with pathToRois={row.get('pathToRois', '')}: missing {', '.join(missing)}")
            continue
        _path_rois = str(row["pathToRois"]).strip()
        rows.append({
            "genotype": str(row["genotype"]).strip(),
            "exp_name": str(row["exp_name"]).strip(),
            "sex": str(row.get("sex", "") or "").strip() or infer_sex(row["exp_name"]),
            "scope": str(row.get("scope", "") or "").strip() or infer_scope(_path_rois),
            "pathToRois": _path_rois,
            "pathToEvents": str(row.get("pathToEvents", "") or "").strip() or derive_events_path(_path_rois),
            "pathToCoords": str(row.get("pathToCoords", "") or "").strip(),
            "islet_rois": parse_roi_id_list(row.get("islet_rois", "")),
            "acinar_rois": parse_roi_id_list(row.get("acinar_rois", "")),
        })
    return rows, errors


def validate_protocol_editor(protocol_df):
    if protocol_df is None or protocol_df.empty:
        return pd.DataFrame(), ["Protocol table is empty."]
    try:
        protocol = normalize_protocol_table(protocol_df, "")
    except Exception as exc:
        return pd.DataFrame(), [str(exc)]

    errors = []
    if protocol["exp_name"].fillna("").astype(str).str.strip().eq("").any():
        errors.append("Protocol table: exp_name is required in every row.")
    if protocol["t_begin_s"].isna().any():
        errors.append("Protocol table: t_begin must be numeric seconds.")
    if errors:
        return pd.DataFrame(), errors
    return protocol, []


def render_run_overview(all_data, rp_all, coords, output_dir):
    n_clustered = len(all_data) if all_data is not None and not all_data.empty else 0
    n_roi_phase = len(rp_all) if rp_all is not None and not rp_all.empty else 0
    n_experiments = 0
    n_phases = 0
    if rp_all is not None and not rp_all.empty:
        n_experiments = rp_all["exp_name"].nunique() if "exp_name" in rp_all.columns else 0
        n_phases = rp_all["phase"].nunique() if "phase" in rp_all.columns else 0
    elif all_data is not None and not all_data.empty:
        n_experiments = all_data["exp_name"].nunique() if "exp_name" in all_data.columns else 0

    n_coords = len(coords) if coords is not None and not coords.empty else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Experiments", f"{n_experiments:,}")
    c2.metric("Phases", f"{n_phases:,}")
    c3.metric("ROI-phase rows", f"{n_roi_phase:,}")
    c4.metric("Clustered ROIs", f"{n_clustered:,}")
    c5.metric("Coordinate rows", f"{n_coords:,}")
    st.caption(f"Output folder: {output_dir}")


def render_phase_coverage(rp_all):
    if rp_all is None or rp_all.empty or "phase" not in rp_all.columns:
        st.info("No ROI-phase table is available.")
        return
    agg = {"n_ROIs": ("roi", "nunique")}
    if "n_events" in rp_all.columns:
        agg["total_events"] = ("n_events", "sum")
    hw_col = find_halfwidth_column(rp_all)
    agg["median_halfwidth"] = (hw_col, "median")
    if "event_rate" in rp_all.columns:
        agg["median_event_rate"] = ("event_rate", "median")
    if "T_stable_min" in rp_all.columns:
        agg["T_stable_min"] = ("T_stable_min", "first")
    coverage = rp_all.groupby("phase").agg(**agg).reset_index()
    order = {phase: i for i, phase in enumerate(ordered_phases(rp_all))}
    coverage["order"] = coverage["phase"].map(order)
    coverage = coverage.sort_values("order").drop(columns="order")
    st.dataframe(coverage.round(3), use_container_width=True)


st.set_page_config(page_title="Calcium analysis", layout="wide")
st.title("Calcium analysis")
st.caption(
    "Clean calcium-imaging analysis for one experiment or pooled WT/HET/MUT datasets. "
    "Fields marked with * are required."
)
st.sidebar.caption(f"App version: {APP_VERSION}")


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    div[data-testid="stMetric"] {
        background: #f7f8fb;
        border: 1px solid #e5e7ef;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7ef;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.65rem 1rem;
    }
    .section-note {
        color: #5c6270;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    .quiet-panel {
        background: #f7f8fb;
        border: 1px solid #e5e7ef;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

mode = st.sidebar.radio(
    "Input mode",
    [
        "Raw rois/events paths",
        "Existing pooled_results folders",
    ],
    help=(
        "Use raw rois/events paths to start from events files plus a manually entered protocol table. "
        "Use existing pooled_results when cluster/ROI-phase CSVs already exist."
    ),
)

import getpass as _getpass
import os as _os
_default_user = _os.environ.get("JUPYTERHUB_USER") or _os.environ.get("USER") or _getpass.getuser()
_default_data_root = f"/data/{_default_user}" if os.path.exists(f"/data/{_default_user}") else "/data"

data_root = Path(st.sidebar.text_input(
    "DATA_ROOT *",
    _default_data_root,
    help=(
        "Base folder for output. Existing pooled_results mode reads from the folders below "
        "and writes comparison outputs to DATA_ROOT/comparison_clean_outputs. "
        "Raw mode also writes pooled_results under DATA_ROOT/pooling_<GENOTYPE>/pooled_results."
    ),
))
output_dir = data_root / "comparison_clean_outputs"

with st.sidebar.expander("Environment check", expanded=False):
    st.caption("Shows the exact Python environment used by this Streamlit app.")
    env_df = environment_diagnostics()
    st.dataframe(env_df, use_container_width=True, hide_index=True, height=260)
    if pd.__version__ != "1.3.5":
        st.warning(
            "This Streamlit process is not using pandas 1.3.5. "
            "Old rois.pkl files may fail to load even if they work in a notebook kernel."
        )
    test_path = st.text_input(
        "Test one pathToRois",
        "",
        help="Paste a full path to 5.6_rois.pkl. This loads the file exactly as the app does.",
    )
    if st.button("Test ROI loading"):
        if not test_path.strip():
            st.error("Paste a pathToRois first.")
        else:
            info, preview = test_rois_loading(test_path)
            st.dataframe(pd.DataFrame([info]), use_container_width=True, hide_index=True)
            if info["status"] == "loaded" and not preview.empty:
                st.dataframe(preview, use_container_width=True, height=180)

if mode == "Existing pooled_results folders":
    st.sidebar.info(
        "Use this mode when the pooled CSV files already exist. "
        "It reads the cluster/ROI-phase tables and creates comparison tables and figures."
    )
    pooled_dirs = {
        genotype: Path(st.sidebar.text_input(
            f"{genotype} pooled_results *",
            str(data_root / f"pooling_{genotype}" / "pooled_results"),
            help=(
                f"Folder containing {CLUSTER_FILES[genotype]}, "
                f"{ROI_PHASE_FILES[genotype]}, and optional coordinate/event CSVs."
            ),
        ))
        for genotype in GENOTYPE_ORDER
    }
    run = st.sidebar.button("Load and create tables", type="primary")
    if run:
        with st.spinner("Loading pooled_results..."):
            all_data, _ = load_existing_cluster_tables(pooled_dirs)
            rp_all = load_roi_phase_tables(pooled_dirs)
            coords = load_coordinate_tables(pooled_dirs)
            st.session_state["all_data"] = all_data
            st.session_state["rp_all"] = rp_all
            st.session_state["coords"] = coords
            st.session_state["coord_status"] = pd.DataFrame()
            st.session_state["pooled_dirs"] = pooled_dirs

else:
    st.sidebar.info(
        "Use this for one experiment or a full pooling batch. "
        "Required columns: genotype, exp_name, pathToRois. "
        "pathToEvents is derived automatically when left empty. "
        "Protocol rows are required below."
    )
    st.warning(
        "Raw mode writes new pooled_results CSVs under DATA_ROOT. "
        f"For a first test, choose a separate DATA_ROOT such as /data/{_default_user}/CLEAN_PIPELINES/test_outputs."
    )
    run_gmm = st.sidebar.checkbox(
        "Run GMM population analysis",
        value=False,
        help=(
            "Use this for WT/HET/MUT comparison datasets. "
            "For pharmacology/single-experiment phase analysis, leave it off and use the phase KDE/overlay outputs."
        ),
    )

    if run_gmm:
        gmm_pop_filter = st.sidebar.selectbox(
            "GMM Cell Population",
            options=["Acinar cells only", "Islet cells only", "Both (Acinar + Islet)"],
            index=0,
            help="Select which cell population to cluster using GMM.",
        )
        if gmm_pop_filter == "Both (Acinar + Islet)":
            st.sidebar.warning(
                "⚠️ Running GMM on both cell types together is not recommended because they are different cell types with differing calcium dynamics."
            )
        
        st.sidebar.markdown("**ACh Concentration Settings**")
        gmm_ach_val = st.sidebar.number_input(
            "ACh Concentration Value",
            min_value=0.001,
            max_value=10000.0,
            value=100.0,
            step=1.0,
            format="%.3f",
            help="Enter the numeric concentration of ACh.",
        )
        gmm_ach_unit = st.sidebar.selectbox(
            "ACh Concentration Unit",
            options=["fM", "pM", "nM", "uM", "µM", "mM"],
            index=2, # default nM
            help="Select the concentration unit.",
        )
    else:
        gmm_pop_filter = "Acinar cells only"
        gmm_ach_val = 100.0
        gmm_ach_unit = "nM"

    if "_raw_exp_list" not in st.session_state:
        st.session_state["_raw_exp_list"] = [
            {"genotype": "WT", "exp_name": "", "pathToRois": "", "sex": "", "scope": "", "pathToEvents": "", "pathToCoords": ""},
            {"genotype": "WT", "exp_name": "", "pathToRois": "", "sex": "", "scope": "", "pathToEvents": "", "pathToCoords": ""},
            {"genotype": "WT", "exp_name": "", "pathToRois": "", "sex": "", "scope": "", "pathToEvents": "", "pathToCoords": ""},
        ]

    if st.sidebar.button("Clear experiment table", help="Reset the experiment table to 3 empty rows."):
        if "_raw_exp_list" in st.session_state:
            del st.session_state["_raw_exp_list"]
        for k in list(st.session_state.keys()):
            if k.startswith(("exp_geno_", "exp_name_", "exp_path_", "exp_sex_", "exp_scope_", "exp_events_", "exp_coords_")):
                del st.session_state[k]
        st.rerun()

    def sync_exp_list():
        if "_raw_exp_list" in st.session_state:
            for i in range(len(st.session_state["_raw_exp_list"])):
                if f"exp_geno_{i}" in st.session_state:
                    st.session_state["_raw_exp_list"][i]["genotype"] = st.session_state[f"exp_geno_{i}"]
                if f"exp_name_{i}" in st.session_state:
                    st.session_state["_raw_exp_list"][i]["exp_name"] = st.session_state[f"exp_name_{i}"]
                if f"exp_path_{i}" in st.session_state:
                    val = st.session_state[f"exp_path_{i}"]
                    st.session_state["_raw_exp_list"][i]["pathToRois"] = val
                    if val.strip() and not st.session_state.get(f"exp_name_{i}", "").strip() and not st.session_state["_raw_exp_list"][i].get("exp_name", "").strip():
                        path_str = val.strip()
                        extracted_name = ""
                        parts = path_str.replace("\\", "/").split("/")
                        
                        # 1. Reversed lookup for experiment directory, skipping files and generic directories
                        for part in reversed(parts):
                            part_clean = part.strip()
                            if not part_clean:
                                continue
                            if part_clean.endswith((".pkl", ".csv", ".nd2", ".lif", ".tiff", ".czi")):
                                continue
                            if part_clean.lower() in ("all", "regions", "rois", "events", "analysis"):
                                continue
                            
                            # Remove suffix like ".nd2_analysis" or "_analysis"
                            part_clean = re.sub(r"\.(nd2|lif|tiff|czi)?_analysis$", "", part_clean, flags=re.IGNORECASE)
                            part_clean = re.sub(r"_analysis$", "", part_clean, flags=re.IGNORECASE)
                            
                            if part_clean.lower() not in ("all", "regions", "rois", "events"):
                                extracted_name = part_clean
                                break
                                
                        # 2. Fallback: search path components for typical experiment pattern (e.g. 279a_F)
                        if not extracted_name or extracted_name.lower() in ("all", "regions"):
                            for part in reversed(parts):
                                match = re.search(r"((?:exp|Experiment)?\d+[a-zA-Z]*(?:_[FM])?)", part, flags=re.IGNORECASE)
                                if match:
                                    extracted_name = match.group(1)
                                    break
                                    
                        if extracted_name:
                            st.session_state["_raw_exp_list"][i]["exp_name"] = extracted_name
                            try:
                                st.session_state[f"exp_name_{i}"] = extracted_name
                            except Exception:
                                pass
                if f"exp_sex_{i}" in st.session_state:
                    st.session_state["_raw_exp_list"][i]["sex"] = st.session_state[f"exp_sex_{i}"]
                if f"exp_scope_{i}" in st.session_state:
                    st.session_state["_raw_exp_list"][i]["scope"] = st.session_state[f"exp_scope_{i}"]
                if f"exp_events_{i}" in st.session_state:
                    st.session_state["_raw_exp_list"][i]["pathToEvents"] = st.session_state[f"exp_events_{i}"]
                if f"exp_coords_{i}" in st.session_state:
                    st.session_state["_raw_exp_list"][i]["pathToCoords"] = st.session_state[f"exp_coords_{i}"]

    def delete_row(idx):
        sync_exp_list()
        st.session_state["_raw_exp_list"].pop(idx)
        for k in list(st.session_state.keys()):
            if k.startswith(("exp_geno_", "exp_name_", "exp_path_", "exp_sex_", "exp_scope_", "exp_events_", "exp_coords_")):
                del st.session_state[k]
        st.rerun()

    def add_row():
        sync_exp_list()
        st.session_state["_raw_exp_list"].append({
            "genotype": "WT", "exp_name": "", "pathToRois": "", "sex": "", "scope": "", "pathToEvents": "", "pathToCoords": ""
        })
        for k in list(st.session_state.keys()):
            if k.startswith(("exp_geno_", "exp_name_", "exp_path_", "exp_sex_", "exp_scope_", "exp_events_", "exp_coords_")):
                del st.session_state[k]
        st.rerun()

    def set_row_count(n):
        sync_exp_list()
        current_len = len(st.session_state["_raw_exp_list"])
        if n > current_len:
            for _ in range(n - current_len):
                st.session_state["_raw_exp_list"].append({
                    "genotype": "WT", "exp_name": "", "pathToRois": "", "sex": "", "scope": "", "pathToEvents": "", "pathToCoords": ""
                })
        elif n < current_len:
            st.session_state["_raw_exp_list"] = st.session_state["_raw_exp_list"][:n]
        for k in list(st.session_state.keys()):
            if k.startswith(("exp_geno_", "exp_name_", "exp_path_", "exp_sex_", "exp_scope_", "exp_events_", "exp_coords_")):
                del st.session_state[k]
        st.rerun()

    sync_exp_list()
    st.write("### Experiment pooling table")
    st.write("Fill one row for a single experiment, or many rows for pooling. Use **🗑️** next to any row to delete it, or set the total row count below.")
    st.markdown(
        """
        **Required:** `genotype`, `exp_name`, `pathToRois`  
        **Optional:** `sex`, `scope`, `pathToEvents`, `pathToCoords`
        """
    )

    col_opts1, col_opts2, col_opts3 = st.columns([2, 2.5, 2])
    with col_opts1:
        current_rows_count = len(st.session_state["_raw_exp_list"])
        new_rows_count = st.number_input(
            "Set total rows count",
            min_value=1,
            max_value=100,
            value=current_rows_count,
            key="set_total_rows_input",
            help="Directly set the number of experiment rows."
        )
        if new_rows_count != current_rows_count:
            set_row_count(new_rows_count)

    with col_opts2:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        show_advanced = st.checkbox(
            "Show advanced paths (Events, Coords)",
            value=False,
            help="Toggle to show advanced fields."
        )

    with col_opts3:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        if st.button("➕ Add row", type="secondary", use_container_width=True):
            add_row()

    if show_advanced:
        col_headers = st.columns([1.8, 2.0, 3.5, 1.0, 1.0, 2.0, 2.0, 0.5])
        headers = ["Genotype *", "Exp Name *", "Path to ROIs *", "Sex", "Scope", "pathToEvents (opt)", "pathToCoords (opt)", ""]
    else:
        col_headers = st.columns([1.8, 2.2, 5.5, 1.0, 1.0, 0.5])
        headers = ["Genotype *", "Exp Name *", "Path to ROIs *", "Sex", "Scope", ""]

    for col_w, header in zip(col_headers, headers):
        if header:
            col_w.markdown(f"<span style='font-size: 0.85rem; font-weight: 600; color: #31333F; white-space: nowrap;'>{header}</span>", unsafe_allow_html=True)

    for i, item in enumerate(st.session_state["_raw_exp_list"]):
        if show_advanced:
            col_geno, col_name, col_path, col_sex, col_scope, col_events, col_coords, col_del = st.columns([1.8, 2.0, 3.5, 1.0, 1.0, 2.0, 2.0, 0.5])
        else:
            col_geno, col_name, col_path, col_sex, col_scope, col_del = st.columns([1.8, 2.2, 5.5, 1.0, 1.0, 0.5])

        try:
            g_idx = GENOTYPE_ORDER.index(item["genotype"])
        except ValueError:
            g_idx = 0
        col_geno.selectbox(
            f"Genotype {i}",
            options=GENOTYPE_ORDER,
            index=g_idx,
            key=f"exp_geno_{i}",
            label_visibility="collapsed",
        )

        col_name.text_input(
            f"Exp Name {i}",
            value=item["exp_name"],
            key=f"exp_name_{i}",
            placeholder="e.g. 279a_F",
            label_visibility="collapsed",
        )

        col_path.text_input(
            f"Path to ROIs {i}",
            value=item["pathToRois"],
            key=f"exp_path_{i}",
            placeholder="Path to 5.6_rois.pkl",
            label_visibility="collapsed",
        )

        try:
            s_idx = ["", "F", "M"].index(item["sex"])
        except ValueError:
            s_idx = 0
        col_sex.selectbox(
            f"Sex {i}",
            options=["", "F", "M"],
            index=s_idx,
            key=f"exp_sex_{i}",
            label_visibility="collapsed",
        )

        try:
            sc_idx = ["", "nd2", "lif", "tiff", "czi", "unknown"].index(item["scope"])
        except ValueError:
            sc_idx = 0
        col_scope.selectbox(
            f"Scope {i}",
            options=["", "nd2", "lif", "tiff", "czi", "unknown"],
            index=sc_idx,
            key=f"exp_scope_{i}",
            label_visibility="collapsed",
        )

        if show_advanced:
            col_events.text_input(
                f"pathToEvents {i}",
                value=item.get("pathToEvents", ""),
                key=f"exp_events_{i}",
                placeholder="Derived automatically",
                label_visibility="collapsed",
            )
            col_coords.text_input(
                f"pathToCoords {i}",
                value=item.get("pathToCoords", ""),
                key=f"exp_coords_{i}",
                placeholder="Optional CSV",
                label_visibility="collapsed",
            )

        if col_del.button("🗑️", key=f"exp_del_btn_{i}", help=f"Delete row {i+1}"):
            delete_row(i)

    # ── Bulk upload / save config expander moved below the table ──
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    with st.expander("📥 Import / Export Configuration (JSON / CSV)", expanded=False):
        # ── Bulk upload CSV & Config JSON ──
        col_up1, col_up2 = st.columns([1, 1])
        with col_up1:
            uploaded_config = st.file_uploader("Upload Bulk CSV / Config JSON", type=["csv", "json"], key="bulk_upload_config")
            if uploaded_config is not None:
                try:
                    if uploaded_config.name.endswith(".json"):
                        import json
                        config_data = json.load(uploaded_config)
                        if isinstance(config_data, list):
                            st.session_state["_raw_exp_list"] = config_data
                            for k in list(st.session_state.keys()):
                                if k.startswith(("exp_geno_", "exp_name_", "exp_path_", "exp_sex_", "exp_scope_", "exp_events_", "exp_coords_")):
                                    del st.session_state[k]
                            st.success("Loaded configuration successfully!")
                            st.rerun()
                    else:
                        csv_df = pd.read_csv(uploaded_config)
                        req = ["genotype", "exp_name", "pathToRois"]
                        missing = [c for c in req if c not in csv_df.columns]
                        if missing:
                            st.error(f"CSV is missing required columns: {', '.join(missing)}")
                        else:
                            new_list = []
                            for _, r in csv_df.iterrows():
                                new_list.append({
                                    "genotype": str(r.get("genotype", "WT")).strip(),
                                    "exp_name": str(r.get("exp_name", "")).strip(),
                                    "pathToRois": str(r.get("pathToRois", "")).strip(),
                                    "sex": str(r.get("sex", "") or "").strip(),
                                    "scope": str(r.get("scope", "") or "").strip(),
                                    "pathToEvents": str(r.get("pathToEvents", "") or "").strip(),
                                    "pathToCoords": str(r.get("pathToCoords", "") or "").strip(),
                                })
                            st.session_state["_raw_exp_list"] = new_list
                            for k in list(st.session_state.keys()):
                                if k.startswith(("exp_geno_", "exp_name_", "exp_path_", "exp_sex_", "exp_scope_", "exp_events_", "exp_coords_")):
                                    del st.session_state[k]
                            st.success(f"Successfully loaded {len(new_list)} experiments from CSV!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        with col_up2:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            sync_exp_list()
            import json
            config_str = json.dumps(st.session_state["_raw_exp_list"], indent=2)
            st.download_button(
                "💾 Save Configuration (JSON)",
                data=config_str,
                file_name="calcium_imaging_pooling_config.json",
                mime="application/json",
                help="Download currently configured table to restore it later.",
                use_container_width=True
            )
            template_df = pd.DataFrame([
                {"genotype": "WT", "exp_name": "279a_F", "pathToRois": f"/data/{_default_user}/exp279/5.6_rois.pkl", "sex": "F", "scope": "nd2"},
                {"genotype": "HET", "exp_name": "280b_M", "pathToRois": f"/data/{_default_user}/exp280/5.6_rois.pkl", "sex": "M", "scope": "lif"}
            ])
            st.download_button(
                "📋 Download CSV Template",
                data=template_df.to_csv(index=False),
                file_name="pooling_experiments_template.csv",
                mime="text/csv",
                help="Download a skeleton template to create your CSV configuration offline.",
                use_container_width=True
            )

    sync_exp_list()
    edited_rows = []
    for item in st.session_state["_raw_exp_list"]:
        edited_rows.append({
            "genotype": item["genotype"],
            "exp_name": item["exp_name"],
            "pathToRois": item["pathToRois"],
            "sex": item["sex"],
            "scope": item["scope"],
            "pathToEvents": item.get("pathToEvents", ""),
            "pathToCoords": item.get("pathToCoords", ""),
        })
    edited = pd.DataFrame(edited_rows)

    # ── ROI populations — separate text areas (data_editor can't handle long pastes) ──
    _roi_exp_names = [
        str(x).strip()
        for x in edited.get("exp_name", pd.Series(dtype=str)).dropna()
        if str(x).strip()
    ]
    with st.expander("ROI populations — islet / acinar (optional)", expanded=bool(_roi_exp_names)):
        if not _roi_exp_names:
            st.info("Fill exp_name in the table above first.")
        else:
            st.caption(
                "Paste ROI ID lists from your notebook. Accepted formats: `1,2,5,7` "
                "or `islet = [1, 2, 5, 7]` or `[1, 2, 5, 7]`. "
                "Acinar (excl. islet) = acinar − islet."
            )
            for _roi_exp in _roi_exp_names:
                st.markdown(f"**{_roi_exp}**")
                _c1, _c2 = st.columns(2)
                with _c1:
                    st.text_area(
                        "islet_rois",
                        key=f"_islet_rois_{_roi_exp}",
                        height=100,
                        placeholder="e.g. 126,132,143  or  islet = [126, 132, 143]",
                        label_visibility="visible",
                    )
                with _c2:
                    st.text_area(
                        "acinar_rois",
                        key=f"_acinar_rois_{_roi_exp}",
                        height=100,
                        placeholder="e.g. 10,15,22  or  acinar = [10, 15, 22]",
                        label_visibility="visible",
                    )
    st.subheader("Protocol table")
    st.markdown(
        """
        Required: `exp_name`, `compound`, `concentration`, `t_begin` — Optional: `t_end`

        Times are in seconds. Each phase ends at the next `t_begin`; `t_end` is used only for the last phase.
        """
    )

    _PROTO_COL_CFG = {
        "exp_name": st.column_config.TextColumn(
            "exp_name *",
            help="Required. Must exactly match the exp_name in the experiment table.",
        ),
        "compound": st.column_config.TextColumn(
            "compound *",
            help="Required. Added compound/stimulus, e.g. Glucose, ACh, Isr.",
        ),
        "concentration": st.column_config.TextColumn(
            "concentration *",
            help="Required. Include units, e.g. 8mM, 100nM, 0.1uM.",
        ),
        "t_begin": st.column_config.NumberColumn(
            "t_begin *",
            help="Required. Phase start time in seconds from recording start.",
        ),
        "t_end": st.column_config.NumberColumn(
            "t_end",
            help="Optional. End time in seconds. The app uses the next t_begin for non-last phase endings.",
        ),
    }

    def _default_proto_rows(exp_name):
        return [
            {"exp_name": exp_name, "compound": "Glucose", "concentration": "6mM", "t_begin": 0.0, "t_end": np.nan},
            {"exp_name": exp_name, "compound": "Glucose", "concentration": "8mM", "t_begin": np.nan, "t_end": np.nan},
            {"exp_name": exp_name, "compound": "ACh", "concentration": "100nM", "t_begin": np.nan, "t_end": np.nan},
        ]

    def _default_proto_text(exp_name):
        name = exp_name or "exp072a"
        return (
            "exp_name,compound,concentration,t_begin,t_end\n"
            f"{name},Glucose,6mM,0,\n"
            f"{name},Glucose,8mM,,\n"
            f"{name},ACh,100nM,,\n"
        )

    def _get_proto_fill_df(exp_name):
        key = f"_proto_fill_df_{exp_name}"
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame(_default_proto_rows(exp_name))
        return st.session_state[key]

    exp_names_for_protocol = [
        str(x).strip()
        for x in edited.get("exp_name", pd.Series(dtype=str)).dropna()
        if str(x).strip()
    ]
    if not exp_names_for_protocol:
        exp_names_for_protocol = [""]

    protocol_source = st.radio(
        "Protocol input method",
        ["Fill table", "Upload CSV", "Paste CSV text"],
        horizontal=True,
        help="All methods require: exp_name, compound, concentration, t_begin. Optional: t_end.",
    )
    protocol_edited = pd.DataFrame()

    def _render_proto_for_exp(exp_name, single=False):
        """Render the protocol input for one experiment; return a DataFrame or empty."""
        tab_df = pd.DataFrame()
        suffix = "" if single else exp_name  # unique key suffix

        if protocol_source == "Fill table":
            init_df = _get_proto_fill_df(exp_name)
            tab_df = st.data_editor(
                init_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config=_PROTO_COL_CFG,
                key=f"_proto_fill_{suffix}",
            )
            st.session_state[f"_proto_fill_df_{exp_name}"] = tab_df

        elif protocol_source == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload protocol CSV" if single else f"Upload protocol CSV for {exp_name}",
                type=["csv"],
                help="CSV columns: exp_name, compound, concentration, t_begin, optional t_end.",
                key=f"_proto_upload_{suffix}",
            )
            if uploaded is not None:
                try:
                    tab_df = pd.read_csv(uploaded)
                    st.dataframe(tab_df, use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not read CSV: {exc}")
            else:
                st.info("Upload a protocol CSV before running the raw pipeline." if single
                        else f"Upload the protocol CSV for {exp_name}.")

        else:  # Paste CSV text — key keeps value alive across re-runs
            tab_df_text = st.text_area(
                "Paste protocol CSV text" if single else f"Paste protocol CSV for {exp_name}",
                value=_default_proto_text(exp_name),
                height=180 if single else 160,
                key=f"_proto_paste_{suffix}",
            )
            try:
                tab_df = pd.read_csv(io.StringIO(tab_df_text))
                st.dataframe(tab_df, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not parse protocol CSV: {exc}")

        return tab_df

    # ── single experiment: flat input ────────────────────────────────────────
    if len(exp_names_for_protocol) <= 1:
        exp_name_single = exp_names_for_protocol[0] if exp_names_for_protocol else ""
        protocol_edited = _render_proto_for_exp(exp_name_single, single=True)

    # ── multiple experiments: one tab per experiment ──────────────────────────
    else:
        st.caption(
            f"{len(exp_names_for_protocol)} experiments detected — each tab below is for one experiment. "
            "The protocols are merged automatically when you run the pipeline."
        )
        proto_tabs = st.tabs(exp_names_for_protocol)
        all_proto_frames = []

        for tab, exp_name in zip(proto_tabs, exp_names_for_protocol):
            with tab:
                tab_df = _render_proto_for_exp(exp_name, single=False)
                if not tab_df.empty:
                    all_proto_frames.append(tab_df)

        if all_proto_frames:
            protocol_edited = pd.concat(all_proto_frames, ignore_index=True)

    # ── Custom ACh concentration and pre-run validation checks ───────────────
    gmm_ach_val_str = f"{int(gmm_ach_val)}" if gmm_ach_val == int(gmm_ach_val) else f"{gmm_ach_val:.3f}".rstrip('0').rstrip('.')
    gmm_ach_concentration = f"{gmm_ach_val_str}{gmm_ach_unit}"
    gmm_ach_phase_name = f"{gmm_ach_concentration} ACh"
    gmm_ach_priority = [gmm_ach_phase_name]

    active_exps = []
    for _, row in edited.iterrows():
        exp_name = str(row.get("exp_name", "")).strip()
        path_rois = str(row.get("pathToRois", "")).strip()
        if exp_name and path_rois:
            active_exps.append(exp_name)

    # Check islet/acinar ROI lists completeness
    empty_islet_exps = []
    empty_acinar_exps = []
    for exp in active_exps:
        islet_str = st.session_state.get(f"_islet_rois_{exp}", "").strip()
        acinar_str = st.session_state.get(f"_acinar_rois_{exp}", "").strip()
        if not islet_str:
            empty_islet_exps.append(exp)
        if not acinar_str:
            empty_acinar_exps.append(exp)

    validation_passed = True
    proceed_partial = False
    has_ach = []
    missing_ach = []

    if empty_islet_exps or empty_acinar_exps:
        validation_passed = False
        st.markdown("### ❌ Missing Required ROIs")
        if empty_islet_exps:
            st.error(f"Please provide an Islet ROI list for the following experiments: {', '.join(empty_islet_exps)}")
        if empty_acinar_exps:
            st.error(f"Please provide an Acinar ROI list for the following experiments: {', '.join(empty_acinar_exps)}")

    elif run_gmm and active_exps and not protocol_edited.empty:
        # Check if the ACh concentration exists in the protocols of all active experiments
        for exp in active_exps:
            sub = protocol_edited[
                (protocol_edited["exp_name"].astype(str).str.strip() == exp) &
                (protocol_edited["compound"].astype(str).str.strip().str.lower() == "ach") &
                (protocol_edited["concentration"].astype(str).str.strip() == gmm_ach_concentration)
            ]
            if not sub.empty:
                has_ach.append(exp)
            else:
                missing_ach.append(exp)
        
        if missing_ach:
            st.markdown("### ⚠️ GMM ACh Concentration Check")
            st.warning(
                f"The requested concentration **{gmm_ach_concentration} ACh** is not present in the protocol of all selected slices.\n\n"
                f"✅ **Slices containing the concentration:** {', '.join(has_ach) if has_ach else 'None'}\n\n"
                f"❌ **Slices lacking the concentration:** {', '.join(missing_ach)}"
            )
            if not has_ach:
                st.error("Cannot proceed because no slice contains the requested concentration in its protocol.")
                validation_passed = False
            else:
                proceed_partial = st.checkbox(
                    "Do you want to proceed only with slices containing the requested concentration?",
                    value=False,
                    help="If checked, analysis will only run on slices where the concentration is present; other slices will be omitted."
                )
                if not proceed_partial:
                    st.info("Please check the box above to continue with the matching slices, or change the concentration/protocols.")
                    validation_passed = False

    run = st.button("Run raw pipeline and create tables", type="primary", disabled=not validation_passed)
    if run:
        # Inject islet/acinar ROI lists from their dedicated text_areas into the table
        edited_with_rois = edited.copy()
        edited_with_rois["islet_rois"] = edited_with_rois["exp_name"].apply(
            lambda n: st.session_state.get(f"_islet_rois_{str(n).strip()}", "")
        )
        edited_with_rois["acinar_rois"] = edited_with_rois["exp_name"].apply(
            lambda n: st.session_state.get(f"_acinar_rois_{str(n).strip()}", "")
        )
        
        # Filter matching experiments and protocols if we have concentration mismatches
        if run_gmm and missing_ach and proceed_partial:
            edited_with_rois = edited_with_rois[edited_with_rois["exp_name"].isin(has_ach)]
            protocol_edited = protocol_edited[protocol_edited["exp_name"].isin(has_ach)]

        experiments, errors = parse_experiment_table(edited_with_rois)
        protocol_df, protocol_errors = validate_protocol_editor(protocol_edited)
        tiff_exps = [e["exp_name"] for e in experiments if str(e.get("scope", "")).lower() == "tiff"]
        if tiff_exps:
            st.warning(
                f"**Note — tiff resolution:** {', '.join(tiff_exps)} "
                f"{'is' if len(tiff_exps) == 1 else 'are'} tiff experiment"
                f"{'s' if len(tiff_exps) != 1 else ''}. "
                "Tiff files typically have lower spatial resolution than nd2/lif scans. "
                "Spatial maps and population comparisons may not be reliable for cross-experiment comparison."
            )
        if errors or protocol_errors:
            for error in errors:
                st.error(error)
            for error in protocol_errors:
                st.error(error)
        elif not experiments:
            st.error("Add at least one experiment with pathToRois.")
        else:
            with st.spinner("Running raw-data pipeline..."):
                try:
                    all_data, rp_all, coords, coord_status, gmm_diag = run_raw_pipeline(
                        experiments,
                        protocol_df,
                        data_root,
                        run_gmm=run_gmm,
                        gmm_pop_filter=gmm_pop_filter,
                        gmm_ach_priority=gmm_ach_priority
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    if coords is not None and not coords.empty:
                        coords.to_csv(output_dir / "roi_coordinates_all.csv", index=False)
                    if coord_status is not None and not coord_status.empty:
                        coord_status.to_csv(output_dir / "coordinate_diagnostics.csv", index=False)
                    st.session_state["all_data"] = all_data
                    st.session_state["rp_all"] = rp_all
                    st.session_state["coords"] = coords
                    st.session_state["coord_status"] = coord_status
                    st.session_state["gmm_diag"] = gmm_diag
                except Exception as exc:
                    st.error(f"Raw pipeline failed: {type(exc).__name__}: {exc}")
                    with st.expander("Full traceback", expanded=True):
                        st.code(traceback.format_exc())
                    st.exception(exc)


all_data = st.session_state.get("all_data", pd.DataFrame())
rp_all = st.session_state.get("rp_all", pd.DataFrame())
coords = st.session_state.get("coords", pd.DataFrame())
coord_status = st.session_state.get("coord_status", pd.DataFrame())

# ── GMM diagnostics — shown whenever run_gmm was used ────────────────────────
_gmm_diag = st.session_state.get("gmm_diag", {})
if _gmm_diag:
    _skipped = {g: d for g, d in _gmm_diag.items() if not d.get("gmm_ran", False)}
    _ran     = {g: d for g, d in _gmm_diag.items() if d.get("gmm_ran", False)}
    if _ran:
        st.success(f"GMM ran for: {', '.join(sorted(_ran.keys()))}")
    if _skipped:
        for g, d in sorted(_skipped.items()):
            reasons = []
            if not d["has_8mM_glucose"]:
                reasons.append(f"no '8mM Glucose' phase (phases found: {d['phases_present']})")
            if not d["has_ach"]:
                reasons.append(f"no ACh phase matching {ACH_PRIORITY}")
            if d["has_8mM_glucose"] and d["has_ach"]:
                reasons.append(
                    f"only {d['n_qualifying']} ROIs had all required features "
                    f"(8mM Glucose ROIs: {d['n_8mM_glucose_rois']}, ACh ROIs: {d['n_ach_rois']}; need ≥20)"
                )
            st.warning(
                f"**{g}: GMM skipped** — {'; '.join(reasons) if reasons else 'unknown reason'}. "
                f"The Figures tab will not show {g}. Phase analysis tab is unaffected."
            )

if (all_data is not None and not all_data.empty) or (rp_all is not None and not rp_all.empty):
    output_dir.mkdir(parents=True, exist_ok=True)
    if all_data is None:
        all_data = pd.DataFrame()
    if rp_all is None:
        rp_all = pd.DataFrame()

if (all_data is not None and not all_data.empty) or (rp_all is not None and not rp_all.empty):
    tables = compute_summary_tables(all_data) if all_data is not None and not all_data.empty else {}
    if tables:
        save_summary_tables(tables, output_dir)

    st.divider()
    st.header("Results")
    render_run_overview(all_data, rp_all, coords, output_dir)

    # --- Global population selector ----------------------------------------
    has_population = (
        coords is not None
        and not coords.empty
        and "population" in coords.columns
        and coords["population"].ne("unknown").any()
    )
    if has_population:
        available_pops = set(coords["population"].unique())
        pop_options = ["All ROIs"]
        if "islet" in available_pops:
            pop_options.append("Islet")
        if "acinar" in available_pops:
            pop_options.append("Acinar (excl. islet)")
        if "islet" in available_pops and "acinar" in available_pops:
            pop_options.append("Islet + Acinar")
        try:
            default_index = pop_options.index("Acinar (excl. islet)")
        except ValueError:
            default_index = 0
        pop_col, _ = st.columns([3, 2])
        with pop_col:
            global_population_mode = st.radio(
                "View population",
                pop_options,
                index=default_index,
                horizontal=True,
                help=(
                    "Filters all analyses (KDE, halfwidth overlay, pooling comparison, spatial maps) "
                    "to the selected cell population. Islet and acinar lists come from the experiment table."
                ),
            )
        n_total = len(coords)
        n_shown = len(filter_coords_by_population(coords, global_population_mode))
        st.caption(f"Showing {n_shown:,} of {n_total:,} ROIs · population: **{global_population_mode}**")
    else:
        global_population_mode = "All ROIs"

    rp_displayed = filter_rp_by_population(rp_all, coords, global_population_mode)
    # -----------------------------------------------------------------------

    overview_tab, tables_tab, figures_tab, phase_tab, files_tab = st.tabs(
        ["Overview", "Tables", "Figures", "Phase analysis", "Saved files"]
    )

    with overview_tab:
        st.subheader("Run summary")
        if rp_all is not None and not rp_all.empty:
            st.markdown('<p class="section-note">Phase coverage from stable ROI-level metrics.</p>', unsafe_allow_html=True)
            render_phase_coverage(rp_all)
        if all_data is None or all_data.empty:
            st.info(
                "No GMM cluster table is available. This is expected if Raw mode was run with "
                "`Run GMM population analysis` turned off. Use the Phase analysis tab for ISR/pharmacology-style outputs."
            )
        if coord_status is not None and not coord_status.empty:
            with st.expander("Coordinate diagnostics", expanded=coords is None or coords.empty):
                st.dataframe(coord_status, use_container_width=True, hide_index=True)
                failed = coord_status[coord_status["status"] != "ok"]
                if not failed.empty:
                    st.warning(
                        "At least one experiment failed coordinate extraction. "
                        "Check the message column, and confirm that the server is running the updated app file."
                    )

    with tables_tab:
        if tables:
            table_names = list(tables.keys())
            table_tabs = st.tabs(table_names)
            for tab, name in zip(table_tabs, table_names):
                with tab:
                    st.dataframe(tables[name], use_container_width=True, height=360)
                    st.download_button(
                        f"Download {name}.csv",
                        tables[name].to_csv(index=False).encode("utf-8"),
                        file_name=f"{name}.csv",
                        mime="text/csv",
                    )
        else:
            st.info("No GMM summary tables are available for this run.")
        if rp_all is not None and not rp_all.empty:
            with st.expander("ROI-phase table preview", expanded=False):
                st.dataframe(rp_all.head(1000), use_container_width=True, height=360)
                st.download_button(
                    "Download roi_phase.csv",
                    rp_all.to_csv(index=False).encode("utf-8"),
                    file_name="roi_phase.csv",
                    mime="text/csv",
                )

    with figures_tab:
        if all_data is not None and not all_data.empty:
            left, right = st.columns(2)
            with left:
                st.subheader("Population structure")
                fig = plot_population(all_data)
                st.pyplot(fig)
                _dl_buttons([("population_structure.png", fig_to_bytes(fig), "image/png")])
            with right:
                st.subheader("Halfwidth by cluster")
                fig = plot_halfwidth(all_data)
                st.pyplot(fig)
                _dl_buttons([("halfwidth_comparison.png", fig_to_bytes(fig), "image/png")])
        else:
            st.info("Core GMM figures are unavailable because no clustered ROI table was generated.")

        st.subheader("Spatial maps")
        st.markdown(
            '<div class="quiet-panel">Population follows the global selector above. '
            'Zoom crops the field; 0 = full tissue, higher values crop to the centre.</div>',
            unsafe_allow_html=True,
        )
        phase_for_spatial = None
        spatial_value = None
        spatial_controls = st.columns([2, 1])
        with spatial_controls[1]:
            spatial_zoom_pct = st.slider(
                "Spatial zoom",
                min_value=0,
                max_value=90,
                value=0,
                step=5,
                help="Crops each slice to the central coordinate range. 0 shows the full field.",
            )
        if all_data is not None and not all_data.empty:
            try:
                with st.spinner("Drawing spatial cluster map..."):
                    fig = plot_spatial_maps(
                        all_data,
                        coords,
                        population_mode=global_population_mode,
                        zoom_pct=spatial_zoom_pct,
                    )
                    st.pyplot(fig, use_container_width=True)
                    _dl_buttons([("spatial_cluster_maps.png", fig_to_bytes(fig), "image/png")])
            except Exception as exc:
                st.warning(f"{type(exc).__name__}: {exc}")
                st.exception(exc)
        else:
            phase_options_for_spatial = ordered_phases(rp_all) if rp_all is not None and not rp_all.empty else []
            metric_options = [c for c in ["mean_halfwidth", "median_halfwidth", "event_rate"] if rp_all is not None and c in rp_all.columns]
            with spatial_controls[0]:
                ctrl_left, ctrl_right = st.columns(2)
                with ctrl_left:
                    if phase_options_for_spatial:
                        phase_for_spatial = st.selectbox(
                            "Phase for spatial metric map",
                            phase_options_for_spatial,
                            index=0,
                            help="Choose which pharmacological phase to draw on the spatial map.",
                        )
                    else:
                        st.info("No phase options are available for a spatial metric map.")
                with ctrl_right:
                    if metric_options:
                        spatial_value = st.selectbox(
                            "Spatial metric",
                            metric_options,
                            index=0,
                            help="Color each ROI by this phase-level metric.",
                        )
                    else:
                        st.info("No spatial metric columns are available yet.")
            if phase_for_spatial and spatial_value:
                try:
                    with st.spinner("Drawing spatial phase map..."):
                        fig = plot_spatial_phase_maps(
                            rp_displayed,
                            coords,
                            phase_for_spatial,
                            spatial_value,
                            population_mode=global_population_mode,
                            zoom_pct=spatial_zoom_pct,
                        )
                        st.pyplot(fig, use_container_width=True)
                        _dl_buttons([("spatial_phase_metric_map.png", fig_to_bytes(fig), "image/png")])
                except Exception as exc:
                    st.warning(f"{type(exc).__name__}: {exc}")
                    st.exception(exc)

    with phase_tab:
        if rp_all is None or rp_all.empty:
            st.info("No ROI-phase data available.")
        else:
            phase_options = ordered_phases(rp_all)
            st.markdown(
                '<div class="quiet-panel">Choose phases once, then switch between the two analyses below. Each figure uses the same selected phases.</div>',
                unsafe_allow_html=True,
            )
            phase_controls_left, phase_controls_right = st.columns([3, 1])
            with phase_controls_left:
                selected_phases = st.multiselect(
                    "Phases to include",
                    phase_options,
                    default=phase_options,
                    help="Select any phases present in the data: ACh, Isr, glucose, or other compounds/concentrations.",
                )
            with phase_controls_right:
                min_rois_for_kde = st.number_input(
                    "Minimum ROIs for KDE",
                    min_value=5,
                    max_value=500,
                    value=20,
                    step=5,
                    help="Groups with fewer ROI-phase rows are shown as insufficient instead of drawing a KDE.",
                )

            bimodality_tab, overlay_tab, pooling_tab, fraction_tab, heatmap_tab, transition_tab = st.tabs([
                "Bimodality KDE", "Halfwidth Overlay", "Pooling Comparison",
                "Active ROI Fraction", "Event Rate Heatmap", "Phase Transitions",
            ])
            with bimodality_tab:
                st.markdown(
                    '<p class="section-note">KDE peak detection for each selected phase and genotype, with a numeric phasic/intermediate/sustained summary.</p>',
                    unsafe_allow_html=True,
                )
                if not selected_phases:
                    st.info("Select at least one phase to generate the bimodality plot.")
                else:
                    try:
                        fig, bimodality_summary = plot_bimodality(
                            rp_displayed,
                            selected_phases=selected_phases,
                            min_n=min_rois_for_kde,
                        )
                        st.pyplot(fig, use_container_width=True)
                        st.dataframe(bimodality_summary, use_container_width=True, height=300)
                        st.text(bimodality_summary_text(bimodality_summary))
                        _png = fig_to_bytes(fig)
                        _csv = bimodality_summary.to_csv(index=False).encode()
                        _dl_buttons([("bimodality_by_phase.png", _png, "image/png"), ("bimodality_summary.csv", _csv, "text/csv")])
                    except Exception as exc:
                        st.warning(f"{type(exc).__name__}: {exc}")
                        st.exception(exc)

            with overlay_tab:
                st.markdown(
                    '<p class="section-note">Overlay of halfwidth distributions across selected phases or individual experiments, useful for tracking peak shifts and slice-to-slice variations.</p>',
                    unsafe_allow_html=True,
                )
                if not selected_phases:
                    st.info("Select at least one phase to generate the overlay.")
                else:
                    overlay_mode = st.radio(
                        "Overlay Mode",
                        ["Overlay phases (pooled experiments)", "Overlay experiments (for a single selected phase)"],
                        horizontal=True,
                        help="Choose whether to compare halfwidth distributions across different pharmacological phases, or across different active slices/experiments for a selected phase."
                    )
                    
                    if overlay_mode == "Overlay phases (pooled experiments)":
                        try:
                            fig, overlay_summary = plot_halfwidth_overlay(
                                rp_displayed,
                                selected_phases=selected_phases,
                                min_n=min_rois_for_kde,
                            )
                            st.pyplot(fig, use_container_width=True)
                            st.dataframe(overlay_summary.round(3), use_container_width=True, height=300)
                            _png = fig_to_bytes(fig)
                            _csv = overlay_summary.to_csv(index=False).encode()
                            _dl_buttons([("halfwidth_overlay_by_phase.png", _png, "image/png"), ("halfwidth_overlay_summary.csv", _csv, "text/csv")])
                        except Exception as exc:
                            st.warning(f"{type(exc).__name__}: {exc}")
                            st.exception(exc)
                    else:
                        selected_phase = st.selectbox(
                            "Select phase to compare experiments",
                            selected_phases,
                            index=0,
                            help="Choose which pharmacological phase to compare different experiments/slices for."
                        )
                        try:
                            fig, overlay_summary = plot_halfwidth_experiment_overlay(
                                rp_displayed,
                                selected_phase=selected_phase,
                                min_n=min_rois_for_kde,
                            )
                            st.pyplot(fig, use_container_width=True)
                            st.dataframe(overlay_summary.round(3), use_container_width=True, height=300)
                            _png = fig_to_bytes(fig)
                            _csv = overlay_summary.to_csv(index=False).encode()
                            _dl_buttons([("halfwidth_overlay_by_experiment.png", _png, "image/png"), ("halfwidth_overlay_by_experiment_summary.csv", _csv, "text/csv")])
                        except Exception as exc:
                            st.warning(f"{type(exc).__name__}: {exc}")
                            st.exception(exc)

            with pooling_tab:
                st.markdown(
                    '<p class="section-note">Compare experiments/slices entered in Raw mode side by side. This is the pooling view built from pathToRois rows, not only from existing pooled_results.</p>',
                    unsafe_allow_html=True,
                )
                metric_options = [c for c in ["mean_halfwidth", "median_halfwidth", "event_rate", "cell_activation_rate"] if c in rp_displayed.columns]
                if not metric_options or not selected_phases:
                    st.info("Select phases and make sure ROI-phase metrics are available.")
                else:
                    comp_left, comp_right = st.columns([1, 1])
                    with comp_left:
                        METRIC_HELP = {
                            "mean_halfwidth": "Average duration (seconds) of a calcium transient at half its peak height.",
                            "median_halfwidth": "Median duration (seconds) of a calcium transient at half its peak height.",
                            "event_rate": "Number of calcium transients per minute during the stable period of the phase.",
                            "cell_activation_rate": (
                                "mean_halfwidth × event_rate. Approximates total activation time per minute "
                                "(seconds of active calcium signal per minute). "
                                "Example: event_rate=5 events/min, halfwidth=10 s → rate=50. "
                                "event_rate=10 events/min, halfwidth=3 s → rate=30. "
                                "The first ROI fires less often but each transient lasts longer; "
                                "this metric captures both dimensions together. "
                                "Note: not bounded, interpret alongside event_rate and halfwidth separately."
                            ),
                        }
                        comparison_metric = st.selectbox(
                            "Metric",
                            metric_options,
                            index=0,
                            help=METRIC_HELP.get(metric_options[0], "Metric used for side-by-side experiment comparison."),
                        )
                        if comparison_metric in METRIC_HELP:
                            st.caption(METRIC_HELP[comparison_metric])
                    with comp_right:
                        comparison_phase = st.selectbox(
                            "Phase",
                            ["All selected phases"] + selected_phases,
                            index=0,
                            help="Compare all selected phases or one phase at a time.",
                        )
                    try:
                        compare_df = rp_displayed[rp_displayed["phase"].isin(selected_phases)].copy()
                        phase_arg = None if comparison_phase == "All selected phases" else comparison_phase
                        fig, comparison_summary = plot_experiment_comparison(compare_df, comparison_metric, phase=phase_arg)
                        st.pyplot(fig, use_container_width=True)
                        st.dataframe(comparison_summary.round(3), use_container_width=True, height=320)
                        _png = fig_to_bytes(fig)
                        _csv = comparison_summary.to_csv(index=False).encode()
                        _dl_buttons([("experiment_pooling_comparison.png", _png, "image/png"), ("experiment_pooling_comparison_summary.csv", _csv, "text/csv")])
                    except Exception as exc:
                        st.warning(f"{type(exc).__name__}: {exc}")
                        st.exception(exc)

            with fraction_tab:
                st.markdown(
                    '<p class="section-note">Fraction of ROIs with event_rate > 0 per experiment and phase. '
                    'Useful to see how many cells are recruited at each stimulus level.</p>',
                    unsafe_allow_html=True,
                )
                if not selected_phases:
                    st.info("Select at least one phase.")
                else:
                    try:
                        _rp_frac = rp_displayed[rp_displayed["phase"].isin(selected_phases)].copy()
                        fig, frac_summary = plot_active_roi_fraction(_rp_frac, coords=coords)
                        st.pyplot(fig, use_container_width=True)
                        st.dataframe(frac_summary.round(3), use_container_width=True)
                        _png = fig_to_bytes(fig)
                        _csv = frac_summary.to_csv(index=False).encode()
                        _dl_buttons([("active_roi_fraction.png", _png, "image/png"),
                                     ("active_roi_fraction.csv", _csv, "text/csv")])
                    except Exception as exc:
                        st.warning(f"{type(exc).__name__}: {exc}")

            with heatmap_tab:
                st.markdown(
                    '<p class="section-note">Median event rate per experiment and phase as a colour-coded matrix. '
                    'Rows are experiments, columns are phases.</p>',
                    unsafe_allow_html=True,
                )
                if not selected_phases:
                    st.info("Select at least one phase.")
                else:
                    try:
                        _rp_hm = rp_displayed[rp_displayed["phase"].isin(selected_phases)].copy()
                        fig, hm_summary = plot_event_rate_heatmap(_rp_hm)
                        st.pyplot(fig, use_container_width=True)
                        st.dataframe(hm_summary.round(3), use_container_width=True)

                        roi_counts = compute_roi_count_table(_rp_hm)
                        if not roi_counts.empty:
                            with st.expander("ROI count per experiment × phase", expanded=False):
                                st.dataframe(roi_counts, use_container_width=True)
                                _count_csv = roi_counts.to_csv(index=False).encode()
                                _dl_buttons([("roi_count_table.csv", _count_csv, "text/csv")])

                        _png = fig_to_bytes(fig)
                        _csv = hm_summary.to_csv(index=False).encode()
                        _dl_buttons([("event_rate_heatmap.png", _png, "image/png"),
                                     ("event_rate_heatmap.csv", _csv, "text/csv")])
                    except Exception as exc:
                        st.warning(f"{type(exc).__name__}: {exc}")

            with transition_tab:
                st.markdown(
                    '<p class="section-note">Mean ± SEM of a metric across phases for each experiment. '
                    'Shows whether a treatment shifts the distribution up or down.</p>',
                    unsafe_allow_html=True,
                )
                if not selected_phases:
                    st.info("Select at least one phase.")
                else:
                    _metric_opts = [c for c in ["event_rate", "mean_halfwidth", "median_halfwidth"]
                                    if c in rp_displayed.columns]
                    _trans_metric = st.selectbox(
                        "Metric for transitions",
                        _metric_opts,
                        index=0,
                        key="transition_metric",
                    )
                    try:
                        _rp_tr = rp_displayed[rp_displayed["phase"].isin(selected_phases)].copy()
                        fig, trans_summary = plot_phase_transitions(_rp_tr, metric_col=_trans_metric)
                        st.pyplot(fig, use_container_width=True)
                        if not trans_summary.empty:
                            st.dataframe(trans_summary.round(3), use_container_width=True)
                        _png = fig_to_bytes(fig)
                        _csv = trans_summary.to_csv(index=False).encode() if not trans_summary.empty else b""
                        items = [("phase_transitions.png", _png, "image/png")]
                        if _csv:
                            items.append(("phase_transitions.csv", _csv, "text/csv"))
                        _dl_buttons(items)
                    except Exception as exc:
                        st.warning(f"{type(exc).__name__}: {exc}")

    with files_tab:
        outputs = st.session_state.get("outputs", {})
        st.subheader("Save to server")
        if not outputs:
            st.info("No outputs generated yet. Generate figures or tables in the other tabs first.")
        else:
            save_path_input = st.text_input(
                "Server directory",
                value=str(output_dir),
                help="Absolute path on the server where selected files will be written.",
            )
            selected_to_save = st.multiselect(
                "Files to save",
                sorted(outputs.keys()),
                default=sorted(outputs.keys()),
            )
            if st.button("Save selected to server", type="primary"):
                save_dir = Path(save_path_input.strip())
                try:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    saved = []
                    for name in selected_to_save:
                        data = outputs[name]
                        if isinstance(data, str):
                            data = data.encode("utf-8")
                        (save_dir / name).write_bytes(data)
                        saved.append(name)
                    st.success(f"Saved {len(saved)} file(s) to {save_dir}")
                except Exception as exc:
                    st.error(f"Save failed: {type(exc).__name__}: {exc}")

        st.divider()
        st.subheader("Existing files in output directory")
        st.caption(str(output_dir))
        files = sorted(output_dir.glob("*")) if output_dir.exists() else []
        if files:
            file_df = pd.DataFrame({
                "file": [p.name for p in files],
                "size_kb": [round(p.stat().st_size / 1024, 1) for p in files],
            })
            st.dataframe(file_df, use_container_width=True, hide_index=True)
        else:
            st.info("No files saved in the output folder yet.")

    st.success(f"Tables and figures are saved to: {output_dir}")
else:
    st.info("Choose an input mode and run the analysis from the sidebar.")
