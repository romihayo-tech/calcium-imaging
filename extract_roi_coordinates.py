from pathlib import Path
import argparse
import sys
import types

import numpy as np
import pandas as pd


def install_pandas_pickle_compat():
    """Allow old pandas pickles to load under newer pandas versions."""
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
    except Exception as exc:
        return False, f"pandas internals not available: {exc}"

    if getattr(blocks.new_block, "_calcium_compat_patch", False):
        return True, "already installed"

    original_new_block = blocks.new_block

    def new_block_compat(values, placement, ndim, refs=None):
        if isinstance(placement, slice):
            placement = BlockPlacement(placement)
        return original_new_block(values, placement=placement, ndim=ndim, refs=refs)

    new_block_compat._calcium_compat_patch = True
    blocks.new_block = new_block_compat
    return True, "installed"


def normalize_path(path):
    return Path(str(path).replace("/local_", "/"))


def extract_coordinates(path_to_rois, exp_name, genotype="", scope=""):
    ok, msg = install_pandas_pickle_compat()
    if not ok:
        raise RuntimeError(msg)

    from islets.Regions import load_regions

    regions = load_regions(str(normalize_path(path_to_rois)))
    try:
        regions.detrend_traces(method="debleach")
    except Exception:
        pass

    if not hasattr(regions, "df") or "peak" not in regions.df.columns:
        raise ValueError("regions.df['peak'] was not found in this rois.pkl")

    df = regions.df.copy()
    coords = pd.DataFrame({
        "roi": df.index,
        "x": df["peak"].apply(lambda p: p[0] if isinstance(p, (list, tuple, np.ndarray)) and len(p) > 0 else np.nan),
        "y": df["peak"].apply(lambda p: p[1] if isinstance(p, (list, tuple, np.ndarray)) and len(p) > 1 else np.nan),
    }).dropna(subset=["roi", "x", "y"])

    coords["exp_name"] = exp_name
    coords["genotype"] = genotype
    coords["scope"] = scope
    coords["x_centered"] = coords["x"] - coords["x"].median()
    coords["y_centered"] = coords["y"] - coords["y"].median()
    coords["radial_dist"] = np.sqrt(coords["x_centered"] ** 2 + coords["y_centered"] ** 2)
    return coords


def main():
    parser = argparse.ArgumentParser(description="Extract ROI coordinates from islets rois.pkl.")
    parser.add_argument("--pathToRois", required=True, help="Full path to 5.6_rois.pkl")
    parser.add_argument("--exp_name", required=True, help="Experiment name used in downstream tables")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--genotype", default="", help="Optional genotype label")
    parser.add_argument("--scope", default="", help="Optional scope label, e.g. nd2/lif/tiff")
    args = parser.parse_args()

    coords = extract_coordinates(args.pathToRois, args.exp_name, args.genotype, args.scope)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    coords.to_csv(out, index=False)
    print(f"Saved {len(coords):,} ROI coordinates to {out}")


if __name__ == "__main__":
    main()
