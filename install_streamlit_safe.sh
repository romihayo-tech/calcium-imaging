#!/bin/bash
# install_streamlit_safe.sh
#
# Install Streamlit dependencies in /opt/conda WITHOUT upgrading pandas.
#
# The trick: pip's --constraint flag tells it "this package version is fixed,
# do not change it" — so pandas stays at 1.3.5 even if a dep would normally
# want a newer version. If a true conflict exists, pip errors loudly instead
# of silently upgrading.
#
# Usage (run from the JupyterLab terminal):
#   bash /data/<your-username>/CLEAN_PIPELINES/install_streamlit_safe.sh
#
# What this script does, step by step:
#   1. Record the current pandas version.
#   2. Write a pip constraints file that pins pandas==1.3.5.
#   3. Install the one confirmed-missing dep (starlette).
#   4. Run "pip install streamlit -c ..." to let pip fill in any other
#      missing Streamlit deps while respecting the pandas pin.
#   5. Verify pandas is still 1.3.5.
#   6. Run a quick Streamlit import test.
#   7. Run a quick load_regions import test.
#
# If step 4 fails with a pandas conflict, it means the installed Streamlit
# version requires pandas >= 2.0. In that case you need to downgrade
# Streamlit — see the README "Troubleshooting" section.

set -e   # exit immediately on any error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON=/opt/conda/bin/python
CONSTRAINT=/tmp/pandas_pin.txt

# ---------------------------------------------------------------------------
# 1. Current state
# ---------------------------------------------------------------------------
echo "=== 1/7  Current environment ==="
$PYTHON -c "import sys, pandas as pd; print('python :', sys.executable); print('pandas :', pd.__version__)"

# ---------------------------------------------------------------------------
# 2. Write the pandas pin
# ---------------------------------------------------------------------------
echo ""
echo "=== 2/7  Writing pandas constraint file -> $CONSTRAINT ==="
echo "pandas==1.3.5" > "$CONSTRAINT"
cat "$CONSTRAINT"

# ---------------------------------------------------------------------------
# 3. Install the confirmed-missing dep: starlette
# ---------------------------------------------------------------------------
echo ""
echo "=== 3/7  Installing starlette (with pandas pinned) ==="
$PYTHON -m pip install starlette -c "$CONSTRAINT"

# ---------------------------------------------------------------------------
# 4. Re-run Streamlit's own dependency resolution to catch any other gaps.
#    --no-deps is NOT used here so pip can install transitive deps, but
#    the constraint file keeps pandas locked.
# ---------------------------------------------------------------------------
echo ""
echo "=== 4/7  Filling remaining Streamlit deps (pandas pinned) ==="
$PYTHON -m pip install streamlit -c "$CONSTRAINT"

# If the line above printed "Requirement already satisfied: streamlit" and
# any dep still fails, uncomment and edit the block below to install that
# dep explicitly:
#
#   $PYTHON -m pip install <missing-package> -c "$CONSTRAINT"

# ---------------------------------------------------------------------------
# 5. Verify pandas
# ---------------------------------------------------------------------------
echo ""
echo "=== 5/7  Verifying pandas version ==="
PANDAS_VERSION=$($PYTHON -c "import pandas as pd; print(pd.__version__)")
echo "pandas: $PANDAS_VERSION"
if [ "$PANDAS_VERSION" != "1.3.5" ]; then
    echo ""
    echo "ERROR: pandas was changed! Expected 1.3.5, got $PANDAS_VERSION."
    echo "Do NOT run the app. Restore pandas with:"
    echo "  $PYTHON -m pip install 'pandas==1.3.5'"
    exit 1
fi
echo "OK — pandas is still 1.3.5"

# ---------------------------------------------------------------------------
# 6. Test Streamlit import
# ---------------------------------------------------------------------------
echo ""
echo "=== 6/7  Testing Streamlit import ==="
$PYTHON -c "import streamlit as st; print('streamlit:', st.__version__)"

# ---------------------------------------------------------------------------
# 7. Test load_regions import
# ---------------------------------------------------------------------------
echo ""
echo "=== 7/7  Testing islets import ==="
$PYTHON -c "from islets.Regions import load_regions; print('load_regions: OK')"

echo ""
echo "=== All checks passed. Run the app with: ==="
echo "  cd $SCRIPT_DIR"
echo "  $PYTHON -m streamlit run thesis_streamlit_app.py"
