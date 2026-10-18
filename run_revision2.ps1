# Reproduces every revision-2 result (Windows PowerShell). Requires the two DLIS
# files under data/raw/ (see README.md) and an activated venv.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
python scripts/rev2_build_data.py
python scripts/rev2_run_cv.py --protocols blocked random stratdepth
python scripts/rev2_run_cv.py --protocols lc --seeds 42 123 456
python scripts/rev2_run_crosswell.py
python scripts/rev2_run_improve.py
python scripts/rev2_run_binary.py
python scripts/rev2_analysis.py
python scripts/rev2_improve_analysis.py
python scripts/rev2_binary_analysis.py
