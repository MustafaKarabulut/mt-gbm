#!/usr/bin/env bash
# Complete analysis. Several hours on an 18-core machine; every step writes to results/ and can be run on its own.
# Usage: bash run_all.sh [N_JOBS]        (python interpreter: $PYTHON, default "python")
set -euo pipefail
JOBS=${1:-8}
PY=${PYTHON:-python}
mkdir -p logs
$PY 01_fetch.py            2>&1 | tee logs/01_fetch.log
$PY 01b_fetch_union.py     2>&1 | tee logs/01b_fetch_union.log
$PY 02_validate_cox.py     2>&1 | tee logs/02_validate_cox.log
$PY 03_gate1.py --jobs "$JOBS"                                    2>&1 | tee logs/03_gate1_main.log
$PY 03_gate1.py --exps grp --n_obs 10 --n_null 100 --jobs "$JOBS" 2>&1 | tee logs/03_gate1_grp.log
$PY 04_gate2.py ab "$JOBS" 2>&1 | tee logs/04_gate2.log          # needs data/cgga_raw (see README)
$PY 04b_audit.py           2>&1 | tee logs/04b_audit.log
$PY 04c_modules_cv.py      2>&1 | tee logs/04c_modules_cv.log
$PY 05_gate3_sim.py --n 30 --jobs "$JOBS"                         2>&1 | tee logs/05_gate3.log
$PY 06_insample_null.py 60 "$JOBS" 2>&1 | tee logs/06_insample_null.log
$PY 07_realcv.py "$JOBS"   2>&1 | tee logs/07_realcv.log
$PY 08_coassoc.py          2>&1 | tee logs/08_coassoc.log
$PY 10_minp.py             2>&1 | tee logs/10_minp.log
$PY 11_crosscohort_genes.py 2>&1 | tee logs/11_crosscohort_genes.log
$PY 12_strata.py "$JOBS"   2>&1 | tee logs/12_strata.log
$PY 13_random_column_sets.py 2>&1 | tee logs/13_random_column_sets.log
$PY 09_figures.py          2>&1 | tee logs/09_figures.log
