# mt-gbm

Permutation-calibrated Cox regression and genetic-algorithm analyses of mechanotransduction (MT) and matrisome gene sets in glioblastoma (TCGA-GBM, with the CGGA cohort as external validation).

The code builds over-/under-expression features from cBioPortal RNA-seq data, fits ridge-penalised Cox models with a fast numba engine (checked against `lifelines`), runs NSGA-II searches for gene, patient and group subsets, and compares every search result with permutation nulls, gene-set level tests, out-of-sample prediction and simulations with a known ground truth.

A companion repository, [mt-pancancer](https://github.com/MustafaKarabulut/mt-pancancer), applies the gene-set framework to 17 TCGA cancers.

## Contents

| Path | Purpose |
|---|---|
| `src/` | Cox engine (`coxcore.py`, `coxfast.py`), data loaders (`dataset.py`, `cgga.py`, `cbio.py`), NSGA-II problems (`ga.py`), gene-set tools (`gset.py`), baselines (`baselines.py`), constants (`common.py`) |
| `resources/mt_genes.txt` | the MT gene list (1,233 entries, 1,229 unique symbols; four blocks A-D defined by position in the file, see `src/common.py`) |
| `01_fetch.py`, `01b_fetch_union.py` | download TCGA-GBM expression and clinical data from cBioPortal into `data/` |
| `02_validate_cox.py` | validate the Cox/C-index engine against lifelines |
| `03_gate1.py` | genetic-algorithm searches on observed vs permuted data |
| `04_gate2.py`, `04b_audit.py`, `04c_modules_cv.py`, `11_crosscohort_genes.py` | feature audit, gene-set tests in TCGA and CGGA, random-set cross-validation, gene-level replication |
| `05_gate3_sim.py` | simulations with known ground truth |
| `06_insample_null.py`, `13_random_column_sets.py`, `12_strata.py`, `10_minp.py`, `08_coassoc.py` | in-sample analyses and their permutation nulls, random column sets, strata, min-p / FDR, co-association |
| `07_realcv.py` | cross-validated and cross-cohort prediction (elastic net, nested GA) |
| `09_figures.py` | figures |
| `run_all.sh` | the whole analysis in order |
| `tests/test_cox_engine.py` | data-free test of the Cox engine |

## Installation

```bash
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate mt-gbm
python tests/test_cox_engine.py          # should print "engine test passed"
```

The numba Cox engine uses its own Cholesky routine and loop-based matrix-vector products; install `numpy`/`scipy` from conda-forge (OpenBLAS builds) if you see LAPACK errors.

## Data

* TCGA-GBM expression (RNA-seq V2 RSEM and z-scores) and clinical data are fetched automatically from the public cBioPortal API by `01_fetch.py` and `01b_fetch_union.py` (about 275 MB in `data/`).
* The CGGA cohort has to be downloaded by hand from the Chinese Glioma Genome Atlas (http://www.cgga.org.cn, Download: mRNAseq_693 and mRNAseq_325, RSEM expression and clinical files). Place these four files in `data/cgga_raw/`:
  `CGGA.mRNAseq_693.RSEM-genes.20200506.txt`, `CGGA.mRNAseq_693_clinical.20200506.txt`, `CGGA.mRNAseq_325.RSEM-genes.20200506.txt`, `CGGA.mRNAseq_325_clinical.20200506.txt`.
* `resources/mt_genes.txt` is included.

## Running

```bash
bash run_all.sh 8        # 8 parallel workers; several hours on 18 cores
```

or step by step, from the repository root:

```bash
python 01_fetch.py && python 01b_fetch_union.py
python 02_validate_cox.py
python 03_gate1.py --jobs 8                      # GA vs permutation null (about 10 min per experiment at 18 cores)
python 04_gate2.py ab 8                          # gene-set tests (needs CGGA files)
python 05_gate3_sim.py --n 30 --jobs 8
python 06_insample_null.py 60 8
python 07_realcv.py 8
python 09_figures.py
```

Results are written to `results/` (pickles), figures to `figures/`, logs to `logs/`. Random seeds are fixed in the scripts. All of these folders are created on first use and are not tracked by git.

## License

MIT, see `LICENSE`.

## Citation

A manuscript describing this work is in preparation. Until then, please cite this repository.