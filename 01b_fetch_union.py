"""Step 1b: complete the TCGA-GBM data set (run after 01_fetch.py).
(a) all-sample RNA-seq z-scores of the two further TCGA GBM releases (gbm_tcga_pub2013, gbm_tcga) for the 'union' cohort;
(b) the patient-level annotations used as covariates (expression subtype, MGMT status, IDH1 mutation, Karnofsky score) in long format.
Output: data/tcga_z_all__<study>.pkl, data/tcga_extra_clinical_long.csv"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import cbio
from common import DATA

t0 = time.time()
cbio.fetch_union_zscores(["gbm_tcga_pub2013", "gbm_tcga"], DATA / "genes.csv", str(DATA))

KEEP = {"gbm_tcga_pub2013": ["EXPRESSION_SUBTYPE", "IDH1_MUTATION", "MGMT_STATUS"], "gbm_tcga": ["KARNOFSKY_PERFORMANCE_SCORE"]}
rows = []
for study, attrs in KEEP.items():
    for level in ("PATIENT", "SAMPLE"):
        df = cbio.clinical(study, level)
        df = df[df.attr.isin(attrs)].copy()
        df["study"], df["level"] = study, level
        rows.append(df[["patientId", "sampleId", "attr", "value", "study", "level"]])
long = pd.concat(rows, ignore_index=True)
long.to_csv(DATA / "tcga_extra_clinical_long.csv", index=False)
print("extra clinical:", long.groupby(["study", "attr"]).size().to_dict())
print(f"done in {time.time()-t0:.0f}s")
