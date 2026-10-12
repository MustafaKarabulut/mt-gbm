"""Step 1: pull TCGA-GBM (PanCan Atlas) expression + clinical data from cBioPortal and cache to disk."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import cbio
from common import DATA, STUDY

t0 = time.time()
genes = cbio.protein_coding_genes()
print(f"protein-coding genes: {len(genes)}  ({time.time()-t0:.0f}s)", flush=True)
genes.to_csv(DATA / "genes.csv", index=False)

out = {}
for key, suffix in [("z_all", "rna_seq_v2_mrna_median_all_sample_Zscores"),
                    ("rsem", "rna_seq_v2_mrna"),
                    ("z_dip", "rna_seq_v2_mrna_median_Zscores")]:
    m = cbio.molecular_matrix(STUDY, suffix, genes.entrez.values)
    print(f"{key}: {m.shape}  ({time.time()-t0:.0f}s)", flush=True)
    m.to_pickle(DATA / f"tcga_{key}.pkl")
    out[key] = m

# clinical -------------------------------------------------------------------------------
merged = cbio.merged_os_clinical(["gbm_tcga", STUDY, "gbm_tcga_pub2013"])
merged.to_csv(DATA / "tcga_clinical_merged.csv", index=False)
print("merged clinical (3 releases):", merged.shape, flush=True)

pat = cbio.clinical(STUDY, "PATIENT").pivot_table(index="patientId", columns="attr", values="value", aggfunc="first")
smp = cbio.clinical(STUDY, "SAMPLE")
smp_w = smp.pivot_table(index="sampleId", columns="attr", values="value", aggfunc="first")
smp_map = smp.drop_duplicates("sampleId")[["sampleId", "patientId"]].set_index("sampleId")
smp_w = smp_w.join(smp_map)
pat.to_csv(DATA / "tcga_pancan_patient_clinical.csv")
smp_w.to_csv(DATA / "tcga_pancan_sample_clinical.csv")
print("pancan patient clinical", pat.shape, "sample clinical", smp_w.shape, flush=True)

idh = cbio.mutated_samples(STUDY, 3417)  # IDH1
idh.to_csv(DATA / "tcga_idh1_mutations.csv", index=False)
print("IDH1 mutations:", len(idh), idh.protein.value_counts().to_dict(), flush=True)
print(f"done in {time.time()-t0:.0f}s")
