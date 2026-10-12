"""Data audit: what do the binary over/under-expression columns measure? Saves numbers for the figures and tables."""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import dataset
from common import RESULTS

warnings.filterwarnings("ignore")
d = dataset.load("union")
G, n = len(d.genes), d.n
under, over = d.B[:, G:].mean(0), d.B[:, :G].mean(0)
med = np.nanmedian(d.rsem, axis=0)
med = np.nan_to_num(med, nan=0.0)
zero = np.nanmean(d.rsem == 0, axis=0)
prev = d.B.mean(0)
elig = (prev >= 0.1127) & (prev <= 0.8873)
silent = med[d.col_gene] < 1
mid = (med[d.col_gene] >= 1) & (med[d.col_gene] < 10)
expr = med[d.col_gene] >= 10
a = dict(n=n, G=G, ncols=len(prev), n_elig=int(elig.sum()), elig_under=int(elig[G:].sum()), elig_over=int(elig[:G].sum()),
         elig_silent=int((elig & silent).sum()), elig_mid=int((elig & mid).sum()), elig_expr=int((elig & expr).sum()),
         n_gt25=int((under > .25).sum()), n_gt50=int((under > .5).sum()), n_gt75=int((under > .75).sum()), over_gt25=int((over > .25).sum()),
         under=under, over=over, med=med, zero=zero, mt=d.gene_is_mt, expressed10=int((med >= 10).sum()), expressed1=int((med >= 1).sum()))
idx = np.where(elig[G:] & silent[G:])[0]
a["flag_equals_zero"] = float(np.mean([np.mean((d.B[:, G + i] == 1) == (np.nan_to_num(d.rsem[:, i], nan=-1) == 0)) for i in idx]))
key = ("MUC2 CCL27 OIT3 MMP21 MUC21 MUC16 WNT8A SERPINA9 CLEC4C CTSE TDGF1 MEP1B GP1BA USP26 BRDT GK2 MAGEA8 UTY SRY OR5B12 ADGRF1 MTNR1B "
       "HIGD2B TMCO2 NLRP10 FBXO40 PLEKHG7 TEX19 C17ORF64 EGFR GFAP SOX2 CD44 CHI3L1").split()
gi = {g: i for i, g in enumerate(d.genes)}
rows = [(g, "MT" if d.gene_is_mt[gi[g]] else "", under[gi[g]], over[gi[g]], med[gi[g]], zero[gi[g]]) for g in key if g in gi]
a["key"] = pd.DataFrame(rows, columns=["gene", "MT", "P_under", "P_over", "median_RSEM", "P_RSEM0"])
pickle.dump(a, open(RESULTS / "audit.pkl", "wb"))
print({k: v for k, v in a.items() if not isinstance(v, (np.ndarray, pd.DataFrame))})
print(a["key"].round(2).to_string(index=False))
