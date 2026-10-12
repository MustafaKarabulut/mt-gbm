"""Load the cached TCGA-GBM data into arrays that the fast Cox engine can consume.

Conventions:
  * SEX: male = 1, female = 0;  OS_STATUS: 1 = deceased
  * over-expression  : z >= +2   -> column  GENE_overexp
  * under-expression : z <= -2   -> column  GENE_underexp
Cohort 'union' merges the three TCGA GBM releases (per patient/gene the FIRST study wins, order
pub2013 -> PanCan Atlas -> Firehose), cohort 'pancan' uses the PanCan Atlas only.
Patients with a missing clinical value are dropped.
Rows are sorted by OS_MONTHS ascending so that any row subset stays time-sorted.
"""
import numpy as np
import pandas as pd

from common import DATA, Z_CUTOFF, mt_blocks, read_mt_genes

STUDY_ORDER = ["gbm_tcga_pub2013", "gbm_tcga_pan_can_atlas_2018", "gbm_tcga"]


class GbmData:
    pass


def _by_patient(df, ent2sym):
    smp = pd.Series(df.index, index=df.index)
    pid = smp.str.slice(0, 12)
    order = np.argsort(~smp.str.endswith("-01").values, kind="stable")
    keep = ~pid.iloc[order].duplicated().values
    sel = df.index[order][keep]
    out = df.loc[sel].copy()
    out.index = pid.loc[sel].values
    out.columns = [ent2sym[c] for c in out.columns]
    return out.loc[:, ~out.columns.duplicated()]


def _extra_clinical(pids):
    long = pd.read_csv(DATA / "tcga_extra_clinical_long.csv")
    ex = pd.DataFrame(index=pids)

    def pick(study, attr):
        s = long[(long.study == study) & (long.attr == attr)].drop_duplicates("patientId")
        return s.set_index("patientId")["value"].reindex(pids)

    ex["subtype"] = pick("gbm_tcga_pub2013", "EXPRESSION_SUBTYPE").values
    m = pick("gbm_tcga_pub2013", "MGMT_STATUS")
    ex["mgmt_meth"] = np.where(m.isna(), np.nan, (m == "METHYLATED").astype(float))
    i = pick("gbm_tcga_pub2013", "IDH1_MUTATION")
    idh = pd.read_csv(DATA / "tcga_idh1_mutations.csv")
    idh_p = set(idh["patientId"])
    ex["idh1_mut"] = np.where(i.isna(), np.where(pd.Index(pids).isin(idh_p), 1.0, np.nan), (i != "WT").astype(float))
    ex["kps"] = pd.to_numeric(pick("gbm_tcga", "KARNOFSKY_PERFORMANCE_SCORE"), errors="coerce").values
    return ex


def load(cohort="union", cutoff=Z_CUTOFF):
    genes_tab = pd.read_csv(DATA / "genes.csv")
    ent2sym = dict(zip(genes_tab.entrez, genes_tab.symbol))
    if cohort == "pancan":
        frames = [_by_patient(pd.read_pickle(DATA / "tcga_z_all.pkl"), ent2sym)]
    else:
        files = {"gbm_tcga_pan_can_atlas_2018": DATA / "tcga_z_all.pkl",
                 "gbm_tcga_pub2013": DATA / "tcga_z_all__gbm_tcga_pub2013.pkl",
                 "gbm_tcga": DATA / "tcga_z_all__gbm_tcga.pkl"}
        frames = [_by_patient(pd.read_pickle(files[s]), ent2sym) for s in STUDY_ORDER]
    Z = frames[0]
    for f in frames[1:]:
        Z = Z.combine_first(f)
    rs = _by_patient(pd.read_pickle(DATA / "tcga_rsem.pkl"), ent2sym)

    clin = pd.read_csv(DATA / "tcga_clinical_merged.csv").set_index("patientId")
    clin = clin.reindex(Z.index)
    complete = ~clin.isna().any(axis=1)
    n_all, n_drop = len(Z), int((~complete).sum())
    Z, clin = Z.loc[complete.values], clin.loc[complete.values]

    d = GbmData()
    d.cohort, d.n_before_drop, d.n_dropped_incomplete = cohort, n_all, n_drop
    order = np.argsort(clin["OS_MONTHS"].values, kind="stable")
    d.order = order
    d.pid = np.array(Z.index)[order]
    d.age = clin["AGE"].values.astype(float)[order]
    d.sex = clin["SEX"].values.astype(float)[order]
    d.time = clin["OS_MONTHS"].values.astype(float)[order]
    d.event = clin["OS_STATUS"].values.astype(np.int64)[order]
    d.genes = np.array(Z.columns)
    d.Z = np.nan_to_num(Z.values.astype(np.float32)[order], nan=0.0)
    d.Z_nan = np.isnan(Z.values.astype(np.float32)[order])
    d.rsem = rs.reindex(index=d.pid, columns=Z.columns).values.astype(np.float32)
    G = len(d.genes)
    d.B = np.hstack([d.Z >= cutoff, d.Z <= -cutoff]).astype(np.uint8)      # n x 2G
    d.colnames = np.array([f"{g}_overexp" for g in d.genes] + [f"{g}_underexp" for g in d.genes])
    d.col_gene = np.concatenate([np.arange(G), np.arange(G)])
    mt = read_mt_genes()
    blocks = mt_blocks(mt)
    d.gene_block = np.array([blocks.get(g) or "" for g in d.genes])
    d.gene_is_mt = d.gene_block != ""
    d.col_is_mt = d.gene_is_mt[d.col_gene]
    d.n = len(d.pid)
    d.mt_listed = len(set(mt))
    d.extra = _extra_clinical(d.pid)
    return d


def summary(d):
    return (f"cohort={d.cohort}  n={d.n} (before dropping incomplete clinical: {d.n_before_drop}, dropped {d.n_dropped_incomplete})  "
            f"events={int(d.event.sum())}  genes={len(d.genes)}  MT genes present={int(d.gene_is_mt.sum())} (listed {d.mt_listed})  "
            f"binary columns={d.B.shape[1]}")
