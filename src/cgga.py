"""CGGA (Chinese Glioma Genome Atlas) mRNAseq_693 + mRNAseq_325 loader: external validation cohort.

Expression is RSEM; we use log2(x+1) and z-score every gene WITHIN each batch (693 / 325) and cohort subset.
Survival is converted from days to months. Gender: male = 1.
"""
import numpy as np
import pandas as pd

from common import DATA

RAW = DATA / "cgga_raw"


def _read(batch):
    clin = pd.read_csv(RAW / f"CGGA.mRNAseq_{batch}_clinical.20200506.txt", sep="\t")
    expr = pd.read_csv(RAW / f"CGGA.mRNAseq_{batch}.RSEM-genes.20200506.txt", sep="\t", index_col="Gene_Name")
    clin["batch"] = batch
    return clin, expr


def load(subset="primary_gbm", genes=None):
    parts = [_read("693"), _read("325")]
    clin = pd.concat([p[0] for p in parts], ignore_index=True)
    ren = {"Censor (alive=0; dead=1)": "event", "Radio_status (treated=1;un-treated=0)": "radio",
           "Chemo_status (TMZ treated=1;un-treated=0)": "chemo", "IDH_mutation_status": "idh",
           "MGMT" "p_methylation_status": "mgmt", "OS": "os_days", "Age": "age", "Gender": "gender"}
    clin = clin.rename(columns=ren)
    if subset == "primary_gbm":
        sel = (clin["Histology"] == "GBM") & (clin["PRS_type"] == "Primary")
    elif subset == "primary_gbm_idhwt":
        sel = (clin["Histology"] == "GBM") & (clin["PRS_type"] == "Primary") & (clin["idh"] == "Wildtype")
    elif subset == "all_gbm":
        sel = clin["Histology"].isin(["GBM", "rGBM", "sGBM"])
    else:
        raise ValueError(subset)
    clin = clin[sel & clin["os_days"].notna() & clin["event"].notna() & clin["age"].notna()].drop_duplicates("CGGA_ID")
    Zs, Rs = [], []
    for batch, (c, e) in zip(["693", "325"], parts):
        ids = [i for i in clin.loc[clin.batch == batch, "CGGA_ID"] if i in e.columns]
        if not ids:
            continue
        x = e[ids].T
        x.columns = [str(g).upper() for g in x.columns]
        x = x.loc[:, ~x.columns.duplicated()]
        L = np.log2(x + 1.0)
        Z = (L - L.mean()) / L.std(ddof=0).replace(0, np.nan)
        Zs.append(Z)
        Rs.append(x)
    Z = pd.concat(Zs)
    R = pd.concat(Rs)
    clin = clin[clin["CGGA_ID"].isin(Z.index)].set_index("CGGA_ID").loc[Z.index]
    if genes is not None:
        Z, R = Z.reindex(columns=genes), R.reindex(columns=genes)
    order = np.argsort(clin["os_days"].values, kind="stable")
    d = type("CggaData", (), {})()
    d.pid = np.array(clin.index)[order]
    d.age = clin["age"].values.astype(float)[order]
    d.sex = (clin["gender"] == "Male").values.astype(float)[order]
    d.time = (clin["os_days"].values / 30.4375)[order]
    d.event = clin["event"].values.astype(np.int64)[order]
    d.genes = np.array(Z.columns)
    d.Z = np.nan_to_num(Z.values.astype(np.float32)[order], nan=0.0)
    d.Z_nan = np.isnan(Z.values.astype(np.float32)[order])
    d.rsem = R.values.astype(np.float32)[order]
    d.n = len(d.pid)
    ex = pd.DataFrame(index=d.pid)
    ex["batch"] = clin["batch"].values[order]
    ex["idh_mut"] = np.where(clin["idh"].isna(), np.nan, (clin["idh"] == "Mutant").astype(float))[order]
    ex["mgmt_meth"] = np.where(clin["mgmt"].isna(), np.nan, (clin["mgmt"] == "methylated").astype(float))[order]
    ex["radio"] = clin["radio"].values[order]
    ex["chemo"] = clin["chemo"].values[order]
    d.extra = ex
    return d
