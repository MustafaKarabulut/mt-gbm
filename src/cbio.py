"""Minimal cBioPortal REST client."""
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

BASE = "https://www.cbioportal.org/api"
_S = requests.Session()


def _req(method, path, retries=5, **kw):
    last = None
    for i in range(retries):
        try:
            r = _S.request(method, BASE + path, timeout=300, **kw)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:  # network hiccup
            last = e
        time.sleep(2 * (i + 1))
    raise last


def get(path, **params):
    return _req("GET", path, params=params)


def post(path, body, **params):
    return _req("POST", path, json=body, params=params)


def protein_coding_genes():
    """DataFrame(symbol, entrez) of the protein-coding genes known to cBioPortal."""
    rows = get("/genes", direction="ASC", pageNumber=0, pageSize=10000000, projection="SUMMARY")
    df = pd.DataFrame([(g["hugoGeneSymbol"].upper(), g["entrezGeneId"]) for g in rows if g["type"] == "protein-coding"],
                      columns=["symbol", "entrez"])
    return df.drop_duplicates("symbol").drop_duplicates("entrez").reset_index(drop=True)


def molecular_matrix(study, profile_suffix, entrez, chunk=1000, workers=4, sample_list=None):
    """Fetch a molecular profile as DataFrame(samples x entrez ids) in chunks."""
    pid = f"{study}_{profile_suffix}"
    sl = sample_list or f"{study}_all"
    parts = [list(map(int, c)) for c in np.array_split(np.asarray(entrez), max(1, int(np.ceil(len(entrez) / chunk))))]

    def one(ids):
        data = post(f"/molecular-profiles/{pid}/molecular-data/fetch", {"entrezGeneIds": ids, "sampleListId": sl},
                    projection="SUMMARY")
        if isinstance(data, dict) and "message" in data:
            raise RuntimeError(data["message"])
        return pd.DataFrame({"sample": [d["sampleId"] for d in data], "entrez": [d["entrezGeneId"] for d in data],
                             "value": [d["value"] for d in data]})

    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(one, parts))
    long = pd.concat(frames, ignore_index=True)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    wide = long.pivot_table(index="sample", columns="entrez", values="value", aggfunc="first")
    return wide


def clinical(study, level="PATIENT"):
    rows = get(f"/studies/{study}/clinical-data", clinicalDataType=level, direction="ASC", pageNumber=0,
               pageSize=10000000, projection="SUMMARY")
    return pd.DataFrame([{"patientId": r["patientId"], "sampleId": r.get("sampleId"),
                          "attr": r["clinicalAttributeId"], "value": r["value"]} for r in rows])


def merged_os_clinical(studies, attrs=("AGE", "SEX", "OS_MONTHS", "OS_STATUS")):
    """Patient-level age, sex and overall survival merged across studies; once a patient is recorded as deceased that status and time are kept."""
    out = {}
    for st in studies:
        df = clinical(st, "PATIENT")
        for r in df.itertuples():
            a = r.attr
            if a not in attrs:
                continue
            v = r.value
            if a == "SEX":
                v = 1 if re.match("female", v, re.I) is None else 0
            elif a == "OS_STATUS":
                v = int(v[0])
            else:
                try:
                    v = float(v)
                except ValueError:
                    continue
            d = out.setdefault(r.patientId, {k: None for k in attrs})
            if d["OS_STATUS"] == 1:
                if a == "OS_STATUS":
                    continue
                if a == "OS_MONTHS" and d["OS_MONTHS"] is not None:
                    continue
            d[a] = v
    return pd.DataFrame.from_dict(out, orient="index").rename_axis("patientId").reset_index()


def mutated_samples(study, entrez_gene_id):
    rows = get(f"/molecular-profiles/{study}_mutations/mutations", sampleListId=f"{study}_all",
               entrezGeneId=entrez_gene_id, projection="SUMMARY")
    return pd.DataFrame([{"sampleId": r["sampleId"], "patientId": r["patientId"], "protein": r.get("proteinChange"),
                          "type": r.get("mutationType")} for r in rows])


def fetch_union_zscores(studies, genes_csv, out_dir):
    """Fetch RNASeqV2 all-sample z-scores of further studies (for the 3-release union cohort)."""
    genes = pd.read_csv(genes_csv)
    t0 = time.time()
    for st in studies:
        m = molecular_matrix(st, "rna_seq_v2_mrna_median_all_sample_Zscores", genes.entrez.values)
        m.to_pickle(f"{out_dir}/tcga_z_all__{st}.pkl")
        print(f"{st}: {m.shape} samples x genes   ({time.time()-t0:.0f}s)", flush=True)
