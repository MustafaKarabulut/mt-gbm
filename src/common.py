"""Shared constants and helpers for the MT-gene / GBM analysis."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
LOGS = ROOT / "logs"
RESOURCES = ROOT / "resources"
for _d in (DATA, RESULTS, FIGS, LOGS):
    _d.mkdir(exist_ok=True)

STUDY = "gbm_tcga_pan_can_atlas_2018"   # cBioPortal study with the RNA-seq V2 data
Z_CUTOFF = 2.0                          # |z| >= 2 -> over-/under-expression flag

# Blocks of the gene list, identified by their first / last gene in file order.
#   A  matrisome-like (ECM glycoproteins, collagens, proteoglycans, ECM-affiliated,
#      regulators, secreted factors)                       ABI3BP ... ZFP91
#   B  matrisome-related pseudogenes / legacy symbols       ADAM1A ... UNQ5830
#   C  stemness / EMT / Notch / Hippo-linked signalling     AXL    ... ZEB2
#   D  mechanotransduction core (channels, adhesion, YAP)   ACTN1  ... WWTR1
BLOCKS = [("A", "ABI3BP", "ZFP91"), ("B", "ADAM1A", "UNQ5830"),
          ("C", "AXL", "ZEB2"), ("D", "ACTN1", "WWTR1")]


def read_mt_genes():
    """Gene symbols of the MT list in file order (resources/mt_genes.txt; 1,233 entries, 1,229 unique)."""
    txt = (RESOURCES / "mt_genes.txt").read_text(encoding="utf-8")
    return [g.strip().upper() for g in txt.split() if g.strip()]


def mt_blocks(genes=None):
    """Return {gene -> block letter}. Genes in several blocks get the most specific one (D > C > A > B)."""
    genes = genes or read_mt_genes()
    pos = {}
    for i, g in enumerate(genes):
        pos.setdefault(g, []).append(i)
    ranges = []
    for name, first, last in BLOCKS:
        s = genes.index(first.upper())
        e = len(genes) - 1 - genes[::-1].index(last.upper())
        ranges.append((name, s, e))
    prio = {"D": 3, "C": 2, "A": 1, "B": 0}
    out = {}
    for g, ps in pos.items():
        best = None
        for name, s, e in ranges:
            if any(s <= p <= e for p in ps):
                if best is None or prio[name] > prio[best]:
                    best = name
        out[g] = best
    return out
