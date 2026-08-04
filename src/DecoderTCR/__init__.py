"""DecoderTCR V0.3. TCR-pMHC masked-LM scoring on ESM2 and ESMC backbones.

Two model families share one scoring contract (masked-peptide pseudo-log-likelihood):

- DecoderTCR  (ESM2 backbone):  DecoderTCR_650M, DecoderTCR_3B
- DecoderTCR-ESMC (ESMC backbone):  DecoderTCR-ESMC_300M, DecoderTCR-ESMC_600M, DecoderTCR-ESMC_6B

Quick start. The default input is V/J genes + CDR3 + HLA allele + peptide (reconstructed and
scored, so you never supply full sequences):
    import DecoderTCR as dt
    df = dt.score_from_components([
        {"trav": "TRAV21", "traj": "TRAJ6", "cdr3a": "CAVRPGGAGPFFVVF",
         "trbv": "TRBV7-9", "trbj": "TRBJ2-7", "cdr3b": "CASSLGQAYEQYF",
         "hla": "HLA-B*27:05", "peptide": "LRVMMLAPF"}], model="DecoderTCR-ESMC_600M")

If you already have full HLA/TCR sequences, score them directly:
    df  = dt.score([{"HLA_a": HLA, "Peptide": "YLQPRTFLL", "TCR_a": TCRA, "TCR_b": TCRB}])
    pll = dt.score_one("YLQPRTFLL", hla_a=HLA, tcr_a=TCRA, tcr_b=TCRB)
    emb = dt.embed(df)                       # (N, d) backbone embeddings

Going the other way, ask what peptides a TCR and HLA expect. No peptide is needed, only a
length. One masked forward gives the per-position amino-acid profile:
    prof = dt.peptide_profile({"HLA_a": HLA, "TCR_a": TCRA, "TCR_b": TCRB}, length=9)
    dt.sequence_logo(prof, save="logo.png")
    designs = dt.design_peptides({"HLA_a": HLA, "TCR_a": TCRA, "TCR_b": TCRB}, length=9)

Lower-level loader + scorer (what the above wraps):
    model, n_layers = dt.load("DecoderTCR-ESMC_600M", device="cuda")
    scores = dt.run_pll_benchmark(model, n_layers, entries, "peptide", "cuda")
"""

__version__ = "0.3.0"

from DecoderTCR.api import score, score_one, embed
from DecoderTCR.design import (design_peptides, iegr, peptide_profile, region_profile,
                               sequence_logo)
from DecoderTCR.reconstruct import score_from_components, list_alleles
from DecoderTCR.utils.model_zoo import load
from DecoderTCR.utils.scoring import run_pll_benchmark

__all__ = ["score_from_components", "list_alleles", "score", "score_one", "embed",
           "peptide_profile", "region_profile", "sequence_logo", "design_peptides", "iegr",
           "load", "run_pll_benchmark", "__version__"]
