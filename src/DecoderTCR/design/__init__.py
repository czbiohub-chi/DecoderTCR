"""Generative use of DecoderTCR: what peptides does a TCR and HLA expect?

    import DecoderTCR as dt

    # per-position amino-acid profile, one masked forward pass
    prof = dt.peptide_profile({"HLA_a": HLA, "TCR_a": TCR_A, "TCR_b": TCR_B}, length=9)
    dt.sequence_logo(prof, save="logo.png")

    # sample a peptide library from that profile: no model, no GPU
    peptides, stats = dt.sample_from_profile(prof, n=1000, temperature=1.2)

    # or profile and sample in one call
    designs = dt.design_peptides({"HLA_a": HLA, "TCR_a": TCR_A, "TCR_b": TCR_B}, length=9)

    # or with the paper's iterative entropy-guided decoding
    designs = dt.design_peptides({...}, length=9, method="iegr")
"""

from DecoderTCR.design.profile import (build_masked_entry, consensus, consensus, peptide_profile,
                                       region_profile, sequence_logo)
from DecoderTCR.design.generate import design_peptides, sample_from_profile
from DecoderTCR.design.iegr import block_gibbs, iegr, iegr_profile

__all__ = ["peptide_profile", "region_profile", "sequence_logo", "consensus",
           "sample_from_profile", "design_peptides", "iegr", "iegr_profile",
           "block_gibbs", "build_masked_entry"]
