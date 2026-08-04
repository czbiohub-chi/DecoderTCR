"""Generative use of DecoderTCR: what peptides does a TCR and HLA expect?

    import DecoderTCR as dt

    # per-position amino-acid profile, one masked forward pass
    prof = dt.peptide_profile({"HLA_a": HLA, "TCR_a": TCR_A, "TCR_b": TCR_B}, length=9)
    dt.sequence_logo(prof, save="logo.png")

    # candidate peptides, sampled from that profile
    designs = dt.design_peptides({"HLA_a": HLA, "TCR_a": TCR_A, "TCR_b": TCR_B}, length=9)

    # or with the paper's iterative entropy-guided decoding
    designs = dt.design_peptides({...}, length=9, method="iegr")
"""

from DecoderTCR.design.profile import (build_masked_entry, consensus, peptide_profile,
                                       region_profile, sequence_logo)
from DecoderTCR.design.generate import design_peptides
from DecoderTCR.design.iegr import iegr

__all__ = ["peptide_profile", "region_profile", "sequence_logo", "consensus",
           "design_peptides", "iegr", "build_masked_entry"]
