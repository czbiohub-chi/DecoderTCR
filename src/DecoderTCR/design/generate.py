"""Generate peptide sequences for a TCR and HLA.

    import DecoderTCR as dt

    designs = dt.design_peptides({"HLA_a": HLA, "TCR_a": TCR_A, "TCR_b": TCR_B},
                                 length=9, n=10)

`method="one_shot"` (default) draws each position from its marginal in a single masked forward
pass, for any number of sequences. `method="iegr"` resamples positions in the context of those
already committed, at one forward pass per residue plus the Gibbs rounds.

Designs carry the masked-peptide `pll` that `dt.score` reports and are sorted best first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from DecoderTCR.constants import AA20, AA20_IDS
from DecoderTCR.model.tokenizer import TCRpMHCTokenizer
from DecoderTCR.design.profile import (build_masked_entry, consensus, masked_logits,
                                       profile_from_logits, region_positions)

METHODS = ("one_shot", "iegr")


def _one_shot(profile: pd.DataFrame, n: int, temperature: float, seed: int) -> list[str]:
    """Draw sequences position by position from an already computed profile."""
    probs = profile[list(AA20)].to_numpy()
    if temperature <= 0:
        return [consensus(profile)]

    if temperature != 1.0:                       # re-sharpen the marginals
        logp = np.log(probs + 1e-12) / temperature
        probs = np.exp(logp - logp.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    letters = np.array(list(AA20))
    seen, out = set(), []
    for _ in range(n * 20):                      # oversample, then keep the unique draws
        seq = "".join(letters[rng.choice(len(AA20), p=row)] for row in probs)
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
        if len(out) >= n:
            break
    return out


# Default sampling temperature per method.
DEFAULT_TEMPERATURE = {"one_shot": 1.0, "iegr": 0.1}


def design_peptides(data, length: int, n: int = 10, method: str = "one_shot",
                    temperature: float | None = None, seed: int = 42, rescore: bool = True,
                    gibbs_rounds: int = 10, gibbs_subset_size: int = 5, model=None, *,
                    num_layers=None, device="cuda", from_genes: bool = False,
                    checkpoint=None, backbone=None, arch=None) -> pd.DataFrame:
    """Generate candidate peptides of `length` for the TCR and HLA in `data`.

    `data` is a dict (or Series, or one-row DataFrame) of sequences (`HLA_a`, `HLA_b`, `TCR_a`,
    `TCR_b`), or of V/J genes plus CDR3 plus an HLA allele when `from_genes=True`. No peptide
    is needed, only the length to generate.

    `method="one_shot"` (default) samples each position from its marginal in a single forward
    pass. `method="iegr"` runs Iterative Entropy-Guided Refinement, which is slower and
    accounts for dependencies between positions.

    `temperature` defaults to 1.0 for one-shot and 0.1 for IEGR. Lower values sharpen sampling
    toward the most probable residues, and 0 returns the consensus peptide alone.

    Returns a DataFrame with `sequence`, `method`, `phase`, `step`, and, when `rescore=True`,
    the masked-peptide `pll` for each design, sorted best first.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE[method]

    if method == "iegr":
        from DecoderTCR.design.iegr import iegr
        out = iegr(data, region="peptide", length=length, n=n, temperature=temperature,
                   gibbs_rounds=gibbs_rounds, gibbs_subset_size=gibbs_subset_size, seed=seed,
                   model=model, num_layers=num_layers, device=device, from_genes=from_genes,
                   checkpoint=checkpoint, backbone=backbone, arch=arch)
    else:
        from DecoderTCR.api import _resolve_model
        entry = build_masked_entry(data, region="peptide", length=length, from_genes=from_genes)
        mdl, n_layers, dev, _, _ = _resolve_model(model, num_layers, device, checkpoint,
                                                  backbone, arch)
        tok = TCRpMHCTokenizer(entry, mask_probs=None, use_sep=False)
        positions = region_positions(tok, "peptide", entry)
        profile = profile_from_logits(masked_logits(mdl, n_layers, tok, positions, dev))
        out = pd.DataFrame({"sequence": _one_shot(profile, n, temperature, seed)})
        out["phase"] = 0
        out["step"] = range(len(out))

    out.insert(1, "method", method)
    if rescore and len(out):
        out["pll"] = _rescore(data, out["sequence"].tolist(), model, num_layers, device,
                              from_genes, checkpoint, backbone, arch)
        out = out.sort_values("pll", ascending=False, ignore_index=True)
    return out


def _rescore(data, sequences, model, num_layers, device, from_genes, checkpoint, backbone, arch):
    """Masked-peptide PLL for each design, the same score `dt.score` reports."""
    from DecoderTCR.api import score

    base = build_masked_entry(data, region="peptide", length=len(sequences[0]),
                              from_genes=from_genes)
    entries = []
    for s in sequences:
        seqs = dict(base["sequences"])
        seqs["Peptide"] = s
        entries.append({"sequences": seqs, "pocket_idx": base["pocket_idx"], "meta_data": {}})
    return score(entries, model, num_layers=num_layers, device=device, checkpoint=checkpoint,
                 backbone=backbone, arch=arch, return_dataframe=False)
