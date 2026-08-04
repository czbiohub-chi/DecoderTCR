"""Conditional amino-acid profiles read out of a masked region.

    import DecoderTCR as dt

    prof = dt.peptide_profile({"HLA_a": HLA, "TCR_a": TCR_A, "TCR_b": TCR_B}, length=9)
    print("".join(prof[list(AA20)].idxmax(axis=1)))   # consensus peptide
    dt.sequence_logo(prof)

Masks the whole target region and runs the model once, returning the marginal distribution at
each position. Takes the region length, not its sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from DecoderTCR.constants import AA20, AA20_IDS, MASK_IDX
from DecoderTCR.model.tokenizer import TCRpMHCTokenizer

# Fills a region that is about to be masked.
PLACEHOLDER_AA = "A"

_SEQ_ALIASES = {
    "HLA_a": ["hla_a", "hla", "mhc", "mhc_a"],
    "HLA_b": ["hla_b", "b2m", "mhc_b"],
    "Peptide": ["peptide", "epitope", "pep"],
    "TCR_a": ["tcr_a", "tcra", "tcr_alpha"],
    "TCR_b": ["tcr_b", "tcrb", "tcr_beta"],
}


def _norm_sequences(data) -> dict[str, str]:
    """Pull canonical sequence fields out of a dict, Series, or one-row DataFrame."""
    if isinstance(data, pd.DataFrame):
        if len(data) != 1:
            raise ValueError(f"expected a single complex, got {len(data)} rows. "
                             f"Loop over rows, or use the CLI with -i for a batch.")
        data = data.iloc[0]
    if isinstance(data, pd.Series):
        data = data.to_dict()
    if not isinstance(data, dict):
        raise TypeError(f"expected a dict, Series, or one-row DataFrame, got {type(data).__name__}")

    lower = {str(k).lower(): v for k, v in data.items()}
    seqs = {}
    for canonical, aliases in _SEQ_ALIASES.items():
        for a in aliases:
            v = lower.get(a)
            if v is not None and not (isinstance(v, float) and v != v):
                seqs[canonical] = str(v).strip()
                break
        seqs.setdefault(canonical, "")
    return seqs


def build_masked_entry(data, region: str = "peptide", length: int | None = None,
                       from_genes: bool = False) -> dict:
    """Build a scoring entry whose `region` can be masked, with no sequence needed there.

    `length` applies to `region="peptide"` and sets how many residues to profile. The CDR3
    regions take their length from the template TCR and reject `length`. With
    `from_genes=True` the TCR chains are stitched from V/J genes plus CDR3 and the HLA is
    looked up by allele, as in `score_from_components`.
    """
    if region == "peptide":
        if length is None:
            raise ValueError("peptide profiling needs `length` (the peptide length to profile)")
        if int(length) < 1:
            raise ValueError(f"`length` must be at least 1, got {length}")
        length = int(length)
    elif length is not None:
        raise ValueError(f"`length` only applies to region='peptide'. Region {region!r} takes "
                         f"its length from the template sequence.")

    if from_genes:
        from DecoderTCR.reconstruct.components import (_entry, _normalize_rows,
                                                       reconstruct_components)
        row = dict(data.iloc[0]) if isinstance(data, pd.DataFrame) else dict(data)
        # Reconstruction requires a peptide field.
        if region == "peptide":
            row["peptide"] = PLACEHOLDER_AA * length
        elif not any(str(row.get(k, "")).strip() for k in ("peptide", "epitope", "pep")):
            raise ValueError(f"region {region!r} designs part of the TCR, so the peptide it "
                             f"should recognize has to be supplied as the conditioning context. "
                             f"Add a `peptide` field, or use region='peptide' to design one.")
        rec = reconstruct_components(_normalize_rows([row]))[0]
        if not rec["ok"]:
            reason = rec.get("tcr_reason") or rec.get("hla_reason") or "unknown"
            raise ValueError(f"could not reconstruct the complex: {reason}")
        return _entry(rec)

    seqs = _norm_sequences(data)
    if region == "peptide":
        seqs["Peptide"] = PLACEHOLDER_AA * length
    if not seqs["Peptide"]:
        raise ValueError("the entry has no peptide. Pass `length` to profile the peptide, or "
                         "supply a Peptide when profiling a CDR3 region.")
    if not (seqs["HLA_a"] or seqs["TCR_a"] or seqs["TCR_b"]):
        raise ValueError("nothing to condition on. Supply at least an HLA_a or a TCR chain.")
    return {"sequences": seqs, "pocket_idx": {}, "meta_data": {}}


_CDR_KEY = {"TCR_a": "CDRa", "TCR_b": "CDRb"}
REGIONS = ("peptide", "cdr3a", "cdr3b", "cdr3", "cdr")


def _cdr3_span(tok: TCRpMHCTokenizer, entry: dict | None, chain: str) -> np.ndarray:
    """Sequence-space indices of one chain's CDR3.

    The CDR3 is the third range in a full CDR1/CDR2/CDR3 annotation, or the only range when
    the chain was reconstructed from genes. Both conventions store an inclusive end.
    """
    sl = tok.region_slices.get(chain)
    ranges = ((entry or {}).get("pocket_idx") or {}).get(_CDR_KEY[chain], [])
    if sl is not None and ranges:
        r = ranges[2] if len(ranges) >= 3 else ranges[-1]
        return np.arange(r[0] + sl.start, r[1] + sl.start + 1, dtype=np.intp)
    attr = tok._cdr3a_indices if chain == "TCR_a" else tok._cdr3b_indices
    return np.asarray(attr, dtype=np.intp)


def region_positions(tok: TCRpMHCTokenizer, region: str, entry: dict | None = None) -> np.ndarray:
    """Sequence-space indices of a target region. Token space is these plus 1 for CLS."""
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}. Choose from: {list(REGIONS)}")
    if region == "peptide":
        idx = np.asarray(tok.peptide_indices, dtype=np.intp)
    elif region == "cdr":
        idx = np.asarray(tok.cdr_indices, dtype=np.intp)
    else:
        chains = {"cdr3a": ["TCR_a"], "cdr3b": ["TCR_b"],
                  "cdr3": ["TCR_a", "TCR_b"]}[region]
        idx = np.concatenate([_cdr3_span(tok, entry, c) for c in chains]) if chains else []
        idx = np.asarray(idx, dtype=np.intp)
    if idx.size == 0:
        raise ValueError(f"region {region!r} is empty for this complex. A CDR3 region needs the "
                         f"CDR3 span, which `from_genes=True` records, and the matching TCR chain.")
    return idx


@torch.no_grad()
def masked_logits(model, num_layers: int, tok: TCRpMHCTokenizer, positions: np.ndarray,
                  device) -> torch.Tensor:
    """Mask `positions` and return the logits there, shape (len(positions), vocab)."""
    ids = tok.original_ids.clone()
    token_idx = torch.as_tensor(positions, dtype=torch.long) + 1   # +1 for CLS
    ids[token_idx] = MASK_IDX
    out = model(ids.unsqueeze(0).to(device), repr_layers=[num_layers], return_contacts=False)
    return out["logits"][0][token_idx].float().cpu()


def profile_from_logits(logits: torch.Tensor) -> pd.DataFrame:
    """Turn region logits into a per-position amino-acid profile plus its entropy.

    The softmax covers the 20 standard amino acids only.
    """
    # float64, required by logomaker.
    probs = torch.softmax(logits[:, AA20_IDS].double(), dim=-1).numpy()
    prof = pd.DataFrame(probs, columns=list(AA20))
    prof.index = pd.RangeIndex(1, len(prof) + 1, name="position")
    prof["entropy"] = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    return prof


def region_profile(data, region: str = "peptide", length: int | None = None, model=None, *,
                   num_layers=None, device="cuda", from_genes: bool = False,
                   checkpoint=None, backbone=None, arch=None) -> pd.DataFrame:
    """Per-position amino-acid profile for a masked region of one complex.

    Returns a DataFrame indexed by position (1-based) with one column per amino acid and an
    `entropy` column. Rows sum to 1 over the 20 amino acid columns. The consensus sequence is
    `"".join(prof[list(AA20)].idxmax(axis=1))`.
    """
    from DecoderTCR.api import _resolve_model

    entry = build_masked_entry(data, region=region, length=length, from_genes=from_genes)
    mdl, n, dev, _, _ = _resolve_model(model, num_layers, device, checkpoint, backbone, arch)
    tok = TCRpMHCTokenizer(entry, mask_probs=None, use_sep=False)
    positions = region_positions(tok, region, entry)
    return profile_from_logits(masked_logits(mdl, n, tok, positions, dev))


def peptide_profile(data, length: int, model=None, *, num_layers=None, device="cuda",
                    from_genes: bool = False, checkpoint=None, backbone=None,
                    arch=None) -> pd.DataFrame:
    """Peptide amino-acid profile conditioned on the HLA and TCR of one complex.

    `data` is a dict (or Series, or one-row DataFrame) of sequences (`HLA_a`, `HLA_b`, `TCR_a`,
    `TCR_b`), or of V/J genes plus CDR3 plus an HLA allele when `from_genes=True`. `length` is
    the peptide length to profile: no peptide sequence is required or used.

    Returns a DataFrame indexed by peptide position (1-based) with one column per amino acid
    plus an `entropy` column. Leave out the TCR chains to profile the HLA alone.
    """
    return region_profile(data, region="peptide", length=length, model=model,
                          num_layers=num_layers, device=device, from_genes=from_genes,
                          checkpoint=checkpoint, backbone=backbone, arch=arch)


def consensus(profile: pd.DataFrame) -> str:
    """Most probable residue at each position of a profile."""
    return "".join(profile[list(AA20)].idxmax(axis=1))


def sequence_logo(profile: pd.DataFrame, units: str = "bits", ax=None, save: str | Path | None = None,
                  title: str | None = None):
    """Draw a sequence logo for a profile and return the matplotlib Axes.

    `units="bits"` scales each stack by its information content. `units="probability"` draws the
    raw probabilities. Pass `save` to write a figure file.
    """
    import logomaker
    import matplotlib.pyplot as plt

    if units not in ("bits", "probability"):
        raise ValueError(f"units must be 'bits' or 'probability', got {units!r}")

    mat = profile[list(AA20)].copy()
    mat.index = range(len(mat))
    if units == "bits":
        mat = logomaker.transform_matrix(mat, from_type="probability", to_type="information")

    if ax is None:
        _, ax = plt.subplots(figsize=(0.6 * len(mat) + 1, 2.4))
    logomaker.Logo(mat, ax=ax)
    ax.set_xticks(range(len(mat)))
    ax.set_xticklabels(range(1, len(mat) + 1))
    ax.set_xlabel("position")
    ax.set_ylabel("bits" if units == "bits" else "probability")
    if units == "bits":
        ax.set_ylim(0, np.log2(len(AA20)))
    if title:
        ax.set_title(title)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.tight_layout()
        ax.figure.savefig(save, dpi=200)
    return ax
