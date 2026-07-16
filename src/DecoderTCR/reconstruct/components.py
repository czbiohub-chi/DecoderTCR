"""Orchestrate: (V/J genes + CDR3 + HLA allele + peptide) -> reconstruct -> PLL score."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from DecoderTCR.reconstruct.hla import lookup_hla
from DecoderTCR.reconstruct.tcr import stitch_tcrs

_TCR_KEYS = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b")

_ALIASES = {
    "trav": ["trav", "tcra_v", "va", "v_alpha"],
    "traj": ["traj", "tcra_j", "ja", "j_alpha"],
    "cdr3a": ["cdr3a_extended", "cdr3a", "tcra_cdr3", "cdr3_a", "cdr3_alpha"],
    "trbv": ["trbv", "tcrb_v", "vb", "v_beta"],
    "trbj": ["trbj", "tcrb_j", "jb", "j_beta"],
    "cdr3b": ["cdr3b_extended", "cdr3b", "tcrb_cdr3", "cdr3_b", "cdr3_beta"],
    "hla": ["hla", "allele", "hla_allele", "mhc"],
    "peptide": ["peptide", "epitope", "pep"],
    "name": ["name", "id", "tcr_name"],
    "label": ["label", "binder", "y"],
}
_REQUIRED = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b", "hla", "peptide")


def _normalize_rows(rows) -> list[dict]:
    """Accept a dict, a list of dicts, or a DataFrame and return canonical component rows.

    Case-insensitive column/key aliases are resolved. Required fields are always present
    (blank when absent). Optional `name`/`label` only when given. A required field blank in
    *every* row raises (a likely missing-column mistake). A blank in a single row flows
    through, and that one row fails reconstruction gracefully (`ok=False`).
    """
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    elif isinstance(rows, dict):
        rows = [rows]
    out = []
    for r in rows:
        lower = {str(k).lower(): k for k in r}
        canon: dict = {}
        for canonical, aliases in _ALIASES.items():
            val = ""
            for a in aliases:
                if a in lower:
                    v = r[lower[a]]
                    val = "" if v is None or (isinstance(v, float) and v != v) else str(v).strip()
                    break
            if canonical in _REQUIRED or val:
                canon[canonical] = val
        if "label" in canon:
            try:
                canon["label"] = int(float(canon["label"]))
            except (ValueError, TypeError):
                canon["label"] = 0
        out.append(canon)
    for req in _REQUIRED:
        if out and all(not r.get(req) for r in out):
            raise ValueError(
                f"no '{req}' values found in the input (missing column/field?). "
                f"Accepted aliases: {_ALIASES[req]}"
            )
    return out


def reconstruct_components(rows: list[dict]) -> list[dict]:
    """Reconstruct full HLA_a/HLA_b/TCR_a/TCR_b for each component row (no model needed).

    Each row needs: trav, traj, cdr3a, trbv, trbj, cdr3b, hla, peptide (+ optional name, label).
    Returns the rows augmented with HLA_a, HLA_b, TCR_a, TCR_b, normalized genes, and
    `ok`/`*_reason` flags. One thimble call stitches all TCRs together.
    """
    named = []
    for i, r in enumerate(rows):
        r = dict(r)
        r.setdefault("name", f"pair_{i}")
        named.append(r)

    stitched = stitch_tcrs([{"name": r["name"], **{k: r[k] for k in _TCR_KEYS}} for r in named])

    out = []
    for r in named:
        st = stitched[r["name"]]
        rec = dict(r)
        rec.update(TCR_a=st["TCR_a"], TCR_b=st["TCR_b"],
                   tcr_ok=st["ok"], tcr_reason=st["reason"],
                   TRAV=st.get("TRAV", ""), TRAJ=st.get("TRAJ", ""),
                   TRBV=st.get("TRBV", ""), TRBJ=st.get("TRBJ", ""))
        try:
            ha, hb = lookup_hla(r["hla"])
            rec.update(HLA_a=ha, HLA_b=hb, hla_ok=True, hla_reason="")
        except KeyError as e:
            rec.update(HLA_a="", HLA_b="", hla_ok=False, hla_reason=str(e))
        rec["ok"] = bool(st["ok"] and rec["hla_ok"])
        out.append(rec)
    return out


def _entry(rec: dict) -> dict:
    """Build a scoring entry, locating CDR3 spans in the stitched chains when present."""
    tcr_a, tcr_b = rec["TCR_a"], rec["TCR_b"]
    pocket = {"HLA_a": [], "HLA_b": [], "CDRa": [], "CDRb": []}
    ca = tcr_a.find(rec["cdr3a"]) if rec.get("cdr3a") else -1
    cb = tcr_b.find(rec["cdr3b"]) if rec.get("cdr3b") else -1
    if ca != -1:
        pocket["CDRa"] = [[ca, ca + len(rec["cdr3a"]) - 1]]
    if cb != -1:
        pocket["CDRb"] = [[cb, cb + len(rec["cdr3b"]) - 1]]
    return {
        "sequences": {"HLA_a": rec["HLA_a"], "HLA_b": rec["HLA_b"],
                      "Peptide": rec["peptide"], "TCR_a": tcr_a, "TCR_b": tcr_b},
        "pocket_idx": pocket,
        "meta_data": {"label": int(rec.get("label", 0) or 0),
                      "epitope": rec.get("peptide", ""),
                      "hla_allele": rec.get("hla", "")},
    }


def score_from_components(
    rows,
    model: str | None = None,
    device: str = "cuda",
    *,
    checkpoint: str | Path | None = None,
    backbone: str | None = None,
    arch: str | None = None,
    mask_region: str = "peptide",
) -> pd.DataFrame:
    """Reconstruct and score from V/J genes + CDR3 + HLA allele + peptide.

    `rows` is a single dict, a list of dicts, or a DataFrame, with columns/keys (case-insensitive,
    aliases accepted) `trav, traj, cdr3a, trbv, trbj, cdr3b, hla, peptide` (optional `name`,
    `label`). Returns a DataFrame with the reconstructed HLA_a/HLA_b/TCR_a/TCR_b, `ok` /
    `*_reason` flags, and a `pll_<model>` column. Rows that fail reconstruction (unknown allele,
    unstitchable/pseudogene TCR) get a NaN score. Pass a registry `model` name (default
    `DecoderTCR-ESMC_600M`) or an explicit `checkpoint` + `backbone` + `arch`.
    """
    from DecoderTCR.utils.model_zoo import load, DEFAULT_MODEL
    from DecoderTCR.utils.scoring import run_pll_benchmark

    recs = reconstruct_components(_normalize_rows(rows))
    valid = [r for r in recs if r["ok"]]
    col = model or (Path(checkpoint).stem if checkpoint else DEFAULT_MODEL)

    if valid:
        if checkpoint is not None:
            mdl, n_layers = load(device=device, checkpoint=checkpoint, backbone=backbone, arch=arch)
        else:
            mdl, n_layers = load(model, device=device)
        scores = run_pll_benchmark(mdl, n_layers, [_entry(r) for r in valid], mask_region, device)
    else:
        scores = []

    for r in recs:
        r[f"pll_{col}"] = np.nan
    for r, sc in zip(valid, scores):
        r[f"pll_{col}"] = float(sc)

    cols = ["name", "label", "hla", "peptide", "trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b",
            "TRAV", "TRAJ", "TRBV", "TRBJ", "HLA_a", "HLA_b", "TCR_a", "TCR_b",
            "ok", "tcr_ok", "tcr_reason", "hla_ok", "hla_reason", f"pll_{col}"]
    df = pd.DataFrame(recs)
    return df[[c for c in cols if c in df.columns]]
