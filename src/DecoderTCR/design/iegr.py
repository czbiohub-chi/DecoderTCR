"""IEGR (Iterative Entropy-Guided Refinement) sequence design.

    import DecoderTCR as dt
    designs = dt.iegr({"HLA_a": HLA, "TCR_a": TCR_A, "TCR_b": TCR_B}, length=9, n=10)

Two phases. Phase 1 masks the target region, then repeatedly commits the lowest-entropy
position and resamples the rest in its context, one forward pass per designed residue. Phase 2
re-masks a random subset and resamples it for `gibbs_rounds` passes per residue, returning a set
of variants. Set `gibbs_rounds=0` for phase 1 alone.

Use an ESMC model. ESM2 applies token dropout, which rescales embeddings by the observed mask
ratio.

`dt.design_peptides` with the default `method="one_shot"` needs a single forward pass in total.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from DecoderTCR.constants import AA20, AA20_IDS, MASK_IDX
from DecoderTCR.model.tokenizer import TCRpMHCTokenizer
from DecoderTCR.design.profile import build_masked_entry, region_positions

_AA_IDS = torch.tensor(AA20_IDS, dtype=torch.long)
_ID_TO_AA = {i: a for i, a in zip(AA20_IDS, AA20)}


def _sample(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> int:
    """Sample one amino acid token id from logits restricted to the 20 standard residues."""
    aa_logits = logits[_AA_IDS].float()
    if temperature <= 0:
        return int(_AA_IDS[int(aa_logits.argmax())])
    probs = torch.softmax(aa_logits / temperature, dim=-1)
    return int(_AA_IDS[int(torch.multinomial(probs, 1, generator=generator))])


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy per row over the 20 standard residues."""
    probs = torch.softmax(logits[:, _AA_IDS].float(), dim=-1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=-1)


@torch.no_grad()
def _forward(model, num_layers: int, ids: torch.Tensor, device) -> torch.Tensor:
    out = model(ids.unsqueeze(0).to(device), repr_layers=[num_layers], return_contacts=False)
    return out["logits"][0].cpu()


def _rebuild(base: torch.Tensor, chosen: dict[int, int], masked: list[int]) -> torch.Tensor:
    """Write the current residues into the complex, then re-mask the unresolved positions."""
    ids = base.clone()
    for t, v in chosen.items():
        ids[t] = v
    for t in masked:
        ids[t] = MASK_IDX
    return ids


def _phase1(model, num_layers, base_ids, token_idx, temperature, generator, device):
    """Entropy-guided greedy fill. One forward per designed residue."""
    logits = _forward(model, num_layers, _rebuild(base_ids, {}, token_idx), device)
    chosen = {t: _sample(logits[t], temperature, generator) for t in token_idx}
    remaining = list(token_idx)

    while len(remaining) > 1:
        rows = logits[torch.tensor(remaining, dtype=torch.long)]
        remaining.pop(int(_entropy(rows).argmin()))          # commit the most confident
        logits = _forward(model, num_layers, _rebuild(base_ids, chosen, remaining), device)
        for t in remaining:
            chosen[t] = _sample(logits[t], temperature, generator)

    ids = _rebuild(base_ids, chosen, [])
    return ids, "".join(_ID_TO_AA[int(ids[t])] for t in token_idx)


def _phase2(model, num_layers, ids, token_idx, temperature, rounds, subset_size,
            generator, rng, device):
    """Gibbs refinement. Returns (token ids, residues) at each checkpoint."""
    n = len(token_idx)
    subset_size = min(subset_size, n)
    interval = max(n - 1, 1)
    idx_array = np.asarray(token_idx)

    current = ids.clone()
    out = []
    for step in range(rounds * n):
        subset = [int(t) for t in rng.choice(idx_array, size=subset_size, replace=False)]
        logits = _forward(model, num_layers, _rebuild(current, {}, subset), device)
        for t in subset:
            current[t] = _sample(logits[t], temperature, generator)
        if (step + 1) % interval == 0:
            out.append((current.clone(),
                        "".join(_ID_TO_AA[int(current[t])] for t in token_idx)))
    return out


def iegr(data, region: str = "peptide", length: int | None = None, n: int = 10,
         temperature: float = 0.1, gibbs_rounds: int = 10, gibbs_subset_size: int = 5,
         seed: int = 42, model=None, *, num_layers=None, device="cuda",
         from_genes: bool = False, checkpoint=None, backbone=None, arch=None) -> pd.DataFrame:
    """Design a region of one complex by Iterative Entropy-Guided Refinement.

    `region` is `peptide`, where `length` sets how many residues to design, or `cdr3a`,
    `cdr3b`, or `cdr3`, which take their length from the template TCR and require the peptide
    as conditioning context. `gibbs_rounds=0` returns the phase 1 design alone.

    `temperature` defaults to 0.1, near greedy. Raise it for more diverse designs.

    Returns up to `n` unique designs as a DataFrame with `sequence`, `phase`, and `step`.
    Reproducible for a given `seed`.
    """
    from DecoderTCR.api import _resolve_model

    entry = build_masked_entry(data, region=region, length=length, from_genes=from_genes)
    mdl, n_layers, dev, _, _ = _resolve_model(model, num_layers, device, checkpoint,
                                              backbone, arch)
    tok = TCRpMHCTokenizer(entry, mask_probs=None, use_sep=False)
    token_idx = [int(p) + 1 for p in region_positions(tok, region, entry)]   # +1 for CLS

    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    ids, seq = _phase1(mdl, n_layers, tok.original_ids, token_idx, temperature, generator, dev)
    rows = [{"sequence": seq, "phase": 1, "step": 0}]

    if gibbs_rounds > 0:
        checkpoints = _phase2(mdl, n_layers, ids, token_idx, temperature, gibbs_rounds,
                              gibbs_subset_size, generator, rng, dev)
        rows += [{"sequence": s, "phase": 2, "step": i}
                 for i, (_, s) in enumerate(checkpoints, start=1)]

    out = pd.DataFrame(rows).drop_duplicates(subset="sequence", ignore_index=True)
    return out.head(n)
