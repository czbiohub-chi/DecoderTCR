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

FORWARD_BUDGET_PER_DESIGN = 20   # default max_forwards per requested design

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


def _profile_row(logits: torch.Tensor) -> np.ndarray:
    """Renormalised distribution over the 20 standard residues for one position."""
    # float64, required by logomaker, matching profile_from_logits.
    return torch.softmax(logits[_AA_IDS].double(), dim=-1).numpy()


def iegr_profile(data, region: str = "peptide", length: int | None = None,
                 temperature: float = 0.1, seed: int = 42, model=None, *, num_layers=None,
                 device="cuda", from_genes: bool = False, checkpoint=None, backbone=None,
                 arch=None) -> pd.DataFrame:
    """Profile a region by entropy-guided refinement instead of one masked forward pass.

    `peptide_profile` reads every position from a single fully-masked pass, so each row is
    independent of the others. This walks the region instead: it commits the lowest-entropy
    position, re-masks the rest, and runs another pass, recording each position's distribution at
    the moment it is committed. Row `i` is therefore conditioned on every position committed
    before it, which is what a one-shot profile cannot express.

    Costs one forward pass per residue rather than one in total. `temperature` shapes the residue
    committed at each step, defaulting to 0.1, near greedy. The returned frame has the same shape
    as `peptide_profile`, so `sample_from_profile` consumes it unchanged.
    """
    from DecoderTCR.api import _resolve_model

    entry = build_masked_entry(data, region=region, length=length, from_genes=from_genes)
    mdl, n_layers, dev, _, _ = _resolve_model(model, num_layers, device, checkpoint,
                                              backbone, arch)
    tok = TCRpMHCTokenizer(entry, mask_probs=None, use_sep=False)
    token_idx = [int(p) + 1 for p in region_positions(tok, region, entry)]   # +1 for CLS
    generator = torch.Generator().manual_seed(seed)

    rows: dict[int, np.ndarray] = {}
    order: list[int] = []
    chosen: dict[int, int] = {}
    remaining = list(token_idx)

    while remaining:
        logits = _forward(mdl, n_layers, _rebuild(tok.original_ids, chosen, remaining), dev)
        sel = torch.tensor(remaining, dtype=torch.long)
        t = remaining[int(_entropy(logits[sel]).argmin())]       # commit the most confident
        rows[t] = _profile_row(logits[t])
        order.append(t)
        chosen[t] = _sample(logits[t], temperature, generator)
        remaining.remove(t)

    probs = np.stack([rows[t] for t in token_idx])
    prof = pd.DataFrame(probs, columns=list(AA20))
    prof.index = pd.RangeIndex(1, len(prof) + 1, name="position")
    prof["entropy"] = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    # Rank at which each position was committed, 1 = most confident. Diagnostic only.
    prof["commit_order"] = [order.index(t) + 1 for t in token_idx]
    return prof


def block_gibbs(data, peptides, region: str = "peptide", k: int = 5, rounds: int = 1,
                n: int | None = None, temperature: float = 1.0, seed: int = 42,
                max_forwards: int | None = None, model=None, *, num_layers=None,
                device="cuda", from_genes: bool = False, checkpoint=None, backbone=None,
                arch=None) -> tuple[list[str], dict]:
    """Refine a peptide library by block Gibbs, re-masking `k` positions at a time.

    One-shot sampling draws every position independently, so it cannot represent dependence
    between positions. This re-masks a random block of `k` positions in a peptide and resamples
    them together from one forward pass. Positions inside a block are still drawn independently of
    each other, but they are drawn in the context of the positions left standing.

    Two modes. Without `n`, each input peptide is walked for exactly `rounds` blocks and only its
    final state is kept, so refined peptides that collide reduce the library. With `n`, the walks
    run round robin and every distinct state they visit is kept, until `n` distinct peptides are
    collected or `max_forwards` passes are spent. Use `n` to hold the library size fixed rather
    than watching it shrink.

    `max_forwards` defaults to `rounds * len(peptides)` without `n`, which is exact, and to
    `FORWARD_BUDGET_PER_DESIGN * n` with it. Needs the model, so unlike `sample_from_profile` it is
    not free, and its forward passes dominate every other cost in a run that uses it.

    Consecutive states of one walk differ by at most `k` positions, so they are correlated. A `k`
    closer to the peptide length decorrelates faster at the cost of keeping less context.

    Returns the peptides alongside `n_input`, `n_returned`, `n_forwards`, `n_changed` and
    `budget_exhausted`, which says the walk stopped on `max_forwards` rather than reaching `n`.
    """
    from DecoderTCR.api import _resolve_model

    if not peptides:
        return [], {"n_input": 0, "n_returned": 0, "n_forwards": 0, "n_changed": 0}
    if k < 1:
        raise ValueError(f"`k` must be at least 1, got {k}")
    if rounds < 1:
        raise ValueError(f"`rounds` must be at least 1, got {rounds}")
    if n is not None and n < 1:
        raise ValueError(f"`n` must be at least 1, got {n}")
    if max_forwards is not None and max_forwards < 1:
        raise ValueError(f"`max_forwards` must be at least 1, got {max_forwards}")

    length = len(peptides[0])
    if any(len(s) != length for s in peptides):
        raise ValueError("every peptide must have the same length")

    entry = build_masked_entry(data, region=region, length=length, from_genes=from_genes)
    mdl, n_layers, dev, _, _ = _resolve_model(model, num_layers, device, checkpoint,
                                              backbone, arch)
    tok = TCRpMHCTokenizer(entry, mask_probs=None, use_sep=False)
    token_idx = [int(q) + 1 for q in region_positions(tok, region, entry)]   # +1 for CLS
    if len(token_idx) != length:
        raise ValueError(f"peptides are length {length} but the region holds {len(token_idx)}")

    aa_to_id = {a: i for a, i in zip(AA20, AA20_IDS)}
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    block = min(k, length)
    idx_array = np.asarray(token_idx)

    budget = max_forwards if max_forwards is not None else (
        rounds * len(peptides) if n is None else FORWARD_BUDGET_PER_DESIGN * n)

    # One chain per input peptide, advanced in place so each walk keeps its own history.
    chains = []
    for pep in peptides:
        ids = tok.original_ids.clone()
        for t, a in zip(token_idx, pep):
            ids[t] = aa_to_id[a]
        chains.append(ids)

    def step(ids):
        """Re-mask one random block and resample it, returning the peptide that results."""
        subset = [int(q) for q in rng.choice(idx_array, size=block, replace=False)]
        logits = _forward(mdl, n_layers, _rebuild(ids, {}, subset), dev)
        for q in subset:
            ids[q] = _sample(logits[q], temperature, generator)
        return "".join(_ID_TO_AA[int(ids[t])] for t in token_idx)

    seen, out, forwards, changed = set(), [], 0, 0
    if n is None:
        for ids, pep in zip(chains, peptides):
            refined = pep
            for _ in range(rounds):
                if forwards >= budget:
                    break
                refined = step(ids)
                forwards += 1
            changed += refined != pep
            if refined not in seen:
                seen.add(refined)
                out.append(refined)
    else:
        # Round robin, so no single seed dominates the library when one chain mixes faster.
        origin = list(peptides)
        i = 0
        while len(out) < n and forwards < budget:
            refined = step(chains[i % len(chains)])
            forwards += 1
            changed += refined != origin[i % len(chains)]
            if refined not in seen:
                seen.add(refined)
                out.append(refined)
            i += 1

    return out, {"n_input": len(peptides), "n_returned": len(out),
                 "n_forwards": forwards, "n_changed": changed,
                 "budget_exhausted": bool(n is not None and len(out) < n)}


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
