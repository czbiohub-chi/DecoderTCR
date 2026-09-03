"""Generate peptide sequences for a TCR and HLA.

    import DecoderTCR as dt

    # one model call: TCR + HLA context in, a position weight matrix out
    prof = dt.peptide_profile({"HLA_a": HLA, "TCR_a": TCR_A, "TCR_b": TCR_B}, length=9)

    # sampling is separate, needs no model, and is where the knobs live
    peptides, stats = dt.sample_from_profile(prof, n=1000, temperature=1.2)
    print(stats["saturated"], stats["n_draws_used"])

    # or both at once
    designs = dt.design_peptides({"HLA_a": HLA, "TCR_a": TCR_A, "TCR_b": TCR_B}, length=9, n=1000)

The split matters: `peptide_profile` is the only part that touches the model, so a profile can be
resampled at a dozen temperatures for free. `method="iegr"` instead resamples positions in the
context of those already committed, at one forward pass per residue plus the Gibbs rounds.

Designs carry the masked-peptide `pll` that `dt.score` reports and are sorted best first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from DecoderTCR.constants import AA20
from DecoderTCR.design.profile import build_masked_entry, consensus, peptide_profile

METHODS = ("one_shot", "iegr")

# Default sampling temperature per method.
DEFAULT_TEMPERATURE = {"one_shot": 1.0, "iegr": 0.1}

DRAW_CAP = 1_000_000       # proposals to spend before giving up on reaching `n` distinct
BATCH = 250_000            # proposals per vectorised batch
CLIP_P = 1e-12             # floor before the log in tempering, so a zero column cannot become -inf
MAX_CODEABLE_LENGTH = 14   # 20**14 < 2**64; dedup packs a peptide into one uint64


def _encode(x: np.ndarray) -> np.ndarray:
    """Pack (M, L) residue indices into (M,) base-20 codes for exact dedup."""
    code = np.zeros(len(x), dtype=np.uint64)
    for j in range(x.shape[1]):
        code = code * np.uint64(20) + x[:, j].astype(np.uint64)
    return code


def sample_from_profile(profile: pd.DataFrame, n: int = 1000, temperature: float = 1.0,
                        seed: int = 42, cap: int = DRAW_CAP) -> tuple[list[str], dict]:
    """Draw `n` distinct peptides from a profile, independently at each position.

    `profile` is what `peptide_profile` returns. Nothing here touches the model, so the same
    profile can be resampled at many temperatures without another forward pass.

    `temperature` reshapes the marginals as `p ** (1 / temperature)` renormalised: below 1 sharpens
    toward the most probable residue, above 1 flattens. A `temperature` of 0 or less returns the
    consensus peptide alone.

    Peptides longer than 14 are refused, because dedup packs each one into a single uint64.

    Sampling draws with replacement and keeps the first occurrence of each distinct peptide, so a
    peaked profile can run out of distinct peptides before reaching `n`. `cap` bounds the proposals
    spent, and a short result is reported rather than hidden.

    The stats carry `n_requested`, `n_returned`, `n_draws_used` and three flags. `saturated` says
    the result is short. `cap_reached` says the proposal budget ran out. `support_exhausted` says
    the last batch produced no peptide that had not already been drawn, so the distribution is out
    of distinct peptides and raising `cap` will not help. A short result with `support_exhausted`
    false is budget-bound, and a larger `cap` will return more.

    Returns the peptides in draw order alongside those stats.
    """
    probs = profile[list(AA20)].to_numpy(dtype=np.float64)
    length = probs.shape[0]
    if length > MAX_CODEABLE_LENGTH:
        raise ValueError(f"peptides longer than {MAX_CODEABLE_LENGTH} are not supported by the "
                         f"dedup packing, got length {length}")
    if n < 1:
        raise ValueError(f"`n` must be at least 1, got {n}")
    if cap < 1:
        raise ValueError(f"`cap` must be at least 1, got {cap}")

    if temperature <= 0:
        seq = consensus(profile)
        return [seq], {"n_requested": n, "n_returned": 1, "n_draws_used": 0,
                       "saturated": n > 1, "cap_reached": False, "support_exhausted": n > 1}

    if temperature != 1.0:
        logp = np.log(np.clip(probs, CLIP_P, None)) / temperature
        probs = np.exp(logp - logp.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    letters = np.array(list(AA20))
    seen = np.zeros(0, dtype=np.uint64)     # sorted, membership only
    kept: list[np.uint64] = []              # draw order, the actual output
    used = 0
    last_novel = None                       # novel peptides in the most recent batch

    while len(kept) < n and used < cap:
        # Draw a few times what is still missing: enough to make progress on a flat profile
        # without spending 250k proposals to collect ten peptides.
        want = n - len(kept)
        size = int(min(BATCH, cap - used, max(1024, 4 * want)))
        draw = np.stack([rng.choice(20, size=size, p=probs[i]) for i in range(length)],
                        axis=1).astype(np.uint8)
        used += size

        code = _encode(draw)
        # np.unique sorts, so first-occurrence indices are what recovers draw order. Truncating
        # the sorted array instead would keep the lexicographically smallest peptides, which is a
        # biased prefix of the sample rather than a sample.
        uniq, first = np.unique(code, return_index=True)
        novel = ~np.isin(uniq, seen)
        last_novel = int(novel.sum())
        if novel.any():
            fresh = uniq[novel]
            take = fresh[np.argsort(first[novel])][:want]
            kept.extend(take.tolist())
            seen = np.union1d(seen, fresh)

    out = []
    for c in kept[:n]:
        idx = np.empty(length, dtype=np.int64)
        v = int(c)
        for j in range(length - 1, -1, -1):
            idx[j] = v % 20
            v //= 20
        out.append("".join(letters[idx]))

    short = len(out) < n
    return out, {"n_requested": n, "n_returned": len(out), "n_draws_used": used,
                 "saturated": short, "cap_reached": used >= cap,
                 "support_exhausted": bool(short and last_novel == 0)}


def design_peptides(data, length: int, n: int = 10, method: str = "one_shot",
                    temperature: float | None = None, seed: int = 42, rescore: bool = True,
                    gibbs_rounds: int = 10, gibbs_subset_size: int = 5, model=None, *,
                    profile_method: str = "one_shot", gibbs_k: int = 0,
                    gibbs_temperature: float | None = None,
                    gibbs_max_forwards: int | None = None,
                    num_layers=None, device="cuda", cap: int = DRAW_CAP,
                    checkpoint=None, backbone=None, arch=None) -> pd.DataFrame:
    """Generate candidate peptides of `length` for the TCR and HLA in `data`.

    `data` is a dict (or Series, or one-row DataFrame) of full sequences: `HLA_a`, `HLA_b`,
    `TCR_a`, `TCR_b`. No peptide is needed, only the length to generate. To start from V/J genes
    and an allele instead, reconstruct the chains first with `reconstruct_components`.

    `method="one_shot"` (default) profiles the peptide, samples every position independently from
    that profile, and optionally refines the result. `method="iegr"` instead runs the end-to-end
    Iterative Entropy-Guided Refinement of the paper, which walks one design rather than sampling
    a library from a profile.

    The one-shot pipeline has a knob at each layer. `profile_method="iegr"` builds the profile by
    committing one position at a time in entropy order, at one forward pass per residue, so each
    row is conditioned on the positions committed before it. `gibbs_k` then re-masks blocks of
    that many positions in the sampled peptides and resamples them together, at
    `gibbs_temperature` (defaulting to `temperature`). `gibbs_k=0` disables it.

    Refinement targets `n` distinct designs rather than shrinking the library: the walks continue
    until `n` are collected or `gibbs_max_forwards` passes are spent. Without that the refined
    peptides collide and the result comes back smaller than requested, so `gibbs_rounds` does not
    apply on this path.

    Both knobs cost forward passes: `profile_method="iegr"` costs `length` of them, and `gibbs_k`
    costs up to `gibbs_max_forwards`, which dominates every other cost in a large run.

    `temperature` defaults to 1.0 for one-shot and 0.1 for IEGR. Lower values sharpen sampling
    toward the most probable residues, and 0 returns the consensus peptide alone. `cap` bounds the
    proposals one-shot sampling will spend chasing `n` distinct peptides.

    Returns a DataFrame with `sequence`, `method`, `phase`, `step`, and, when `rescore=True`, the
    masked-peptide `pll` for each design, sorted best first. For one-shot the frame's `.attrs`
    carry `n_draws_used` and `saturated`; a saturated result means the profile could not supply
    `n` distinct peptides.
    """
    from DecoderTCR.api import _resolve_model

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if profile_method not in METHODS:
        raise ValueError(f"profile_method must be one of {METHODS}, got {profile_method!r}")
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE[method]
    if gibbs_temperature is None:
        gibbs_temperature = temperature

    # Resolve once and hand the module to both the profiling and rescoring passes, so a single
    # design call loads the checkpoint once rather than twice.
    mdl, n_layers, dev, _, _ = _resolve_model(model, num_layers, device, checkpoint,
                                              backbone, arch)
    stats: dict = {}

    if method == "iegr":
        from DecoderTCR.design.iegr import iegr
        out = iegr(data, region="peptide", length=length, n=n, temperature=temperature,
                   gibbs_rounds=gibbs_rounds, gibbs_subset_size=gibbs_subset_size, seed=seed,
                   model=mdl, num_layers=n_layers, device=dev,
                   checkpoint=checkpoint, backbone=backbone, arch=arch)
        # IEGR yields one design per checkpoint, so it is structurally capped well below a large
        # `n`. Carry the same keys as the one-shot path so a short result is never silent.
        stats = {"n_requested": n, "n_returned": len(out), "n_draws_used": 0,
                 "saturated": len(out) < n, "cap_reached": False, "support_exhausted": False}
    else:
        if profile_method == "iegr":
            from DecoderTCR.design.iegr import iegr_profile
            profile = iegr_profile(data, region="peptide", length=length, temperature=temperature,
                                   seed=seed, model=mdl, num_layers=n_layers, device=dev)
        else:
            profile = peptide_profile(data, length=length, model=mdl, num_layers=n_layers,
                                      device=dev)
        sequences, stats = sample_from_profile(profile, n=n, temperature=temperature,
                                               seed=seed, cap=cap)
        stats["profile_method"] = profile_method
        if gibbs_k > 0:
            from DecoderTCR.design.iegr import block_gibbs
            # Target `n` distinct designs so refinement holds the library size instead of
            # collapsing it, which is what plain round-based refinement does.
            sequences, gstats = block_gibbs(data, sequences, region="peptide", k=gibbs_k,
                                            n=n, max_forwards=gibbs_max_forwards,
                                            temperature=gibbs_temperature,
                                            seed=seed, model=mdl, num_layers=n_layers,
                                            device=dev)
            stats.update({f"gibbs_{key}": v for key, v in gstats.items()},
                         gibbs_k=gibbs_k, gibbs_temperature=gibbs_temperature)
            # Refinement replaces the library, so the sampler's counts no longer describe what
            # the caller is holding.
            stats["n_returned"] = len(sequences)
            stats["saturated"] = len(sequences) < n
        out = pd.DataFrame({"sequence": sequences})
        out["phase"] = 2 if gibbs_k > 0 else 0
        out["step"] = range(len(out))

    out.insert(1, "method", method)
    if rescore and len(out):
        out["pll"] = _rescore(data, out["sequence"].tolist(), mdl, n_layers, dev,
                              checkpoint, backbone, arch)
        out = out.sort_values("pll", ascending=False, ignore_index=True)
    out.attrs.update(stats)
    return out


def _rescore(data, sequences, model, num_layers, device, checkpoint, backbone, arch):
    """Masked-peptide PLL for each design, the same score `dt.score` reports."""
    from DecoderTCR.api import score

    base = build_masked_entry(data, region="peptide", length=len(sequences[0]))
    entries = []
    for s in sequences:
        seqs = dict(base["sequences"])
        seqs["Peptide"] = s
        entries.append({"sequences": seqs, "pocket_idx": base["pocket_idx"], "meta_data": {}})
    return score(entries, model, num_layers=num_layers, device=device, checkpoint=checkpoint,
                 backbone=backbone, arch=arch, return_dataframe=False)
