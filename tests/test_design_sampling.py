"""Tests for peptide sampling: profile in, peptide library out.

    uv run pytest tests/ -q

`sample_from_profile` takes a position weight matrix, so every test here builds one by hand. No
test loads a model, downloads weights or needs a GPU, which is the point of keeping the sampler
model-free.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from DecoderTCR.constants import AA20
from DecoderTCR.design.generate import sample_from_profile

L = 9


def make_profile(mat: np.ndarray) -> pd.DataFrame:
    """Wrap a (L, 20) matrix in the frame `peptide_profile` returns."""
    mat = mat / mat.sum(axis=1, keepdims=True)
    prof = pd.DataFrame(mat, columns=list(AA20))
    prof.index = pd.RangeIndex(1, len(prof) + 1, name="position")
    prof["entropy"] = -(mat * np.log(mat + 1e-12)).sum(axis=1)
    return prof


def flat_profile(length: int = L) -> pd.DataFrame:
    return make_profile(np.full((length, 20), 1.0 / 20))


def one_hot_profile(length: int = L) -> pd.DataFrame:
    """A profile with all its mass on one residue: exactly one peptide is reachable."""
    mat = np.full((length, 20), 1e-12)
    mat[:, 0] = 1.0
    return make_profile(mat)


def two_residue_profile(length: int = L) -> pd.DataFrame:
    """Two equiprobable residues per position, so the support is exactly 2 ** length."""
    mat = np.full((length, 20), 1e-12)
    mat[:, :2] = 0.5
    return make_profile(mat)


def test_saturation_is_reported_not_hidden():
    """The defect this suite exists for: a profile that cannot supply `n` distinct peptides used to
    return short with no signal, so a caller could not tell whether the distribution ran out or the
    draw budget did."""
    seqs, stats = sample_from_profile(one_hot_profile(), n=50, seed=0, cap=20_000)
    assert seqs == ["A" * L]
    assert stats["n_returned"] == 1
    assert stats["n_requested"] == 50
    assert stats["saturated"] is True
    assert stats["n_draws_used"] == 20_000, "a saturated draw should spend its whole budget"


def test_not_saturated_when_the_profile_can_supply_n():
    seqs, stats = sample_from_profile(flat_profile(), n=200, seed=0)
    assert len(seqs) == 200
    assert stats["saturated"] is False
    assert stats["n_draws_used"] < 20_000, "a flat profile should not need many proposals"


def test_support_is_respected_exactly():
    """Two residues at each of 9 positions is 512 reachable peptides, and no draw budget invents a
    513th."""
    seqs, stats = sample_from_profile(two_residue_profile(), n=5000, temperature=0.5,
                                      seed=0, cap=200_000)
    assert len(set(seqs)) == len(seqs) == 2 ** L
    assert stats["saturated"] is True


def test_cap_bounds_the_work():
    _, small = sample_from_profile(one_hot_profile(), n=10, seed=0, cap=2_000)
    _, large = sample_from_profile(one_hot_profile(), n=10, seed=0, cap=8_000)
    assert small["n_draws_used"] == 2_000
    assert large["n_draws_used"] == 8_000


def test_same_seed_is_reproducible():
    a, sa = sample_from_profile(flat_profile(), n=100, temperature=1.3, seed=7)
    b, sb = sample_from_profile(flat_profile(), n=100, temperature=1.3, seed=7)
    assert a == b
    assert sa == sb


def test_different_seeds_differ():
    a, _ = sample_from_profile(flat_profile(), n=100, seed=1)
    b, _ = sample_from_profile(flat_profile(), n=100, seed=2)
    assert a != b


@pytest.mark.parametrize("length", [8, 9, 10, 11])
def test_output_shape_and_alphabet(length):
    seqs, _ = sample_from_profile(flat_profile(length), n=50, seed=0)
    assert all(len(s) == length for s in seqs)
    assert set("".join(seqs)) <= set(AA20)
    assert len(set(seqs)) == len(seqs), "peptides are distinct"


def test_distinct_count_is_non_decreasing_in_temperature():
    """Flattening the profile can only widen the reachable set, so a higher temperature must not
    yield fewer distinct peptides."""
    counts = [sample_from_profile(two_residue_profile(), n=5000, temperature=t, seed=0,
                                  cap=50_000)[1]["n_returned"]
              for t in (0.25, 0.5, 1.0, 2.0, 4.0)]
    assert counts == sorted(counts), counts


def test_zero_temperature_returns_the_consensus_alone():
    mat = np.full((L, 20), 0.01)
    mat[:, 3] = 0.5
    seqs, stats = sample_from_profile(make_profile(mat), n=25, temperature=0.0)
    assert seqs == [AA20[3] * L]
    assert stats["saturated"] is True
    assert stats["n_draws_used"] == 0, "the consensus needs no draws"


def test_empirical_frequencies_track_the_profile():
    """At T=1 the residues a position emits most should be the ones it weights most.

    Compared as a rank correlation rather than an argmax match: the draw is deduplicated, so it is
    not an unbiased frequency estimate, and two near-tied residues swap places freely. On the seed
    below, position 1 weights 0.1318 and 0.1317 on two residues, which no finite sample orders
    reliably.
    """
    rng = np.random.default_rng(0)
    mat = rng.dirichlet(np.full(20, 0.5), size=L)
    prof = make_profile(mat)
    seqs, _ = sample_from_profile(prof, n=4000, temperature=1.0, seed=0, cap=400_000)

    idx = {a: i for i, a in enumerate(AA20)}
    counts = np.zeros((L, 20))
    for s in seqs:
        for j, ch in enumerate(s):
            counts[j, idx[ch]] += 1

    def rank(v):
        order = np.empty(len(v), dtype=float)
        order[np.argsort(v)] = np.arange(len(v))
        return order

    for j in range(L):
        r = np.corrcoef(rank(counts[j]), rank(mat[j]))[0, 1]
        assert r > 0.9, f"position {j + 1} rank correlation {r:.3f} against its own profile"


def test_rejects_a_nonsense_request():
    with pytest.raises(ValueError, match="`n` must be at least 1"):
        sample_from_profile(flat_profile(), n=0)


def test_rejects_peptides_too_long_to_pack():
    with pytest.raises(ValueError, match="not supported by the dedup packing"):
        sample_from_profile(flat_profile(15), n=5)


def test_matches_a_reference_implementation_where_neither_saturates():
    """The rewrite batches its draws, so it consumes the RNG in a different order and cannot match
    the old loop element for element. What must not change is the distribution: at a temperature
    and size where neither implementation saturates, the two agree on which residues each position
    favours."""
    prof = flat_profile()
    probs = prof[list(AA20)].to_numpy()

    rng = np.random.default_rng(3)                   # the pre-rewrite loop, inlined
    letters = np.array(list(AA20))
    seen, reference = set(), []
    for _ in range(200 * 20):
        seq = "".join(letters[rng.choice(20, p=row)] for row in probs)
        if seq not in seen:
            seen.add(seq)
            reference.append(seq)
        if len(reference) >= 200:
            break

    new, stats = sample_from_profile(prof, n=200, seed=3)
    assert stats["saturated"] is False and len(reference) == 200
    assert all(len(s) == L for s in new)
    ref_counts = np.bincount([AA20.index(c) for c in "".join(reference)], minlength=20)
    new_counts = np.bincount([AA20.index(c) for c in "".join(new)], minlength=20)
    # Both sample the same uniform profile, so neither residue histogram should be extreme.
    assert ref_counts.min() > 0 and new_counts.min() > 0


def test_design_peptides_chains_profile_into_sampler(monkeypatch):
    """`design_peptides` should be a thin wrapper: profile once, sample from it, and surface the
    sampler's saturation stats on the frame. Stubs the model so this stays CPU-only."""
    from DecoderTCR.design import generate

    calls = {}

    def fake_resolve(model, num_layers, device, checkpoint, backbone, arch):
        calls["resolved"] = calls.get("resolved", 0) + 1
        return "MODEL", 36, "cpu", "pll_stub", "esmc"

    def fake_profile(data, length, model=None, **kw):
        calls["profile"] = calls.get("profile", 0) + 1
        calls["model_passed"] = model
        return two_residue_profile(length)

    monkeypatch.setattr("DecoderTCR.api._resolve_model", fake_resolve)
    monkeypatch.setattr(generate, "peptide_profile", fake_profile)

    out = generate.design_peptides({"HLA_a": "AAA"}, length=L, n=20, rescore=False, seed=0)

    assert list(out.columns) == ["sequence", "method", "phase", "step"]
    assert (out["method"] == "one_shot").all()
    assert len(out) == 20
    assert out.attrs["n_requested"] == 20 and out.attrs["saturated"] is False
    assert calls["profile"] == 1, "the profile is computed once"
    assert calls["resolved"] == 1, "the checkpoint is resolved once, not once per pass"
    assert calls["model_passed"] == "MODEL", "the resolved module is reused, not re-resolved"


def test_design_peptides_reports_saturation(monkeypatch):
    from DecoderTCR.design import generate

    monkeypatch.setattr("DecoderTCR.api._resolve_model",
                        lambda *a, **k: ("MODEL", 36, "cpu", "pll_stub", "esmc"))
    monkeypatch.setattr(generate, "peptide_profile",
                        lambda data, length, model=None, **kw: one_hot_profile(length))

    out = generate.design_peptides({"HLA_a": "AAA"}, length=L, n=100, rescore=False,
                                   cap=5_000)
    assert len(out) == 1
    assert out.attrs["saturated"] is True
    assert out.attrs["n_returned"] == 1


def test_sample_sequence_pairs_is_shipped_and_usable():
    """The design API takes full sequences, so the repo has to ship a full-sequence example. The
    gene-level `genes_pairs.csv` is no longer enough on its own."""
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "Demo" / "sample_data" / "sequence_pairs.csv"
    assert path.exists(), f"missing {path}"
    df = pd.read_csv(path)
    assert {"name", "HLA_a", "HLA_b", "TCR_a", "TCR_b"} <= set(df.columns)
    assert len(df) >= 1
    for col in ("HLA_a", "HLA_b", "TCR_a", "TCR_b"):
        assert df[col].str.len().min() > 50, f"{col} looks truncated"
        assert df[col].map(lambda s: set(s) <= set(AA20)).all(), f"{col} has non-standard residues"


# --- block Gibbs: the argument contract, which is checked before any model is touched ---

def test_block_gibbs_on_an_empty_library_is_a_no_op():
    """Sampling can return nothing when a profile saturates at zero, so refinement must not
    demand a model just to discover it has no work."""
    from DecoderTCR.design import block_gibbs
    out, stats = block_gibbs({"HLA_a": "AAA"}, [], k=3)
    assert out == [] and stats["n_returned"] == 0 and stats["n_forwards"] == 0


@pytest.mark.parametrize("kwargs,message", [
    ({"k": 0}, "`k` must be at least 1"),
    ({"k": -1}, "`k` must be at least 1"),
    ({"rounds": 0}, "`rounds` must be at least 1"),
])
def test_block_gibbs_rejects_a_nonsense_request(kwargs, message):
    from DecoderTCR.design import block_gibbs
    with pytest.raises(ValueError, match=re.escape(message)):
        block_gibbs({"HLA_a": "AAA"}, ["ACDEFGHIK"], **kwargs)


def test_block_gibbs_rejects_ragged_peptides():
    """A ragged library would silently write past the region, so it is refused up front."""
    from DecoderTCR.design import block_gibbs
    with pytest.raises(ValueError, match="same length"):
        block_gibbs({"HLA_a": "AAA"}, ["ACDEFGHIK", "ACDEF"], k=2)


def test_design_peptides_rejects_an_unknown_profile_method():
    from DecoderTCR.design import generate
    with pytest.raises(ValueError, match="profile_method must be one of"):
        generate.design_peptides({"HLA_a": "AAA"}, length=9, profile_method="gibbs")


# --- gaps the audit proved: these four mutations previously survived the whole suite ---

def peaked_profile(length=L, dominant=0.9):
    """One dominant residue per position, so tempering has something to actually change."""
    mat = np.full((length, 20), (1 - dominant) / 19)
    mat[:, 0] = dominant
    return make_profile(mat)


def _mean_hamming_to_consensus(seqs, consensus):
    return float(np.mean([sum(a != b for a, b in zip(s, consensus)) for s in seqs]))


def test_temperature_actually_reshapes_the_distribution():
    """Distinct-peptide counts alone cannot see tempering: on a two-residue profile the support is
    2**L at every T > 0. Measure the shape instead, or a no-op `temperature` passes."""
    prof = peaked_profile()
    consensus = "A" * L
    cold, _ = sample_from_profile(prof, n=200, temperature=0.5, seed=0, cap=2_000_000)
    hot, _ = sample_from_profile(prof, n=200, temperature=2.0, seed=0, cap=2_000_000)
    d_cold = _mean_hamming_to_consensus(cold, consensus)
    d_hot = _mean_hamming_to_consensus(hot, consensus)
    assert d_hot > d_cold + 1.0, f"tempering had no effect: cold {d_cold:.2f}, hot {d_hot:.2f}"


def test_a_budget_bound_short_result_is_not_reported_as_support_exhaustion():
    """A flat 9-mer reaches 20**9 peptides, so a short result there is the cap binding and a bigger
    cap would help. Conflating the two tells the caller to give up when they should spend more."""
    _, stats = sample_from_profile(flat_profile(), n=2000, temperature=1.0, seed=0, cap=1500)
    assert stats["n_returned"] == 1500
    assert stats["saturated"] is True and stats["cap_reached"] is True
    assert stats["support_exhausted"] is False


def test_support_exhaustion_is_distinguished_from_the_cap():
    """A one-hot profile has exactly one peptide, so no budget can ever satisfy n > 1."""
    _, stats = sample_from_profile(one_hot_profile(), n=50, seed=0, cap=20_000)
    assert stats["support_exhausted"] is True


def test_stats_keys_are_the_same_on_every_path():
    """The zero-temperature branch returns early, so it has to carry the same contract."""
    keys = {"n_requested", "n_returned", "n_draws_used", "saturated", "cap_reached",
            "support_exhausted"}
    _, sampled = sample_from_profile(flat_profile(), n=10, seed=0)
    _, consensus_only = sample_from_profile(flat_profile(), n=10, temperature=0.0)
    assert set(sampled) == keys and set(consensus_only) == keys


def test_rescore_reuses_the_resolved_module(monkeypatch):
    """The chain tests all passed rescore=True's leg over, so re-resolving the checkpoint inside
    _rescore, the exact double load this change fixed, survived the suite."""
    from DecoderTCR.design import generate

    calls = {"resolved": 0, "scored": 0, "models": []}

    def fake_resolve(model, num_layers, device, checkpoint, backbone, arch):
        calls["resolved"] += 1
        calls["models"].append(model)
        return "MODEL", 36, "cpu", "pll_stub", "esmc"

    def fake_score(entries, model, **kw):
        calls["scored"] += 1
        calls["score_model"] = model
        return np.linspace(-1.0, -0.1, len(entries))

    monkeypatch.setattr("DecoderTCR.api._resolve_model", fake_resolve)
    monkeypatch.setattr("DecoderTCR.api.score", fake_score)
    monkeypatch.setattr(generate, "peptide_profile", lambda data, length, **kw:
                        two_residue_profile(length))
    monkeypatch.setattr(generate, "build_masked_entry", lambda data, region, length: {
        "sequences": {"HLA_a": "AAA", "HLA_b": "", "TCR_a": "", "TCR_b": "", "Peptide": "A" * length},
        "pocket_idx": {}})

    out = generate.design_peptides({"HLA_a": "AAA"}, length=L, n=8, rescore=True, seed=0)

    assert calls["resolved"] == 1, "the checkpoint is resolved once across profiling and rescoring"
    assert calls["scored"] == 1 and calls["score_model"] == "MODEL"
    assert "pll" in out.columns and out["pll"].is_monotonic_decreasing


def test_every_profile_source_is_float64():
    """logomaker writes information values into the frame in place, which raises on a float32
    column. profile_from_logits documents this; iegr_profile's row builder must match it or
    sequence_logo dies only on the IEGR path, and only once a GPU is involved."""
    import torch
    from DecoderTCR.design.iegr import _profile_row
    from DecoderTCR.design.profile import profile_from_logits

    logits = torch.randn(4, 64)
    assert profile_from_logits(logits)[list(AA20)].to_numpy().dtype == np.float64
    assert _profile_row(logits[0]).dtype == np.float64


# --- block Gibbs targeting a library size, stubbed so the walk itself is testable on CPU ---

def _stub_gibbs(monkeypatch, length=L, constant=False):
    """Stub everything block_gibbs needs from the model so the walk logic runs on the CPU."""
    import importlib
    import torch
    # DecoderTCR.design rebinds the name `iegr` to the FUNCTION, shadowing the module, so the
    # module has to come from importlib rather than attribute access.
    ig = importlib.import_module("DecoderTCR.design.iegr")

    token_idx = np.arange(length)
    entry = {"sequences": {"HLA_a": "AAA", "HLA_b": "", "TCR_a": "", "TCR_b": "",
                           "Peptide": "A" * length}, "pocket_idx": {}}

    class Tok:
        original_ids = torch.zeros(length + 1, dtype=torch.long)

    calls = {"forwards": 0}
    gen = np.random.default_rng(0)
    # A fixed logit table makes the entropy ordering deterministic, which is what pins the
    # commit order. Without `constant` each call is fresh, which exercises a mixing walk.
    fixed = torch.as_tensor(np.random.default_rng(7).normal(size=(length + 1, 64)),
                            dtype=torch.float32)

    def fake_forward(model, num_layers, ids, device):
        calls["forwards"] += 1
        if constant:
            return fixed
        return torch.as_tensor(gen.normal(size=(length + 1, 64)), dtype=torch.float32)

    monkeypatch.setattr("DecoderTCR.api._resolve_model",
                        lambda *a, **k: ("MODEL", 36, "cpu", "stub", "esmc"))
    monkeypatch.setattr(ig, "build_masked_entry", lambda *a, **k: entry)
    monkeypatch.setattr(ig, "TCRpMHCTokenizer", lambda *a, **k: Tok())
    monkeypatch.setattr(ig, "region_positions", lambda *a, **k: token_idx)
    monkeypatch.setattr(ig, "_forward", fake_forward)
    return calls


def test_block_gibbs_reaches_the_requested_library_size(monkeypatch):
    """Without `n` the walk shrinks the library when refined peptides collide. With `n` it keeps
    going, which is the whole point of asking for a size."""
    from DecoderTCR.design import block_gibbs
    _stub_gibbs(monkeypatch)
    seeds = ["ACDEFGHIK", "LMNPQRSTV"]
    out, stats = block_gibbs({"HLA_a": "AAA"}, seeds, k=3, n=25, seed=0)
    assert stats["n_returned"] == 25 == len(out)
    assert len(set(out)) == 25, "the library must be distinct"
    assert stats["budget_exhausted"] is False


def test_block_gibbs_without_n_keeps_the_old_shape(monkeypatch):
    """Backward compatibility: one final state per seed, at most len(peptides) of them."""
    from DecoderTCR.design import block_gibbs
    calls = _stub_gibbs(monkeypatch)
    seeds = ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH"]
    out, stats = block_gibbs({"HLA_a": "AAA"}, seeds, k=3, rounds=2, seed=0)
    assert len(out) <= len(seeds) and stats["n_forwards"] == len(seeds) * 2
    assert calls["forwards"] == 6


def test_block_gibbs_stops_on_the_forward_budget(monkeypatch):
    """A target the walk cannot reach must stop and say so, not spin forever."""
    from DecoderTCR.design import block_gibbs
    _stub_gibbs(monkeypatch)
    out, stats = block_gibbs({"HLA_a": "AAA"}, ["ACDEFGHIK"], k=2, n=10_000,
                             max_forwards=40, seed=0)
    assert stats["n_forwards"] == 40
    assert stats["budget_exhausted"] is True and len(out) < 10_000


def test_block_gibbs_default_budget_is_per_requested_design(monkeypatch):
    """With no explicit cap the budget is FORWARD_BUDGET_PER_DESIGN * n. A single position has
    only 20 reachable peptides, so a request for 50 must exhaust exactly that budget."""
    from DecoderTCR.design import block_gibbs
    from DecoderTCR.design.iegr import FORWARD_BUDGET_PER_DESIGN
    _stub_gibbs(monkeypatch, length=1)
    out, stats = block_gibbs({"HLA_a": "AAA"}, ["A"], k=1, n=50, seed=0)
    assert stats["n_forwards"] == FORWARD_BUDGET_PER_DESIGN * 50
    assert stats["budget_exhausted"] is True
    assert len(out) <= 20, "a single position cannot exceed the 20 residue alphabet"


@pytest.mark.parametrize("kwargs,message", [
    ({"n": 0}, "`n` must be at least 1"),
    ({"max_forwards": 0}, "`max_forwards` must be at least 1"),
])
def test_block_gibbs_rejects_a_nonsense_target(kwargs, message):
    from DecoderTCR.design import block_gibbs
    with pytest.raises(ValueError, match=re.escape(message)):
        block_gibbs({"HLA_a": "AAA"}, ["ACDEFGHIK"], k=2, **kwargs)


def test_gene_input_is_named_not_a_vendored_keyerror():
    """The old README shipped design_peptides(clone, from_genes=True) with exactly this dict. A
    returning user who drops the removed keyword used to get KeyError: '*27:05' out of a vendored
    alphabet file, which names neither the field nor the fix."""
    from DecoderTCR.design.profile import build_masked_entry
    clone = {"trav": "TRAV21", "traj": "TRAJ6", "cdr3a": "CAVRPGGAGPFFVVF",
             "trbv": "TRBV7-9", "trbj": "TRBJ2-7", "cdr3b": "CASSLGQAYEQYF",
             "hla": "HLA-B*27:05"}
    with pytest.raises(ValueError, match="V/J gene input"):
        build_masked_entry(clone, region="peptide", length=9)


def test_an_allele_name_in_a_sequence_field_names_the_field():
    from DecoderTCR.design.profile import build_masked_entry
    with pytest.raises(ValueError, match="HLA_a is not an amino-acid sequence"):
        build_masked_entry({"HLA_a": "HLA-B*27:05"}, region="peptide", length=9)


# --- iegr_profile invariants. The docs sell one property: lowest entropy is committed first. ---

def test_iegr_profile_commits_lowest_entropy_first(monkeypatch):
    """With a fixed logit table every forward pass sees the same entropies, so the commit order
    must be exactly ascending entropy. Inverting the argmin flips this."""
    from DecoderTCR.design import iegr_profile
    _stub_gibbs(monkeypatch, constant=True)
    prof = iegr_profile({"HLA_a": "AAA"}, length=L, seed=0)
    order = list(prof.sort_values("entropy")["commit_order"])
    assert order == list(range(1, L + 1)), f"commit order not ascending in entropy: {order}"


def test_iegr_profile_shape_and_normalisation(monkeypatch):
    from DecoderTCR.design import iegr_profile
    _stub_gibbs(monkeypatch)
    prof = iegr_profile({"HLA_a": "AAA"}, length=L, seed=0)
    assert list(prof.index) == list(range(1, L + 1)) and prof.index.name == "position"
    assert sorted(prof["commit_order"]) == list(range(1, L + 1)), "must be a permutation of 1..L"
    mat = prof[list(AA20)].to_numpy()
    assert mat.dtype == np.float64 and np.allclose(mat.sum(axis=1), 1.0)


def test_iegr_profile_costs_one_forward_per_residue(monkeypatch):
    """The documented cost. A mutation that stops re-masking would change it."""
    from DecoderTCR.design import iegr_profile
    calls = _stub_gibbs(monkeypatch)
    iegr_profile({"HLA_a": "AAA"}, length=L, seed=0)
    assert calls["forwards"] == L


# --- the two headline knobs must actually be wired, not silently ignored ---

def _record_stages(monkeypatch):
    from DecoderTCR.design import generate
    seen = {"one_shot": 0, "iegr_profile": 0, "block_gibbs": 0, "gibbs_kwargs": None}

    monkeypatch.setattr("DecoderTCR.api._resolve_model",
                        lambda *a, **k: ("MODEL", 36, "cpu", "stub", "esmc"))

    def fake_oneshot(data, length, **kw):
        seen["one_shot"] += 1
        return two_residue_profile(length)

    def fake_iegr_profile(data, region="peptide", length=None, **kw):
        seen["iegr_profile"] += 1
        return two_residue_profile(length)

    def fake_block_gibbs(data, peptides, **kw):
        seen["block_gibbs"] += 1
        seen["gibbs_kwargs"] = kw
        return list(peptides), {"n_input": len(peptides), "n_returned": len(peptides),
                                "n_forwards": 1, "n_changed": 0, "budget_exhausted": False}

    import importlib
    ig = importlib.import_module("DecoderTCR.design.iegr")   # the module, not the function
    monkeypatch.setattr(generate, "peptide_profile", fake_oneshot)
    monkeypatch.setattr(ig, "iegr_profile", fake_iegr_profile)
    monkeypatch.setattr(ig, "block_gibbs", fake_block_gibbs)
    return seen


def test_profile_method_iegr_actually_uses_iegr_profile(monkeypatch):
    """Ignoring profile_method makes a user pay `length` extra forward passes for the one-shot
    answer. Nothing else in the suite notices."""
    from DecoderTCR.design import generate
    seen = _record_stages(monkeypatch)
    generate.design_peptides({"HLA_a": "AAA"}, length=L, n=8, rescore=False,
                             profile_method="iegr", seed=0)
    assert seen["iegr_profile"] == 1 and seen["one_shot"] == 0

    seen = _record_stages(monkeypatch)
    generate.design_peptides({"HLA_a": "AAA"}, length=L, n=8, rescore=False, seed=0)
    assert seen["one_shot"] == 1 and seen["iegr_profile"] == 0


def test_gibbs_k_is_wired_and_carries_n_and_the_budget(monkeypatch):
    """gibbs_k=0 must not refine. gibbs_k>0 must refine AND pass n, which is the fix that stops
    the library collapsing."""
    from DecoderTCR.design import generate
    seen = _record_stages(monkeypatch)
    generate.design_peptides({"HLA_a": "AAA"}, length=L, n=8, rescore=False, gibbs_k=0, seed=0)
    assert seen["block_gibbs"] == 0

    seen = _record_stages(monkeypatch)
    generate.design_peptides({"HLA_a": "AAA"}, length=L, n=8, rescore=False, gibbs_k=3,
                             gibbs_max_forwards=99, seed=0)
    assert seen["block_gibbs"] == 1
    kw = seen["gibbs_kwargs"]
    assert kw["k"] == 3, "k must reach the refiner"
    assert kw["n"] == 8, "the design target must reach the refiner, or the library collapses"
    assert kw["max_forwards"] == 99


# --- block Gibbs correctness. A random stub forward satisfies "returns n distinct peptides" no
# --- matter how broken the walk is, so these use a stub whose output depends on the input.

def _stub_policy(monkeypatch, policy, length=L, target_peptide=None):
    """Stub the model with a forward whose logits are a known function of the token index.

    `policy="recover"` peaks row t on `target_peptide[t-1]`, so at temperature 0 the walk is a
    fixed point that must reproduce that peptide. Row t is deliberately keyed to the TOKEN index,
    so reading without the CLS offset lands on the wrong residue. `policy="always_W"` peaks every
    row on tryptophan regardless of position.
    """
    import importlib
    import torch
    ig = importlib.import_module("DecoderTCR.design.iegr")
    from DecoderTCR.constants import AA20_IDS

    entry = {"sequences": {"HLA_a": "AAA", "HLA_b": "", "TCR_a": "", "TCR_b": "",
                           "Peptide": "A" * length}, "pocket_idx": {}}
    w_id = AA20_IDS[AA20.index("W")]

    class Tok:
        original_ids = torch.zeros(length + 1, dtype=torch.long)

    def fake_forward(model, num_layers, ids, device):
        logits = torch.full((length + 1, 64), -30.0)
        for t in range(length + 1):
            if policy == "always_W":
                target = w_id
            elif t == 0 or target_peptide is None:
                target = w_id
            else:
                target = AA20_IDS[AA20.index(target_peptide[t - 1])]
            logits[t, target] = 30.0
        return logits

    monkeypatch.setattr("DecoderTCR.api._resolve_model",
                        lambda *a, **k: ("MODEL", 36, "cpu", "stub", "esmc"))
    monkeypatch.setattr(ig, "build_masked_entry", lambda *a, **k: entry)
    monkeypatch.setattr(ig, "TCRpMHCTokenizer", lambda *a, **k: Tok())
    monkeypatch.setattr(ig, "region_positions", lambda *a, **k: np.arange(length))
    monkeypatch.setattr(ig, "_forward", fake_forward)


def test_block_gibbs_round_trips_the_seed(monkeypatch):
    """A forward that predicts whatever is already there makes the walk a fixed point, so the seed
    must come back byte-identical. This is the round trip through seed encoding, the CLS offset,
    the residue map and decoding. Drop the +1 or scramble the map and it breaks."""
    from DecoderTCR.design import block_gibbs
    seed = "ACDEFGHIK"
    _stub_policy(monkeypatch, "recover", target_peptide=seed)
    out, _ = block_gibbs({"HLA_a": "AAA"}, [seed], k=4, rounds=3, temperature=0.0, seed=0)
    assert out == [seed], f"seed did not survive a fixed-point walk: {out}"


def test_block_gibbs_changes_exactly_k_positions(monkeypatch):
    """One block at temperature 0 against an always-W model must set exactly k positions to W.
    Resampling every position instead of k is then visible."""
    from DecoderTCR.design import block_gibbs
    for k in (1, 3, 5):
        _stub_policy(monkeypatch, "always_W")
        out, _ = block_gibbs({"HLA_a": "AAA"}, ["ACDEFGHIK"], k=k, rounds=1,
                             temperature=0.0, seed=0)
        assert out[0].count("W") == k, f"k={k} changed {out[0].count('W')} positions: {out[0]}"


def test_block_gibbs_advances_the_chain_across_visits(monkeypatch):
    """Each chain must carry its history. Restarting from the seed every visit caps the drift at
    k positions no matter how long the walk runs."""
    from DecoderTCR.design import block_gibbs
    _stub_policy(monkeypatch, "always_W")
    out, _ = block_gibbs({"HLA_a": "AAA"}, ["ACDEFGHIK"], k=2, rounds=6,
                         temperature=0.0, seed=0)
    assert out[0].count("W") > 2, f"chain did not accumulate change: {out[0]}"


def test_block_gibbs_visits_every_seed(monkeypatch):
    """Round robin. The two seeds use disjoint residue sets, so every collected peptide carries
    the fingerprint of the chain it came from. Advancing only chain 0 never yields the second."""
    from DecoderTCR.design import block_gibbs
    _stub_policy(monkeypatch, "always_W")
    seeds = ["ACDEFGHIK", "LMNPQRSTV"]
    out, _ = block_gibbs({"HLA_a": "AAA"}, seeds, k=1, n=8, temperature=0.0,
                         max_forwards=8, seed=0)
    fams = {i for i in (0, 1)
            for q in out if set(q) - {"W"} <= set(seeds[i])}
    assert fams == {0, 1}, f"only chain(s) {fams} contributed: {out}"


def test_block_gibbs_uses_the_temperature(monkeypatch):
    """At temperature 0 an always-W model drives every chain to the same fixed point, so no budget
    can reach several distinct peptides. Ignoring temperature makes every walk behave this way."""
    from DecoderTCR.design import block_gibbs
    _stub_policy(monkeypatch, "always_W")
    _, cold = block_gibbs({"HLA_a": "AAA"}, ["ACDEFGHIK"], k=3, n=25, temperature=0.0,
                          max_forwards=200, seed=0)
    _stub_policy(monkeypatch, "always_W")
    _, hot = block_gibbs({"HLA_a": "AAA"}, ["ACDEFGHIK"], k=3, n=25, temperature=50.0,
                         max_forwards=200, seed=0)
    assert cold["budget_exhausted"] is True, "a deterministic walk cannot reach 25 distinct"
    assert hot["n_returned"] > cold["n_returned"], (
        f"temperature had no effect: cold {cold['n_returned']}, hot {hot['n_returned']}")
