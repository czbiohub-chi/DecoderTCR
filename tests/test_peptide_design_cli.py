"""Tests for the JSON design CLI's input handling.

Everything here runs on the CPU with no model: `load_complexes` and `check` decide what reaches
the GPU, so they are worth testing on their own. The forward pass itself is covered by
scripts/smoke_test.py.
"""

from __future__ import annotations

import json

import pytest

from DecoderTCR.utils.peptide_design import _build_parser, _slug, check, load_complexes

REAL = "ACDEFGHIKLMNPQRSTVWY" * 3


def write(tmp_path, obj):
    p = tmp_path / "complexes.json"
    p.write_text(json.dumps(obj))
    return p


def good(**over):
    e = {"HLA_a": REAL, "HLA_b": REAL, "TCR_a": REAL, "TCR_b": REAL}
    e.update(over)
    return e


def test_reads_a_list_and_names_unnamed_entries(tmp_path):
    entries = load_complexes(write(tmp_path, [good(), good(name="mine")]))
    assert [e["name"] for e in entries] == ["complex_0", "mine"]


def test_accepts_a_single_object(tmp_path):
    """A user with one complex should not have to wrap it in a list."""
    entries = load_complexes(write(tmp_path, good(name="solo")))
    assert len(entries) == 1 and entries[0]["name"] == "solo"


@pytest.mark.parametrize("payload,message", [
    ([], "no complexes"),
    ("a string", "expected a JSON list"),
    ([{"HLA_a": REAL}, 7], "expected an object"),
])
def test_rejects_malformed_input(tmp_path, payload, message):
    with pytest.raises(SystemExit, match=message):
        load_complexes(write(tmp_path, payload))


def test_accepts_a_complete_complex():
    assert check(good()) is None


def test_accepts_hla_only_and_tcr_only():
    """build_masked_entry needs one side, not both, so the CLI must not demand both."""
    assert check({"HLA_a": REAL}) is None
    assert check({"TCR_b": REAL}) is None


def test_rejects_a_complex_with_nothing_to_condition_on():
    assert "nothing to condition on" in check({"length": 9})


def test_rejects_non_standard_residues():
    """A stray X or a nucleotide sequence should be named, not fed to the tokenizer."""
    assert "outside the 20 standard" in check(good(TCR_a="ACDEFX"))
    assert "TCR_b" in check(good(TCR_b="ACGTACGT-ACGT"))


def test_rejects_unknown_fields():
    """A typo in an override would otherwise be silently ignored and the default used."""
    reason = check(good(temperatur=1.5))
    assert "unknown field" in reason and "temperatur" in reason


@pytest.mark.parametrize("field,value", [("length", 9), ("n", 100), ("temperature", 1.2),
                                         ("seed", 7), ("name", "clone")])
def test_documented_overrides_are_accepted(field, value):
    assert check(good(**{field: value})) is None


def test_slug_makes_a_safe_filename():
    assert _slug("AS4.3-91mer") == "AS4.3-91mer"
    assert _slug("HLA-B*27:05 clone") == "HLA-B_27_05_clone"
    assert _slug("///") == "complex"


def test_parser_requires_input_and_output():
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])
    args = _build_parser().parse_args(["-i", "a.json", "-o", "out"])
    assert args.length == 9 and args.num == 1000 and args.temperature == 1.0
    assert args.method == "one_shot" and args.cap == 1_000_000


@pytest.mark.parametrize("field,value", [("profile_method", "iegr"), ("profile_method", "one_shot"),
                                         ("gibbs_k", 3), ("gibbs_rounds", 2),
                                         ("gibbs_temperature", 1.5)])
def test_iegr_and_gibbs_overrides_are_accepted(field, value):
    """The knobs are per-entry, so one file can mix pipelines across complexes."""
    assert check(good(**{field: value})) is None


def test_parser_exposes_the_pipeline_knobs():
    args = _build_parser().parse_args(["-i", "a.json", "-o", "out"])
    assert args.profile_method == "one_shot"
    assert args.gibbs_k == 0                      # block Gibbs is off unless asked for
    assert args.gibbs_temperature is None         # falls back to --temperature
    args = _build_parser().parse_args(["-i", "a.json", "-o", "out", "--profile-method", "iegr",
                                       "--gibbs-k", "3", "--gibbs-temperature", "1.5"])
    assert args.profile_method == "iegr" and args.gibbs_k == 3 and args.gibbs_temperature == 1.5


def test_parser_rejects_an_unknown_profile_method():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-i", "a.json", "-o", "out", "--profile-method", "gibbs"])


# --- the audit proved a bad override value crashed the run and discarded every artifact ---

@pytest.mark.parametrize("override,fragment", [
    ({"n": 0}, "n must be at least 1"),
    ({"length": 0}, "length must be 1 to 14"),
    ({"length": 15}, "length must be 1 to 14"),
    ({"length": "nine"}, "length is not a int"),
    ({"temperature": -1}, "temperature must be at least 0"),
    ({"gibbs_rounds": 0}, "gibbs_rounds must be at least 1"),
    ({"profile_method": "gibbs"}, "profile_method must be one of"),
])
def test_bad_override_values_are_a_skip_reason_not_a_crash(override, fragment):
    """These used to reach the model and raise mid-run, discarding every complex already done."""
    reason = check(good(**override))
    assert reason is not None and fragment in reason


def test_a_null_chain_does_not_become_the_string_none():
    """`entry.get(k, "")` only defaults on a MISSING key, so an explicit JSON null used to reach
    the tokenizer as the literal 'NONE'."""
    entry = {"name": "x", "HLA_a": REAL, "HLA_b": None, "TCR_a": REAL, "TCR_b": REAL}
    assert check(entry) is None
    seqs = {k: str(entry.get(k) or "").upper() for k in ("HLA_a", "HLA_b", "TCR_a", "TCR_b")}
    assert seqs["HLA_b"] == "", f"null became {seqs['HLA_b']!r}"


def test_length_limit_matches_the_sampler():
    """The CLI must refuse what sample_from_profile's dedup packing cannot encode, before the GPU."""
    from DecoderTCR.design.generate import MAX_CODEABLE_LENGTH
    assert check(good(length=MAX_CODEABLE_LENGTH)) is None
    assert "length must be" in check(good(length=MAX_CODEABLE_LENGTH + 1))


# --- a short result has several causes and each needs a different response from the user ---

@pytest.mark.parametrize("rec,fragment", [
    ({"short": False, "n_returned": 50}, ""),
    ({"short": True, "n_returned": 43, "gibbs_collapsed": 7}, "block Gibbs collapsed 7"),
    ({"short": True, "n_returned": 1, "support_exhausted": True}, "no more distinct peptides"),
    ({"short": True, "n_returned": 1500, "cap_reached": True}, "raise --cap above 1500"),
    ({"short": True, "n_returned": 12}, "structurally capped"),
])
def test_short_reason_names_the_cause(rec, fragment):
    from DecoderTCR.utils.peptide_design import short_reason
    out = short_reason(rec, requested=50, cap=1500)
    assert fragment in out
    if fragment:
        assert out.startswith("  SHORT, ")


def test_gibbs_collapse_is_not_silent():
    """The smoke returned 43 of 50 after Gibbs dedup while `saturated` was false, so keying the
    notice off saturation alone reported a complete run."""
    from DecoderTCR.utils.peptide_design import short_reason
    rec = {"short": True, "n_returned": 43, "saturated": False, "cap_reached": False,
           "support_exhausted": False, "gibbs_collapsed": 7}
    assert "SHORT, 43 of 50" in short_reason(rec, requested=50, cap=1_000_000)


def test_short_reason_keys_are_actually_produced_by_the_pipeline():
    """short_reason branched on `gibbs_budget_exhausted` while the record builder never wrote it,
    so the budget branch was dead and a budget-bound run was mislabelled as Gibbs collapse. Pin
    the key names on both sides of that handoff."""
    import inspect
    from DecoderTCR.design import generate
    from DecoderTCR.utils import peptide_design as pdm

    src = inspect.getsource(pdm.main)
    reason_src = inspect.getsource(pdm.short_reason)
    gibbs_keys = {k for k in ("gibbs_budget_exhausted", "gibbs_collapsed")
                  if f'"{k}"' in reason_src or f"'{k}'" in reason_src}
    assert gibbs_keys, "short_reason should branch on at least one gibbs key"
    for k in gibbs_keys:
        assert k in src, f"short_reason reads {k} but main() never writes it into the record"

    # And the underlying stat exists on the design frame, under the prefixed name.
    assert 'f"gibbs_{key}"' in inspect.getsource(generate.design_peptides)
