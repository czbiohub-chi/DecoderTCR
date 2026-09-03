#!/usr/bin/env python3
"""Design peptide libraries for TCR-HLA complexes supplied as full sequences.

Input is a JSON list of complexes. Each entry carries the four chains and may override the
run-wide settings, so one file can mix peptide lengths and temperatures.

    [
      {"name": "AS8.4", "HLA_a": "GSHSMRY...", "HLA_b": "MIQRTPK...",
       "TCR_a": "METLLGL...", "TCR_b": "MGFRLLC...", "length": 9},
      {"name": "Aga1", "HLA_a": "...", "HLA_b": "...", "TCR_a": "...", "TCR_b": "...",
       "length": 11, "temperature": 1.5}
    ]

Design a library for every complex:
    python -m DecoderTCR.utils.peptide_design -i complexes.json -o out/ -n 10000 --temperature 1.25

Profile only, no sampling:
    python -m DecoderTCR.utils.peptide_design -i complexes.json -o out/ --profile-only

Four artifacts are written into the output directory:
    designs.csv    one row per designed peptide, with its masked-peptide PLL, best first
    profiles.csv   the per-position amino-acid distribution and entropy, one block per complex
    manifest.json  settings, per-complex statistics, and the saturation flag
    logos/         one sequence logo per complex

Per-entry overrides: name, length, n, temperature, seed, profile_method, gibbs_k,
gibbs_rounds, gibbs_temperature. Everything else comes from the flags.

The peptide is masked in full, so no peptide is supplied and no anchor is fixed. To start from
V/J genes instead of sequences, use DecoderTCR.utils.peptide_profile, which reconstructs each row.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch

from DecoderTCR.design.generate import MAX_CODEABLE_LENGTH, METHODS
from DecoderTCR.utils.model_zoo import MODEL_ZOO

SEQ_KEYS = ("HLA_a", "HLA_b", "TCR_a", "TCR_b")
OVERRIDES = ("length", "n", "temperature", "seed", "profile_method", "gibbs_k",
             "gibbs_rounds", "gibbs_temperature", "gibbs_max_forwards")
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DecoderTCR: design peptide libraries from full TCR and HLA sequences",
        epilog="Per-entry JSON overrides: " + ", ".join(OVERRIDES))
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="JSON list of complexes, each with HLA_a, HLA_b, TCR_a, TCR_b")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="output directory, created if absent")

    p.add_argument("-L", "--length", type=int, default=9, help="peptide length (default 9)")
    p.add_argument("-n", "--num", type=int, default=1000, metavar="N",
                   help="distinct peptides to design per complex (default 1000)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="sampling temperature, 0 gives the consensus peptide (default 1.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap", type=int, default=1_000_000,
                   help="proposals to spend per complex chasing N distinct (default 1000000)")
    p.add_argument("--method", choices=["one_shot", "iegr"], default="one_shot",
                   help="design method (default one_shot, a single forward pass)")

    g = p.add_argument_group("one-shot pipeline knobs")
    g.add_argument("--profile-method", choices=["one_shot", "iegr"], default="one_shot",
                   help="how the profile is built. one_shot is a single masked forward pass. "
                        "iegr commits one position at a time in entropy order, at one forward "
                        "pass per residue (default one_shot)")
    g.add_argument("--gibbs-k", type=int, default=0, metavar="K",
                   help="block Gibbs: re-mask K positions of each sampled peptide and resample "
                        "them together, running until N distinct designs are collected. "
                        "0 disables it. Its forward passes dominate a large run (default 0)")
    g.add_argument("--gibbs-temperature", type=float, default=None,
                   help="temperature for the block Gibbs resampling (default: --temperature)")
    g.add_argument("--gibbs-max-forwards", type=int, default=None, metavar="F",
                   help="cap on block Gibbs forward passes per complex. Refinement runs until N "
                        "distinct designs are collected or this many passes are spent "
                        "(default 20 per requested design)")
    p.add_argument("--profile-only", action="store_true",
                   help="write the profile and logo, skip sampling")
    p.add_argument("--no-rescore", action="store_true",
                   help="skip scoring the designs with masked-peptide PLL")
    p.add_argument("--no-logo", action="store_true", help="skip the sequence logos")
    p.add_argument("--logo-units", choices=["bits", "probability"], default="bits")
    p.add_argument("--gibbs-rounds", type=int, default=10,
                   help="blocks resampled per peptide for --gibbs-k, and rounds for --method iegr")
    p.add_argument("--gibbs-subset-size", type=int, default=5, help="--method iegr only")

    p.add_argument("-m", "--model", default=None,
                   help=f"Registry model name. One of: {list(MODEL_ZOO)}")
    p.add_argument("-c", "--checkpoint", default=None,
                   help="Explicit checkpoint path (requires --backbone and --arch)")
    p.add_argument("--backbone", choices=["esm2", "esmc"], default=None)
    p.add_argument("--arch", default=None)
    p.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def load_complexes(path: Path) -> list[dict]:
    """Read the JSON list, naming any entry that did not name itself."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):                       # tolerate a single complex
        data = [data]
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON list of complexes, got {type(data).__name__}")
    if not data:
        raise SystemExit(f"{path}: no complexes")
    for i, e in enumerate(data):
        if not isinstance(e, dict):
            raise SystemExit(f"{path}: entry {i} is {type(e).__name__}, expected an object")
        # A present-but-null override is not a value. Strip nulls so they read as absent rather
        # than reaching int()/float() outside the per-complex guard and killing the whole run.
        for k in [k for k, v in e.items() if v is None]:
            del e[k]
        e.setdefault("name", f"complex_{i}")
    return data


# Per-entry override validation: (caster, minimum, maximum). A value outside this is a reason to
# skip the complex, never a traceback that discards the artifacts already written.
_LIMITS = {
    "length": (int, 1, MAX_CODEABLE_LENGTH),
    "n": (int, 1, None),
    "seed": (int, 0, None),
    "gibbs_k": (int, 0, None),
    "gibbs_rounds": (int, 1, None),
    "gibbs_max_forwards": (int, 1, None),
    "temperature": (float, 0.0, None),
    "gibbs_temperature": (float, 0.0, None),
}


def check(entry: dict) -> str | None:
    """Reason this complex cannot be designed for, or None when it is usable."""
    unknown = set(entry) - set(SEQ_KEYS) - set(OVERRIDES) - {"name"}
    if unknown:
        return f"unknown field(s): {sorted(unknown)}"
    for key, (cast, lo, hi) in _LIMITS.items():
        if entry.get(key) is None:
            continue
        try:
            v = cast(entry[key])
        except (TypeError, ValueError):
            return f"{key} is not a {cast.__name__}: {entry[key]!r}"
        if v < lo or (hi is not None and v > hi):
            bound = f"{lo} to {hi}" if hi is not None else f"at least {lo}"
            return f"{key} must be {bound}, got {v}"
    if entry.get("profile_method") not in (None, *METHODS):
        return f"profile_method must be one of {list(METHODS)}, got {entry['profile_method']!r}"
    for k in SEQ_KEYS:
        v = entry.get(k) or ""
        if v and not isinstance(v, str):
            return f"{k} must be a string, got {type(v).__name__}"
        if v and not AA_RE.match(str(v).upper()):
            return f"{k} has residues outside the 20 standard amino acids"
    if not (entry.get("HLA_a") or entry.get("TCR_a") or entry.get("TCR_b")):
        return "nothing to condition on, supply at least HLA_a or a TCR chain"
    return None


def short_reason(rec: dict, requested: int, cap: int) -> str:
    """Console notice for a result that came back short, naming the cause, or an empty string.

    A short result has several causes and they call for different responses, so reporting only
    sampling saturation leaves block Gibbs collapse and the structural IEGR ceiling silent.
    """
    if not rec.get("short"):
        return ""
    if rec.get("gibbs_budget_exhausted"):
        cause = ("block Gibbs ran out of forward passes, raise --gibbs-max-forwards")
    elif rec.get("gibbs_collapsed"):
        cause = f"block Gibbs collapsed {rec['gibbs_collapsed']} into duplicates"
    elif rec.get("support_exhausted"):
        cause = "the profile has no more distinct peptides"
    elif rec.get("cap_reached"):
        cause = f"the draw budget ran out, raise --cap above {cap}"
    else:
        cause = "the method is structurally capped below the request"
    return f"  SHORT, {rec['n_returned']} of {requested}, {cause}"


def _slug(name: str) -> str:
    """Filesystem-safe stem for a per-complex artifact."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "complex"


def main():
    args = _build_parser().parse_args()
    if args.checkpoint and not (args.backbone and args.arch):
        raise SystemExit("--checkpoint requires --backbone and --arch")
    if args.num < 1:
        raise SystemExit(f"-n/--num must be at least 1, got {args.num}")
    if not 1 <= args.length <= MAX_CODEABLE_LENGTH:
        raise SystemExit(f"-L/--length must be 1 to {MAX_CODEABLE_LENGTH}, got {args.length}")
    if args.cap < 1:
        raise SystemExit(f"--cap must be at least 1, got {args.cap}")

    from DecoderTCR.api import _resolve_model
    from DecoderTCR.design import design_peptides, peptide_profile, sequence_logo
    from DecoderTCR.design.profile import consensus

    entries = load_complexes(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(entries)} complexes from {args.input}")

    model, n_layers, device, name, _ = _resolve_model(args.model, None, args.device,
                                                      args.checkpoint, args.backbone, args.arch)
    common = dict(model=model, num_layers=n_layers, device=device)
    print(f"model {name} ({n_layers} layers) on {device}\n")

    designs, profiles, records = [], [], []
    for entry in entries:
        label = entry["name"]
        reason = check(entry)
        if reason:
            print(f"  {label:20} SKIPPED  {reason}")
            records.append(dict(name=label, ok=False, reason=reason))
            continue

        length = int(entry.get("length", args.length))
        n = int(entry.get("n", args.num))
        temperature = float(entry.get("temperature", args.temperature))
        seed = int(entry.get("seed", args.seed))
        profile_method = str(entry.get("profile_method", args.profile_method))
        gibbs_k = int(entry.get("gibbs_k", args.gibbs_k))
        gibbs_rounds = int(entry.get("gibbs_rounds", args.gibbs_rounds))
        gibbs_temperature = entry.get("gibbs_temperature", args.gibbs_temperature)
        gibbs_temperature = temperature if gibbs_temperature is None else float(gibbs_temperature)
        gibbs_max_forwards = entry.get("gibbs_max_forwards", args.gibbs_max_forwards)
        gibbs_max_forwards = None if gibbs_max_forwards is None else int(gibbs_max_forwards)
        seqs = {k: str(entry.get(k) or "").upper() for k in SEQ_KEYS}

        try:
            rec = dict(name=label, ok=True, length=length, temperature=temperature, seed=seed,
                       profile_method=profile_method, gibbs_k=gibbs_k)
            if gibbs_k:
                rec.update(gibbs_rounds=gibbs_rounds, gibbs_temperature=gibbs_temperature)
            if profile_method == "iegr":
                from DecoderTCR.design import iegr_profile
                prof = iegr_profile(seqs, region="peptide", length=length,
                                    temperature=temperature, seed=seed, **common)
            else:
                prof = peptide_profile(seqs, length=length, **common)
            seq = consensus(prof)
            rec.update(consensus=seq, mean_entropy=round(float(prof.entropy.mean()), 4))

            block = prof.reset_index()
            block.insert(0, "name", label)
            profiles.append(block)

            if not args.profile_only:
                out = design_peptides(seqs, length=length, n=n, method=args.method,
                                      temperature=temperature, seed=seed, cap=args.cap,
                                      rescore=not args.no_rescore, profile_method=profile_method,
                                      gibbs_k=gibbs_k, gibbs_rounds=gibbs_rounds,
                                      gibbs_temperature=gibbs_temperature,
                                      gibbs_max_forwards=gibbs_max_forwards,
                                      gibbs_subset_size=args.gibbs_subset_size, **common)
                out.insert(0, "name", label)
                out.insert(1, "rank", range(1, len(out) + 1))
                designs.append(out)
                rec.update(n_requested=n, n_returned=len(out),
                           n_draws_used=out.attrs.get("n_draws_used"),
                           saturated=bool(out.attrs.get("saturated", len(out) < n)),
                           cap_reached=bool(out.attrs.get("cap_reached", False)),
                           support_exhausted=bool(out.attrs.get("support_exhausted", False)))
                if gibbs_k:
                    rec.update(gibbs_forwards=out.attrs.get("gibbs_n_forwards"),
                               gibbs_changed=out.attrs.get("gibbs_n_changed"),
                               gibbs_budget_exhausted=bool(
                                   out.attrs.get("gibbs_budget_exhausted", False)),
                               gibbs_collapsed=out.attrs.get("gibbs_n_input", 0)
                               - out.attrs.get("gibbs_n_returned", 0))
                rec["short"] = len(out) < n

            if not args.no_logo:
                logos = args.output / "logos"
                logos.mkdir(exist_ok=True)
                path = logos / f"{_slug(label)}.png"
                sequence_logo(prof, units=args.logo_units, save=path, title=str(label))
                rec["logo"] = str(path.relative_to(args.output))

            flag = short_reason(rec, n, args.cap)
            print(f"  {label:20} L={length:<3d} consensus={seq}"
                  + (f"  designs={rec['n_returned']}" if not args.profile_only else "") + flag)

        except Exception as e:                       # keep the artifacts already built
            print(f"  {label:20} FAILED  {type(e).__name__}: {e}")
            records.append(dict(name=label, ok=False,
                                reason=f"{type(e).__name__}: {e}"))
            continue

        records.append(rec)

    if profiles:
        pd.concat(profiles, ignore_index=True).to_csv(args.output / "profiles.csv", index=False)
    if designs:
        pd.concat(designs, ignore_index=True).to_csv(args.output / "designs.csv", index=False)

    manifest = dict(
        model=name, checkpoint=args.checkpoint, device=str(device), method=args.method,
        defaults=dict(length=args.length, n=args.num, temperature=args.temperature,
                      seed=args.seed, cap=args.cap, rescore=not args.no_rescore,
                      profile_method=args.profile_method, gibbs_k=args.gibbs_k,
                      gibbs_rounds=args.gibbs_rounds,
                      gibbs_temperature=args.gibbs_temperature,
                      gibbs_max_forwards=args.gibbs_max_forwards),
        n_complexes=len(entries), n_ok=sum(r["ok"] for r in records),
        n_failed=sum(not r["ok"] for r in records),
        n_saturated=sum(bool(r.get("saturated")) for r in records),
        n_short=sum(bool(r.get("short")) for r in records),
        complexes=records)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nwrote {args.output}/  "
          + ", ".join(p for p, ok in [("profiles.csv", bool(profiles)),
                                      ("designs.csv", bool(designs)),
                                      ("manifest.json", True),
                                      ("logos/", not args.no_logo)] if ok))
    print(f"complexes: {manifest['n_ok']} ok, {manifest['n_failed']} skipped, "
          f"{manifest['n_short']} short of the request")
    raise SystemExit(1 if manifest["n_failed"] else 0)


if __name__ == "__main__":
    main()
