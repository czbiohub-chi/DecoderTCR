#!/usr/bin/env python3
"""Profile the peptides a TCR and HLA expect, and optionally design new ones.

Supply the TCR's V/J genes and CDR3s plus the HLA allele. No peptide is needed, only the
length to profile. DecoderTCR reconstructs the complex, masks the whole peptide, and reads out
the per-position amino-acid distribution in a single forward pass.

Reconstruction (stitchr) is installed by default. Fetch IMGT germline data once:
    uv run stitchrdl -s human

Profile a 9-mer and draw the logo:
    python -m DecoderTCR.utils.peptide_profile \\
        --trav TRAV21 --traj TRAJ6 --cdr3a CAVRPGGAGPFFVVF \\
        --trbv TRBV7-9 --trbj TRBJ2-7 --cdr3b CASSLGQAYEQYF \\
        --hla 'HLA-B*27:05' --length 9 -o profile.csv --logo-out logo.png -d cuda:0

Also design 20 candidate peptides:
    python -m DecoderTCR.utils.peptide_profile ... --length 9 --design 20 -o designs.csv

Use the paper's iterative entropy-guided decoding instead of one-shot sampling:
    python -m DecoderTCR.utils.peptide_profile ... --design 20 --method iegr -o designs.csv

Batch (a CSV of complexes in, one output per row):
    python -m DecoderTCR.utils.peptide_profile -i genes.csv --length 9 -o profiles.csv

CSV columns / flags (case-insensitive, aliases accepted): trav, traj, cdr3a, trbv, trbj, cdr3b,
hla (optional name). Any peptide column is ignored.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from DecoderTCR.utils.model_zoo import MODEL_ZOO

_FIELDS = ("trav", "traj", "cdr3a", "trbv", "trbj", "cdr3b", "hla")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DecoderTCR: peptide profile and design for a TCR and HLA")
    p.add_argument("-i", "--input", type=Path, default=None,
                   help="input CSV of complexes (batch mode)")
    p.add_argument("-o", "--output", type=Path, default=None, help="output CSV")
    g = p.add_argument_group("single complex (use instead of -i)")
    for f in _FIELDS:
        g.add_argument(f"--{f}", default=None, help=f"{f} for a single complex")
    g.add_argument("--name", default=None, help="optional name for the single complex")

    p.add_argument("-L", "--length", type=int, default=9, help="peptide length (default 9)")
    p.add_argument("--logo-out", type=Path, default=None, help="write a sequence logo here")
    p.add_argument("--logo-units", choices=["bits", "probability"], default="bits")
    p.add_argument("--design", type=int, default=0, metavar="N",
                   help="also generate N candidate peptides (default 0, profile only)")
    p.add_argument("--method", choices=["one_shot", "iegr"], default="one_shot",
                   help="design method (default one_shot, a single forward pass)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="sampling temperature, 0 gives the consensus peptide")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gibbs-rounds", type=int, default=10, help="IEGR only")
    p.add_argument("--gibbs-subset-size", type=int, default=5, help="IEGR only")
    p.add_argument("--no-rescore", action="store_true",
                   help="skip scoring the designs with masked-peptide PLL")

    p.add_argument("-m", "--model", default=None,
                   help=f"Registry model name. One of: {list(MODEL_ZOO)}")
    p.add_argument("-c", "--checkpoint", default=None,
                   help="Explicit checkpoint path (requires --backbone and --arch)")
    p.add_argument("--backbone", choices=["esm2", "esmc"], default=None)
    p.add_argument("--arch", default=None)
    p.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def _rows(args) -> list[dict]:
    """One dict per complex, from the single-complex flags or the input CSV."""
    single = {f: getattr(args, f) for f in _FIELDS}
    if any(v is not None for v in single.values()):
        missing = [f for f, v in single.items() if v is None]
        if missing:
            raise SystemExit(f"single-complex mode needs all of {list(_FIELDS)}, missing: {missing}")
        if args.name:
            single["name"] = args.name
        return [single]
    if args.input is None:
        raise SystemExit("supply either -i/--input or the single-complex flags")
    print(f"Loaded rows from {args.input}")
    return pd.read_csv(args.input).to_dict("records")


def main():
    args = _build_parser().parse_args()
    if args.checkpoint and not (args.backbone and args.arch):
        raise SystemExit("--checkpoint requires --backbone and --arch")

    from DecoderTCR.design import design_peptides, peptide_profile, sequence_logo

    rows = _rows(args)
    common = dict(model=args.model, device=args.device, from_genes=True,
                  checkpoint=args.checkpoint, backbone=args.backbone, arch=args.arch)
    frames = []
    for i, row in enumerate(rows):
        name = row.get("name", f"complex_{i}")
        if args.design:
            out = design_peptides(row, length=args.length, n=args.design, method=args.method,
                                  temperature=args.temperature, seed=args.seed,
                                  rescore=not args.no_rescore, gibbs_rounds=args.gibbs_rounds,
                                  gibbs_subset_size=args.gibbs_subset_size, **common)
        else:
            out = peptide_profile(row, length=args.length, **common).reset_index()
        out.insert(0, "name", name)
        frames.append(out)

        if args.logo_out:
            prof = (peptide_profile(row, length=args.length, **common)
                    if args.design else out.set_index("position").drop(columns="name"))
            path = (args.logo_out if len(rows) == 1
                    else args.logo_out.with_name(f"{args.logo_out.stem}_{name}{args.logo_out.suffix}"))
            sequence_logo(prof, units=args.logo_units, save=path, title=str(name))
            print(f"Wrote {path}")

    result = pd.concat(frames, ignore_index=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Wrote {args.output}  ({len(result)} rows)")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
