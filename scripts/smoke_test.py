"""Per-model load, score and design smoke test for the DecoderTCR V0.3 release.

For each registry model: reconstruct and score the gene-input sample (genes_pairs.csv) with
masked-peptide PLL, assert the scores are finite, and report per-epitope binder/non-binder
separation; then profile and design peptides for the full-sequence sample (sequence_pairs.csv)
and assert the designs are well formed. The 6B model needs an 80 GB GPU.

    uv run python scripts/smoke_test.py                 # all 5 models
    uv run python scripts/smoke_test.py DecoderTCR_650M DecoderTCR-ESMC_300M
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
import DecoderTCR as dt                                       # noqa: E402
from DecoderTCR.constants import AA20                         # noqa: E402
from DecoderTCR.design.profile import consensus               # noqa: E402
from DecoderTCR.utils.model_zoo import MODEL_ZOO              # noqa: E402

DESIGN_LENGTH = 9
DESIGN_N = 20

# docs/peptide_design.md documents this consensus for the B*27:05 demo clone under the default
# model, so a mismatch is a regression in the profile path rather than a smoke-test flake.
DOCUMENTED_CONSENSUS = {"DecoderTCR-ESMC_600M": ("B2705_clone", "LRVMMLAPF")}


def design_smoke(name, device, pairs):
    """Profile and design for one clone, returning a one-line report.

    Loads the checkpoint once and hands the module to both passes. `score_from_components` takes
    no `num_layers`, so it still resolves its own.
    """
    row = pairs.iloc[0]
    seqs = {k: row[k] for k in ("HLA_a", "HLA_b", "TCR_a", "TCR_b")}
    mdl, n_layers = dt.load(name, device=device)

    prof = dt.peptide_profile(seqs, length=DESIGN_LENGTH, model=mdl, num_layers=n_layers,
                              device=device)
    assert len(prof) == DESIGN_LENGTH, f"profile has {len(prof)} rows, want {DESIGN_LENGTH}"
    mass = prof[list(AA20)].to_numpy().sum(axis=1)
    assert np.allclose(mass, 1.0, atol=1e-5), f"profile rows sum to {mass}"
    cons = consensus(prof)

    designs = dt.design_peptides(seqs, length=DESIGN_LENGTH, n=DESIGN_N, model=mdl,
                                 num_layers=n_layers, device=device)
    seq = designs["sequence"].tolist()
    assert len(seq) == len(set(seq)), "design returned duplicates"
    assert all(len(s) == DESIGN_LENGTH for s in seq), "design returned a wrong length"
    assert all(set(s) <= set(AA20) for s in seq), "design returned a non-standard residue"
    assert np.isfinite(designs["pll"].to_numpy()).all(), "non-finite design PLL"
    assert "saturated" in designs.attrs, "saturation was not reported"

    want = DOCUMENTED_CONSENSUS.get(name)
    if want and row["name"] == want[0]:
        assert cons == want[1], (f"consensus is {cons}, docs/peptide_design.md documents "
                                 f"{want[1]}")

    return (f"    design {row['name']} | consensus {cons} | {len(seq)}/{DESIGN_N} designs | "
            f"saturated={designs.attrs['saturated']} draws={designs.attrs['n_draws_used']} | "
            f"pll [{designs.pll.min():.3f},{designs.pll.max():.3f}]")


def main():
    names = sys.argv[1:] or list(MODEL_ZOO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  models={names}\n")
    pairs = pd.read_csv(ROOT / "Demo/sample_data/genes_pairs.csv")
    seq_pairs = pd.read_csv(ROOT / "Demo/sample_data/sequence_pairs.csv")

    ok = True
    for name in names:
        spec = MODEL_ZOO[name]
        print(f"=== {name}  [{spec.backbone} / {spec.arch}] ===")
        try:
            scored = dt.score_from_components(pairs, model=name, device=device)
            valid = scored[scored.ok]
            s = valid[f"pll_{name}"].to_numpy()
            assert len(s) and np.isfinite(s).all(), "empty or non-finite scores"
            sep = []
            for pep, g in valid.groupby("peptide"):
                if g.label.nunique() == 2:
                    sep.append(f"{pep} pos={g[g.label == 1][f'pll_{name}'].mean():.2f} "
                               f"neg={g[g.label == 0][f'pll_{name}'].mean():.2f}")
            print(f"    {len(valid)}/{len(scored)} ok | PLL [{s.min():.3f},{s.max():.3f}] | "
                  + " | ".join(sep))
            print(design_smoke(name, device, seq_pairs))
        except Exception as e:
            ok = False
            print(f"    FAILED: {type(e).__name__}: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print()

    print("SMOKE OK" if ok else "SMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
