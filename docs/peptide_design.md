# Peptide profiles and design

Design runs in two layers, with the model boundary at the profile. Sampling touches no model, so
one profile can be resampled at many temperatures and library sizes without another forward pass.
The two optional stages, entropy-guided profiling and block Gibbs refinement, do need it.

| layer | call | forward passes |
|---|---|---|
| profile | `dt.peptide_profile(seqs, length)` | 1 |
| profile, alternative | `dt.iegr_profile(seqs, length=L)` | `length` |
| sample | `dt.sample_from_profile(profile, n, temperature)` | 0 |
| refine, optional | `dt.block_gibbs(seqs, peptides, k=..., n=...)` | up to `max_forwards` |
| all of it | `dt.design_peptides(seqs, length, n)` | 1, or more with the knobs |

## Input

Design takes full sequences: `HLA_a`, `HLA_b`, `TCR_a` and `TCR_b`. No peptide is supplied, only
the length to design. [`Demo/sample_data/sequence_pairs.csv`](../Demo/sample_data/sequence_pairs.csv)
ships three clones in this format.

```python
import pandas as pd
import DecoderTCR as dt

pair = pd.read_csv("Demo/sample_data/sequence_pairs.csv").iloc[0]
seqs = {k: pair[k] for k in ("HLA_a", "HLA_b", "TCR_a", "TCR_b")}
```

## Profiling

`dt.peptide_profile` masks the whole peptide and returns the per-position amino-acid distribution
from a single forward pass, indexed by 1-based `position`, with the 20 residue columns summing to 1
plus a per-position `entropy` column.

```python
prof = dt.peptide_profile(seqs, length=9, device="cuda:0")
print(dt.consensus(prof))
dt.sequence_logo(prof, save="logo.png")
```

For the HLA-B\*27:05 clone shipped as `B2705_clone` the consensus is `LRVMMLAPF`, the epitope it
recognizes, with arginine at position 2 at 0.98, the canonical B\*27:05 anchor. Omit the TCR chains
to profile the HLA alone, which here raises the mean per-position entropy from 0.26 to 2.29.

## Sampling

`dt.sample_from_profile` draws each position independently from the profile and returns distinct
peptides in draw order, alongside statistics about the draw. It needs no model, no GPU and no
weights.

```python
peptides, stats = dt.sample_from_profile(prof, n=1000, temperature=1.2)
print(stats["n_returned"], stats["saturated"])
```

| argument | default | meaning |
|---|---|---|
| `n` | 1000 | distinct peptides to return |
| `temperature` | 1.0 | below 1 sharpens toward the most probable residue, above 1 flattens |
| `seed` | 42 | same seed gives identical output |
| `cap` | 1000000 | proposals to spend before giving up on reaching `n` |

Peptides longer than 14 are refused, because dedup packs each one into a single uint64.

`temperature=0` returns the consensus peptide alone.

### Saturation

Sampling draws with replacement and keeps distinct peptides, so a request can come back short
either because the profile ran out of distinct peptides or because the proposal budget ran out.
Those need different responses, so they are reported separately. The statistics carry
`n_requested`, `n_returned`, `n_draws_used` and three flags:

| flag | meaning | what to do |
|---|---|---|
| `saturated` | the result is short of `n` | look at the other two |
| `cap_reached` | the proposal budget ran out | raise `cap` |
| `support_exhausted` | the last batch produced nothing new | raise `temperature`, `cap` will not help |

A short result with `support_exhausted` false is budget-bound, so a larger `cap` returns more. With
it true the distribution genuinely has no more distinct peptides, and only a flatter profile will.

## Both at once

`dt.design_peptides` chains the two layers and scores each design with the masked-peptide PLL that
`dt.score` reports, sorted best first.

```python
designs = dt.design_peptides(seqs, length=9, n=20, device="cuda:0")
print(designs.attrs["saturated"], designs.attrs["n_draws_used"])
```

The returned frame has `sequence`, `method`, `phase`, `step` and `pll`. The sampling statistics are
attached to `designs.attrs`. Pass `rescore=False` to skip the PLL pass.

## Entropy-guided profiling

`dt.iegr_profile` builds the profile by walking the peptide instead of reading it from one pass. It
commits the lowest-entropy position, re-masks the rest, runs another pass, and records each
position's distribution at the moment it is committed. Row `i` is therefore conditioned on every
position committed before it, which a one-shot profile cannot express.

```python
prof = dt.iegr_profile(seqs, length=9, device="cuda:0")
peptides, stats = dt.sample_from_profile(prof, n=1000, temperature=1.2)
```

It returns the same frame as `peptide_profile` plus a `commit_order` column, so everything
downstream consumes it unchanged. It costs one forward pass per residue rather than one in total.

## Block Gibbs refinement

One-shot sampling draws every position independently, so it cannot represent dependence between
positions. `dt.block_gibbs` takes a sampled library, re-masks a random block of `k` positions in a
peptide, and resamples that block together from one forward pass.

Pass `n` to hold the library size. The walks then run round robin and every distinct peptide they
visit is kept, until `n` distinct designs are collected or `max_forwards` passes are spent:

```python
refined, stats = dt.block_gibbs(seqs, peptides, k=3, n=1000, temperature=1.0)
print(stats["n_returned"], stats["n_forwards"], stats["budget_exhausted"])
```

Without `n`, each input peptide is walked for exactly `rounds` blocks and only its final state is
kept, so peptides that refine onto each other shrink the library.

| argument | default | meaning |
|---|---|---|
| `k` | 5 | positions re-masked per block |
| `n` | none | target number of distinct designs. Without it, the library can shrink |
| `rounds` | 1 | blocks per peptide, used only when `n` is not set |
| `max_forwards` | `20 * n`, or `rounds * len(peptides)` without `n` | cap on forward passes, so a target that cannot be reached still stops |

This needs the model, so unlike `sample_from_profile` it is not free, and its forward passes
dominate every other cost in a large run. The statistics carry `n_input`, `n_returned`,
`n_forwards` and `budget_exhausted`, the last of which says the walk stopped on `max_forwards`
rather than reaching `n`.

Consecutive states of one walk differ by at most `k` positions, so they are correlated. A `k`
closer to the peptide length decorrelates faster at the cost of keeping less context.

Both knobs are reachable from `dt.design_peptides` without calling the stages yourself:

```python
designs = dt.design_peptides(seqs, length=9, n=1000, profile_method="iegr",
                             gibbs_k=3, gibbs_temperature=1.0)
```

Refinement here targets the same `n` you asked for, so the library does not shrink.

| argument | default | meaning |
|---|---|---|
| `profile_method` | `"one_shot"` | `"iegr"` builds the profile position by position |
| `gibbs_k` | 0 | positions re-masked per block. 0 disables refinement |
| `gibbs_temperature` | `temperature` | temperature for the resampled blocks |
| `gibbs_max_forwards` | `20 * n` | cap on refinement forward passes |
| `gibbs_rounds` | 10 | ignored here, since refinement stops on `n`. Applies to `method="iegr"` |

## End-to-end IEGR

`method="iegr"` is a different algorithm from the knobs above. Rather than sampling a library from
a profile, it walks a single design through both phases of the paper's decoder.

```python
designs = dt.design_peptides(seqs, length=9, n=20, method="iegr", device="cuda:0")
```

## Command line

`DecoderTCR.utils.peptide_design` takes a JSON list of complexes and writes a whole run to a directory.

```bash
python -m DecoderTCR.utils.peptide_design \
    -i Demo/sample_data/complexes.json -o out/ \
    -n 10000 --temperature 1.25 -d cuda:0
```

```json
[
  {"name": "AS8.4", "HLA_a": "GSHSMRY...", "HLA_b": "MIQRTPK...",
   "TCR_a": "METLLGL...", "TCR_b": "MGFRLLC...", "length": 9},
  {"name": "Aga1", "HLA_a": "...", "HLA_b": "...", "TCR_a": "...", "TCR_b": "...",
   "length": 11, "temperature": 1.5}
]
```

Any entry may override `name`, `length`, `n`, `temperature`, `seed`, `profile_method`,
`gibbs_k`, `gibbs_rounds`, `gibbs_temperature` and `gibbs_max_forwards`, so one file can mix
peptide lengths, temperatures and pipelines. Everything else comes from the flags. An unknown
field is an error rather than a silent fallback to the default, so a typo in an override cannot
quietly leave the run on its default.

Four artifacts are written into the output directory:

| file | contents |
|---|---|
| `designs.csv` | `name`, `rank`, `sequence`, `method`, `phase`, `step`, and `pll` unless `--no-rescore`. Best first when rescored, draw order otherwise |
| `profiles.csv` | `name`, `position`, the 20 residue columns, `entropy`, and `commit_order` when `--profile-method iegr` |
| `manifest.json` | settings, and per complex the consensus, mean entropy and sampling statistics |
| `logos/<name>.png` | one sequence logo per complex |

Use `--profile-only` to skip sampling, `--no-logo` to skip the figures, `--no-rescore` to skip the
PLL pass, and `--method iegr` for the end-to-end decoder. `--profile-method iegr` switches the
profile to entropy-guided, and `--gibbs-k K` with `--gibbs-temperature` and `--gibbs-max-forwards`
adds block Gibbs refinement, which runs until N distinct designs are collected. Each of these is
also a per-entry override. A complex that cannot be designed for is named with a reason in the
manifest and the run continues, exiting non-zero at the end.
