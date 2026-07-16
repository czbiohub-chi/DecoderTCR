# V0.3 benchmark results

V0.3 results across four TCR-pMHC evaluations: **TCRvdb**, **IMMREP23**, **Viral** (ePytope-TCR), and
**PRP** (Peptide Recognition Profiling, HLA-B\*27:05). Models score a pair by masked-peptide PLL
(higher means more binder-like), and metrics are per epitope or clone, macro-averaged, using the
released V0.3 checkpoints. Figures are rendered by `eval/scripts/{plot_macro_comparisons,plot_prp}.py`
in the research project (shown in the main README), and each `.png` has a `.pdf` next to it.

## Summary

Macro AUROC across the three balanced benchmarks (TCRvdb, IMMREP23, Viral), with an ESMC scaling panel
(300M, 600M, 6B). PRP is excluded, because at ~0.4% prevalence AUROC is uninformative. PRP is reported
separately by macro AUPRC and recall@K below. Figure: [summary](figures/summary_dotplot_scaling.png).

## TCRvdb

Functionally validated TCR-pMHC data ([Messemaker et al., bioRxiv 2025](https://doi.org/10.1101/2025.04.28.651095),
[repo](https://github.com/schumacherlab/TCRvdb)). Two epitopes (YLQ, GLC) are scored label-free by
masked-peptide PLL and read out as per-epitope AUROC. These TCRs are seen in training (VDJdb), so this
is a recognition check rather than a generalization test, and larger models do not help.

**per-epitope AUROC** (YLQ and GLC): DecoderTCR-ESMC 300M 0.853, 600M 0.819, 6B 0.779. Figure:
[tcrvdb](figures/tcrvdb_roc.png).

## IMMREP23

Per-epitope AUROC over near-novel TCRs (~0.3% train overlap), from the IMMREP23 challenge
([Nielsen et al., *ImmunoInformatics* 2024](https://doi.org/10.1016/j.immuno.2024.100045)).

**macro AUROC** (20 epitopes): DecoderTCR-ESMC 600M 0.698, 6B 0.687, 300M 0.659. Figure:
[immrep23](figures/immrep23_macro_comparison_with_esmc.png).

### Score IMMREP23 from the raw CSV

`immrep23/immrep23_test.csv` is a clean copy of the IMMREP23 challenge test set (3484 pairs, 20
epitopes), taken unmodified from the official repository
[justin-barton/IMMREP23](https://github.com/justin-barton/IMMREP23) (`data/solutions.csv`, MIT). Score
it to a per-pair score CSV with the CLI. This writes a `pll_DecoderTCR-ESMC_600M` column.

```bash
python -m DecoderTCR.utils.predict_from_genes \
    -i results/immrep23/immrep23_test.csv \
    -o immrep23_scored.csv \
    -m DecoderTCR-ESMC_600M -d cuda
```

The other model sizes run the same way with `-m DecoderTCR-ESMC_300M` or `-m DecoderTCR-ESMC_6B`. Model
weights are fetched by `scripts/download_weights.py` (see the main README). Pass `-d cpu` to run without
a GPU. Cite Nielsen et al., ImmunoInformatics 2024 (https://doi.org/10.1016/j.immuno.2024.100045).

## Viral (ePytope-TCR)

From the ePytope-TCR benchmark ([Drost et al., *Cell Genomics* 2025](https://www.cell.com/cell-genomics/fulltext/S2666-979X(25)00202-2)),
an external comparison of about twenty published tools on viral-epitope TCR specificity. Metric is
per-method macro-average AUROC over the viral epitopes, with missing predictions scored at 0.5.

**macro AUROC** (14 epitopes): DecoderTCR-ESMC 600M 0.657, 6B 0.644, 300M 0.640. Figure:
[viral](figures/viral_macro_comparison_with_esmc.png).

## PRP (Peptide Recognition Profiling)

From [Deep peptide recognition profiling decodes TCR specificity and enables
disease-associated antigen discovery](https://www.nature.com/articles/s41587-026-03128-x)
(*Nature Biotechnology*, 2026): 16 HLA-B\*27:05-restricted TCR clones screened against an anchor-fixed
peptide library. A retrieval task at ~0.4% prevalence, so AUROC is uninformative and the reported
scalar is **macro AUPRC** (per-clone average precision, chance ~0.004).

- **macro AUPRC** ([prp_macro_auprc](figures/prp_macro_auprc.png)): DecoderTCR-ESMC 6B 0.391,
  600M 0.351, 300M 0.303. V0.1, the untrained backbones, and all third-party tools sit ≤0.02.
- **recall@K**, the fraction of a TCR's true binders in its top-K, mean over 16 clones
  ([prp_topk_recall](figures/prp_topk_recall.png)): 6B recall@500 ≈ 0.62, recall@100 ≈ 0.35
  (random ~0.01 and 0.003), 600M lower, 300M lower still. V0.1, the untrained backbones, and the
  third-party tools track the random line.
- **per-clone recall@100** ([prp_per_clone_recall](figures/prp_per_clone_recall.png)): the 16
  clones by method, seen • and held-out ○.
