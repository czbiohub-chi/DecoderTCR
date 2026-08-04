"""Constants for DecoderTCR V0.3. Shared across both backbones.

Both the ESM2 (DecoderTCR) and ESMC (DecoderTCR-ESMC) backbones use the SAME Meta
ESM-1b alphabet at runtime. This is intentional: the proven masking + `+1` CLS-offset
logic in `DecoderTCR/model/tokenizer.py` runs unchanged for both. The Meta `Alphabet`
is sourced once, from the ESM2 tree (`esm.data`). ESMC's own HuggingFace
tokenizer (under `esmc/tokenization/`) produces byte-identical token IDs for
slots 0-30 and 32. Only slot 31 diverges (`<null_1>` vs `|`) and the model never
emits it, so a single tokenizer source is safe.
"""

from esm.data import Alphabet

# ---------------------------------------------------------------------------
# Tokenizer alphabet (Meta ESM-1b, the single runtime tokenizer for both backbones)
# ---------------------------------------------------------------------------

ALPHABET = Alphabet.from_architecture("ESM-1b")
BATCH_CONVERTER = ALPHABET.get_batch_converter()

CLS_IDX = ALPHABET.cls_idx
EOS_IDX = ALPHABET.eos_idx
PAD_IDX = ALPHABET.padding_idx
MASK_IDX = ALPHABET.mask_idx
SEP_PLACEHOLDER = "."  # Single-char placeholder in sequence string (index 29)
SEP_IDX = ALPHABET.eos_idx  # 2, replaced with <eos> after tokenization
VOCAB_SIZE = len(ALPHABET)

# The 20 standard amino acids, alphabetical, with their token ids. Shared by both backbones.
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA20_IDS = [ALPHABET.tok_to_idx[a] for a in AA20]

# ---------------------------------------------------------------------------
# ESM2 backbone architectures (DecoderTCR)
# ---------------------------------------------------------------------------

ESM2_ARCH = {
    "ESM2_8M":   dict(num_layers=6,  embed_dim=320,  attention_heads=20),
    "ESM2_35M":  dict(num_layers=12, embed_dim=480,  attention_heads=20),
    "ESM2_150M": dict(num_layers=30, embed_dim=640,  attention_heads=20),
    "ESM2_650M": dict(num_layers=33, embed_dim=1280, attention_heads=20),
    "ESM2_3B":   dict(num_layers=36, embed_dim=2560, attention_heads=40),
    "ESM2_15B":  dict(num_layers=48, embed_dim=5120, attention_heads=40),
}

# ---------------------------------------------------------------------------
# ESMC backbone architectures (DecoderTCR-ESMC)
# ---------------------------------------------------------------------------
# Base weights (Chan Zuckerberg Biohub ESMC, github.com/Biohub/esm, fp32):
#   checkpoints/base_models/ESMC_300M.pt  (1.3 GB)
#   checkpoints/base_models/ESMC_600M.pt  (2.2 GB)
#   checkpoints/base_models/ESMC_6B.pt    (24 GB)
# Each loads with strict=True into ESMC(d_model=…, n_heads=…, n_layers=…).

DECODERTCRC_ARCH = {
    "DecoderTCRC_300M": dict(d_model=960,  n_heads=15, n_layers=30),
    "DecoderTCRC_600M": dict(d_model=1152, n_heads=18, n_layers=36),
    "DecoderTCRC_6B":   dict(d_model=2560, n_heads=40, n_layers=80),
}

# ---------------------------------------------------------------------------
# Sequence component order (canonical concatenation order for the full complex)
# ---------------------------------------------------------------------------

SEQUENCE_ORDER = ["HLA_a", "HLA_b", "Peptide", "TCR_a", "TCR_b"]

# Region annotation keys in the JSON pocket_idx field
POCKET_REGIONS = ["HLA_a", "HLA_b"]               # MHC binding pocket regions
CDR_REGIONS = {"TCR_a": "CDRa", "TCR_b": "CDRb"}  # TCR chain → CDR key mapping
