"""Where everything lives, relative to the repository root. The README's "Install" section says how it gets there."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# third-party code and binaries (setup.sh)
TED_CONSENSUS = ROOT / "vendor/ted-tools/ted_consensus_1.0"     # TED's released pipeline, run unmodified
MERIZO_SEARCH = ROOT / "vendor/merizo_search/merizo_search"     # Foldclass network
TMALIGN = MERIZO_SEARCH / "programs/Foldclass/tmalign"
MMSEQS = ROOT / "tools/bin/mmseqs"
MMSEQS_GPU = ROOT / "tools/bin/mmseqs-gpu"
FOLDSEEK = ROOT / "tools/bin/foldseek8"                         # release 8-ef4e960, the closest to TED's commit a435618
VENV = ROOT / ".venv"

# ESMFold weights (downloaded from Hugging Face on first use)
HF_HOME = ROOT / "models/hf"
ESMFOLD_MODEL = "facebook/esmfold_v1"

# CATH 4.3 reference. TED's "SSG5" target list is not public; the S40 set is the closest public one.
CATH_LABELS = ROOT / "db/cath/cath43_labels.tsv"
CATH_NAMES = ROOT / "db/cath/cath43_names.tsv"
CATH_FOLDSEEK_DB = ROOT / "data/cath/foldseek_s40/cath43_s40"
FOLDCLASS_DB = ROOT / "db/cath/foldclass_s40"

# lookups built from TED's published tables (scripts/build_lookup_dbs.sbatch)
TED_MD5_INDEX = ROOT / "db/ted_md5"
TED_MMSEQS_DB = ROOT / "db/mmseqs/ted100"
MMSEQS_SERVER_FILE = ROOT / "db/mmseqs/SERVER"                  # address of a running MMseqs server
