#!/bin/bash
#SBATCH -J ted_build_dbs
#SBATCH -A kutem
#SBATCH --qos=kutem
#SBATCH -p kutem_gpu
#SBATCH -c 24
#SBATCH --mem=220G
#SBATCH -t 2-00:00:00
#SBATCH -o %x-%j.out
# Build everything cath_label looks things up in (CPU only, about 2 hours). Submit from the repository root.
#   db/cath/               CATH 4.3 labels and names, Foldclass embeddings of the S40 domains
#   db/ted_md5/            md5 of every TED domain sequence -> TED's label (365M domains)
#   db/mmseqs/ted100       MMseqs2 DB of the 120.7M representatives of TED's sequence clusters, labels in the headers
#   db/mmseqs/ted100.idx   its k-mer index, for the CPU server
#   db/mmseqs/ted100_pad   its padded copy, for MMseqs2-GPU
# Each step is skipped if its output exists. Inputs: the files listed under "Data" in the README.
set -euo pipefail
PY=.venv/bin/python
export TMPDIR=$PWD/db/tmp SORT_MEM=60G
mkdir -p "$TMPDIR"

test -s db/cath/foldclass_s40.pt || $PY scripts/build_cath_reference.py
test -s db/ted_md5/labels.json || $PY scripts/build_ted_md5_index.py
$PY scripts/build_ted_mmseqs_db.py
test -s db/mmseqs/ted100_pad.dbtype || tools/bin/mmseqs-gpu makepaddedseqdb db/mmseqs/ted100 db/mmseqs/ted100_pad --threads 24
test -s db/mmseqs/ted100.idx.dbtype || tools/bin/mmseqs createindex db/mmseqs/ted100 "$TMPDIR" --threads 24 --split-memory-limit 150G
