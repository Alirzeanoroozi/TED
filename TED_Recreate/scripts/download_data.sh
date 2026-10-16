#!/bin/bash
# Download the published data the lookups are built from into data/ (about 80 GB; allow a few hours).
# Run from the repository root after setup.sh. Files that are already there are skipped.
# Afterwards build the lookups: sbatch scripts/build_lookup_dbs.sbatch
set -eu

get() {     # get URL FILE, with a few tries because figshare turns some requests away
    mkdir -p "$(dirname "$2")"
    [ -s "$2" ] && return
    for try in 1 2 3 4 5; do
        curl -fsSL -o "$2.part" "$1" && mv "$2.part" "$2" && return
        sleep 10
    done
    echo "could not download $1" >&2
    exit 1
}

# TED's tables (Zenodo record 13908086): every domain with its label, the sequence clusters, the TED-redundant ids
for f in ted_365m.domain_summary.cath.globularity.taxid.tsv.gz ted_324m_seq_clustering.cathlabels.tsv.gz \
         ted_redundant_40m_domain_id.list.gz; do
    get https://zenodo.org/api/records/13908086/files/$f/content data/ted/$f
done

# The sequences of all 365M TED domains. TED does not publish them; Foldseek's TED database holds them as one
# 50 GB member of a much larger archive, so only that member is unpacked and the download stops after it.
if [ ! -s data/foldseek_teddb/teddb ]; then
    mkdir -p data/foldseek_teddb
    curl -fsSL https://foldseek.steineggerlab.workers.dev/teddb.tar.gz |
        tar -xz --occurrence=1 --strip-components=1 -C data/foldseek_teddb teddb/teddb
fi

# CATH 4.3: labels, names, and the S40 non-redundant domains as a Foldseek database
CATH=https://download.cathdb.info/cath/releases/all-releases/v4_3_0
get $CATH/cath-classification-data/cath-domain-list-v4_3_0.txt data/cath/cath-domain-list-v4_3_0.txt
get $CATH/cath-classification-data/cath-names-v4_3_0.txt data/cath/cath-names-v4_3_0.txt
get $CATH/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.list data/cath/cath-dataset-nonredundant-S40-v4_3_0.list
if [ ! -s data/cath/foldseek_s40/cath43_s40.dbtype ]; then
    mkdir -p data/cath/s40_pdb data/cath/foldseek_s40
    curl -fsSL $CATH/non-redundant-data-sets/cath-dataset-nonredundant-S40-v4_3_0.pdb.tgz | tar -xz -C data/cath/s40_pdb
    tools/bin/foldseek8 createdb data/cath/s40_pdb/dompdb data/cath/foldseek_s40/cath43_s40
fi

# Foldclass embeddings of the CATH 4.3 domains: Merizo-search's CATH database, from figshare
get https://ndownloader.figshare.com/files/50846193 data/merizo_search_cath/cath-4.3-foldclassdb.pt
get https://ndownloader.figshare.com/files/50846196 data/merizo_search_cath/cath-4.3-foldclassdb.index

echo "done: data/ is ready"
