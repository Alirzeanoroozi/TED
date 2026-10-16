#!/bin/bash
# Download the third-party code and programs this repository runs into vendor/ and tools/ (about 1.5 GB).
# Run once from anywhere: bash setup.sh. A part that is already there is left alone.
set -euo pipefail
cd "$(dirname "$0")"

fetch() {   # fetch URL DIR: unpack a .tar.gz into DIR, without the archive's top-level folder
    mkdir -p "$2"
    curl -fsSL "$1" | tar -xz -C "$2" --strip-components=1
}

# TED's released pipeline, which includes Merizo and Chainsaw, plus the Merizo weights its own setup.sh downloads
TED=vendor/ted-tools/ted_consensus_1.0
if [ ! -d vendor/ted-tools ]; then
    fetch https://github.com/psipred/ted-tools/archive/f07e24ef4a19a722f2c3c3001cbd03d477ba3f31.tar.gz vendor/ted-tools
    for part in 0 1 2; do
        curl -fsSL --create-dirs -o $TED/programs/merizo/weights/weights_part_$part.pt \
            https://github.com/psipred/Merizo/raw/main/weights/weights_part_$part.pt
    done
fi

# UniDoc, the third parser, which TED's code expects in programs/unidoc. TED's script runs bin/UniDoc_struct.
# The UniDoc package ships two builds of that program, and TED's published UniDoc results match the 2022 one
# (UniDoc_structure: identical for 97 of 106 proteins, against 38 for the newer UniDoc_struct), so the name is
# pointed at that build. TED's script itself stays as released.
if [ ! -d $TED/programs/unidoc ]; then
    fetch https://yanglab.qd.sdu.edu.cn/UniDoc/download/UniDoc_2023.tgz $TED/programs/unidoc
    cp $TED/scripts/Run_UniDoc_from_scratch_structure_afdb.py $TED/programs/unidoc/
    ln -sf UniDoc_structure $TED/programs/unidoc/bin/UniDoc_struct
fi

# Merizo-search, for its Foldclass network and TM-align
if [ ! -d vendor/merizo_search ]; then
    fetch https://github.com/psipred/merizo_search/archive/c7dbbd1bbfd58a34fdb58a8100da7c3eec54b829.tar.gz vendor/merizo_search
fi

# Foldseek release 8, the closest release to the commit TED used, and MMseqs2 in its CPU and GPU builds
tool() {    # tool NAME URL: unpack a release into tools/src/NAME and link its program as tools/bin/NAME
    [ -e tools/bin/$1 ] && return
    fetch "$2" tools/src/$1
    mkdir -p tools/bin
    ln -s ../src/$1/bin/"$(ls tools/src/$1/bin)" tools/bin/$1
}
MMSEQS=https://mmseqs.com/archive/564f40d8857f4eca4e1dfe100c67c155b1933e70
tool foldseek8 https://github.com/steineggerlab/foldseek/releases/download/8-ef4e960/foldseek-linux-avx2.tar.gz
tool mmseqs $MMSEQS/mmseqs-linux-avx2.tar.gz
tool mmseqs-gpu $MMSEQS/mmseqs-linux-gpu.tar.gz

echo "done: vendor/ and tools/ are ready"
