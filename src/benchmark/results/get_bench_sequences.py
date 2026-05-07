import pandas as pd

df = pd.read_csv('chainsaw_model_v3_on_cath1363_test.csv')

print(df.head())
print(df.shape)
print(df.columns)

import requests

def get_uniprot_sequence(uniprot_id, chain=None):
    """
    Fetches the protein sequence from UniProt for a given UniProt ID.
    If a chain is specified, attempts to fetch the sequence for that chain
    using the UniProt API (for PDB cross-references).
    """
    # UniProt API endpoint for FASTA
    url = f"https://www.rcsb.org/fasta/entry/{uniprot_id}"
    # url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch sequence for UniProt ID {uniprot_id}")
    fasta = response.text
    if chain is None:
        # Return the full sequence
        lines = fasta.splitlines()
        seq = "".join(line.strip() for line in lines if not line.startswith(">"))
        return seq
    else:
        # Try to fetch chain-specific sequence via UniProt's PDB cross-references
        # This is a best-effort approach; UniProt does not always provide chain-specific sequences
        # You may need to use SIFTS or RCSB PDB API for precise chain sequences
        # Here, we attempt to parse the FASTA header for chain info
        lines = fasta.splitlines()
        header = lines[0]
        if f"Chain {chain}" in header or f"chain {chain}" in header:
            seq = "".join(line.strip() for line in lines[1:])
            return seq
        else:
            # Fallback: return full sequence with a warning
            print(f"Warning: Chain {chain} not found in UniProt FASTA header. Returning full sequence.")
            seq = "".join(line.strip() for line in lines[1:])
            return seq

# Example usage:
# seq = get_uniprot_sequence("4w7s".upper())  # Hemoglobin subunit alpha
# seq_chain = get_uniprot_sequence("4w7s".upper(), chain="A")
# print(seq)
# print(seq_chain)

# exit()
new_df = pd.DataFrame()
new_df['uniprot_id'] = df['chain_id'].apply(lambda x: x[:-1])
new_df['chain'] = df['chain_id'].apply(lambda x: x[-1])
new_df['sequence'] = None

for index, row in new_df.iterrows():
    try:
        seq = get_uniprot_sequence(row['uniprot_id'], row['chain'])
    except Exception as e:
        print(f"Error processing {row['uniprot_id']} {row['chain']}: {e}")
        seq = None
    new_df.loc[index, 'sequence'] = seq
    print(f"Processed {index+1}/{len(df)}")

print(new_df.head())

new_df.to_csv('benchmark_sequences.csv', index=False)