import os
import tempfile
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import biotite.structure.io as bsio
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
inputs = tokenizer(["MLKNVQVQLV"], return_tensors="pt", add_special_tokens=False).to(device)
with torch.no_grad():
    outputs = model(**inputs)

# Convert to PDB
final_atom_positions = atom14_to_atom37(outputs.positions[-1], outputs)
outputs_np = {k: v.cpu().numpy() for k, v in outputs.items()}
final_atom_positions = final_atom_positions.cpu().numpy()
atom37_mask = outputs_np["atom37_atom_exists"]

pred = OFProtein(
    aatype=outputs_np["aatype"][0],
    atom_positions=final_atom_positions[0],
    atom_mask=atom37_mask[0],
    residue_index=outputs_np["residue_index"][0] + 1,
    b_factors=outputs_np["plddt"][0],
    chain_index=outputs_np.get("chain_index", [None])[0],
)
pdb_string = to_pdb(pred)

# Load PDB and save as CIF
with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
    f.write(pdb_string)
    temp_path = f.name
struct = bsio.load_structure(temp_path)
os.unlink(temp_path)
bsio.save_structure("result.cif", struct)
print("Saved result.cif")
