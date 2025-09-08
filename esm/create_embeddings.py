from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import json
import torch
from tqdm import tqdm
import os
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def get_domains(sequence, spans):
    domains = {}
    caths = []
    for i, span in enumerate(spans.split("*")):
        domain  = span.strip().split("|")[0]
        caths.append(span.strip().split("|")[1].strip())
        split_domains = domain.split("_")
        for split_domain in split_domains:
            start, end = split_domain.split("-")
            if i in domains:
                domains[i].append([int(start),int(end)])
            else:
                domains[i] = [[int(start),int(end)]]
    # print(domains, caths)
    # print(len(domains), len(caths))
    seq_domains = []
    for i, domain_list in domains.items():
        seq_i = sequence[domain_list[0][0]:domain_list[0][1]]
        last_end = domain_list[0][1]
        if len(domain_list) > 1:
            for domain in domain_list[1:]:
                seq_i += "<unk>" * (domain[0] - last_end)
                seq_i += sequence[domain[0]:domain[1]]
                last_end = domain[1]
        seq_domains.append((seq_i, caths[i]))
    # print(seq_domains)
    return seq_domains

os.makedirs("esm/embeddings", exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

DATA_PATH = "dlp/jsons/ted_validation.json"      # JSONL with {"sequence": "...", "spans": "11-12_34-34"}
data = {
    "sequence": [],
    "spans": [],
}
with open(DATA_PATH, "r") as f:
    for i, line in enumerate(f):
        if i < 150:
            continue
        line_dict = json.loads(line.strip())
        data["sequence"].append(line_dict["Sequence"])
        data["spans"].append(line_dict["label"])

        
# create a dictionary of domains and their embeddings
all_domains = {}
for i, sequence in tqdm(enumerate(data["sequence"])):
    domains = get_domains(sequence, data["spans"][i])
    for j, (domain, cath) in enumerate(domains):
        if cath != "-":
            all_domains[str(i)+"_"+str(j)] = (domain, cath)
            print(str(i)+"_"+str(j), domain, cath)
exit()
# client = ESMC.from_pretrained("esmc_300m").to(device) # or "cpu"

# all_embeddings = {}
# for name, (domain, cath) in tqdm(all_domains.items()):
#     protein = ESMProtein(sequence=domain)
#     protein_tensor = client.encode(protein)
#     logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))

#     embedding_to_save = logits_output.embeddings[0].detach().cpu().numpy().mean(axis=0)
#     all_embeddings[name] = embedding_to_save

# Save embeddings to file
# np.save(f"esm/embeddings/domain_embeddings.npy", list(all_embeddings.values()))
# np.save(f"esm/embeddings/domain_names.npy", list(all_embeddings.keys()))
# np.save(f"esm/embeddings/cath_classes.npy", [v[1] for v in list(all_domains.values())])

# load embeddings
all_embeddings = np.load("esm/embeddings/domain_embeddings.npy", allow_pickle=True)
all_domains = np.load("esm/embeddings/domain_names.npy", allow_pickle=True)


print(f"Generated embeddings for {len(all_embeddings)} domains")
print(f"Embedding dimension: {all_embeddings[0].shape[0]}")
print(f"Unique CATH classes: {len(set([v[1].split('.')[0] for v in all_domains]))}")

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(np.array(all_embeddings))

# Plot the t-SNE results
plt.figure(figsize=(10, 8))
# Create color mapping for CATH classes
cath_classes = [v[1].split(".")[0] for v in all_domains]
unique_classes = list(set(cath_classes))
color_map = {cls: i for i, cls in enumerate(unique_classes)}
colors = [color_map[cls] for cls in cath_classes]

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, c=colors, cmap='tab10', alpha=0.7)
plt.title("t-SNE of ESM Embeddings by CATH Class")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()

# Add legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=plt.cm.tab10(i/len(unique_classes)), 
                              markersize=8, label=cls) 
                  for i, cls in enumerate(unique_classes)]
plt.legend(handles=legend_elements, title="CATH Class", loc='best')
plt.savefig("esm/embeddings/tsne_plot.png")
plt.close()
