import torch
import esm
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE

def average_product_correction(attn_map):
    row_mean = attn_map.mean(dim=-1, keepdim=True)
    col_mean = attn_map.mean(dim=-2, keepdim=True)
    total_mean = attn_map.mean()
    return attn_map - (row_mean * col_mean / total_mean)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

model.to(device)

seq = "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"
domains = {
    "D1": (5, 82),
    "D2": (88, 225),
}

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [("protein", seq)]
print(len(seq))
batch_labels, batch_strs, batch_tokens = batch_converter(data)
print(model.layers, "model.layers")

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens.to(device), repr_layers=range(6), return_contacts=True)
print(results.keys())
print(results["logits"].shape) # (batch_size, seq_len, vocab_size)
logits = results["logits"]  # shape: (batch, seq_len, vocab_size)
pred_indices = logits.argmax(dim=-1)  # shape: (batch, seq_len)
decoded_seq = [alphabet.get_tok(toks) for toks in pred_indices[0]]
print("decoded_seq", "".join(decoded_seq))
clean_seq = decoded_seq[1:-1]
print("clean_seq", "".join(clean_seq))

print("--------------------------------")
for i, (c, d) in enumerate(zip(seq, clean_seq)):
    if c != d:
        print(i, c, d)
print("--------------------------------")

all_reprs = results['representations']  # dict: layer_idx -> (batch, seq_len, emb_dim)
n_layers = len(all_reprs)
fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4), squeeze=False)
axes = axes[0]

for li, (layer_name, reprs) in tqdm(enumerate(sorted(all_reprs.items()))):
    resid_embs = reprs[0]  # (seq_len, emb_dim)
    # t-SNE: Per-residue
    tsne = TSNE(n_components=2, random_state=42, init="random")
    # Slicing to use only AAs between BOS and EOS if desired; here keep all for demonstration
    embs_2d = tsne.fit_transform(resid_embs.cpu().numpy())
    ax = axes[li]
    # Assign color to each point based on its domain membership
    scatter_colors = []
    domain_names = list(domains.keys())
    domain_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink", "tab:brown"]
    for idx in range(len(decoded_seq)):
        # residue_position is 1-based in decoded_seq, seq is 0-based
        residue_position = idx
        color = "mediumorchid"  # not in a domain by default
        for dom_name, (dom_start, dom_end) in domains.items():
            if dom_start <= residue_position <= dom_end:
                color = domain_colors[domain_names.index(dom_name) % len(domain_colors)]
                break
        scatter_colors.append(color)
    ax.scatter(embs_2d[:,0], embs_2d[:,1], c=scatter_colors, s=16, alpha=0.7, edgecolors='k', linewidths=0.1)
    for idx, label in enumerate(decoded_seq):
        if idx==0 or idx==len(decoded_seq)-1:
            # Often BOS/EOS are at position 0 and at end
            continue

        # Assign color and domain label for each residue
        residue_position = idx  # 1-based in decoded_seq, seq is 0-based
        color = 'tab:blue'
        domain_label = "Non-domain"
        for dom_name, (dom_start, dom_end) in domains.items():
            if dom_start <= residue_position <= dom_end:
                # assign a unique color for each domain
                domain_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink", "tab:brown"]
                domain_names = list(domains.keys())
                color = domain_colors[domain_names.index(dom_name) % len(domain_colors)]
                domain_label = dom_name
                break

        ax.text(
            embs_2d[idx,0], embs_2d[idx,1],
            f"{idx} {label}",
            fontsize=6,
            alpha=0.7,
            color=color
        )

        # build up a legend of used domains/colors
        if not hasattr(ax, '_domain_legend'):
            ax._domain_legend = {}
        ax._domain_legend[domain_label] = color
    ax.set_title(f"Layer {layer_name} t-SNE")
    ax.axis('off')
plt.tight_layout()
plt.legend()
plt.savefig("tsne_repr_per_layer.png")
plt.close()

# Plot an image of (layers x heads) for each attention matrix
attentions = results["attentions"][0]  # (n_layers, n_heads, L, L)

attention_maps = []
apc_attention_maps = []
for layer_idx, layer_attn in enumerate(attentions):
    for head_idx, attn_map in enumerate(layer_attn):
        attention_maps.append((layer_idx, head_idx, attn_map.cpu().numpy()))
        apc_attention_maps.append((layer_idx, head_idx, average_product_correction(attn_map).cpu().numpy()))

n_layers = attentions.shape[0]
n_heads = attentions.shape[1]
fig, axes = plt.subplots(n_layers, n_heads, figsize=(2*n_heads, 2*n_layers), squeeze=False)
for (layer_idx, head_idx, attn_map) in attention_maps:
    ax = axes[layer_idx, head_idx]
    im = ax.imshow(attn_map, cmap='viridis', origin='upper')
    ax.set_title(f"L{layer_idx+1} H{head_idx+1}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('attention_maps.png')
plt.close()

fig, axes = plt.subplots(n_layers, n_heads, figsize=(2*n_heads, 2*n_layers), squeeze=False)
for (layer_idx, head_idx, attn_map) in apc_attention_maps:
    ax = axes[layer_idx, head_idx]
    im = ax.imshow(attn_map, cmap='viridis', origin='upper')
    ax.set_title(f"L{layer_idx+1} H{head_idx+1} APC")
    ax.axis('off')
plt.tight_layout()
plt.savefig('attention_maps_apc.png')
plt.close()

contacts = results["contacts"].detach().cpu().numpy()[0]  # (L, L)

fig, ax = plt.subplots(figsize=(8, 8))
image = ax.imshow(contacts, cmap='Blues', interpolation='nearest', origin='upper')
for start, end in domains.values():
    for border in (start - 1.5, end - 0.5):
        if -0.5 <= border <= contacts.shape[0] - 0.5:
            ax.hlines(border, xmin=-0.5, xmax=contacts.shape[1] - 0.5,
                      colors='tab:orange', linestyles='--', linewidth=1.5)
fig.colorbar(image, ax=ax, label='Contact Probability')
ax.set_title('Predicted Contacts')
ax.set_xlabel('Residue')
ax.set_ylabel('Residue')
plt.tight_layout()
plt.savefig('predicted_contacts_domains.png')
plt.close()
