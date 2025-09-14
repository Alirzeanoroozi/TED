import os
os.makedirs("jsons", exist_ok=True)
import json
from data_access import PQDataAccess

def create_label(chopping_star, cath_list):
    assert len(chopping_star.split("*")) == len(cath_list)
    labels = []
    for domain, cath in zip(chopping_star.split("*"), cath_list):
        labels.append(f"{domain} | {cath}")
    return " * ".join(labels)

def export_pq_to_jsonl(pq_path, jsonl_path, max_rows, batch_size, max_seq_len):
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
    data_access = PQDataAccess(pq_path, batch_size=batch_size)
    number_of_rows = 0
    while True:
        batch = data_access.get_batch()
        if not batch:
            break
        number_of_rows += len(batch)
        print(f"number_of_rows = {number_of_rows}")
        with open(jsonl_path, "a") as f:
            for row in batch:
                row_dict = row.to_dict()
                if row_dict.get("Sequence") is None or row_dict.get("chopping_star") is None or row_dict.get("cath_list") is None or len(row_dict.get("Sequence")) > max_seq_len:
                    continue
                output_dict = {
                    "Sequence": row_dict.get("Sequence"),
                    "label": create_label(row_dict.get("chopping_star"), row_dict.get("cath_list"))
                }
                json.dump(output_dict, f)
                f.write("\n")
        if number_of_rows > max_rows:
            break

# Export train set
# export_pq_to_jsonl(
#     "../data/export_pqt_0_ted_new/corpus_chains_2048_unique",
#     "jsons/train.json",
#     max_rows=1000000,
#     batch_size=1024,
#     max_seq_len=2048
# )

# Export validation set
export_pq_to_jsonl(
    "../data/export_pqt_0_ted_new/corpus_chains_2048_unique",
    "jsons/validation.json",
    max_rows=100,
    batch_size=1024,
    max_seq_len=2048
)

