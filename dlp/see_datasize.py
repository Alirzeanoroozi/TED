from data_access import PQDataAccess

def export_pq_to_jsonl(pq_path, jsonl_path, max_rows=1000000, batch_size=128, max_seq_len=500):
    data_access = PQDataAccess(pq_path, batch_size=batch_size)
    number_of_rows = 0
    total_tokens = 0
    while True:
        batch = data_access.get_batch()
        if not batch:
            break
        number_of_rows += len(batch)

        for row in batch:
            row_dict = row.to_dict()
            if row_dict.get("Sequence") is None or row_dict.get("chopping_star") is None or row_dict.get("cath_list") is None or len(row_dict.get("Sequence")) > max_seq_len:
                continue
            total_tokens += len(row_dict.get("Sequence"))

        if number_of_rows > max_rows:
            break
        print(f"number of rows = {number_of_rows}, total_tokens = {total_tokens}")

# Export train set
export_pq_to_jsonl(
    "data/export_pqt_0_ted/corpus_chains_sequence",
    "dlp/jsons/ted_train.json",
    max_rows=1000000,
    batch_size=1024,
    max_seq_len=2048
)