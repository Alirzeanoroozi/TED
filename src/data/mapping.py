import requests
import time
import csv
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

GOOGLE_DRIVE_MOUNT_POINT = Path("/scratch/erkmenerken22/google-drive")
DEFAULT_OUTPUT_DIR = Path("/scratch/erkmenerken22/TED_sequences")

def _resolve_output_dir(
    preferred_mount=GOOGLE_DRIVE_MOUNT_POINT,
    fallback_dir=DEFAULT_OUTPUT_DIR
):
    preferred_mount = Path(preferred_mount)
    fallback_dir = Path(fallback_dir)

    # only use "google-drive" if it is actually mounted
    if os.path.ismount(str(preferred_mount)):
        return preferred_mount / "TED_sequences"

    return fallback_dir


def uniprotkb_to_uniparc(uniprot_id):
    """
    Map a UniProtKB accession to UniParc using the UniProt ID mapping API.
    Returns a JSON response from the API.
    """
    url = "https://rest.uniprot.org/idmapping/run"
    data = {
        "from": "UniProtKB_AC-ID",
        "to": "UniParc",
        "ids": uniprot_id
    }
    response = requests.post(url, data=data)
    if response.ok:
        return response.json()
    else:
        print(f"Error: {response.status_code} {response.text}")
        return None

def get_uniparc_results(job_json):
    """
    Given a JSON response from the idmapping/run endpoint,
    poll the UniProt API for results of the mapping job and return them as JSON.
    """
    if not job_json or "jobId" not in job_json:
        print("No jobId found in response")
        return None

    job_id = job_json["jobId"]
    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    result_url = f"https://rest.uniprot.org/idmapping/uniparc/results/{job_id}"

    # Poll for job completion
    while True:
        status_response = requests.get(status_url)
        if not status_response.ok:
            print(f"Error checking status: {status_response.status_code} {status_response.text}")
            return None
        status = status_response.json()
        if status.get("jobStatus") == "RUNNING":
            time.sleep(1)
        elif status.get("jobStatus") == "FINISHED":
            break
        elif status.get("jobStatus") == "FAILED":
            print("Job failed.")
            return None
        else:
            # Handle unknown status or already completed jobs
            break

    # Retrieve results
    results_response = requests.get(result_url)
    if results_response.ok:
        return results_response.json()
    else:
        print(f"Error retrieving results: {results_response.status_code} {results_response.text}")
        return None


def _extract_uniprot_ids(csv_file):
    """
    Read UniProt IDs from a CSV file.
    Expected column: uniprot_id (case-insensitive).
    """
    with open(csv_file, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No headers found in CSV: {csv_file}")

        # Resolve the UniProt column name case-insensitively.
        column_lookup = {name.strip().lower(): name for name in reader.fieldnames}
        if "uniprot_id" not in column_lookup:
            raise ValueError(
                f"'uniprot_id' column not found in CSV headers: {reader.fieldnames}"
            )
        uniprot_column = column_lookup["uniprot_id"]

        ids = []
        for row in reader:
            raw_id = (row.get(uniprot_column) or "").strip()
            if raw_id:
                ids.append(raw_id)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(ids))


def _download_uniprot_fasta(uniprot_id, output_dir, timeout=30):
    """
    Download FASTA for a single UniProt accession and save to <uniprot_id>.fasta.
    Returns (uniprot_id, success, message).
    """
    fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    output_path = output_dir / f"{uniprot_id}.fasta"

    try:
        response = requests.get(fasta_url, timeout=timeout)
        if response.status_code == 200 and response.text.startswith(">"):
            output_path.write_text(response.text, encoding="utf-8")
            return uniprot_id, True, str(output_path)
        return uniprot_id, False, f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        return uniprot_id, False, str(exc)


def download_sequences_from_csv(csv_file, output_dir, max_workers=8):
    """
    Download UniProt FASTA sequences for all IDs in the provided CSV file.

    - csv_file: path to chains.ted.csv (or similar)
    - output_dir: directory where FASTA files are written
    - max_workers: number of parallel workers (default 8)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uniprot_ids = _extract_uniprot_ids(csv_file)
    if not uniprot_ids:
        print("No UniProt IDs found in CSV.")
        return

    print(f"Found {len(uniprot_ids)} unique UniProt IDs.")
    print(f"Downloading sequences with {max_workers} workers...")

    success_count = 0
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(_download_uniprot_fasta, uniprot_id, output_dir): uniprot_id
            for uniprot_id in uniprot_ids
        }
        for future in as_completed(future_to_id):
            uniprot_id, success, message = future.result()
            if success:
                success_count += 1
            else:
                failed.append((uniprot_id, message))

    print(f"Download complete. Success: {success_count}, Failed: {len(failed)}")

    if failed:
        failed_log = output_dir / "failed_downloads.txt"
        with open(failed_log, "w", encoding="utf-8") as handle:
            for uniprot_id, reason in failed:
                handle.write(f"{uniprot_id}\t{reason}\n")
        print(f"Failed IDs logged to: {failed_log}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download UniProt sequences from a CSV file of UniProt IDs."
    )
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).with_name("chains.ted.csv")),
        help="Path to CSV file containing a 'uniprot_id' column.",
    )
    parser.add_argument(
        "--outdir",
        default=str(_resolve_output_dir()),
        help="Output directory where FASTA files will be saved.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    args = parser.parse_args()

    download_sequences_from_csv(args.csv, args.outdir, max_workers=args.workers)
