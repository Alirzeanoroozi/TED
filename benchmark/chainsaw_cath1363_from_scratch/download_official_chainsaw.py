#!/usr/bin/env python3
"""Download the official CHAINSAW CATH1363 benchmark files from GitHub."""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "data" / "raw"
DEFAULT_REF = "main"
REPO_RAW_BASE = "https://raw.githubusercontent.com/JudeWells/chainsaw/{ref}/data_and_benchmarking"

FILES = {
    "README.md": "README.md",
    "chainsaw_cath1363_train_test_splits.json": "chainsaw_cath1363_train_test_splits.json",
    "chainsaw_model_v3_on_cath1363_test.csv": "chainsaw_model_v3_on_cath1363_test.csv",
}


def fetch_bytes(url, timeout, retries):
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "TEDPred CHAINSAW benchmark setup"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(5.0, 0.5 * (attempt + 1)))
    raise RuntimeError("failed to download {}: {}".format(url, last_error))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ref", default=DEFAULT_REF, help="GitHub branch/tag/commit to download")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "downloaded_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_repo": "https://github.com/JudeWells/chainsaw",
        "source_ref": args.ref,
        "files": [],
    }

    base = REPO_RAW_BASE.format(ref=args.ref)
    for output_name, source_name in FILES.items():
        url = "{}/{}".format(base, source_name)
        out_path = args.out_dir / output_name
        if out_path.exists() and not args.force:
            data = out_path.read_bytes()
            status = "already_exists"
        else:
            data = fetch_bytes(url, args.timeout, args.retries)
            out_path.write_bytes(data)
            status = "downloaded"

        manifest["files"].append(
            {
                "name": output_name,
                "url": url,
                "path": str(out_path),
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "status": status,
            }
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
