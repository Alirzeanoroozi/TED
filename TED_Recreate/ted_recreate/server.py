"""The "MMseqs server": keeps the TED lookup DB in memory so a search takes seconds instead of minutes.

MMseqs2 has no query daemon for CPU search. The usual setup, as in the ColabFold MSA server, is a precomputed index
that stays in memory between searches:
  CPU   `mmseqs touchdb` loads the index into the page cache; searches with --db-load-mode 2 then map it instead of
        reading ~100 GB from disk each time.
  GPU   `mmseqs gpuserver` holds the padded DB in GPU memory; searches with --gpu-server 1 hand their queries to it.
Either way the memory belongs to one node, so the searches must run on that node. This module is a small HTTP
service that runs there (scripts/mmseqs_server.sbatch) and does the searches for whoever asks:

    POST /search  {"sequences": {"name": "SEQ", ...}}  ->  {"hits": {"name": [hit, ...]}}
    GET  /health

It writes its address to db/mmseqs/SERVER, where ted_recreate.classify finds it.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config
from .classify import mmseqs_search


class Handler(BaseHTTPRequestHandler):
    gpu = False
    threads = 8
    slots = threading.Semaphore(2)      # searches allowed to run at once

    def reply(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"ok": True, "mode": "gpu" if self.gpu else "cpu"})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/search":
            return self.reply(404, {"error": "not found"})
        try:
            seqs = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["sequences"]
            with self.slots, tempfile.TemporaryDirectory(prefix="ted_search_") as work:
                start = time.time()
                hits = mmseqs_search(seqs, Path(work), self.threads, gpu=self.gpu, server=True)
            self.reply(200, {"hits": hits, "seconds": round(time.time() - start, 2)})
        except Exception as e:      # tell the client what went wrong instead of dropping the connection
            self.reply(500, {"error": repr(e)[:2000]})

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()
    Handler.gpu, Handler.threads = args.mode == "gpu", args.threads

    gpuserver = None
    if Handler.gpu:
        print("starting mmseqs gpuserver ...", flush=True)
        gpuserver = subprocess.Popen([str(config.MMSEQS_GPU), "gpuserver", f"{config.TED_MMSEQS_DB}_pad",
                                      "--max-seqs", "300", "--db-load-mode", "2"])
        time.sleep(30)              # it needs a moment to load the DB before it accepts queries
    else:
        print("loading the index into the page cache (mmseqs touchdb) ...", flush=True)
        subprocess.run([str(config.MMSEQS), "touchdb", str(config.TED_MMSEQS_DB), "--threads", str(args.threads)],
                       check=True)        # given the DB, touchdb loads its index

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://{socket.getfqdn()}:{args.port}"
    config.MMSEQS_SERVER_FILE.write_text(json.dumps({"url": url, "mode": args.mode,
                                                     "slurm_job": os.environ.get("SLURM_JOB_ID")}))
    print(f"TED MMseqs server ({args.mode}) listening at {url}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))       # Slurm stops jobs with SIGTERM: leave through `finally`
    try:
        httpd.serve_forever()
    finally:
        file = config.MMSEQS_SERVER_FILE
        if file.exists() and json.loads(file.read_text())["url"] == url:    # unless a newer server took over
            file.unlink()
        if gpuserver:
            gpuserver.terminate()


if __name__ == "__main__":
    main()
