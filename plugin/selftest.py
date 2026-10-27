#!/usr/bin/env python3
"""Exercise backend.allocate_shared_table against the REAL prepared artifact.

The spike proved the mechanism on a 400 MB slice. This proves it at the served
size — 51,200,245,760 B — through the exact function the plugin installs, and it
checks something the slice could not: that the preparer laid the **128 shards out
shard-major** (shard k -> rows [k*2,500,012, ...)) by sampling rows drawn from
every shard and comparing against the original checkpoint files.

Run inside the served image with the shared table on tmpfs. `--manifest` is
required (it is what the layout and content are checked against), and the image
must provide `torch` and `safetensors` — this file imports them before it parses
arguments, so a missing one fails on the import, not on usage:
    docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<uuid> \
      -v /mnt/fast:/mnt/fast -v /dev/shm/ple-shared-spike:/shmtab:rw \
      -v /var/tmp/ple-shared-ram:/tools:ro --entrypoint python3 <image> \
      /tools/selftest.py --path /shmtab/layer-0.bin --manifest /ple/ssd-stream.json \
                         --checkpoint /model --verify full
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from sglang_shared_ram_ple import backend, config  # noqa: E402

ROWS, COLS = 320_001_536, 160
SHARD_ROWS = 2_500_012
SHARD_RE = re.compile(r"ngram_embedding\.shard_(\d+)\.weight$")


def shard_map(checkpoint_dir: str):
    """flat row -> (file, byte offset) using the LIVE checkpoint, as an
    independent authority on the shard-major layout."""
    index = json.load(open(os.path.join(checkpoint_dir, "model.safetensors.index.json")))
    files = {}
    for key, fn in index["weight_map"].items():
        m = SHARD_RE.search(key)
        if not m:
            continue
        shard = int(m.group(1))
        path = os.path.join(checkpoint_dir, fn)
        with open(path, "rb") as fh:
            import struct
            hlen = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(hlen).decode("utf-8"))
        ent = head[key]
        base = 8 + hlen + ent["data_offsets"][0]
        assert ent["dtype"] == "F8_E4M3" and list(ent["shape"]) == [SHARD_ROWS, COLS]
        files[shard] = (path, base)
    missing = [s for s in range(128) if s not in files]
    if missing:
        raise SystemExit("index lacks shards %s" % missing[:8])
    return files


def struct_unpack(fh):
    import struct
    raw = fh.read(8)
    return struct.unpack("<Q", raw)[0]


def read_source_row(files, row: int) -> bytes:
    shard, local = divmod(row, SHARD_ROWS)
    path, base = files[shard]
    with open(path, "rb") as fh:
        fh.seek(base + local * COLS)
        blob = fh.read(COLS)
    if len(blob) != COLS:
        raise SystemExit("short read shard=%d local=%d" % (shard, local))
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", default="/mnt/fast/models/qwen38-flash-next-nvfp4-mtpfix")
    ap.add_argument("--verify", default="full")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--outfile", default="")
    a = ap.parse_args()

    torch.cuda.set_device(0)
    dev = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({"event": "pre", "gpu_free_gib": round(free / 2**30, 2),
                      "gpu_total_gib": round(total / 2**30, 2),
                      "memavail_gib": round(int(open("/proc/meminfo").read()
                      .split("MemAvailable:")[1].split()[0]) / 2**20, 1)}), flush=True)

    t0 = time.time()
    table = backend.allocate_shared_table(
        (ROWS, COLS), torch.float8_e4m3fn,
        path=a.path, manifest=a.manifest, verify=a.verify,
        log=lambda m: print("[backend] %s" % m, flush=True))
    alloc_s = time.time() - t0
    diag = backend.diagnostics()
    print("SELFTEST|allocate|" + json.dumps({**diag, "seconds": round(alloc_s, 2),
                                            "dtype": str(table.dtype),
                                            "shape": list(table.shape)}, sort_keys=True),
          flush=True)

    # layout proof across ALL 128 shards, against the live checkpoint
    files = shard_map(a.checkpoint)
    u8 = table.view(torch.uint8)
    bad = checked = 0
    for shard in range(128):
        for local in (0, SHARD_ROWS // 2, SHARD_ROWS - 1)[: a.samples]:
            row = shard * SHARD_ROWS + local
            want = torch.frombuffer(read_source_row(files, row), dtype=torch.uint8)
            got = u8[row].cpu()
            checked += 1
            if not torch.equal(want, got):
                bad += 1
                if bad < 4:
                    print("SELFTEST|MISMATCH|shard=%d local=%d" % (shard, local), flush=True)
    # The loop's size is driven by --samples, so `--samples 0` checked nothing and the
    # layout proof still exited 0. A vacuous pass is worse than a skip: it reads as a
    # certification. Require every shard to have been sampled at least once.
    if checked < 128 * max(1, a.samples):
        raise SystemExit("SELFTEST|FAIL|layout coverage too low: checked=%d over 128 shards "
                         "(--samples=%d); the layout proof is vacuous" % (checked, a.samples))
    print("SELFTEST|layout|" + json.dumps({"checked": checked, "mismatch": bad,
                                           "shards_covered": 128}), flush=True)

    # device reachability of the pinned pages: DMA out of the registered buffer.
    # (ids must stay on CPU to index the CPU-side shared mapping; the .to() is what
    # crosses the link.)
    idx_cpu = torch.tensor([0, SHARD_ROWS, ROWS - 1, ROWS // 2], dtype=torch.int64)
    t1 = time.time()
    host_rows = u8[idx_cpu]
    ondev = host_rows.to(dev)
    torch.cuda.synchronize()
    ok_gpu = bool(torch.equal(ondev.cpu(), host_rows))
    print("SELFTEST|device_read|" + json.dumps({"rows": idx_cpu.tolist(), "match": ok_gpu,
                                                "bytes": int(host_rows.numel()),
                                                "seconds": round(time.time() - t1, 4)}), flush=True)

    # the gather kernel's own access path: a flat fp8 load through the host pointer
    print("SELFTEST|vmstat|" + json.dumps(backend._vmstat_pin()), flush=True)
    rec = {"tag": "SELFTEST_FULL", "path": a.path, "verify": a.verify,
           "alloc_seconds": round(alloc_s, 2), "diag": diag,
           "layout_checked": checked, "layout_mismatch": bad, "device_read_ok": ok_gpu}
    print("SELFTEST|" + json.dumps(rec, sort_keys=True), flush=True)
    if a.outfile:
        with open(a.outfile, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    if bad or not ok_gpu:
        sys.exit(6)


if __name__ == "__main__":
    main()
