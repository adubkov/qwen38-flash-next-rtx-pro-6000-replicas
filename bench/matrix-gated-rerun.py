#!/usr/bin/env python3
"""Gated re-run of CONTAMINATED matrix cells (flash-next, serving host).

Attempt-4 of the post-swap matrix ran while llmproxy clients (qwen "dreaming"
sessions) + co-tenant management-agent health traffic shared the GPU: 15/19 cells show
achieved_concurrency != requested. This wrapper re-runs exactly those cells
against a quiet gate and leaves the 4 clean cells untouched.

Gate, per cell:
  pre  : no foreign POST /v1/chat/completions in the last 45 s (health is
         allowed — it costs no GPU). Wait up to --max-wait per cell.
  post : zero foreign chat POSTs inside the cell's real wall window
         (recorded around the harness call, so calibration + cold/warm
         prefill passes are covered too, not just the concurrent phase),
         AND achieved_concurrency == C AND not saturated.
  Fail => cull the row from the journal and retry (<= --retries). After that
  the row is kept but tagged contaminated_accepted=true so the artifact is
  honest about the compromise.

Harness is invoked once per cell (--cs C --ctxs L); it is resumable, so
completed clean cells are never recomputed.
"""
import argparse, json, os, subprocess, sys, time

CT = "flash-next-pair-flash-next-a-1"
OUTDIR = "/var/tmp/phase0/matrix"
JOURNAL = os.path.join(OUTDIR, "matrix.jsonl")
HARNESS = "/var/tmp/phase0/bench-matrix.py"
# Addresses of clients that may legitimately touch the replica besides this harness
# (a proxy, a monitoring agent). Operator-supplied on purpose: with the list empty,
# foreign_chat() below returns 0 for every cell and the whole matrix certifies clean.
FOREIGN_IPS = tuple(os.environ.get("FOREIGN_IPS", "").replace(",", " ").split())
if not FOREIGN_IPS:
    raise SystemExit("FOREIGN_IPS env required (space/comma-separated peer IPs) — the purity gate "
                     "cannot see other clients without it and would report cells clean")

SUSPECT = [(1, 16384), (1, 65536), (1, 131072), (1, 262144), (1, 393216), (1, 520192),
           (2, 1024), (2, 4096), (2, 16384), (2, 65536), (2, 131072), (2, 262144),
           (4, 16384), (4, 65536), (4, 131072)]


def foreign_chat(secs):
    out = subprocess.run(["docker", "logs", "--since", "%ds" % secs, CT],
                         capture_output=True, text=True).stderr
    n = 0
    for line in out.splitlines():
        if any(ip in line for ip in FOREIGN_IPS) and "POST /v1/chat/completions" in line:
            n += 1
    return n


def load_rows():
    rows = {}
    if os.path.exists(JOURNAL):
        for l in open(JOURNAL):
            try:
                r = json.loads(l)
                rows[(r["C"], r["ctx_target"])] = r
            except Exception:
                pass
    return rows


def cull(cell):
    rows = load_rows()
    rows.pop(cell, None)
    tmp = JOURNAL + ".tmp"
    with open(tmp, "w") as fh:
        for r in rows.values():
            fh.write(json.dumps(r) + "\n")
    os.replace(tmp, JOURNAL)


def wait_quiet(need_s, max_wait):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if foreign_chat(45) == 0:
            return True
        print("  gate: foreign chat active, waiting", flush=True)
        time.sleep(15)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-wait", type=int, default=1800)
    a = ap.parse_args()

    # backup journal once
    if not os.path.exists(JOURNAL):
        print("no journal at %s — nothing to gate" % JOURNAL, flush=True)
        sys.exit(1)
    os.system("cp -n %s /var/tmp/phase0/matrix/matrix.gated-pre.jsonl" % JOURNAL)
    # start from the clean 4: drop all suspect rows so the harness re-executes them
    # (its resume design would otherwise skip the stale contaminated rows).
    for cell in SUSPECT:
        cull(cell)
    print("journal primed, remaining cells=%s" % sorted(load_rows().keys()), flush=True)

    for i, cell in enumerate(SUSPECT, 1):
        C, L = cell
        for attempt in range(1, a.retries + 2):
            if not wait_quiet(45, a.max_wait):
                print("[%d/%d] C=%d L=%d gate TIMEOUT (%ds) — running anyway, will flag"
                      % (i, len(SUSPECT), C, L, a.max_wait), flush=True)
            wall0 = time.time()
            r = subprocess.run([sys.executable, HARNESS, "--cs", str(C), "--ctxs", str(L)],
                               capture_output=True, text=True)
            wall = max(10, int(time.time() - wall0) + 5)
            row = load_rows().get(cell)
            if r.returncode != 0 or row is None:
                print("[%d/%d] C=%d L=%d harness rc=%s row=%s — aborting cell"
                      % (i, len(SUSPECT), C, L, r.returncode, row is None), flush=True)
                print(r.stdout[-500:], r.stderr[-500:], flush=True)
                break
            fcount = foreign_chat(wall)
            achv = row.get("achieved_concurrency")
            pure = (fcount == 0 and achv == C and not row.get("saturated"))
            print("[%d/%d] C=%d L=%-7d attempt=%d pure=%s foreign_in_window=%d achv=%s dec=%s accept=%s"
                  % (i, len(SUSPECT), C, L, attempt, pure, fcount, achv,
                     row.get("decode_per_stream_med_tps"), row.get("accept_len_mean")), flush=True)
            if pure or attempt == a.retries + 1:
                if not pure:
                    row["contaminated_accepted"] = True
                    rows = load_rows()
                    with open(JOURNAL + ".tmp", "w") as fh:
                        for rr in rows.values():
                            if (rr["C"], rr["ctx_target"]) == cell:
                                rr = row
                            fh.write(json.dumps(rr) + "\n")
                    os.replace(JOURNAL + ".tmp", JOURNAL)
                break
            cull(cell)
        time.sleep(2)

    # CSV rebuilt here (never via the harness — a harness call would EXECUTE
    # any still-missing cell ungated).
    import csv as _csv
    rows = load_rows()
    with open(os.path.join(OUTDIR, "matrix.csv"), "w", newline="") as fh:
        keys = ["C", "ctx_target", "prompt_tokens", "prefill_cold_tps", "cached_prefix_tps",
                "cold_ttft_s", "ttft_worst_s", "decode_per_stream_med_tps",
                "decode_per_stream_min_tps", "decode_agg_tps", "decode_window_s",
                "accept_len_mean", "achieved_concurrency", "saturated", "cell_s",
                "contaminated_accepted"]
        w = _csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows.values(), key=lambda x: (x["C"], x["ctx_target"])):
            w.writerow(r)
    missing = [c for c in SUSPECT if c not in rows]
    print("MATRIX2-DONE missing=%s" % missing, flush=True)


if __name__ == "__main__":
    main()
