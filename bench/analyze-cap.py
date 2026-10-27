#!/usr/bin/env python3
"""Join the two capability arms (control 4/24 vs treatment 8/48) and judge them.

Reads the matrix.jsonl journals written by bench-matrix.py and answers exactly
two questions, in the order the deployment cares about them:

  1  ADMISSION -- did raising max_running_requests actually put more streams in
     the same batch? The honest column here is `achieved_concurrency` /
     `engine_running_max`, not throughput: throughput is confounded by the
     card's power cap (see below) and by the harness's own short generation
     window at long context.
  2  PRICE -- what did the extra state cost, read from the boot log rather than
     estimated: `KV Cache is allocated. #tokens:` is the capacity that
     long-context concurrency is made of, and C x (L + gen) must fit inside it.

Deliberately NOT claimed:
  * decode parity at ctx >= 65k. decode_maxtok() shortens generation to keep a
    cell minutes instead of tens, so the window is dominated by sibling
    PREFILL; `decode_agg_tps` there is not a decode metric. Those cells say
    "nothing catastrophic", not "parity".
  * any regression verdict without a same-config drift bracket. A cell only
    clears as a regression when it is BOTH >3% and >3x the run-to-run drift
    measured from the repeat journal, per this box's measurement rules.
  * anything about clocks from these rows. clocks are read from the
    probe-clock-attribution arm, because concurrency raises the prefill duty
    cycle and prefill is what pins the card at its enforced power limit.
"""
import argparse, json, os, sys

KV_BYTES_PER_TOKEN = 12_288 + 1_024   # target (12 layers x 2 kv x 256 x 1B x2) + 1-layer draft


def load(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "C" in r and "ctx_target" in r:
            rows[(r["C"], r["ctx_target"])] = r
    return rows


def num(v):
    return v if isinstance(v, (int, float)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--trt", required=True)
    ap.add_argument("--ctrl-repeat", default="",
                    help="same-config second capture, for the drift bracket")
    ap.add_argument("--pool-ctrl", type=int, required=True,
                    help="KV pool tokens read from the CONTROL boot log")
    ap.add_argument("--pool-trt", type=int, required=True,
                    help="KV pool tokens read from the TREATMENT boot log")
    a = ap.parse_args()

    ctrl, trt, rpt = load(a.ctrl), load(a.trt), load(a.ctrl_repeat)
    # load() returns {} for a missing file, so this analyzer used to print a fully
    # formatted comparison table with zero rows AND a "from the boot log (not
    # estimated)" PRICE block computed from argparse defaults: publishable-looking
    # capacity numbers from a run that never happened.
    for name, path, got in (("ctrl", a.ctrl, ctrl), ("trt", a.trt, trt)):
        if not os.path.exists(path):
            raise SystemExit("FAIL: %s journal does not exist: %s" % (name, path))
        if not got:
            raise SystemExit("FAIL: %s journal has no usable rows: %s" % (name, path))
    keys = sorted(set(ctrl) | set(trt), key=lambda k: (k[0], k[1]))
    if not keys:
        raise SystemExit("FAIL: no comparable (C, ctx) cells in either journal")

    hdr = "%-3s %-8s %-6s %-6s %-8s | %-7s %-7s %-6s | %-7s %-7s %-6s | %s" % (
        "C", "ctx", "achvC", "achvT", "runT", "dec/strC", "dec/strT", "d%", "aggC", "aggT", "d%", "purity / notes")
    print("admission + throughput, control(max_running_requests=4, mamba=24) vs treatment(8, 48)")
    print(hdr)
    print("-" * len(hdr))
    for (C, L) in keys:
        c, t = ctrl.get((C, L)), trt.get((C, L))
        def g(r, k):
            return num(r.get(k)) if r else None
        ds_c, ds_t = g(c, "decode_per_stream_med_tps"), g(t, "decode_per_stream_med_tps")
        ag_c, ag_t = g(c, "decode_agg_tps"), g(t, "decode_agg_tps")
        dpct = ("%+.1f" % ((ds_t - ds_c) / ds_c * 100)) if (ds_c and ds_t) else "-"
        apct = ("%+.1f" % ((ag_t - ag_c) / ag_c * 100)) if (ag_c and ag_t) else "-"

        notes = []
        # purity: the bench's own streams arrive via the bridge gateway, so read
        # the source map rather than trusting the bare count.
        for tag, r in (("C", c), ("T", t)):
            src = (r or {}).get("foreign_srcs")
            if src:
                notes.append("%s:%s" % (tag, src.replace(";", ","))[:46])
        if L >= 65536:
            notes.append("agg not a decode metric (short window)")
        # drift bracket, when a same-config repeat exists
        if (C, L) in ctrl and (C, L) in rpt:
            r0, r1 = ctrl[(C, L)], rpt[(C, L)]
            v0, v1 = num(r0.get("decode_agg_tps")), num(r1.get("decode_agg_tps"))
            if v0 and v1:
                drift = abs(v1 - v0) / v0 * 100
                notes.append("drift=%.1f%%" % drift)
        print("%-3s %-8s %-6s %-6s %-8s | %-7s %-7s %-6s | %-7s %-7s %-6s | %s" % (
            C, L, g(c, "achieved_concurrency"), g(t, "achieved_concurrency"),
            C, ds_c, ds_t, dpct, ag_c, ag_t, apct, "; ".join(notes)))

    print()
    d = a.pool_ctrl - a.pool_trt
    print("PRICE of the extra state, from the boot log (not estimated):")
    print("  KV pool      %d -> %d tokens  (%+.1f%%, %d tokens)" % (
        a.pool_ctrl, a.pool_trt, -100.0 * d / a.pool_ctrl, d))
    print("  KV bytes     %.2f GB -> %.2f GB  (at %d B/token incl. draft)" % (
        a.pool_ctrl * KV_BYTES_PER_TOKEN / 1e9, a.pool_trt * KV_BYTES_PER_TOKEN / 1e9, KV_BYTES_PER_TOKEN))
    print("  bought       max_running_requests 4->8, mamba checkpoints 24->48,"
          " cuda-graph batch sizes [1..4]->[1..8]")
    print()
    print("What each limit binds, given the pool (C x (L+gen) <= pool):")
    print("  %-9s %-12s %-12s" % ("context", "max C @ctrl", "max C @trt"))
    for L in (4096, 16384, 65536, 131072, 262144, 393216, 520192):
        gc = min(a.pool_ctrl // (L + 256), 10**6)
        gt = min(a.pool_trt // (L + 256), 10**6)
        print("  %-9s %-12s %-12s" % ("%dk" % (L // 1024), gc, gt))
    print("  (max_running_requests caps the left column at 4, the right at 8;"
          " below ~131k the request cap binds, above it the pool does)")


if __name__ == "__main__":
    main()
