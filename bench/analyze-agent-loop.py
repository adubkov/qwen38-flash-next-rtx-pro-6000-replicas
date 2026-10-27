#!/usr/bin/env python3
"""Report the agent-loop growth bench: per-depth warm-vs-cold = the price of the prefix cache.

Join is on (depth_target, arm). Rows carrying `error` (mis-sized cell or missing
ReqTimeStats) are printed but excluded from the totals, because a cell whose cache
state is unknown cannot price anything.
"""
import argparse, json, sys

FIELDS = ("prompt_tokens", "cache_reuse_frac", "ttft_s", "total_s", "decode_tps",
          "prefill_tok_s")


def num(r, k):
    v = r
    for part in k.split("."):
        v = (v or {}).get(part)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--csv", help="optional path to dump the flat table as CSV")
    a = ap.parse_args()
    rows = json.load(open(a.json_path))
    by = {}
    for r in rows:
        by.setdefault((r["depth_target"], r["arm"]), r)
    depths = sorted({d for d, _ in by})
    arms = [x for x in ("warm", "cold") if any(arms_ == x for _, arms_ in by)]

    hdr = ("%9s %9s | %9s %9s %7s %8s %9s | %9s %9s %7s %8s %9s | %8s %8s" % (
        "target", "prompt", "w:reuse", "w:ttft", "w:dec", "w:total", "w:pf/s",
        "c:reuse", "c:ttft", "c:dec", "c:total", "c:pf/s", "ttft x", "wall x"))
    print(hdr)
    print("-" * len(hdr))
    tw = tc = 0.0
    good = 0
    skipped = []
    csv = []
    for d in depths:
        w, c = by.get((d, "warm")), by.get((d, "cold"))
        if not w or not c:
            skipped.append("%d(%s)" % (d, "+".join(k[1] for k in by if k[0] == d)))
            continue
        if w.get("error") or c.get("error"):
            skipped.append("%d[%s|%s]" % (d, (w.get("error") or "")[:18], (c.get("error") or "")[:18]))
        if w.get("error") or c.get("error"):
            print("%9d %9s | %-51s | %-51s |   (excluded: %s)" % (
                d, w.get("prompt_tokens"), "instrument-err" if w.get("error") else
                "%9s %9s %7s %8s %9s" % tuple(num(w, k) for k in FIELDS[1:]),
                "instrument-err" if c.get("error") else
                "%9s %9s %7s %8s %9s" % tuple(num(c, k) for k in FIELDS[1:]),
                (w.get("error") or c.get("error"))[:60]))
            continue
        good += 1
        wt, ct = num(w, "ttft_s"), num(c, "ttft_s")
        ww, cw = num(w, "total_s"), num(c, "total_s")
        tw += ww or 0
        tc += cw or 0
        print("%9d %9s | %9s %9s %7s %8s %9s | %9s %9s %7s %8s %9s | %8s %8s" % (
            d, w.get("prompt_tokens"),
            num(w, "cache_reuse_frac"), wt, num(w, "decode_tps"), ww, num(w, "prefill_tok_s"),
            num(c, "cache_reuse_frac"), ct, num(c, "decode_tps"), cw, num(c, "prefill_tok_s"),
            round(ct / wt, 1) if (wt and ct) else None,
            round(cw / ww, 1) if (ww and cw) else None))
        csv.append({"depth_target": d, "prompt_tokens": w.get("prompt_tokens"),
                    "warm_ttft_s": wt, "cold_ttft_s": ct,
                    "warm_total_s": ww, "cold_total_s": cw,
                    "warm_reuse": num(w, "cache_reuse_frac"), "cold_reuse": num(c, "cache_reuse_frac"),
                    "warm_decode_tps": num(w, "decode_tps"), "cold_decode_tps": num(c, "decode_tps")})
    print("-" * len(hdr))
    print("priced cells=%d  warm pass total=%.1fs  cold pass total=%.1fs  cache saves %.1fx"
          % (good, tw, tc, (tc / tw) if tw else None))
    if skipped:
        print("excluded/not-paired: %s" % ", ".join(skipped))
    badfp = [(r["depth_target"], r["arm"], r["foreign_posts"]) for r in rows if r.get("foreign_posts")]
    print("contaminated cells: %s" % (badfp or "none"))
    errs = [(r["depth_target"], r["arm"], r["error"]) for r in rows if r.get("error")]
    for e in errs:
        print("  ERR %s" % (e,))
    if a.csv:
        with open(a.csv, "w") as f:
            ks = list(csv[0].keys()) if csv else []
            f.write(",".join(ks) + "\n")
            for r in csv:
                f.write(",".join("" if r[k] is None else str(r[k]) for k in ks) + "\n")
        print("csv -> %s" % a.csv)


main()
