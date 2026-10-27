#!/usr/bin/env python3
"""flash-next concurrency sweep THROUGH LLMPROXY (C1..C8, pool of replicas A+B).

Same client contract as bench-matrix.py (identical prompts, greedy, thinking-off
pin, unique leading markers per stream, resumable journal) with three deliberate
deviations for the proxy path:
  * BASE = the Mac-local llmproxy (:8787); served model name is the proxy's
    `qwen38-flash-next`, not the engines' `pennyroyal`.
  * inbound key comes from env PROXY_KEY (never argv).
  * accept-len is sampled from BOTH replicas' logs via ssh (a proxy-routed cell
    spans both cards).
Consequences to remember when judging rows: /metrics and /flush_cache are not
proxied -> achieved_concurrency reads None (purity gate intentionally NOT used
here; contamination is judged post-hoc against `docker logs llmproxy` upstream
attribution), and flush_status is meaningless as before. Sticky session key is
content-derived (session.Derive), so unique leading markers genuinely spread
streams across the pair while each stream keeps its warm KV on one card.
The plan filter stays SINGLE-pool conservative (C*(L+dt) <= one KV pool) even
though the pool spans two boxes: a skewed sticky spread must never be able to
oversubscribe one replica's 824,384-token pool.
"""
import argparse, csv, json, os, re, statistics as st, threading, time, urllib.request

BASE = "http://127.0.0.1:8787"
MODEL = "qwen38-flash-next"
CTX_KV_POOL = 824384
CONTEXT_MAX = 524288
FILLER = ("Distributed consensus requires a reliable message layer and a failure detector. "
          "Each replica must reach agreement on a total order of commands despite crashes. "
          "Quorum-based reads and writes trade availability for strong consistency. "
          "Vector clocks capture causal partial order across processes without a global clock. "
          "Sharding partitions state by key range to spread load and bound per-node memory. "
          "Write amplification in an LSM tree trades read amplification for space amplification. "
          "Backpressure must propagate to producers or queues grow without bound. ")
CTK_OFF = {"enable_thinking": False, "preserve_thinking": False}
CONT_A = "flash-next-pair-flash-next-a-1"
CONT_B = "flash-next-pair-flash-next-b-1"


class Client:
    def __init__(self, key):
        self.key = key

    def _req(self, path, payload=None, method="POST"):
        data = json.dumps(payload).encode() if payload is not None else b"{}"
        r = urllib.request.Request(BASE + path, data=data, method=method,
                                   headers={"Content-Type": "application/json",
                                            "Authorization": "Bearer " + self.key})
        return urllib.request.urlopen(r, timeout=3600)

    def flush(self):
        try:
            with self._req("/flush_cache") as resp:
                return resp.status
        except Exception as e:
            return "ERR:%s" % type(e).__name__

    def running(self):
        try:
            with self._req("/metrics", method="GET") as r:
                txt = r.read().decode()
            m = re.search(r"^sglang:num_running_reqs\{.*?\}\s+([\d.]+)", txt, re.M)
            return float(m.group(1)) if m else None
        except Exception:
            return None

    def stream(self, text, max_tokens):
        payload = {"model": MODEL, "messages": [{"role": "user", "content": text}],
                   "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
                   "stream_options": {"include_usage": True}, "chat_template_kwargs": CTK_OFF}
        t0 = time.time(); first = last = None
        comp = prompt = 0
        with self._req("/v1/chat/completions", payload) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    o = json.loads(d)
                except Exception:
                    continue
                if o.get("usage"):
                    comp = o["usage"].get("completion_tokens", comp)
                    prompt = o["usage"].get("prompt_tokens", prompt)
                ch = o.get("choices") or []
                if ch:
                    dl = ch[0].get("delta") or {}
                    if dl.get("content") or dl.get("reasoning_content"):
                        now = time.time()
                        first = now if first is None else first
                        last = now
        total = time.time() - t0
        return {"t0": t0, "t_first": first, "t_last": last, "ttft": (first - t0) if first else total,
                "total_s": total, "prompt_tokens": prompt, "completion_tokens": comp,
                "window": (last - first) if (first and last and last > first) else 0.0}


def build(cpt, repeats, marker):
    return marker + " " + FILLER * repeats + " Summarize the engineering tradeoffs above."


def calibrate(cl, target, cache):
    if str(target) in cache:
        return cache[str(target)]
    reps = max(1, int(target * 4.0 / (len(FILLER) * 1.0)))
    best = None
    for _ in range(6):
        r = cl.stream(build(4.0, reps, "[calib]"), 1)
        got = r["prompt_tokens"]
        if not got:
            break
        best = {"repeats": reps, "prompt_tokens": got, "err_pct": round(abs(got - target) / target * 100, 2)}
        if best["err_pct"] <= 3.0:
            break
        reps = max(1, int(reps * target / got))
    cache[str(target)] = best
    return best


def decode_maxtok(L):
    return 256 if L <= 65536 else 128


def run_cell(cl, C, L, cpt_cache, outdir, log_since):
    cal = calibrate(cl, L, cpt_cache)
    if not cal:
        return {"C": C, "ctx_target": L, "error": "calibration failed"}
    results = [None] * C
    fc = cl.flush()
    coldtag = "[cell-%d-%d]" % (C, L)
    cold = cl.stream(build(4.0, cal["repeats"], coldtag), 8)
    prefill_cold = round(cold["prompt_tokens"] / cold["ttft"], 1) if cold["ttft"] > 0.01 else None
    warm = cl.stream(build(4.0, cal["repeats"], coldtag), 8)
    cached_prefix_tps = round(warm["prompt_tokens"] / warm["ttft"], 1) if warm["ttft"] > 0.01 else None

    maxtok = decode_maxtok(L)
    sampler = {"max": 0.0, "vals": []}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            v = cl.running()
            if v is not None:
                sampler["vals"].append(v)
                sampler["max"] = max(sampler["max"], v)
            time.sleep(0.4)

    th = threading.Thread(target=sample, daemon=True); th.start()
    t_start = time.time()

    def worker(i):
        results[i] = cl.stream(build(4.0, cal["repeats"], "[c%d-l%d-s%d]" % (C, L, i)), maxtok)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
    for t in ts: t.start()
    for t in ts: t.join()
    stop.set(); th.join(timeout=3)
    t_end = time.time()

    good = [r for r in results if r and r["window"] > 0.05 and r["completion_tokens"]]
    per_stream = [r["completion_tokens"] / r["window"] for r in good]
    firsts = [r["t_first"] for r in good if r.get("t_first")]
    lasts = [r["t_last"] for r in good if r.get("t_last")]
    dwin = (max(lasts) - min(firsts)) if (firsts and lasts) else 0.0
    agg = sum(r["completion_tokens"] for r in good) / max(0.001, dwin)
    acc = []
    try:
        txt = log_since(int(t_start))
        for line in txt.splitlines():
            if "Decode batch" not in line:
                continue
            m = re.search(r"accept len:\s*([\d.]+)", line)
            if m: acc.append(float(m.group(1)))
    except Exception:
        pass
    return {
        "C": C, "ctx_target": L, "prompt_tokens": good[0]["prompt_tokens"] if good else cal["prompt_tokens"],
        "cal_err_pct": cal["err_pct"], "decode_maxtok": maxtok, "via": "llmproxy",
        "flush_status": fc,
        "prefill_cold_tps": prefill_cold, "cached_prefix_tps": cached_prefix_tps,
        "decode_window_s": round(dwin, 1),
        "cold_ttft_s": round(cold["ttft"], 3),
        "ttft_worst_s": round(max(r["ttft"] for r in good), 3) if good else None,
        "decode_per_stream_med_tps": round(st.median(per_stream), 1) if per_stream else None,
        "decode_per_stream_min_tps": round(min(per_stream), 1) if per_stream else None,
        "decode_agg_tps": round(agg, 1),
        "accept_len_mean": round(st.mean(acc), 3) if acc else None,
        "accept_len_n": len(acc),
        "achieved_concurrency": round(max(sampler["vals"]), 1) if sampler["vals"] else None,
        "requested_C": C, "saturated": bool(sampler["vals"] and max(sampler["vals"]) < C - 0.5),
        "cell_s": round(t_end - t_start, 1), "ts": int(t_start),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--ctxs", default="1024,4096,16384,65536,131072,262144,393216,520192")
    ap.add_argument("--outdir", default="./matrix-proxy")
    a = ap.parse_args()
    key = os.environ.get("PROXY_KEY", "")
    if not key:
        raise SystemExit("PROXY_KEY env required")
    os.makedirs(a.outdir, exist_ok=True)
    cl = Client(key)
    cachef = os.path.join(a.outdir, "ctxcal.json")
    cache = json.load(open(cachef)) if os.path.exists(cachef) else {}

    def log_since(t):
        import subprocess
        secs = max(1, int(time.time() - t) + 5)
        cmd = ("docker logs --since %ds %s 2>&1; docker logs --since %ds %s 2>&1"
               % (secs, CONT_A, secs, CONT_B))
        # Deliberately no hostname default: a guessed host yields an empty log, which
        # the purity gate below would read as "no foreign traffic" and certify clean.
        host = os.environ.get("BENCH_SSH_HOST")
        if not host:
            raise SystemExit("BENCH_SSH_HOST env required — it is how the purity gate reads the "
                             "replicas' request logs; unset means contamination is invisible")
        return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                              capture_output=True, text=True).stdout

    done = {}
    jsonf = os.path.join(a.outdir, "matrix.jsonl")
    if os.path.exists(jsonf):
        for line in open(jsonf):
            try:
                r = json.loads(line); done[(r["C"], r["ctx_target"])] = r
            except Exception:
                pass

    plan = []
    for C in [int(x) for x in a.cs.split(",")]:
        for L in [int(x) for x in a.ctxs.split(",")]:
            dt = decode_maxtok(L)
            if L + dt > CONTEXT_MAX:
                continue
            if C * (L + dt) > CTX_KV_POOL * 0.95:
                continue
            plan.append((C, L))
    todo = [p for p in plan if p not in done]
    print("plan=%d cells, already done=%d, todo=%d" % (len(plan), len(done), len(todo)), flush=True)
    for i, (C, L) in enumerate(todo, 1):
        t0 = time.time()
        row = run_cell(cl, C, L, cache, a.outdir, log_since)
        json.dump(cache, open(cachef, "w"))
        with open(jsonf, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print("[%d/%d] C=%d L=%-7d prefill_cold=%s cached=%s dec_stream=%s agg=%s accept=%s (%.0fs)"
              % (i, len(todo), C, L, row.get("prefill_cold_tps"), row.get("cached_prefix_tps"),
                 row.get("decode_per_stream_med_tps"), row.get("decode_agg_tps"),
                 row.get("accept_len_mean"), time.time() - t0), flush=True)

    bykey = {}
    if os.path.exists(jsonf):
        for line in open(jsonf):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "C" in r and "ctx_target" in r:
                bykey[(r["C"], r["ctx_target"])] = r
    rows = list(bykey.values())
    with open(os.path.join(a.outdir, "matrix.csv"), "w", newline="") as fh:
        keys = ["C", "ctx_target", "prompt_tokens", "prefill_cold_tps", "cached_prefix_tps",
                "cold_ttft_s", "ttft_worst_s", "decode_per_stream_med_tps",
                "decode_per_stream_min_tps", "decode_agg_tps", "decode_window_s",
                "accept_len_mean", "saturated", "cell_s"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["C"], x["ctx_target"])):
            w.writerow(r)
    print("PROXY-MATRIX-DONE", flush=True)


if __name__ == "__main__":
    main()
