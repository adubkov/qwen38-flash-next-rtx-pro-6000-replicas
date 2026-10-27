#!/usr/bin/env python3
"""flash-next single-GPU baseline matrix (on the serving host).

Grid: concurrency C x context length L, at the production power posture.
Per cell records: cold prefill tok/s, warm prefill tok/s, per-stream and
aggregate decode tok/s, accept-len / accept-rate, achieved concurrency, and
KV/token-usage peak -- everything needed to compare a future 2x RTX PRO 6000
run against THIS box with the same client, prompts and kwargs.

Deliberate design choices (all recorded in the row so the artifact is self-describing):
  * chat_template_kwargs pinned (thinking off) -> accept-len is content- AND
    mode-dependent, so an unpinned baseline is not repeatable.
  * each concurrent stream gets a UNIQUE marker so C streams do not collapse
    onto one shared radix prefix; otherwise "C concurrent" silently measures 1.
  * /flush_cache before the cold-prefill pass; warm pass runs immediately after.
  * decode rate uses first-token -> last-token only, so prefill never enters it.
  * context ladder is KV-feasible: C*L must fit --max-total-tokens (824,384).
  * resumable: completed (C,L) cells are skipped, so a long sweep survives
    restarts and needs only coarse polling (few assistant turns = little
    self-inflicted load on the GPU under test).
"""
import argparse, csv, json, os, re, statistics as st, threading, time, urllib.request

BASE = "http://127.0.0.1:30073"
CTX_KV_POOL = 824384          # --max-total-tokens
CONTEXT_MAX = 524288          # --context-length
FILLER = ("Distributed consensus requires a reliable message layer and a failure detector. "
          "Each replica must reach agreement on a total order of commands despite crashes. "
          "Quorum-based reads and writes trade availability for strong consistency. "
          "Vector clocks capture causal partial order across processes without a global clock. "
          "Sharding partitions state by key range to spread load and bound per-node memory. "
          "Write amplification in an LSM tree trades read amplification for space amplification. "
          "Backpressure must propagate to producers or queues grow without bound. ")
CTK_OFF = {"enable_thinking": False, "preserve_thinking": False}


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
        """-> dict with ttft, decode window, prompt/completion tokens."""
        payload = {"model": "pennyroyal", "messages": [{"role": "user", "content": text}],
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
    # Marker LEADS the prompt on purpose. A trailing marker leaves the long FILLER
    # prefix shared across streams, so "C concurrent" silently measures ONE
    # prefill plus C cache hits -- inflating every concurrent cell.
    return marker + " " + FILLER * repeats + " Summarize the engineering tradeoffs above."


def calibrate(cl, target, cache):
    """Find `repeats` so prompt_tokens ~= target (within 3%). Cached per target."""
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
    """Decode length per stream. Short at long contexts so one cell stays minutes,
    not tens -- steady-state rate is visible within the first ~100 tokens."""
    return 256 if L <= 65536 else 128


def run_cell(cl, C, L, cpt_cache, outdir, log_since):
    cal = calibrate(cl, L, cpt_cache)
    if not cal:
        return {"C": C, "ctx_target": L, "error": "calibration failed"}
    streams = []
    # cold prefill pass: one stream, cache flushed. The warm pass reuses the SAME
    # marker so it genuinely measures the radix-cache-served TTFT.
    fc = cl.flush()
    coldtag = "[cell-%d-%d]" % (C, L)
    cold = cl.stream(build(4.0, cal["repeats"], coldtag), 8)
    prefill_cold = round(cold["prompt_tokens"] / cold["ttft"], 1) if cold["ttft"] > 0.01 else None
    warm = cl.stream(build(4.0, cal["repeats"], coldtag), 8)
    cached_prefix_tps = round(warm["prompt_tokens"] / warm["ttft"], 1) if warm["ttft"] > 0.01 else None

    # concurrent decode: C streams, each with a DISTINCT leading marker (never
    # pre-warmed by the cold/warm passes) so all C pay a real prefill.
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
    results = [None] * C

    def worker(i):
        results[i] = cl.stream(build(4.0, cal["repeats"], "[c%d-l%d-s%d]" % (C, L, i)), maxtok)
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
    for t in ts: t.start()
    for t in ts: t.join()
    stop.set(); th.join(timeout=3)
    t_end = time.time()

    good = [r for r in results if r and r["window"] > 0.05 and r["completion_tokens"]]
    per_stream = [r["completion_tokens"] / r["window"] for r in good]
    # aggregate over the DECODE window only (earliest first-token -> latest
    # last-token); wall time would fold in prefill serialisation and understate
    # C-way throughput, which is the number the 2-GPU run gets compared on.
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
        "cal_err_pct": cal["err_pct"], "decode_maxtok": maxtok,
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
    ap.add_argument("--key-file", required=True)   # no default: it is a credential path
    ap.add_argument("--cs", default="1,2,4")
    ap.add_argument("--ctxs", default="1024,4096,16384,65536,131072,262144,393216,520192")
    ap.add_argument("--outdir", default="/var/tmp/phase0/matrix")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cl = Client(open(a.key_file).read().strip())
    cachef = os.path.join(a.outdir, "ctxcal.json")
    cache = json.load(open(cachef)) if os.path.exists(cachef) else {}

    done = {}
    jsonf = os.path.join(a.outdir, "matrix.jsonl")
    if os.path.exists(jsonf):
        for line in open(jsonf):
            try:
                r = json.loads(line); done[(r["C"], r["ctx_target"])] = r
            except Exception:
                pass

    def log_since(t):
        import subprocess
        secs = max(1, int(time.time() - t) + 5)
        return subprocess.run(["docker", "logs", "--since", "%ds" % secs,
                               "flash-next-pair-flash-next-b-1"],
                              capture_output=True, text=True).stderr

    plan = []
    for C in [int(x) for x in a.cs.split(",")]:
        for L in [int(x) for x in a.ctxs.split(",")]:
            dt = decode_maxtok(L)
            # generation tokens count against BOTH limits -- a cell that fits at
            # prompt length alone can still 400 on context-length or retract on KV.
            if L + dt > CONTEXT_MAX:
                print("DROP C%d L%d: prompt+gen %d exceeds context %d" % (C, L, L + dt, CONTEXT_MAX), flush=True)
                continue
            if C * (L + dt) > CTX_KV_POOL * 0.95:
                print("DROP C%d L%d: %d tokens would need %.0f%% of the %d-token KV pool"
                      % (C, L, C * (L + dt), 100.0 * C * (L + dt) / CTX_KV_POOL, CTX_KV_POOL), flush=True)
                continue
            plan.append((C, L))
    # CTX_KV_POOL / CONTEXT_MAX below describe one engine's boot. On any other pool
    # the long-context rows used to vanish silently and the run still printed
    # MATRIX-DONE, so an outsider read a short ladder as the intended experiment.
    if not plan:
        raise SystemExit("no cell fits: every (C, ctx) was dropped above. Pass --kv-pool/--context-max "
                         "matching YOUR engine's boot (see its 'KV pool' line), or narrow --cs/--ctxs.")
    todo = [p for p in plan if p not in done]
    print("plan=%d cells, already done=%d, todo=%d" % (len(plan), len(done), len(todo)), flush=True)
    for i, (C, L) in enumerate(todo, 1):
        t0 = time.time()
        row = run_cell(cl, C, L, cache, a.outdir, log_since)
        json.dump(cache, open(cachef, "w"))
        with open(jsonf, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print("[%d/%d] C=%d L=%-7d prefill_cold=%s cached=%s dec_stream=%s agg=%s accept=%s achv=%s (%.0fs)"
              % (i, len(todo), C, L, row.get("prefill_cold_tps"), row.get("cached_prefix_tps"),
                 row.get("decode_per_stream_med_tps"), row.get("decode_agg_tps"),
                 row.get("accept_len_mean"), row.get("achieved_concurrency"), time.time() - t0), flush=True)

    # rebuild the CSV from the journal alone (last write wins), never by slicing
    # `done` -- an unparseable history line would silently misalign every row.
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
                "accept_len_mean", "achieved_concurrency", "saturated", "cell_s"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["C"], x["ctx_target"])):
            w.writerow(r)
    print("MATRIX-DONE", flush=True)


if __name__ == "__main__":
    main()
