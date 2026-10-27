#!/usr/bin/env python3
"""Agent-workflow growth bench: ONE append-only conversation, C1, measured per turn.

The conversation grows by appending deterministic blocks; each depth is one more turn of
the same conversation, so the prompt at depth D+1 shares its whole prefix with depth D.
That is the case the user asked to measure: what a real agent loop costs as context
climbs ~1k -> ~520k, where radix/prefix cache should make each turn cheap.

Two arms, both measuring the SAME bytes:
  warm = nothing flushed  -> the agent case: prefix reused, only the new tail is prefilled
  cold = /flush_cache first -> the counterfactual: the whole prompt is re-prefilled
warm-vs-cold at the same depth IS the price of the prefix cache.

Arms are OUTER and depths INNER: the warm pass must be one uninterrupted conversation,
so a flush can never land between two warm depths.

Sizing: prompt_tokens is exactly linear in the filler-line count once (a) every line is
fixed width and (b) the line count is a multiple of the chunk size, so two chunk-aligned
probes fit the whole curve and NO cell needs a calibration probe afterwards -- a probe
sends the same bytes as the request it calibrates, so it would warm the very tail the
warm arm exists to price. Cells that come back >20% off target carry an `error` instead
of quietly becoming a point on the curve.

Instrument notes (each of these silently produced garbage in an earlier version):
  * this engine returns usage.prompt_tokens_details = null and prints no "Prefill batch"
    line for a fully-cached request; cache reuse comes from the per-request
    ReqTimeStats(... cached_input_len ...) line;
  * engine metrics ride the container's STDERR, access lines ride stdout -- scraping only
    .stdout reads back empty and looks identical to "no cache";
  * filler line count must be an explicit budget, not a loop counter (a loop counter made
    history grow quadratically: "16k" arrived as 101k);
  * flush must come after any probe, or the probe re-prefills and the cold arm is warm;
  * regex groups are NAMED because positional indexing into this line has already been
    off by one once.
thinking stays off; temperature 0; C1 only.
"""
import argparse, json, os, re, subprocess, sys, threading, time, urllib.request, urllib.error

SYSTEM = ("You are a precise senior infrastructure engineer. Reason about capacity, scheduling "
          "and memory layout. Answer concretely and cite the numbers you used. ") * 5
VOCAB = ("shard replica quorum epoch ledger segment offset index compaction flush checkpoint "
         "snapshot journal barrier lease topology partition rebalance backpressure window "
         "residency eviction admission preemption quota burst").split()
PER_MSG_SCAFFOLD = "Absorb this and reply with a one-line acknowledgement."
FINAL_Q = "Summarise the operational risks across everything above in under 120 words."
# SELF_IPS: the source addresses this harness's own requests arrive from (typically the
# docker bridge the proxy is reached through). Anything else in the replica's log is
# foreign. Both lists are operator-supplied: empty SELF_IPS would count every request as
# foreign, empty FOREIGN_IPS would count none — neither may silently report a cell clean.
SELF_IPS = tuple(os.environ.get("SELF_IPS", "").replace(",", " ").split())
FOREIGN_WATCH = tuple(os.environ.get("FOREIGN_IPS", "").replace(",", " ").split())
if not SELF_IPS or not FOREIGN_WATCH:
    raise SystemExit("SELF_IPS and FOREIGN_IPS env required (space/comma-separated IPs) — the "
                     "self/foreign split is what makes a cell trustworthy; it cannot be guessed")

RTS_RE = re.compile(
    r"ReqTimeStats\(rid=(?P<rid>\w+), input_len=(?P<in>\d+), cached_input_len=(?P<cached>\d+), "
    r"output_len=(?P<out>\d+), attempts=(?P<attempts>\d+), type=(?P<type>\w+)\): "
    r"queue_duration=(?P<q>[\d.]+)(?P<qu>ms|s), initial_prefill_elapsed=(?P<pf>[\d.]+)(?P<pfu>ms|s), "
    r"post_prefill_elapsed=(?P<post>[\d.]+)(?P<postu>ms|s), forward_duration=(?P<fwd>[\d.]+)(?P<fwdu>ms|s), "
    r"entry_time=(?P<et>[\d.]+)")


def _ms(value, unit):
    return float(value) * (1000.0 if unit == "s" else 1.0)


def lines_block(n, seed):
    """Exactly n deterministic, unique lines. Both counters are fixed width on purpose:
    unpadded ints make deep lines cost more tokens than shallow ones (8-digit vs 2-digit),
    which breaks the linear sizing fit. Uniqueness comes from the padded line index."""
    import random
    r = random.Random(1000 + seed)
    return "\n".join("[c%03d-%08d] %s %08d" % (seed, i,
                        " ".join(r.choice(VOCAB) for _ in range(8)), (i * 104729) % 100000000)
                     for i in range(n))


def build_messages(total_lines, chunk_lines=60):
    """Append-only: the first k chunks are byte-identical at every depth, which is what
    lets the radix cache carry a conversation forward instead of re-reading it."""
    msgs = [{"role": "system", "content": SYSTEM}]
    left, c = total_lines, 0
    while left > 0:
        take = min(chunk_lines, left)
        msgs.append({"role": "user", "content": lines_block(take, c) + "\n" + PER_MSG_SCAFFOLD})
        msgs.append({"role": "assistant", "content": "Acknowledged block %d." % c})
        left -= take
        c += 1
    msgs.append({"role": "user", "content": FINAL_Q})
    return msgs


class Engine:
    def __init__(self, base, key):
        self.base, self.key = base, key

    def _req(self, path, payload=None, method="POST"):
        data = json.dumps(payload).encode() if payload is not None else b"{}"
        r = urllib.request.Request(self.base + path, data=data, method=method,
                                   headers={"Content-Type": "application/json",
                                            "Authorization": "Bearer " + self.key})
        return urllib.request.urlopen(r, timeout=1800)

    def flush(self):
        try:
            with self._req("/flush_cache") as resp:
                return resp.status
        except Exception as e:
            return "ERR:%s" % type(e).__name__

    def stream(self, messages, maxtok):
        """One turn. Returns (ttft, wall, usage, error); usage carries prompt/completion tokens."""
        payload = {"model": "m", "messages": messages, "max_tokens": maxtok, "temperature": 0,
                   "stream": True, "stream_options": {"include_usage": True},
                   "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}}
        t0 = time.time(); first = None; usage = {}; chars = 0
        try:
            with self._req("/v1/chat/completions", payload) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        d = json.loads(body)
                    except Exception:
                        continue
                    if d.get("usage"):
                        usage = d["usage"]
                    for ch in d.get("choices") or []:
                        dl = ch.get("delta") or {}
                        piece = dl.get("content") or dl.get("reasoning_content") or ""
                        if piece:
                            if first is None:
                                first = time.time() - t0
                            chars += len(piece)
            return first, time.time() - t0, usage, None
        except urllib.error.HTTPError as e:
            return None, time.time() - t0, {}, "HTTP %s: %s" % (e.code, e.read().decode()[:240])
        except Exception as e:
            return None, time.time() - t0, {}, "%s: %s" % (type(e).__name__, e)


def _logs(container, since_s, floor=0):
    try:
        r = subprocess.run(["docker", "logs", "--since", "%ds" % max(floor, since_s), container],
                           capture_output=True, text=True, timeout=60, errors="replace")
        return (r.stdout or "") + "\n" + (r.stderr or "")
    except Exception:
        return ""


def engine_reqstats(container, want_in, not_before, window_s):
    """Per-request truth for cache reuse and queue wait. Matched on input_len plus
    entry_time >= our own start, newest wins: the engine's output_len is NOT
    usage.completion_tokens under speculative decoding (measured 66 vs 64 on the same
    request), so requiring them to agree drops most lines. Returns (None, "none") when
    nothing landed -- an instrument failure, which must never be read as cached=0."""
    for gap in (0, 3, 6):
        if gap:
            time.sleep(gap)
        cands = []
        for m in RTS_RE.finditer(_logs(container, window_s, floor=90)):
            g = m.groupdict()
            if int(g["in"]) != want_in or float(g["et"]) < not_before:
                continue
            cands.append((float(g["et"]), g))
        if cands:
            _, g = max(cands, key=lambda x: x[0])
            return {"rid": g["rid"], "input_len": int(g["in"]), "cached_input_len": int(g["cached"]),
                    "new_input_len": int(g["in"]) - int(g["cached"]), "output_len": int(g["out"]),
                    "type": g["type"],
                    "queue_ms": round(_ms(g["q"], g["qu"]), 2),
                    "initial_prefill_ms": round(_ms(g["pf"], g["pfu"]), 2),
                    "post_prefill_ms": round(_ms(g["post"], g["postu"]), 2),
                    "forward_ms": round(_ms(g["fwd"], g["fwdu"]), 2),
                    "entry_time": float(g["et"])}, "exact"
    return None, "none"


def traffic(container, window_s):
    """Self vs foreign POST counts for the cell window. Self (the bridge this harness
    arrives from) is expected; anything from the proxy/Mac/management-agent IPs means the card was
    not drained and the cell is contaminated."""
    self_, foreign = {}, {}
    for line in _logs(container, window_s).splitlines():
        if "POST /v1/" not in line:
            continue
        m = re.search(r'(\d{1,3}(?:\.\d{1,3}){3}|\[[0-9a-f:]+\]):\d+[^"]*"(POST /v1/\S+)', line)
        if not m:
            continue
        ip = m.group(1).strip("[]")
        if ip.startswith("127.") or ip == "::1":
            continue
        k = "%s %s" % (ip, m.group(2).rsplit("/v1/", 1)[-1])
        (self_ if ip in SELF_IPS else foreign)[k] = (self_ if ip in SELF_IPS else foreign).get(k, 0) + 1
    return self_, foreign


def gpu_sample(uuid, stop, sink):
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--id=" + uuid,
                                "--query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True,
                               timeout=20).stdout.strip()
            if o:
                p, c, t, u = [x.strip() for x in o.split(",")]
                sink.append((float(p), float(c), float(t), float(u)))
        except Exception:
            pass
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30073")
    ap.add_argument("--container", default="flash-next-pair-flash-next-b-1")
    ap.add_argument("--gpu-uuid", default="", help="nvidia-smi GPU UUID to sample at ~1 Hz "
                                                   "(power draw, SM clocks, temperature, util)")
    ap.add_argument("--no-gpu-sample", action="store_true",
                    help="bench without instrumenting the card. Cells then cannot be attributed "
                         "to a clock/power state and must not be published as certified numbers.")
    ap.add_argument("--key-file", required=True)   # no default: it is a credential path
    ap.add_argument("--depths", default="2048,4096,8192,16384,32768,65536,131072,196608,"
                                        "262144,327680,393216,458752,507904")
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--chunk-lines", type=int, default=60)
    ap.add_argument("--arms", default="warm,cold")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    if not a.gpu_uuid and not a.no_gpu_sample:
        raise SystemExit("--gpu-uuid is required. Every cell has to be attributable to a clock and "
                         "power state or it is not a publishable number (docs/benchmarking.md). "
                         "Pass --no-gpu-sample to proceed anyway; the run is exploratory.")
    if a.no_gpu_sample:
        print("WARNING: --no-gpu-sample: no power/clock record for any cell. Exploratory only — "
              "these numbers cannot be certified as taken at a known cap.", file=sys.stderr)
    os.makedirs(a.outdir, exist_ok=True)
    kf = a.key_file
    key = (open(os.path.join(kf, os.listdir(kf)[0])) if os.path.isdir(kf) else open(kf)).read().strip()
    eng = Engine(a.base, key)
    C = a.chunk_lines
    cp, rp = a.outdir + "/cal.json", a.outdir + "/agent-loop.json"
    cal = json.load(open(cp)) if os.path.exists(cp) else {}
    rows = json.load(open(rp)) if os.path.exists(rp) else []
    depths = [int(x) for x in a.depths.split(",")]

    # (n, prompt_tokens) observations; the fit is slope over chunk-aligned line counts.
    pts = [(int(n), int(q)) for n, q in cal.get("points", [])]
    if len(pts) < 2:
        for n in (C * 2, C * 5):
            _f, _t, u0, e0 = eng.stream(build_messages(n, C), 1)
            q = (u0 or {}).get("prompt_tokens")
            if not q:
                print("FIT-ABORT probe n=%d err=%s" % (n, e0), flush=True)
                raise SystemExit(2)
            pts.append((n, int(q)))
        json.dump({"points": pts[-6:], "chunk_lines": C}, open(cp, "w"))
    (n1, q1), (n2, q2) = pts[-2], pts[-1]
    L = (q2 - q1) / float(n2 - n1)                      # tokens per filler line, chunk term folded in
    A0 = q1 - L * n1                                    # system + final question overhead
    print("FIT L=%.3f tok/line  a0=%.1f  from %s" % (L, A0, pts[-2:]), flush=True)

    def size(depth):
        return C * max(1, int(round((depth - A0) / L / C)))

    for arm in [x.strip() for x in a.arms.split(",")]:
        for depth in depths:
            total = size(depth)
            if arm == "cold" or depth == depths[0]:
                # cold flushes before every cell (that IS the counterfactual); warm flushes once
                # before its first cell so the only cache it can read is the one THIS
                # conversation builds -- otherwise reuse gets credited to an earlier run's
                # identical bytes instead of to the append-only growth being measured.
                fl = eng.flush()
                if not str(fl).startswith("2"):
                    print("[%s] depth=%d FLUSH-FAILED %s" % (arm, depth, fl), flush=True)
            stop = threading.Event(); sink = []
            th = threading.Thread(target=gpu_sample, args=(a.gpu_uuid, stop, sink)); th.start()
            t0 = time.time()
            first, wall, u, err = eng.stream(build_messages(total, C), a.gen)
            window = int(time.time() - t0) + 10
            prow = (u or {}).get("prompt_tokens")
            ct = (u or {}).get("completion_tokens") or 0
            rs, rs_how = engine_reqstats(a.container, prow, t0 - 2.0, window)
            slf, frn = traffic(a.container, window)
            stop.set(); th.join(timeout=10)
            off = abs(prow - depth) / depth if prow else None
            decode_tps = round(ct / (wall - first), 1) if (first and ct and wall > first) else None
            row = {"arm": arm, "depth_target": depth, "lines": total, "chunks": -(-total // C),
                   "prompt_tokens": prow, "completion_tokens": ct,
                   "ttft_s": round(first, 2) if first else None, "total_s": round(wall, 2),
                   "decode_tps": decode_tps, "req_err": err, "engine": rs, "engine_match": rs_how,
                   "cache_reuse_frac": round(rs["cached_input_len"] / prow, 3) if (rs and prow) else None,
                   "prefill_tok_s": round(rs["new_input_len"] / (rs["initial_prefill_ms"] / 1000.0), 0)
                   if (rs and rs["initial_prefill_ms"] and rs["new_input_len"] > 0) else None,
                   "ts": int(t0), "cell_s": int(time.time() - t0),
                   "self_posts": slf, "foreign_posts": frn,
                   "gpu_draw_med_w": round(sorted(x[0] for x in sink)[len(sink) // 2], 1) if sink else None,
                   "gpu_sm_min_mhz": min((x[1] for x in sink), default=None),
                   "gpu_samples": len(sink)}
            if rs is None:
                row["error"] = "engine ReqTimeStats line missing -- cache_reuse is unknown, not zero"
            if off is not None and off > 0.20:
                row["error"] = "sizing off %.0f%% (fit L=%.2f a0=%.0f) -- not the depth it claims" % (
                    100.0 * off, L, A0)
            rows.append(row)
            print("[%-4s] pt=%-7d ch=%-5d cached=%-7s reuse=%-5s ttft=%-7s total=%-7s dec=%-6s "
                  "pfill=%-8s q=%-7s tok/s=%-7s W=%-6s foreign=%s%s"
                  % (arm, prow or -1, row["chunks"], (rs or {}).get("cached_input_len"),
                     row["cache_reuse_frac"], row["ttft_s"], row["total_s"], decode_tps,
                     (rs or {}).get("initial_prefill_ms"), (rs or {}).get("queue_ms"),
                     row["prefill_tok_s"], row["gpu_draw_med_w"], list(frn.keys()) or "-",
                     "  ERR:" + row["error"] if row.get("error") else ""), flush=True)
            json.dump(rows, open(rp, "w"), indent=1)
    print("AGENT-LOOP-DONE cells=%d outdir=%s" % (len(rows), a.outdir))


main()
