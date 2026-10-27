#!/usr/bin/env python3
"""Attribute the SM-clock collapse under concurrency, and locate the real KV wall.

WHY THIS EXISTS. The C x context sweep showed draw pinned at exactly 300.0 W in
every cell while SM clock median fell from ~2235 MHz (C4/4k) to ~1275 MHz
(C4/16k). Two different stories fit that: (a) more concurrent streams make
decode itself expensive, or (b) every stream must PREFILL its own prompt and
those chunks interleave with decode, and prefill is what saturates the power
budget. The sweep cannot separate them because each concurrent stream carries a
unique LEADING marker (that is deliberate -- shared leading markers would make
"C concurrent" collapse into one prefill plus C cache hits). So the sweep's
concurrent window always contains C prefills.

  A  warm-prefix decode. Train the radix on one leading prefix, then admit C
     streams that reuse it and differ only in a short TRAILING suffix. Admission
     is then a cache hit, so the measurement window holds (near) no prefill.
     If the clock recovers toward its C1 value, prefill -- not concurrency --
     owns the power budget.
  B  pool wall. Admit C streams with DISTINCT markers whose combined
     prompt+generation tokens exceed --max-total-tokens, and record what the
     scheduler really does (queue, retract, or serve short) instead of trusting
     the arithmetic. Read together with the engine's own `mamba num:` samples,
     this says which limit binds first at a given context length.

Every leg records 1 Hz power draw, SM clock and the ENFORCED power limit, plus
the engine's own Decode-batch lines, because a leg whose enforced limit drifted
is not comparable (300 -> 450 W moved clocks 4.3% on this box) and because
clocks_event_reasons reads 0x0 on this driver even while the card is capped --
so capping is judged by draw-vs-limit proximity plus clock depression, never by
the throttle counters.
"""
import argparse, json, os, re, statistics as st, subprocess, threading, time, urllib.request

FILLER = ("Distributed consensus requires a reliable message layer and a failure detector. "
          "Each replica must reach agreement on a total order of commands despite crashes. "
          "Quorum-based reads and writes trade availability for strong consistency. "
          "Vector clocks capture causal partial order across processes without a global clock. "
          "Sharding partitions state by key range to spread load and bound per-node memory. "
          "Write amplification in an LSM tree trades read amplification for space amplification. "
          "Backpressure must propagate to producers or queues grow without bound. ")
CTK_OFF = {"enable_thinking": False, "preserve_thinking": False}
BASE = "http://127.0.0.1:30073"


class Client:
    def __init__(self, key, container, gpu):
        self.key, self.container, self.gpu = key, container, gpu

    def _req(self, path, payload=None, method="POST"):
        data = json.dumps(payload).encode() if payload is not None else b"{}"
        r = urllib.request.Request(BASE + path, data=data, method=method,
                                   headers={"Content-Type": "application/json",
                                            "Authorization": "Bearer " + self.key})
        return urllib.request.urlopen(r, timeout=3600)

    def gpu_once(self):
        out = subprocess.run(["nvidia-smi", "--id=" + self.gpu,
                              "--query-gpu=power.draw,enforced.power.limit,clocks.sm,"
                              "temperature.gpu,clocks_throttle_reasons.active",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        v = [x.strip() for x in out.stdout.strip().split(",")]
        return {"W": float(v[0]), "Wlim": float(v[1]), "sm": float(v[2]),
                "C": float(v[3]), "thr": v[4]}

    def engine_lines(self, since_s):
        try:
            return subprocess.run(["docker", "logs", "--since", "%ds" % since_s, self.container],
                                  capture_output=True, text=True, timeout=60).stderr
        except Exception:
            return ""

    def stream(self, text, max_tokens):
        payload = {"model": "pennyroyal", "messages": [{"role": "user", "content": text}],
                   "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
                   "stream_options": {"include_usage": True}, "chat_template_kwargs": CTK_OFF}
        t0 = time.time(); first = last = None; comp = prompt = 0
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
        return {"t_first": first, "t_last": last, "prompt_tokens": prompt,
                "completion_tokens": comp,
                "window": (last - first) if (first and last and last > first) else 0.0,
                "ttft": (first - t0) if first else time.time() - t0}


def leg(cl, C, text_of, maxtok):
    """Run one C-way leg, sampling the card at 1 Hz."""
    vals, stop, samp_err = [], threading.Event(), []

    def samp():
        while not stop.is_set():
            try:
                vals.append(cl.gpu_once())
            except Exception as e:
                samp_err.append(repr(e))
            time.sleep(1.0)

    th = threading.Thread(target=samp, daemon=True); th.start()
    t0 = time.time()
    res = [None] * C

    def worker(i):
        res[i] = cl.stream(text_of(i), maxtok)
    ws = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
    for w in ws: w.start()
    for w in ws: w.join()
    stop.set(); th.join(timeout=3)
    # An empty sample list means the instrument was broken, not that the card was
    # steady. The whole point of this probe is attributing a leg to a clock/power
    # state, so a leg with no samples must not be emitted as a result.
    if not vals:
        raise SystemExit("FAIL: zero GPU samples for a C%d leg — the sampler never read "
                         "nvidia-smi (bad --gpu-uuid, no nvidia-smi, or non-numeric output). "
                         "First errors: %s" % (C, "; ".join(samp_err[:3]) or "none raised"))
    el = time.time() - t0
    good = [r for r in res if r and r["window"] > 0.5 and r["completion_tokens"]]
    per = [r["completion_tokens"] / r["window"] for r in good]
    lines = cl.engine_lines(int(el) + 3)
    dec = []
    for line in lines.splitlines():
        if "Decode batch" in line:
            m = re.search(r"mamba num:\s*(\d+)", line)
            q = re.search(r"#queue-req:\s*(\d+)", line)
            u = re.search(r"full token usage:\s*([\d.]+)", line)
            g = re.search(r"gen throughput \(token/s\):\s*([\d.]+)", line)
            a = re.search(r"accept len:\s*([\d.]+)", line)
            r_ = re.search(r"#running-req:\s*(\d+)", line)
            dec.append({"mamba": int(m.group(1)) if m else None,
                        "queue": int(q.group(1)) if q else None,
                        "usage": float(u.group(1)) if u else None,
                        "gen_tps": float(g.group(1)) if g else None,
                        "accept": float(a.group(1)) if a else None,
                        "running": int(r_.group(1)) if r_ else None})
    retr = [l for l in lines.splitlines() if re.search(r"retract", l, re.I)]
    acc = [d["accept"] for d in dec if d["accept"] is not None]
    return {"C": C, "leg_s": round(el, 1), "streams_ok": len(good),
            "dec_per_stream_med": round(st.median(per), 1) if per else None,
            "dec_per_stream_min": round(min(per), 1) if per else None,
            "ttft_max_s": round(max(r["ttft"] for r in good), 2) if good else None,
            "prompt_tok_med": int(st.median([r["prompt_tokens"] for r in good])) if good else None,
            "completion_tok_med": int(st.median([r["completion_tokens"] for r in good])) if good else None,
            "W_med": round(st.median([v["W"] for v in vals]), 1) if vals else None,
            "Wlim_set": sorted({v["Wlim"] for v in vals}) if vals else None,
            "sm_med": round(st.median([v["sm"] for v in vals])) if vals else None,
            "sm_min": round(min(v["sm"] for v in vals)) if vals else None,
            "sm_max": round(max(v["sm"] for v in vals)) if vals else None,
            "temp_max": max(v["C"] for v in vals) if vals else None,
            "thr_bits": sorted({v["thr"] for v in vals}) if vals else None,
            "gpu_samples": len(vals),
            "engine_running_max": max([d["running"] for d in dec if d["running"] is not None], default=None),
            "engine_mamba_max": max([d["mamba"] for d in dec if d["mamba"] is not None], default=None),
            "engine_queue_max": max([d["queue"] for d in dec if d["queue"] is not None], default=None),
            "engine_usage_max": max([d["usage"] for d in dec if d["usage"] is not None], default=None),
            "engine_accept_med": round(st.median(acc), 3) if acc else None,
            "decode_lines": len(dec), "retract_lines": len(retr)}


def calibrate(cl, target):
    reps = max(1, int(target * 4.0 / (len(FILLER) * 1.0)))
    for _ in range(6):
        r = cl.stream("[calib] " + FILLER * reps + " Summarize the engineering tradeoffs above.", 1)
        got = r["prompt_tokens"]
        if not got:
            break
        if abs(got - target) / target <= 0.03:
            return reps
        reps = max(1, int(reps * target / got))
    return reps


def main():
    global BASE
    ap = argparse.ArgumentParser()
    # No defaults on --key-file/--container/--gpu-uuid: a default here is what made
    # this probe report a clean power/clock record for a card that did not exist.
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--gpu-uuid", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--outdir", default="./clock-attr")
    ap.add_argument("--warm-ctx", type=int, default=16384)
    ap.add_argument("--cs", default="1,2,4,8")
    ap.add_argument("--maxtok", type=int, default=384)
    ap.add_argument("--pool-probe", default="2,420000",
                    help="C,L for the over-pool admission probe; empty skips it")
    a = ap.parse_args()
    BASE = a.base.rstrip("/")
    os.makedirs(a.outdir, exist_ok=True)
    cl = Client(open(a.key_file).read().strip(), a.container, a.gpu_uuid)

    def emit(row, tag):
        with open(os.path.join(a.outdir, "probe.jsonl"), "a") as fh:
            fh.write(json.dumps(dict(row, arm=tag)) + "\n")
        print(json.dumps(dict(row, arm=tag)), flush=True)

    reps = calibrate(cl, a.warm_ctx)
    body = FILLER * reps
    cs = [int(x) for x in a.cs.split(",")]

    # Arm A0: the cold-prefill control, so A1 is compared against the same
    # harness and prompts rather than against the earlier sweep.
    for C in cs:
        emit(leg(cl, C, lambda i: ("[cold%d]" % i) + " " + body + " Summarize the tradeoffs above.",
                a.maxtok), "cold-prefill C%d" % C)

    # Arm A1: train one leading prefix, then vary only a trailing marker, so
    # admission is a radix hit and the window holds (near) no prefill.
    cl.stream("[shared] " + body + " Summarize the tradeoffs above.", 8)
    for C in cs:
        emit(leg(cl, C, lambda i: "[shared] " + body + " Summarize the tradeoffs above. [s%d]" % i,
                a.maxtok), "warm-prefix C%d" % C)

    # Arm B: demand more KV than the pool holds, distinct markers so the streams
    # cannot share a prefix, and let the scheduler reveal which limit it honours.
    if a.pool_probe:
        C, L = [int(x) for x in a.pool_probe.split(",")]
        body2 = FILLER * calibrate(cl, L)
        emit(leg(cl, C, lambda i: ("[pool%d]" % i) + " " + body2 + " Summarize the tradeoffs above.",
                128), "pool-wall C%d L%d" % (C, L))
    print("PROBE-DONE", flush=True)


if __name__ == "__main__":
    main()
