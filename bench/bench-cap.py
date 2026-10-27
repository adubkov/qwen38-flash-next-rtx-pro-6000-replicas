#!/usr/bin/env python3
"""Phase 0.1 C1 decode client for flash-next (on the serving host).

Lineage: flashnext-pub-config/bench-pub.py (produced the documented 119.8 tok/s).
Kept identical on purpose: same essay prompt bytes, temperature 0.0,
max_tokens 1200, decode window measured strictly first-content-token ->
last-content-token so TTFT/prefill never enters the rate.
Changed: target 127.0.0.1:30071 + Bearer auth, explicit chat_template_kwargs
(otherwise every leg silently inherits --default-chat-template-kwargs and the
legs are not comparable), per-request epoch stamps, dropped prefill section.

Runs on the serving host, sequentially (C1 = one stream at a time, back-to-back
with no gap so --sleep-on-idle and clock ramp-down stay out of the data).
"""
import argparse, hashlib, json, os, sys, time, urllib.request

# Verbatim from bench-pub.py -- do not "improve" this, it is the comparability anchor.
DECODE_MSGS = [{"role": "user", "content":
    "Write a very long, detailed, multi-section technical essay on distributed systems, "
    "covering consensus, replication, sharding, and failure recovery. Keep generating "
    "substantive detail for the entire allowed length."}]


def post(base, key, payload):
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return urllib.request.urlopen(req, timeout=1800)


def one_request(base, key, model, max_tokens, ctk):
    payload = {"model": model, "messages": DECODE_MSGS, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": ctk}
    t0 = time.time()
    first = last = None
    completion = prompt = 0
    text = []
    with post(base, key, payload) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            u = obj.get("usage")
            if u:
                completion = u.get("completion_tokens", completion)
                prompt = u.get("prompt_tokens", prompt)
            ch = obj.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                c = d.get("content") or d.get("reasoning_content")
                if c:
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
                    text.append(c)
    total = time.time() - t0
    # decode-only window: exclude prefill/TTFT entirely
    win = (last - first) if (first and last and last > first) else 0.0
    body = "".join(text)
    return {"t0": round(t0, 3), "t_first": round(first, 3) if first else None,
            "t_last": round(last, 3) if last else None, "total_s": round(total, 3),
            "decode_window_s": round(win, 3), "completion_tokens": completion,
            "prompt_tokens": prompt,
            "decode_tps": round(completion / win, 2) if win > 0.05 else None,
            "text_sha1": hashlib.sha1(body.encode()).hexdigest()[:12],
            "chars": len(body)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:30071/v1")
    ap.add_argument("--model", default="pennyroyal")
    ap.add_argument("--key-file", required=True)   # no default: it is a credential path
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--warmup", type=int, default=2, help="discarded leading requests")
    ap.add_argument("--requests", type=int, default=12)
    # Pin the server default explicitly so legs are comparable across runs.
    ap.add_argument("--ctk", default='{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.key_file) as f:
        key = f.read().strip()
    if not key:
        sys.exit("empty api key")
    ctk = json.loads(a.ctk)

    print("warmup %d (discarded)" % a.warmup, flush=True)
    for i in range(a.warmup):
        r = one_request(a.base, key, a.model, a.max_tokens, ctk)
        print("  warm%d tps=%s tok=%s" % (i + 1, r["decode_tps"], r["completion_tokens"]), flush=True)

    print("measured %d" % a.requests, flush=True)
    with open(a.out, "w") as fh:
        for i in range(a.requests):
            r = one_request(a.base, key, a.model, a.max_tokens, ctk)
            r["i"] = i
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print("  req%d tps=%s tok=%s sha=%s" % (i, r["decode_tps"], r["completion_tokens"], r["text_sha1"]), flush=True)
    print("wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
