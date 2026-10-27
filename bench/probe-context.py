#!/usr/bin/env python3
"""Long-context acceptance probe for flash-next (post-swap question: does a healthy
draft KEEP acceptance as prompt context grows?). Three tiers of the SAME structured
task (count 1..400 — the known accept discriminator) behind filler text of ~0/8k/32k
tokens. Leading unique SESSION marker per (tag,tier) defeats radix prefix sharing
(bench-matrix lesson: markers must be at the LEAD and never reused across arms).
Accept from the container's own `Decode batch` log lines (this build has no /metrics
accept series); tps = decode window (first->last token) only. C1, greedy, thinking off.
Runs on the serving host (needs docker CLI for logs); zero GPU work of its own."""
import argparse, json, re, statistics as st, subprocess, time, urllib.request

CTR = "flash-next-pair-flash-next-a-1"
TIERS = [("short", 0), ("8k", 520), ("32k", 2100)]   # filler note count, ~15 tok/note

def filler(n, salt):
    return " ".join(f"Note {salt}-{i}: the quick brown fox jumps over the lazy dog." for i in range(n))

def call(key, text, maxtok):
    p = {"model": "pennyroyal", "messages": [{"role": "user", "content": text}], "max_tokens": maxtok,
         "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}}
    r = urllib.request.Request("http://127.0.0.1:30071/v1/chat/completions", data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time(); first = last = None; comp = 0
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for raw in resp:
            l = raw.decode("utf-8", "replace").strip()
            if not l.startswith("data:"):
                continue
            d = l[5:].strip()
            if d == "[DONE]":
                break
            try:
                o = json.loads(d)
            except Exception:
                continue
            if o.get("usage"):
                comp = o["usage"].get("completion_tokens", comp)
            ch = o.get("choices") or []
            if ch:
                dl = ch[0].get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content"):
                    n = time.time(); first = n if first is None else first; last = n
    return t0, first, last, comp

def accepts_since(t0):
    secs = max(1, int(time.time() - t0) + 3)
    out = subprocess.run(["docker", "logs", "--since", "%ds" % secs, CTR],
                         capture_output=True, text=True).stderr
    return [float(m.group(1)) for m in
            (re.search(r"accept len:\s*([\d.]+)", x) for x in out.splitlines() if "Decode batch" in x) if m]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", required=True)   # no default: it is a credential path
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = open(a.key_file).read().strip()
    call(key, "warmup: reply with one word", 16)
    fh = open(a.out, "a")
    for tier, n in TIERS:
        text = f"SESSION {a.tag} T{tier} " + (filler(n, a.tag + tier) + " " if n else "") + \
               "Output the integers from 1 to 400 separated by commas, begin now:"
        t0, first, last, comp = call(key, text, 600)
        win = (last - first) if (first and last and last > first) else 0.0
        tps = comp / win if win > 0.05 else 0.0
        v = accepts_since(t0)
        acc = st.mean(v) if v else None
        row = {"tag": a.tag, "tier": tier, "filler_notes": n,
               "approx_prompt_tokens": int(len(text) / 3.9),
               "completion_tokens": comp, "decode_tps": round(tps, 1),
               "accept_len": round(acc, 3) if acc else None, "accept_n": len(v),
               "ms_per_step": round(1000 * acc / tps, 2) if (acc and tps) else None,
               "thinking": "off", "ts": int(t0)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("%-6s ptok~%-6d tps=%-7s accept=%-6s n=%-3d ms/step=%s" % (
            tier, row["approx_prompt_tokens"], row["decode_tps"], row["accept_len"],
            row["accept_n"], row["ms_per_step"]), flush=True)
        time.sleep(3)
    fh.close()

if __name__ == "__main__":
    main()
