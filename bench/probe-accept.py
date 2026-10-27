#!/usr/bin/env python3
"""Speculative-decode acceptance probe, reusable across config variants.
Appends one JSON row per prompt to --out, tagged with --tag (the config under test).
accept comes from the container's own `Decode batch` log lines (this build exposes no
spec metrics on /metrics); ms/step is derived, since tok/s = 1000*accept/ms_step."""
import argparse, json, re, statistics as st, subprocess, time, urllib.request

CTR = "flash-next-pair-flash-next-a-1"
PROMPTS = [
    ("echo-200x", "Repeat exactly this line 200 times, nothing else:\nhello world\n"),
    ("nums-1..400", "Output the integers from 1 to 400 separated by commas, begin now:"),
    ("countries-300", "List 300 country names, one per line, begin now:"),
    ("json-array-150", 'Output a JSON array of 150 objects {"id":N,"name":"item N","ok":true} for N=1..150:'),
    ("code-15fn", "Write 15 small Python functions (add,sub,mul,div,mod,pow,neg,abs_,sqr,cube,inc,dec,dbl,half,sq) "
                  "each with a one-line docstring and a return:"),
    ("essay", "Write a very long detailed technical essay on distributed systems: consensus, replication, "
              "sharding, failure recovery."),
    ("poem", "Write an avant-garde poem in which every line is a genuinely unexpected image, using rare vocabulary."),
]


def call(key, text, maxtok, thinking):
    p = {"model": "pennyroyal", "messages": [{"role": "user", "content": text}], "max_tokens": maxtok,
         "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": thinking, "preserve_thinking": thinking}}
    r = urllib.request.Request("http://127.0.0.1:30071/v1/chat/completions", data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time(); first = last = None; comp = 0
    with urllib.request.urlopen(r, timeout=900) as resp:
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
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--thinking", default="off")
    a = ap.parse_args()
    key = open(a.key_file).read().strip()
    think = a.thinking == "on"
    call(key, "warmup: reply with one word", 16, think)
    fh = open(a.out, "a")
    for name, txt in PROMPTS:
        t0, first, last, comp = call(key, txt, a.max_tokens, think)
        win = (last - first) if (first and last and last > first) else 0.0
        tps = comp / win if win > 0.05 else 0.0
        v = accepts_since(t0)
        acc = st.mean(v) if v else None
        row = {"tag": a.tag, "prompt": name, "completion_tokens": comp,
               "decode_tps": round(tps, 1), "accept_len": round(acc, 3) if acc else None,
               "accept_n": len(v),
               "ms_per_step": round(1000 * acc / tps, 2) if (acc and tps) else None,
               "thinking": a.thinking, "ts": int(t0)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("%-16s tps=%-7s accept=%-6s n=%-4s ms/step=%s" % (
            name, row["decode_tps"], row["accept_len"], row["accept_n"], row["ms_per_step"]), flush=True)
    fh.close()


if __name__ == "__main__":
    main()
