#!/usr/bin/env python3
"""Does /flush_cache actually evict, and can a turn reuse more than the conversation put there?

The 26-cell ladder has one internally inconsistent pair: the warm pass flushed before its
first cell, that first cell read cached=0 (so the flush took), yet the NEXT cell read
cached=4,160 -- more tokens than the tree could hold, since only a 2,186-token prompt plus a
512-token completion had been inserted since. The only source for 4,160 is the pre-flush
bootstrap probe (120 lines = 4,240 tokens, byte-identical to that cell).

This script reproduces just that sequence with the bench's OWN builders (exec'd from its
source, so the bytes are identical) and prints the engine's ReqTimeStats per step:
  1. probe  = 120-line conversation   (fills the tree)
  2. flush                            (the disputed claim)
  3. cell1  = 60-line conversation    (a prefix of probe's chunk 0 -> reads 0 if eviction took)
  4. cell2  = 120-line conversation   (byte-identical to probe)
Verdict lines: FLUSH-EFFECTIVE / FLUSH-INEFFECTIVE and PROBE-RESIDUE-CONFIRMED.
"""
import json, os, subprocess, time, urllib.request

# Env, not constants. These named paths and a container on the single machine this
# was written on AND were read at import time, so importing the file raised
# FileNotFoundError before a single argument was parsed: unrunnable anywhere else.
BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:30073")
CTR = os.environ.get("BENCH_CONTAINER", "")
HARNESS = os.environ.get("BENCH_HARNESS", "bench/bench-agent-loop.py")
KEY_FILE = os.environ.get("BENCH_KEY_FILE", "")
key = None
build_messages = None
engine_reqstats = None


def _load_harness():
    """Take build_messages/engine_reqstats from the real harness rather than a copy
    of them that can drift away from what production actually sends."""
    if not os.path.exists(HARNESS):
        raise SystemExit("BENCH_HARNESS=%r does not exist (the default is repo-relative: run "
                         "from the repo root, or point it at bench-agent-loop.py)" % HARNESS)
    src = open(HARNESS).read()
    if "main()" not in src:
        raise SystemExit("BENCH_HARNESS=%r has no `main()` marker: refusing to exec an "
                         "unknown slice of it" % HARNESS)
    i = src.rindex("main()")
    ns = {"__name__": "notmain"}
    exec(compile(src[:i], HARNESS, "exec"), ns)
    return ns["build_messages"], ns["engine_reqstats"]


def _read_key():
    if not KEY_FILE:
        raise SystemExit("BENCH_KEY_FILE is unset. Without a key every request here 401s, and "
                         "the verdicts below would then read cached=0 as a real result.")
    k = open(KEY_FILE).read().strip()
    if not k:
        raise SystemExit("BENCH_KEY_FILE=%s is empty: requests would authenticate as nothing"
                         % KEY_FILE)
    return k


def post(path, payload=None):
    """flush_cache answers with a bare string, not JSON -- don't parse it."""
    data = json.dumps(payload).encode() if payload is not None else b"{}"
    r = urllib.request.Request(BASE + path, data=data, method="POST",
                               headers={"Content-Type": "application/json",
                                        "Authorization": "Bearer " + key})
    with urllib.request.urlopen(r, timeout=600) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")[:80]


def stream(messages, maxtok):
    payload = {"model": "m", "messages": messages, "max_tokens": maxtok, "temperature": 0,
               "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False}}
    data = json.dumps(payload).encode()
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=data, method="POST",
                               headers={"Content-Type": "application/json",
                                        "Authorization": "Bearer " + key})
    usage = {}
    with urllib.request.urlopen(r, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            b = line[5:].strip()
            if b == "[DONE]":
                break
            try:
                d = json.loads(b)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
    return usage


def note(tag, usage, rs, extra=""):
    print("%-8s pt=%-7s cached=%-7s new=%-6s reuse=%-6s prefill_ms=%-9s q_ms=%-7s %s" % (
        tag, usage.get("prompt_tokens"), (rs or {}).get("cached_input_len"),
        (rs or {}).get("new_input_len"),
        round((rs or {}).get("cached_input_len", 0) / max(1, usage.get("prompt_tokens") or 1), 3)
        if rs else None,
        (rs or {}).get("initial_prefill_ms"), (rs or {}).get("queue_ms"), extra), flush=True)
    return usage.get("prompt_tokens"), (rs or {}).get("cached_input_len")


def flushed_at():
    o = subprocess.run(["docker", "logs", "--since", "3m", CTR], capture_output=True,
                       text=True, errors="replace")
    t = (o.stdout or "") + "\n" + (o.stderr or "")
    return t.count("Cache flushed successfully!")


def main():
    global key, build_messages, engine_reqstats
    build_messages, engine_reqstats = _load_harness()
    key = _read_key()
    if not CTR:
        raise SystemExit("BENCH_CONTAINER is unset: every engine-log read below would come back "
                         "empty and be reported as cached=0 / no flush observed")
    print("flush-count-before=%d" % flushed_at())
    t0 = time.time()
    u = stream(build_messages(120, 60), 1)
    rs, _ = engine_reqstats(CTR, u.get("prompt_tokens"), t0 - 2, int(time.time() - t0) + 10)
    p1, c1 = note("probe", u, rs, "(tree should now hold this chain)")

    t0 = time.time()
    st, body = post("/flush_cache")
    n = flushed_at()
    print("flush    status=%s body=%r flush-lines-last3m=%d" % (st, body, n), flush=True)

    t0 = time.time()
    u = stream(build_messages(60, 60), 1)
    rs, _ = engine_reqstats(CTR, u.get("prompt_tokens"), t0 - 2, int(time.time() - t0) + 10)
    p3, c3 = note("cell1", u, rs, "(60 lines: prefix of the probe's chunk 0)")

    t0 = time.time()
    u = stream(build_messages(120, 60), 1)
    rs, _ = engine_reqstats(CTR, u.get("prompt_tokens"), t0 - 2, int(time.time() - t0) + 10)
    p4, c4 = note("cell2", u, rs, "(byte-identical to probe)")

    print()
    # An instrument that could not read the log yields None. Printing `c3 == 0` turned
    # that into `False`, so "flush is broken" and "I could not read anything" printed
    # identically -- the exact confusion bench-agent-loop.py documents as forbidden.
    unk = "unknown - instrument read failed"
    print("VERDICT flush-took-effect = %s   (cell1 read cached=%s; a surviving probe chain "
          "would have given ~%s)" % (unk if c3 is None else str(c3 == 0), c3, (p1 or 0) // 64 * 64))
    print("VERDICT probe-residue-survived-flush = %s  (cell2, identical to the pre-flush probe, "
          "read cached=%s)" % (unk if c4 is None else str((c4 or 0) > ((p3 or 0) // 64 * 64 or 0) + 64), c4))
    print("VERDICT ladder-low-rows-inflated = %s  (4,160/8,256 are THIS run's probe-prefix "
          "lengths, not a universal constant -- recompute if the prompt builder changes)"
          % (unk if c4 is None else str(c4 == 4160)))
    print("elapsed=%.1fs" % (time.time() - t0))


main()
