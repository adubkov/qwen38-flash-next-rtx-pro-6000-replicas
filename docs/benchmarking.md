# How to measure this box without fooling yourself

Everything here exists because we got it wrong first. Each rule below is priced by
an incident that cost a run — most of them a run we initially believed.

## Required environment

The harnesses deliberately ship **no defaults** for these. A guessed value here
produces a bench that reports success, which is worse than one that refuses to run.

| variable | used by | why it has no default |
|---|---|---|
| `SELF_IPS` | `bench-agent-loop.py` | the source addresses *your* requests arrive from. Empty ⇒ every request counts as foreign. |
| `FOREIGN_IPS` | `bench-agent-loop.py`, `matrix-gated-rerun.py` | peers that may legitimately touch the replica. Empty ⇒ **nothing** counts as foreign, so every cell certifies clean forever. |
| `BENCH_SSH_HOST` | `bench-proxy.py` | how the purity gate reads the replicas' request logs. A guessed host returns an empty log ⇒ "no contamination". |
| `PROXY_KEY` | `bench-proxy.py` | key for the proxy route, distinct from the engine key. |
| `--key-file` | all engine-side harnesses | file (or directory containing one) holding the engine API key. It is bind-mounted into the containers as `/api-key`, so it never appears in argv. |

Both purity gates and the sampling gate were verified to *fail* — refuse to start
with the variable missing — not merely to be documented.

## The rules

**1. Drain the replica under measurement from the proxy. Bypassing the proxy is necessary and not sufficient.**
Pointing a bench at `:30073` does not stop a session-sticky proxy from routing
*your own assistant traffic* to that same card. Measured 2026-10-05: 24 real
generations of 9.4–106 s in a 20-minute window, all sticky, overlapping **8 of 12
control cells** — voiding a control arm after the fact.

The remedy that worked:
- remove the measured replica's upstream entry for the **whole** window (there is
  usually no `upstream.disabled` knob; `health.disabled` only means "never exclude
  via probing"),
- pre-validate the edited config in a throwaway container (`--check-config`) before
  applying it,
- **recreate** the proxy, do not restart it: a single-file bind mount tracks the host
  *inode*, so a restart can keep serving the old file,
- drive the bench at the replica's own published port so the proxy's per-card
  concurrency cap doesn't bind,
- leave the **other** replica in the pool — it serves whatever is driving the trial,
  and you do not want it recreated as collateral.

**2. Sample the card at ~1 Hz for the whole window.** Power draw, SM clocks,
temperature, utilisation. Clocks drift 2302 → ~1965 MHz *within* a single run at
the cap, and sustained draw sagged 294 → 269 W across a bracket whose members were
supposed to be identical. A cell with no clock/power record cannot be attributed
to a power state and is not publishable — `--gpu-uuid` is therefore mandatory in
`bench-agent-loop.py`, with `--no-gpu-sample` as a loud, exploratory-only escape.

**3. Never bench both cards at once.** The second card hits its cap first, so the
pair's numbers become a function of which card throttled when. Benching GPU1 while
GPU0 ran production chat produced a spurious −13% and taught us this the hard way.
One replica measured, the sibling drained or idle.

**4. Know which columns are invalid.** At C≥2 with context ≥65k the `aggregate`
figure is dominated by a sibling stream's prefill inside the 128-token decode
window. The shipped row for `C2 65k` (`bench/baseline/matrix-post-swap.csv:13`)
prints per-stream **112.3** against aggregate **62.3** — aggregate *below*
per-stream, and flagged `contaminated_accepted`. (RESULTS §3's C2/65k row, a
different run, prints its aggregate as `—` for the same reason.) Right instrument:
time-per-output-token over windows containing no prefill, which is why the agent
harness reports decode and TTFT separately.

**5. The purity gate can lie in three specific ways.** All three happened on
2026-10-05, all printed `CLEAN`:
- reading proxy logs on the **wrong host** (the proxy ran on a different machine than
  the engine, so the stream was empty and every cell looked pristine);
- a `pgrep -f <harness>` liveness check inside the remote command that **matches its
  own command line**, reporting a dead job as still alive — use
  `ps -eo cmd | grep "[b]ench-matrix"`;
- a foreign-request counter that includes the bench's own streams, ignores
  `GET /health`, or reads only one docker-log stream (access lines ride stdout,
  engine metrics ride stderr).

Bench streams are **not loopback**: connecting from the host to a published port
means docker-proxy is the peer, so the engine logs the bridge-gateway address. A
bare "non-loopback POST" counter blamed 6–8 foreign requests on a 7-second cell
that contained only the bench's own 4 streams. Print a **per-source breakdown** and
gate on the explicit `SELF_IPS`/`FOREIGN_IPS` split.

**6. A health probe can be real inference.** Some dashboards' "is it alive?" check
is an actual `POST /v1/chat/completions` at 4–5/min with bursts to 17/min. It takes
a scheduler slot and interleaves prefill, and it is invisible to a stderr-only gate.
Enumerate what else points at the endpoint before trusting a quiet window.

**7. Bracket before you claim parity.** Repeat one cell of the **same** config and
treat that spread as the noise floor. The bracket the measured record carries is in
RESULTS §2: a same-config control (2×1k and 2×65k, one arm re-run) that moved
**1.4%** and **0.4%**. The four-figure set quoted at 1k/4k/65k/262k (1.4 / 1.9 /
0.4 / ~15–18%, RESULTS reading rule 3) is attributed there to the paired A/B's own
drift bracket — no repeat-run dataset for either lands in this tree, and the
"median of 96 same-config replays across 10 generations" this rule used to cite is
backed by no shipped artifact, so it is withdrawn. The one noise statement this repo
does ship is `bench/baseline/BASELINE-PIN.md:124`: long-context cells carry ±30%
per-attempt noise; compare medians across attempts, never single cells. Nothing
below the band is a finding. A single run per cell supports "nothing catastrophic",
never "equal".

**8. Fresh output directory per arm.** The matrix harnesses resume per outdir;
reusing a directory silently compares an arm against cells measured in an **earlier
boot**. Two arms must not share a journal.

**9. Reject a first-cell outlier against its own same-fixture sibling.** The first
cell of a fresh boot read 162.1 tok/s where its sibling read 221.5 (−27%) after
~2 s of a nominal 20 s stall. The instrument was healthy — median 1965 MHz, 296 W —
so the card was at its cap, not idle; median over six same-fixture C1 cells sits at
227. That row is excluded as a rejected outlier, and re-runs are the only
explanation allowed to stand.

**10. Attach at token granularity for agent workloads.** A first version of the
agent harness read the streamed response **line-buffered**, flushing only at turn
end — which folded inter-token gaps into TTFT and quadratic-sized the context
growth. Its output directory is still named `…INVALID-v1-quadratic-sizing` on the
box to keep the lesson attached to the artifact.

## Running the harnesses

Real flags only, taken from the files in `bench/`.

```bash
# 1. Engine-side C×context matrix, one replica, direct on its published port
python3 bench/bench-matrix.py \
    --base http://127.0.0.1:30071 --cs 1,2,4 \
    --ctxs 1024,4096,16384,65536,131072,262144,393216,520192 \
    --container flash-next-pair-flash-next-a-1 \
    --key-file /path/to/api-key --outdir /var/tmp/phase0/matrix

# 2. Pair-wide ladder through the session-sticky proxy (the run we still owe)
BENCH_SSH_HOST=<serving-host> PROXY_KEY=<key> \
python3 bench/bench-proxy.py --cs 1,2,3,4,5,6,7,8 --outdir ./matrix-proxy

# 3. Agent-workflow growth bench — refuses to run without the card instrumented
SELF_IPS=<bridge-gw> FOREIGN_IPS=<peer…> \
python3 bench/bench-agent-loop.py \
    --base http://127.0.0.1:30073 --container flash-next-pair-flash-next-b-1 \
    --gpu-uuid $(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n 2p) \
    --depths 2048,16384,65536,262144,507904 --gen 512 --arms warm,cold \
    --outdir ./agentloop-v3

# 4. Single-point capacity probe (12 requests, 2 discarded warm-ups)
python3 bench/bench-cap.py --base http://127.0.0.1:30071/v1 --out ./cap.jsonl

# 5. The paired shared-vs-private trial driver: boots the trial container,
#    benches inside the trial window, then restores production under an EXIT trap
bash bench/trial-boot.sh
```

Analyzers take the journal: `analyze-agent-loop.py`, `analyze-cap.py`, and
`probe-clock-attribution.py` (which is what proved the 2-second stall in rule 9
above — the row is also excluded in RESULTS §5 — was a request arriving while the
card was at the cap, not an idle card).

## What counts as publishable

A number enters `RESULTS.md` only if it has: its **scope** (one replica or the pair),
its **cap**, its **n**, its fixture, and — for anything called a delta — a same-config
drift bracket beside it. Publish only from a **completed** run, and give every run its
own tree; a partial run that inherits a previous tree will be mistaken for a finished
one and published anyway.
