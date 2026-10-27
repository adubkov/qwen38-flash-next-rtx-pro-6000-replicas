# Measured record

Every number in this file carries **scope** (one replica or the pair), **cap**
(the GPU power limit it was taken at) and **n** (repeats). Those three are not
decoration: the same configuration produces different numbers when a card is at
a different power cap, and an `aggregate` column computed across a stream that
also contains a prefill is not a decode number at all.

**Reading rules**

1. **Scope.** Everything **measured here** is one replica, benched directly on its
   published port with the sibling drained from the proxy. The only rows in this
   file labelled `pair` are the three third-party citations in §8 — they are not
   our measurements. No whole-pair number produced by our own harness appears in
   this file at all. The `aggregate` column is the sum of the C concurrent streams
   on that **one** card. Two cards would be roughly double — and that measurement
   does not exist yet on this configuration (see [Open items](#open-items)).
2. **Cap.** All of it is **300 W** data. The live cap on the box can drift away
   from what its systemd unit declares, so before comparing anything to these
   rows, read `nvidia-smi --query-gpu=index,power.limit,power.draw,clocks.sm` and
   reconcile. Appendix A is what a cap difference is actually worth.
3. **n.** The paired shared-vs-private table is **n=1 per cell**. Its
   same-config drift bracket (a repeat of the *same* config, which is what bounds
   the instrument) was 1.4%, 1.9%, 0.4% and ~15–18% at 1k/4k/65k/262k. So only
   the 1k–65k cells bound a regression at about 3%; 262k and every C≥2 cell
   resolve nothing below ±8–15%. Those rows mean "nothing catastrophic", never
   "parity certified".
4. **Invalid columns.** At C≥2 and context ≥65k the `aggregate` column is
   dominated by a sibling stream's prefill landing inside the 128-token decode
   window. It is printed as `—` wherever that applies. Read
   [benchmarking.md](benchmarking.md) before adding a row.

---

## 1. The single-replica contract

Source artifact: `bench/baseline/BASELINE-PIN.md` (identity pins + this table),
with the derived CSVs beside it.

Scope: 1 replica (`:30071`, direct) · cap 300 W · n=1 · 2026-10-03 00:5xZ ·
**each replica pinning its own table** (pre-shared) · pool 824,384 @ 0.990 ·
thinking off, temperature 1.0, 128 completion tokens, `inference_context`
prefix replayed across 11 warm-up runs.

| ctx | dec tok/s | prefill tok/s | accept | memory.current |
|---|---|---|---|---|
| 1,024 | 192.0 | 1,882 | 1.500 | 107.35 GiB |
| 4,096 | 188.8 | 6,740 | 1.428 | 107.50 |
| 16,384 | 192.3 | 24,769 | 1.452 | 107.67 |
| 65,536 | 178.7 | 78,199 | 1.494 | 107.58 |
| 131,072 | 180.4 | 168,648 | 1.443 | 107.84 |
| 262,144 | 180.7 | 164,704 | 1.423 | 107.82 |
| 393,216 | 190.2 | 140,421 | 1.451 | 107.84 |
| 428,208 | 179.7 | — | 1.405 | — |

Two things to take from it: decode is **flat to 428k on one card** (mean 185,
sd 4.8, zero slope against context — the hybrid's recurrent state instead of KV),
and **prefill saturates at ~160–165k tok/s by 131k**. The accept column is the
symptom of the artifact defect described in `architecture.md`; absolute speed
below `1.42` is stale for today's build.

## 2. What the artifact swap bought (not the shared table)

Scope: 1 replica · cap 300 W · n=1 · 2026-10-03 ~08:00Z. Same boot, same fixture,
one variable: the **artifact swap** to a BF16-MTP reshard. The fp8-serialised MTP
draft was loading without its expert scales (all 1,536 `weight_scale_inv` dropped),
which is the closed root cause of the flat ~1.42 accept length — see
[architecture.md](architecture.md) §3. The same reshard also raised the PLE vocab the
recipe reads — the 151,936 → 155,691/157,312 pair, a claim no shipped artifact
carries ([LICENSING.md](../LICENSING.md), "One claim we cannot source"); that second
change was never isolated by A/B, so attribute the delta to the draft fix, not to
the vocab line.

| metric | pre-swap | post-swap | delta |
|---|---|---|---|
| C1/1k decode | 192.0 | 213.9 | **+11.4%** |
| accept length | 1.42 | 2.49 | +0.62 spec tokens/iter |
| C1/1k TTFT | 599 ms | 436 ms | −27% |

Post-swap accept runs ~2.4–2.8 across the matrix. A same-config control bracket
(2×1k and 2×65k, one arm re-run) moved 1.4% and 0.4%, so it brackets the
instrument's noise and nothing in that span is signal.

## 3. Shared table vs private table — the paired A/B

Scope: 1 replica per row (**replica B**) · cap 300 W · **n=1 per cell** ·
2026-10-04 00:54–01:11Z · both arms same boot, alternating, fresh outdir each.
Topology to keep straight: **A ran the private table throughout** (its
`created` and `restarts` were the reproducibility control), so this is
B-shared-versus-B-private, not a whole-pair A/B.

**How the `agg delta` column was made — read this before quoting a delta.** It is a
derived figure, and this table does not derive it the same way for every row:

- **Aggregate-derived (7 rows):** C1 1k, C1 4k, C1 65k, C1 262k, C2 1k, C2 4k, C4 1k —
  computed from the two `agg` values printed in the row.
- **Decode-derived (4 rows):** C2 65k, C2 262k, C4 4k, C4 65k — these print `—` for
  aggregate (reading rule 4), so their delta comes from the two per-stream columns
  instead. The column name is a misnomer for them.
- Recomputing each row from the two values printed beside it: **10 of 11 reproduce**;
  the closest miss is `C1 4k`, printing −2.2% where its pair gives −2.1%.
- **`C2 4k` does not reproduce from either pair and is left as printed.** Its
  aggregates (230.6 / 362.0) give −36.3% and its decode pair (191.1 / 193.7) gives
  −1.3%; neither is the −8.7% in the row. The run journal that would settle it is not
  in this repo, so it is flagged, not reconciled.
- **`C4 65k` carried a sign error.** It printed +4.1% where its own printed pair
  (89.4 → 93.2) is a decrease; that is unambiguous from the row, so it is corrected
  to **−4.1%** here. This is the only value this pass changed.

| cell | shared dec / agg | private dec / agg | agg delta | accept sh/pr |
|---|---|---|---|---|
| C1 1k | **236.0** / 236.0 | 227.6 / 227.6 | +3.7% | 2.675 / 2.665 |
| C1 4k | 231.5 / 231.5 | 236.4 / 236.4 | −2.2% | 2.42 / 2.476 |
| C1 65k | 216.7 / 216.7 | 212.5 / 212.5 | +2.0% | 2.42 / 2.611 |
| C1 262k | 226.7 / 226.7 | 236.4 / 236.4 | −4.1% | 2.363 / 2.381 |
| C2 1k | 202.6 / **402.3** | 191.4 / 372.2 | +8.1% | 2.62 / 2.657 |
| C2 4k | 191.1 / 230.6 | 193.7 / 362.0 | −8.7% | 2.478 / 2.557 |
| C2 65k | 106.4 / — | 124.8 / — | −14.7% | 1.323 / 2.035 |
| C2 262k | 94.1 / — | 87.9 / — | +7.1% | 1.545 / 1.317 |
| C4 1k | 150.4 / **589.2** | 154.7 / 606.2 | −2.8% | 2.553 / 2.503 |
| C4 4k | 132.2 / — | 136.5 / — | −3.2% | 2.498 / 2.481 |
| C4 65k | 89.4 / — | 93.2 / — | −4.1% | 1.714 / 1.693 |

Best shared-mode figures on record: **236.0 tok/s per stream** (C1/1k),
**589.2 tok/s single-card aggregate** (C4/1k), accept ~2.5.
Cached-prefix prefill, control arm: 8.1k / 27.4k / 221k / 283.8k tok/s at
1k / 4k / 65k / 262k — the 65k number is the KV-reuse headline.

**Retraction.** A first pass of these notes read the C2/1k `+8.1%` as an
acceptance/throughput win. It does not survive contact with the run's own noise
floor: the three C4 cells moved −2.8%, −3.2% and −4.1% at their own contexts, so
mean |aggregate delta| across the 11 cells is **5.52%**, recomputed from the table
above, and an 8% claim cannot be made from n=1 rows whose sibling cells swing
either side of that. **Withdrawn figure:** this retraction first printed that mean
as **2.55%**; that was a mis-computation, not a different measurement, and the
corrected 5.52% is what the table yields. The retraction's conclusion is unchanged
— if anything the corrected arithmetic weakens any speed story further, since the
cells are further from zero and in both directions than the withdrawn figure
implied. Shared PLE is a memory result.

## 4. What shared PLE actually buys — the certified claim

| quantity | before | after |
|---|---|---|
| private page-locked PLE table | 47.68 GiB **per replica** | one shared tmpfs table |
| replica cgroup `anon` | 53.35 GiB | ~6 GiB under load (1.4–1.6 GiB idle) |
| host RAM used | 179 GiB | 89 GiB |
| host RAM available | 64 GiB | 155 GiB |
| both replicas on one table | — | identical `inode=3` in both boot logs |
| table registration | — | 2.1 s (13 s with a second replica concurrently hashing) |
| sha256 verify through the read-only mapping | — | 12.9 s |
| `oom_kill` at the 24 GiB caps | — | 0, across a real cold boot |

This is the claim that is certified: it is a direct measurement of a state
change, reproduced at a cold boot, not a throughput delta inferred from a noisy
cell.

## 5. Post-promotion per-card ladders (both replicas shared)

Scope: 1 replica at a time, sibling drained · cap 300 W · n=1 per cell ·
2026-10-06 01:0xZ · pool 983,360 @ 0.998.

| cell | value |
|---|---|
| C1, any context | **195–255 tok/s** per stream |
| C2, ≤16k | 87–176 tok/s per stream |
| C2 @ 491,456 | achieved concurrency 2.0, 87.7 tok/s per stream, TTFT 152 s |
| C≥4 | achieved concurrency **saturates at 3.0** |
| prefill (throttled) | ~8,000 tok/s |
| TTFT model | ≈ Σprompt-tokens ÷ ~8k tok/s |

The 3.0 ceiling is structural, not a knob: max **3** concurrent streams above
16k per card. Raising `--max-running-requests` cannot exceed it — the admission
engine won't admit a fourth. One row is deliberately excluded: a first cell
reading 162.1 tok/s where its same-fixture sibling read 221.5 (−27%) after ~2 s
of a 20 s stall. The instrument was healthy (median 1965 MHz, 296 W); median over
six same-fixture C1 cells sits at 227, so that cell is a **rejected outlier**,
not a result, and re-runs are the only admissible explanation test.

## 6. Agent-workload behaviour under shared PLE

Scope: 1 replica · cap 300 W · n=1 per cell · 2026-10-06 · 8 turns, 192-token
output cap, 8k context budget, attached at token granularity.

| observation | value |
|---|---|
| C1 decode, 2k → 508k context | **152–187 tok/s** (~10% spread) |
| steady state at ~500k | TTFT **2.0–3.8 s/turn** |
| cache reuse at ~500k | **0.976–0.992** |
| whole 1k→508k conversation, warm vs cold | **94.6 s vs 339.6 s (3.6×)** |

Decode is nearly **depth-blind**: context length costs you time-to-first-token,
almost never decode rate. The apparent collapse to 64 tok/s at 48k was a
measurement artifact (a `max_tokens` refusal-retry loop, one turn counted twice)
and is not a data point. The correct reading is: **48k is the depth at which a
cache miss starts costing 30+ seconds**, which is why a mis-routed agent turn
feels broken rather than merely slow.

## 7. Concurrency arms trial (shared pair, one card)

Scope: 1 replica · cap 300 W · 2026-10-05 · arms
`--max-running-requests 4 --max-mamba-cache-size 24` (live) vs `8 / 48`.

| context | aggregate at 8/48 | verdict |
|---|---|---|
| 1,024 | +3.5% | tie |
| 4,096 | **+36.7%** | real (pre-pool was ~1/40 of demand, mamba 6/8) |
| 16,384 | +6.8% | marginal |
| 65,536 | −8.2% | real loss |
| 262,144 | −0.1% | tie, n=1 |

**Not adopted.** The 65k loss was real and unexplained at the time, and the
steadiest explanation is eviction pressure (mamba slots 24→48 on 11.8 GiB of
device buffers). The honest summary: at the live cap of 4 requests per card,
admission was never binding for C ≤ 4 — every one of those cells had *identical*
engine admission (C=2 executing as 2 streams, max single-stream 239.8), so
nothing there was an admission win.

## 8. Public anchors (cited, not measured here)

Quoted so a reader can position the box; the methodology differs from ours, so
these are context, not a comparison table. **Every row is marked †: cited from our
private notes, with no URL, run id, or date carried here — not independently
checkable in this repo.** The one public anchor that *is* checkable is credited by
name elsewhere: the 231 tok/s single-card headline of
`gabrielolympie/sglang-flashnext-sm120` ([LICENSING.md](../LICENSING.md)).

| source | configuration | number |
|---|---|---|
| H3PO Labs (SM120 pair) † | single card | C1 ≈ 188 tok/s |
| H3PO Labs (SM120 pair) † | pair | C8 ≈ 1,110 tok/s |
| untcoder2 (SM120 pair) † | pair, thinking off | C6 775.8 (3.18×), C8 821.5 (3.37×) vs single card |
| untcoder2 (SM120 pair) † | pair, thinking on | C14 1,093.4 |

Both report the same qualitative shape we do: pair scaling is real at short
context and per-stream decode barely degrades until admission binds.

---

## Open items

Stated plainly because the absence is load-bearing:

1. **No pair-wide measurement exists on the shared configuration.** The only
   pair-wide ladder we have (C4/1k 690.9 agg ×2.37; C8/1k **1,138.8** ×3.9,
   per-stream still ~157) was taken 2026-10-03 ~21:2xZ — roughly four hours
   *before* the shared-table cutover, with each replica pinning its own table.
   The promoted configuration has never been benched end-to-end through the
   proxy. The harness for that run exists (`bench/bench-proxy.py`).
2. **The retraction gate's two arms have not been run** — 3 repeats each of
   C1@262144, C2@1k, C2@4k to test whether the +8.1%/−8.7% pair is bias or noise.
3. **Nothing here has n≥3.** The ladder above is one run per cell; the numbers
   are exact but single-shot.
4. The proxy's `aggregate` column at long context has never been measured
   alongside a per-output-token instrument, so the long-context rows in §3 are
   decode-only readings.

---

## Appendix A — what a power-cap difference is worth

Scope: 1 replica each, both cards benched in the same pass · cap 350 W ·
2026-10-02. Cards are at the **thermal wall**, not the electrical one: sustained
draw under bench load sat at 294–300 W with clocks dropping from 2302 to ~1965
MHz, so raising the cap buys clocks back.

| arm | C1/1k | C1/4k |
|---|---|---|
| GPU0 @ 350 W (16 lanes) | 217.9 | 218.1 |
| GPU1 @ 350 W (8 lanes, idle) | 228.7 | 217.1 |
| same cards @ 300 W | 191.8 | 187.9 |

Measured **2026-10-07**, the live limit on both cards read **275.00 W** while the
box's power-cap unit still specified `-pl 300` — an out-of-band writer, reverted
silently at the next boot. Treat the unit file as a wish and the sensor as the
fact.
