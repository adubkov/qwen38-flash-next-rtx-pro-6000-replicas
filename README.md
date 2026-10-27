# Qwen3.8-Flash-Next on 2× RTX PRO 6000 — the replica-pair playbook

Serving **Qwen3.8-Flash-Next (180B-A10B, NVFP4)** as **two independent TP1
replicas** on two RTX PRO 6000 Blackwell cards, where both replicas pin **one**
copy of the model's 47.68 GiB host-side PLE table instead of each pinning its own.

This repo is the deployable config, the plugin that makes the sharing work, the
benchmark harness, and the measured record — including the parts of the record
that turned out not to mean what we first said they did.

**Start with [docs/RESULTS.md](docs/RESULTS.md).** Every figure there is labelled
with its scope (one replica or the pair), its power cap, and its n. The same
number read without those three labels has caused most of the confusion in this
project, including inside our own notes.

---

## The honest summary

| claim | status |
|---|---|
| One shared pinned table halves host-RAM cost of the second replica: `anon` 53.35 → **~6 GiB**, host used 179 → **89 GiB**, available 64 → **155 GiB**, proven across a real cold boot at 24 GiB cgroup caps with `oom_kill 0` | **certified** |
| It also makes decode faster | **not supported.** Mean \|aggregate delta\| across the 11 paired cells is **5.52%**, recomputed from the table in [docs/RESULTS.md](docs/RESULTS.md) §3; the smaller figure this row first carried was a mis-computation and is withdrawn there. The run's own same-config noise floor was 1.4–1.9% and ~15–18% at long context, n=1 per cell |
| Per-replica throughput is unaffected | **supported at short context** (1k–65k bound a regression at ~3%); the 262k and C≥2 cells resolve nothing below ±8–15% |
| The pair-wide scaling number for this configuration | **never measured.** The existing ×2.37/×3.9 ladder predates the shared table by ~4 h and used private tables. See [Open items](docs/RESULTS.md#open-items) |

We published an "8% acceptance win" from one cell of that table and retracted it
the same day. Treat the retraction as part of the deliverable.

## Numbers at a glance

All **one replica**, direct on its published port, **300 W**, **n=1** unless noted.

| what | value |
|---|---|
| decode, C1/1k, shared table | **236.0 tok/s** |
| decode, C1, any context (post-promotion) | **195–255 tok/s** |
| aggregate, C4/1k, one card | **589.2 tok/s** |
| MTP accept length | **~2.4–2.8** |
| cached-prefix prefill at 65k | **221k tok/s** |
| decode at 2k → **508k** context, C1 | **152–187 tok/s** (depth-blind) |
| steady-state agent turn at ~500k | TTFT **2.0–3.8 s**, cache reuse **0.976–0.992** |
| whole 1k→508k conversation, warm vs cold | **94.6 s vs 339.6 s (3.6×)** |
| public anchor, single card | 231 tok/s (upstream project's own headline) |

## Pins

| field | value |
|---|---|
| checkpoint | upstream `nvidia/Qwen3.8-Flash-Next-NVFP4`; **measured on a third-party community reshard** — `dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix` (BF16-MTP, 141 shards). See [LICENSING.md](LICENSING.md) |
| PLE table | 51,200,245,760 B · sha256 `b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3` |
| engine | SM120 SGLang-class build, self-reported `sglang 0.5.19+g<rev>` — **supply your own**, see [LICENSING.md](LICENSING.md) |
| torch / CUDA / cuDNN | `2.13.0+cu130` / `13.0` / `92000` |
| driver | `595.91.07`, NVIDIA open kernel module |
| host | 4×64 GB DDR5-6000; both cards capped 300 W (thermic, not electrical, wall) |
| admission (as run) | pool `983,360` @ `--mem-fraction-static 0.998`, `--max-running-requests 4`, `--max-mamba-cache-size 24`, `--context-length 524288` |

## Layout

```
plugin/    the shared-RAM PLE plugin (entry, mmap backend, config), selftest, integrity check
deploy/    compose for the pair, table staging, boot + hourly-integrity systemd units
bench/     the harnesses (engine matrix, proxy ladder, agent loop, capacity probe),
           analyzers, the instrument probes that told us our own gates were lying,
           and baseline/ (the pinned contract table + derived matrix CSVs behind §1–2
           of docs/RESULTS.md)
docs/      architecture · deploy · benchmarking · RESULTS
```

## Quickstart

1. **Bring an engine build.** `deploy/compose.yaml` defaults to the open-source
   image at its own registry (`lmsysorg/sglang`, digest-pinned) — the version and
   CUDA lane we measured on, **not** the vendor fork our numbers came from, and it
   will not accept this checkpoint's serve flags. Override `ENGINE_IMAGE` with a
   build you are entitled to run (see [LICENSING.md](LICENSING.md)). Nothing works
   without such a build and our license implies nothing about it.
2. **Place the checkpoint — and check which MTP draft it carries.** The certified
   cause of accept length pinning at **~1.42** on any content was the community's
   defective fp8-serialised MTP artifact: the loader raw-casts fp8 codes into bf16
   expert params and silently drops all 1,536 `weight_scale_inv`, so the drafter is
   noise the target dutifully rejects. The BF16-MTP reshard swap took C1/1k from
   192.0 to **213.9 tok/s** and accept to ~2.5. Separately, confirm your recipe reads
   the PLE table's real row count — the claim on file is **155,691** against the
   published reference's 151,936, and it rests on prose plus the unshipped recipe,
   not on any artifact in this tree (see [LICENSING.md](LICENSING.md), "One claim we
   cannot source"). That under-read silently *clips* n-gram context, but no A/B on
   hand separates its contribution from the draft defect above.
   See [docs/architecture.md](docs/architecture.md) §3.
3. **Bring the pair up.** The `ple-stage` one-shot runs first (`depends_on`) and
   publishes the table into tmpfs, verifying sha256 against the manifest before an
   atomic rename — you do not invoke the stager by hand in the normal path:
   ```bash
   docker compose -p flash-next-pair up -d
   ```
   For the host boot path, `deploy/ple-stage-onboot.sh` calls the same stager with
   host paths (all overridable via `SRC` / `MAN` / `DST` / `PUB` / `GID`).
4. **Install the units** — boot-time staging and the hourly integrity check. The
   staging unit is conditional on Docker's own on-disk restart intent, so a box that
   runs other models does not strand 47.68 GiB of tmpfs for a pair that is not
   starting. The stager will not accept an already-present tmpfs payload without
   hashing it, so a right-size/wrong-bytes file is replaced rather than republished;
   before that hardening it was republished and only met a hash after a replica had
   pinned it. [docs/deploy.md](docs/deploy.md) §2.
5. **Prove the sharing**, which is the only claim worth verifying on your box:
   ```bash
   for c in flash-next-pair-flash-next-a-1 flash-next-pair-flash-next-b-1; do
     for p in $(docker top $c -o pid | tail -n +2); do
       m=$(grep -h layer-0.bin /proc/$p/maps | head -1)
       [ -n "$m" ] && echo "$c $(echo "$m" | awk '{print "perms="$2,"dev="$4,"inode="$5}')"
     done
   done
   ```
   PASS = one line per replica, **identical `dev:inode`**, `perms=r--s`. Gate on
   inode and size, **never on mtime**: our notes record both a header write that
   leaves mtime unchanged and an accepted payload whose mtime/ctime moved while its
   bytes hashed identical, so mtime is unreliable evidence in either direction.

## The recipe that produced these numbers

The pair runs a config **fragment** that overrides one baked into the engine
image. It is third-party (lineage credited in [LICENSING.md](LICENSING.md)) and
quoted, not shipped. These are the lines that matter, verbatim:

```bash
CONTEXT_LENGTH=524288
PAGE_SIZE=64
# TP1 is the qualified default; TP_SIZE=2 opts into the two-GPU experimental
# (not yet hardware-qualified) path
TP_SIZE="${TP_SIZE:-1}"
COMPUTE_DTYPE=bfloat16
KV_DTYPE=fp8_e4m3
MAMBA_SSM_DTYPE=bfloat16
MAMBA_CONV_DTYPE=bfloat16
MAMBA_TRACK_INTERVAL=64
configure_max_total_tokens 824384

  --page-size "$PAGE_SIZE" --max-running-requests "$MAX_RUNNING_REQUESTS" --sleep-on-idle \
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE" --gdn-mtp-cache-mode none \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant \
  --speculative-token-map "$TOKEN_MAP" --watchdog-timeout 1800
```

Three host-fit guards sit on top of the upstream file, and two of them are why the
box boots at all: fp8 PLE embedding dtype in `json-model-override-args`, and the
HiCache/NIXL tiers **stripped** because leaving them in trips a host-RAM OOM at
this size. We run `--mem-fraction-static 0.998` and pool `983,360` by env override,
not by editing the fragment.

One line quoted above does nothing. `--speculative-draft-model-quantization unquant`
is **inert**: draft boot auto-detects the draft's real quant config and the detection
wins. It is quoted for fidelity, not efficacy — and realising it was inert
retroactively voided an earlier A/B of ours that had treated it as a variable.

## Why two TP1 replicas and not TP2

Because on this board TP2 buys almost nothing at the operating point that matters.
Cited anchors for this checkpoint family — third-party records, we never ran TP2
here ([docs/architecture.md](docs/architecture.md) §4): TP2 gives ~**1.19–1.28×**
decode at C1 over PCIe, with one independent report flat (≈1.0×), and ~**2.5–4×**
the KV pool. The **3.9×** belongs to the two-engine pair, not to TP2: it is our own
C8/1k ladder through the proxy, taken with private tables and never re-run on the
shared one. Two independent single-GPU engines behind a session-sticky
proxy keep native admission and pay no inter-GPU traffic; the recurrent state and
the PLE table never cross the bus. TP2 is our documented **long-context fallback**,
not the default — and note that "2×" here means two single-GPU engines, **not**
tensor parallelism. No shipped file sets `--tp-size 2`: the fragment above defaults
it to 1 and honours a larger value only if you export one.

## Measuring it yourself

Read [docs/benchmarking.md](docs/benchmarking.md) first. Its rules are each priced
by a run we had to throw away — including draining the measured replica from the
proxy (bypassing the proxy is necessary and **not** sufficient), sampling clocks
and power at 1 Hz because clocks drift 2302→1965 MHz mid-run, and why three
different purity instruments printed `CLEAN` while the cells were contaminated.

## Security

The 47.68 GiB payload is a tmpfs file two engines read: `root:975 0660` with
`group_add: ["975"]` rather than a mode widening, and read-only safety restored
by the engine's unconditional `mprotect(PROT_READ)` after registration; the
hourly integrity check hashes **through the live mapping**, not against the
source file. The shipped mechanisms live in `deploy/compose.yaml` and
[docs/deploy.md](docs/deploy.md).

## License

Our code and docs: **MIT** ([LICENSE](LICENSE)). The model, the engine build, and
the quoted vendor configuration are **not** covered by it — read
[LICENSING.md](LICENSING.md) before reusing anything.
