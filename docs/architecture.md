# Architecture — why this deployment looks the way it does

Read this before touching config. It explains the shape of the deployment — two independent single-GPU engines, one
page-locked table in host RAM, a session-sticky proxy in front — and the measurements behind it. Instructions live in
`deploy/`.

**Citation convention.** `(note: <file>:<line>)` = a record in our internal serving notes (the research/plan documents
this repo summarises; not vendored here). `(repo: <path>:<line>)` = a file in this repo. Lines are as those records
stood on 2026-10-08; where a claim has no measurement behind it, this document says so rather than quote a figure.
**A `(note:)` citation is not a link you can follow.** Most of them point at three private prose documents (the
`flashnext-*` files listed under Sources) that this repo declares unpublished; nothing
here lets you check them, and a claim resting only on one is our assertion, not a verifiable record. The exception is
`phase0-baseline/BASELINE-PIN.md`: its shipped copy is `bench/baseline/BASELINE-PIN.md`, and those citations can be
checked there.

**Naming trap, up front.** "2×" here means **two independent single-GPU (TP1) engines behind a session-sticky proxy**.
It is not tensor parallelism: the pair shares one PLE table in host RAM and shares nothing on the GPUs. TP2 is
something we have read about in other people's benchmarks and never run here; §4 says why.

## 1. The model, part by part, and what each part costs a card

Served artifact `qwen4_exp`: 48 hidden layers laid out 3 recurrent + 1 full-attention on an interval of 4 →
**36 Gated-DeltaNet (Mamba-style) + 12 full-attention layers**; hidden 2560; 24 query heads / 2 KV heads; head_dim
256; MoE **512 experts, top-10 plus one shared**, expert intermediate 640; vocab 248,320
`(note: flashnext-dual-gpu-research-2026-10-02.md:30)`. Launched `--quantization modelopt_fp4` (NVFP4 weights),
`--kv-cache-dtype fp8_e4m3`, `--mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 24`,
`--max-running-requests 4 --page-size 64`, `--ple-offload-embedding`,
`--speculative-algorithm NEXTN --speculative-num-draft-tokens 4`
`(note: phase0-baseline/BASELINE-PIN.md:49)`, `(note: flashnext-dual-gpu-research-2026-10-02.md:27)`.

Three things consume the card; only one grows with the conversation. **Weights:** at the shipped operating point
weights + context + CUDA graphs hold ~80.8 GiB of a 95.6 GiB card `(note: flashnext-dual-gpu-research-2026-10-02.md:265)`.
**KV — 12 layers, not 48:** per token per side, 12 full-attention layers × 2 KV heads × 256 head_dim × 1 B fp8 =
**6,144 B**, so K and V together are **12,288 B/token**; the draft layer (1 layer, same head geometry) adds
**1,024 B/token** `(note: flashnext-dual-gpu-research-2026-10-02.md:243-246)`. Check it from the other direction: the
engine's boot line reports K 4.72 + V 4.72 GiB over 824,384 tokens = 6,144 B/side
`(note: flashnext-hicache-plan-2026-10-04.md:182-184)`. **Recurrent state — per stream, not per token:** the 36 GDN
layers carry state that does not scale with context, budgeted in slots at **~4 checkpoints per stream**, which is why
`--max-mamba-cache-size 24` admits roughly 6 streams `(note: flashnext-hicache-plan-2026-10-04.md:232)`; under TP2 it
head-shards to ≈0.7 GB/rank at 24 slots `(note: flashnext-dual-gpu-research-2026-10-02.md:38)`.

So **KV is the capacity wall (grows with tokens) while the state pool and `--max-running-requests` are the concurrency
ceiling (grows with streams)**: short context hits the ceiling, and from ~262k up the pool — not the scheduler —
decides `(note: flashnext-hicache-plan-2026-10-04.md:187-196)`. **MTP is worth most of the machine's decode rate** — the same
card at C1 decodes 221–243 tok/s with a healthy draft and ~128 tok/s with speculation off, ×1.7–1.9 net of
draft+verify overhead, at accept length 2.5–2.8 `(note: phase0-baseline/BASELINE-PIN.md:143-150)`. Acceptance moves
with content and reasoning mode, so every figure here pins `chat_template_kwargs`
`(note: flashnext-dual-gpu-research-2026-10-02.md:153)`.

## 2. The PLE table: 47.68 GiB that cannot live in VRAM

The PLE (per-layer token embedding) is a large **n-gram embedding table** on the embedding path — `ngram_size 3`,
`heads_per_ngram 8`, `split_ngram_parts 128`, `ple_conv_kernel_size 4` `(note: phase0-baseline/BASELINE-PIN.md:33)` —
held in the engine as a `VocabParallelEmbedding` whose storage is one contiguous host buffer a gather kernel
dereferences over PCIe `(note: phase0-baseline/BASELINE-PIN.md:32)`. Its size, four ways, all agreeing (effective dtype
`float8_e4m3fn`, forced at runtime by a model-override argument rather than the config string,
`(note: phase0-baseline/BASELINE-PIN.md:34)`):

```
logical    [20,000,096 rows, 2560 cols] fp8 = 51,200,245,760 B = 47.68 GiB
                                            (base 20,000,000 + 96 pad rows)
physical   128 shards of [2,500,012, 160] F8_E4M3 = 400,001,920 B each
           128 × 400,001,920 = 51,200,245,760
manifest   rows 320,001,536 × columns 160      (= 20,000,096 × 16 head rows)
page-lock  51,200,245,760 / 4096 = 12,500,060 pages, exactly
```
`(note: phase0-baseline/BASELINE-PIN.md:30-32)`, `(note: flashnext-shared-ram-ple-plan.md:173-175,219)`.

**Why host RAM:** at the shipped memory fraction the card has ~3.5 GiB free against a 47.68 GiB table — that gap is why
`--ple-offload-embedding` exists `(note: flashnext-shared-ram-ple-plan.md:157)`. **Why *pinned*, and this is the crux:**
a normal allocation is not enough. The spike that settled it measured an **unregistered private anon mapping at
289 GB/s** — the driver had migrated those pages into VRAM, which production cannot afford — while the registered
mapping read an honest 6.9 GB/s, and the decode-scale gather costs 0.010 ms against an ~11 ms step
`(note: flashnext-shared-ram-ple-plan.md:157)`. Unregistered pages are also reclaimable and swappable mid-decode.
Pinning makes the bytes **unreclaimable and invisible to the accounting you would otherwise trust**: a genuinely
registered 47.7 GiB table leaves cgroup `memory.stat unevictable` at **0** `(note: phase0-baseline/BASELINE-PIN.md:85)`,
`(note: flashnext-shared-ram-ple-plan.md:13)`, and swap cannot absorb them
`(note: flashnext-dual-gpu-research-2026-10-02.md:99)`. The table's cost is therefore not its bytes but the fact that
nothing may reclaim them — and **that cost is paid once per process that pins it.**

What a pin demands here, measured: substrate must be **shmem** (tmpfs or memfd; a block-device page-cache file is
rejected outright); the VMA must be **CPU-writable**, because `cudaDevAttrHostRegisterReadOnlySupported=0` — which is
why the payload bind is `rw`; `cudaHostRegister(..., 0x01|0x02)` succeeds, `nr_foll_pin_acquired` advances by exactly
`bytes/4096`, and `cudaHostGetDevicePointer` returns the host pointer so the gather kernel is unchanged; the CPU
mapping is then `mprotect(PROT_READ)`-ed, fatally on failure `(note: flashnext-shared-ram-ple-plan.md:143-166)`,
`(repo: plugin/sglang_shared_ram_ple/backend.py:7-30)`. Registration is bounded: **2.3–2.4 s** warm, **15.4 s** on first touch of cold tmpfs
pages, beside a ~150 s boot `(note: flashnext-shared-ram-ple-plan.md:181,228)`.

> **One claim we cannot source.** Our working explanation for *why* a pin costs so much more than a plain allocation
> here is the driver's untagged long-term pin path (`dma_map_sg`-style pinning of every page in the mapping) colliding
> with an engine that already holds a large `cudaMallocAsync` heap. **No controlled measurement of that mechanism is on
> file** — no pin-vs-plain A/B with the engine heap resident, no per-stage timing inside the driver. It is a hypothesis
> about *why*; the consequences above are what is measured.

## 3. The vocabulary defect

The table is indexed by a packed n-gram vocabulary, and that row count comes from the recipe, not the checkpoint.
**The claim — not a fact this repo carries evidence for — is that the vendor's published recipe read 151,936 while the
checkpoint carries 155,691, and that the community BF16-MTP reshard the measurements ran on
(`dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix`) reads `ple_packed_vocab=157312`**
`(repo: LICENSING.md, "One claim we cannot source")`. No shipped file, log or CSV holds those numbers; the recipe that
would is deliberately unpublished. Getting it wrong fails silently, which is how it survived: the table is **clipped**, so
n-gram context whose packed index falls at or above the wrong bound is never read and the drafter is conditioned on
less context than the model was trained with. The symptom was MTP accept length pinned at **~1.42** where this stack
produces **~2.5** with a healthy draft `(note: flashnext-shared-ram-ple-plan.md:253)`,
`(note: phase0-baseline/BASELINE-PIN.md:122)`.

Plainly, as it was: **no engine-side A/B had been run when this was found.** The comparisons on hand could not separate
the hypotheses, and one earlier "refutation" was a flag the loader ignores — an inert toggle, not an experiment
`(note: flashnext-dual-gpu-research-2026-10-02.md:178)`. A second, independent defect in the same draft path
(fp8-serialised MTP experts loaded without their scales, ≈3,686× off in magnitude) is separately documented as the
closed root cause of the 1.42 floor, fixed by the checkpoint swap `(note: flashnext-dual-gpu-research-2026-10-02.md:173-191)`,
`(note: phase0-baseline/BASELINE-PIN.md:35)`. Both were real; both sat between the model and the number we were
staring at. Hence the rule in `docs/benchmarking.md`: when acceptance is flat, suspect the artifact before the hardware.

## 4. Why 2× TP1 replicas rather than TP2

| what TP2 buys | the record |
|---|---|
| C1 decode | ×1.19 headline, ×1.28 tuned (259.1→308.3→332.2 tok/s); an independent report is flat, ~201–204 vs best TP1 207 `(note: flashnext-dual-gpu-research-2026-10-02.md:10)` |
| why so little | bandwidth is a minority of the step; TP2 halves the slice and adds ~100 NCCL all-reduces per step over x8 links `(note: flashnext-dual-gpu-research-2026-10-02.md:45)` |
| KV pool | ~2.5–4× — 534,912 → 2,254,464 tokens in one published run `(note: flashnext-dual-gpu-research-2026-10-02.md:10)` |
| host RAM | ~none: the PLE table is a `VocabParallelEmbedding`, so each rank pins only its ≈23.84 GiB shard `(note: flashnext-dual-gpu-research-2026-10-02.md:11)` |
| price | both cards in one engine, one container, one failure domain; a hard startup requirement (`--disable-custom-all-reduce`) and an unverified P2P path `(note: flashnext-dual-gpu-research-2026-10-02.md:13)` |

Our own TP1 contract is the other half of the call: C1 decode 195–255 tok/s per stream across the context ladder
`(note: flashnext-dual-gpu-research-2026-10-02.md:286)` and 133.7 tok/s per stream at C4/1k (292.1 aggregate)
`(note: phase0-baseline/BASELINE-PIN.md:98-116)`, with achieved concurrency saturating at **3.0** for every context ≥16k
`(note: phase0-baseline/BASELINE-PIN.md:120)`. Two independent engines
scale the way independent engines do: measured through the proxy **with private tables** — the only two-card figures we
have — C4/1k reached 690.9 tok/s aggregate (×2.37 vs single-card) and C8/1k 1,138.8 (3.9× the single-card peak) at
~157 tok/s per stream `(note: phase0-baseline/BASELINE-PIN.md:132)`. They work because the proxy is **session-sticky**,
keying affinity on the request-body prefix rather than round-robin, so each conversation keeps its warm radix subtree
on one card `(note: phase0-baseline/BASELINE-PIN.md:136)`. **Decision rule: TP2 is the long-context fallback, not the
default** — its durable asset is pool capacity, not speed, and it stays behind a bench bar it has never been asked to
pass here (C1 ≥1.15×, C4 ≥1.3×, 8K/64K prefill ≥1.25×, accept parity, tolerance-based output parity, 24 h soak)
`(note: flashnext-dual-gpu-research-2026-10-02.md:105)`.

## 5. Why two replicas were impossible before the shared table

Because the pin was private: two TP1 engines meant **two** 47.68 GiB page-locked tables — 95.4 GiB of unreclaimable RAM
`(note: flashnext-dual-gpu-research-2026-10-02.md:12,89)`. The inference envelope in force was ~130 GiB (host total
minus the storage VM, the CI runner at its 85 GiB worst case, cache reserve, host) and a staggered two-load peak of
~165 GiB did not fit `(note: flashnext-dual-gpu-research-2026-10-02.md:79,89)`. Two cautions when redoing that
arithmetic: the envelope was derived from 250 GiB but the host is **244 GiB**
`(note: flashnext-shared-ram-ple-plan.md:40)`, and the single live replica already sat at
`memory.peak == memory.max` (108 GiB) `(note: phase0-baseline/BASELINE-PIN.md:79)`. Weights were never the obstacle:
they arrive through the page cache, shared between processes reading the same files `(note: flashnext-dual-gpu-research-2026-10-02.md:98)`.

The incident record is why this was a wall, not a tuning question. The container had **no swap ceiling**:
`limits.memory: 108G` with no `memswap_limit` allowed a further +108 GiB of swap, and 3.86 GiB was already on a
swapfile on the same NVMe as the weights and the PLE payload `(note: flashnext-shared-ram-ple-plan.md:10)`,
`(note: phase0-baseline/BASELINE-PIN.md:82)`; the host KV tier had itself been **stripped as an OOM guard** in an
earlier 115 GiB era `(note: flashnext-dual-gpu-research-2026-10-02.md:145)`; and a crash-looping replica re-loading
~100 GiB per attempt is the recorded shape of a past incident — hence `restart: "no"`
`(note: flashnext-dual-gpu-research-2026-10-02.md:119)`. Now the same container holds **6.01 GB of anon** shared
instead of the **53.35 GiB** it allocated privately in `ram` mode, same container, same run
`(repo: deploy/compose.yaml:22-24)`, `(note: flashnext-shared-ram-ple-plan.md:222,283)`. One honest caveat from the
time: while replica A still pinned its own copy, the apparent gain was **three copies becoming two**, not a saving
`(note: flashnext-shared-ram-ple-plan.md:262)`.

## 6. What shared PLE buys — and what it does not

Be precise, because an earlier draft of our notes called a noise-level result an ~8% win. **That claim is withdrawn.**
The most favourable cell in the only paired A/B we have was **+8.1% at C2/1k**
`(note: flashnext-shared-ram-ple-plan.md:242)`, against a same-config drift bracket of 1.4 / 1.9 / 0.4 / ~15–18% at
1k / 4k / 65k / 262k `(note: flashnext-shared-ram-ple-plan.md:258)` — a cell inside its own drift floor is the drift,
not an effect. What **is** certified is host RAM economy: one physical table, mapped and registered by both engines at
the same inode; per-replica anon ~53 GiB → ~6 GiB; host `used` 179 → 89 GiB and `available` 64 → 155 GiB at the
cutover; `pin_pages` exact and no hidden private copy `(note: flashnext-shared-ram-ple-plan.md:283)`,
`(repo: deploy/compose.yaml:16)`. What is **not** certified, and must not be repeated as measured, is any pair-wide
throughput on the shared configuration: recomputed from the table in [RESULTS.md](RESULTS.md) §3, the paired C1/C2/C4
ladder averages **5.52%** |aggregate delta| (the smaller figure a first pass of these notes printed was a
mis-computation; the withdrawal is recorded there, in open form, with both numbers). The run's own verdict
was "no measurable throughput cost; the memory result is the claim that is actually certified"
`(note: flashnext-shared-ram-ple-plan.md:251)`, and the review of that run adds that the 262k and every C2/C4 cell are
unresolvable at n=1 below roughly ±8–15% and that **the promoted configuration was never benchmarked at all**
`(note: flashnext-shared-ram-ple-plan.md:258-261)`. That is what physics predicts: with registration free at every
access scale (§2), that bench could not have seen a difference even if one existed. Shared PLE is justified by what it
frees and by the second replica it makes affordable — it carries no speed claim.

## 7. What shared PLE does not fix

The long-context concurrency wall is per card, in VRAM, and out of reach of anything in host RAM: a host tier moves KV
there for *retention after eviction*, never for capacity `(note: flashnext-hicache-plan-2026-10-04.md:259-263)`. The
numbers that draw the wall: `pool = min(ask, profiler estimate)` → **983,360 tokens** at `MEM_FRACTION_STATIC 0.998`
`(note: flashnext-dual-gpu-research-2026-10-02.md:258-260)`; the rate is ~7,160 tokens per +0.001 of fraction, and
1,049,984 tokens would need a 13.02 GiB pool at fraction 1.006 — unreachable, so the configured ask is inert
`(note: flashnext-dual-gpu-research-2026-10-02.md:263-266)`; admission obeys `C × (prompt + generated) ≤ pool`, so two
520k streams on one card (1,040,896 tokens) cannot be admitted at any fraction — one per card is the ceiling, and that
is a routing property, not a memory one `(note: flashnext-dual-gpu-research-2026-10-02.md:270-271)`. Per-stream decode
falls as context and concurrency multiply: C1 195–255 · C2 87–176 · C3 30 @65k, 8 @131k, 3.8 @262k
`(note: flashnext-dual-gpu-research-2026-10-02.md:286)`. "3–4 concurrent at long context" is therefore not a two-card,
300 W outcome `(note: flashnext-dual-gpu-research-2026-10-02.md:288-289)`; long-context concurrency does not fail
here, it **serialises**, and the bill arrives as TTFT `(note: flashnext-hicache-plan-2026-10-04.md:196-200)`. The
levers are more pool (cards, or TP2 for its ~2.5–4×) or a shared **KV** tier — a separate plan with its own gates, not
run `(note: flashnext-hicache-plan-2026-10-04.md:259-263)`. Shared PLE and shared KV solve different problems; do not
let one be sold as the other.

## Sources

Serving notes (internal corpus), cited by filename and line: `models/qwen3.8-flash-next/flashnext-dual-gpu-research-2026-10-02.md`
— 10-13, 27-30, 38, 45, 79, 89, 98-99, 105, 119, 123-124, 145, 153, 173-191, 243-246, 258-289 ·
`models/qwen3.8-flash-next/flashnext-shared-ram-ple-plan.md` — 1-4, 10, 13, 40, 143-166, 173-181, 219, 222, 228, 242,
251, 253, 258-262, 283 · `models/qwen3.8-flash-next/flashnext-hicache-plan-2026-10-04.md` — 182-184, 187-200, 232,
259-263 · `phase0-baseline/BASELINE-PIN.md` — 30-35, 49, 79, 82, 85, 98-122, 132, 136, 143-150.

In this repo: `LICENSING.md` (checkpoint provenance; the "One claim we cannot source" block) · `deploy/compose.yaml` 16, 22-24, 67, 95, 107, 146-147, 167-171, 192-195 ·
`deploy/ple-stage.sh` 1-30 · `plugin/sglang_shared_ram_ple/backend.py` 1-40 · `plugin/sglang_shared_ram_ple/config.py` 1-40 · `plugin/sglang_shared_ram_ple/plugin.py` 1-60.
