# Phase 0.4 — Baseline pin (flash-next TP1, the-serving-host)

Captured 2026-10-03 00:58–01:05 UTC (the-serving-host local), **before** any Phase 0 mutation.
This is the contract every later phase diffs against. Companion records are our internal serving notes (`flashnext-shared-ram-ple-plan.md` §0.4, `flashnext-dual-gpu-research-2026-10-02.md` §1) — not vendored here, so a claim resting only on them is not independently checkable in this repo.

## Identity

| field | value |
|---|---|
| container | `qwen38-flash-next-flash-next-opt-1` (project `qwen38-flash-next`) |
| compose (source of truth) | `deploy/compose.yaml` in this repo; the host keeps its own copy at the project directory |
| image | measured on the vendor SM120/NVFP4 fork of `sglang 0.5.19+g5cd0818abf27`; not redistributable. Open-source reference: `lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9` (tag `v0.5.19-cu130`, Docker Hub) — same version and CUDA lane, NOT the fork, so it will not serve these flags. See `LICENSING.md` |
| image digest | `sha256:294953f40d847b5ff78e89fab41e8657ca925709cb78893d51607792bf51a0d3` |
| pennyroyal git rev | `5cd0818abf27` (editable install `/opt/pennyroyal`) |
| sglang version | `0.5.19+g5cd0818abf27` |
| torch / CUDA / cuDNN | `2.13.0+cu130` / `13.0` / `92000` |
| driver / vBIOS | `595.91.07` / `98.02.81.00.07` |
| GPU A | `GPU_UUID_A` (yours; `nvidia-smi -L`), LnkCap 32GT/s **x16**, LnkSta (idle) 32GT/s **x8 (downgraded)** |
| GPU B | `GPU_UUID_B`, **free** (2 MiB, 0%), LnkSta idle 2.5GT/s x8 (power-managed) |
| endpoint | `http://127.0.0.1:30071` → container 8001; served-model-name `pennyroyal` |

## Model / PLE artifact

| field | value |
|---|---|
| dir | **served now:** `/mnt/fast/models/qwen38-flash-next-nvfp4-mtpfix` (BF16-MTP reshard, `dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix`, 141 shards, live since 2026-10-03 07:32Z). Pre-swap original `/mnt/fast/models/qwen38-flash-next-nvfp4` (XFS NVMe, 124 G, 26 files) was retained at swap time and **deleted 2026-10-03 after promotion was verified** (124 G reclaimed). Rollback = re-download `nvidia/Qwen3.8-Flash-Next-NVFP4` main using the pattern in `docs/deploy.md` and verify against the truncated LFS sha256s recorded below. L2 proof: ALL 296,473 non-mtp tensors byte-identical between the dirs (main NVFP4, dense bf16, PLE 132 keys) — only the mtp experts differ (fp8+`weight_scale_inv` → fused BF16). |
| identity hash | `07384cd01fc7db50` = sha256 over sorted `path size` of *.safetensors/*.json/*.jinja/*.pt (first 16) |
| PLE container file | `model-fp8-mtp-ple.safetensors` = **53,717,551,730 B** |
| PLE tensor | `model.language_model.layers.1.ple.ple_embedding.ngram_embedding` |
| PLE math | logical `[20,000,096, 2560]` fp8 = **51,200,245,760 B = 47.68 GiB** ✔ plan's qualified figure, verified from the safetensors header |
| PLE padding | **96 pad rows** exist above `ngram_vocab_size_base: 20,000,000`. (An earlier revision of this file claimed `padded == base, no pad rows` because 20,000,000 is already divisible by `make_ngram_vocab_size_divisible_by: 128` — **that was wrong**: divisibility by 128 is not the only padding source, and the on-disk table is 96 rows larger than base. Phase B's manifest must carry the observed `num_embeddings_padded = 20,000,096`, not a computed guess.) |
| PLE physical layout | stored as **128 shards of `[2,500,012, 160] F8_E4M3`** = 400,001,920 B each; `128 × 400,001,920 = 51,200,245,760` exactly. Reconciles with config: `2,500,012 = 20,000,096 / heads_per_ngram(8)` and `160 = ple_embed_dim(2560) / 16`, with `split_ngram_parts: 128 = 8 × 16`. **Consequence for Phase A/B:** RESOLVED 2026-10-03 by code-read (`qwen4_exp.py:492-517,~2080-2145` in the image, plus the private contract file above): the fork assembles ONE contiguous host buffer `[320,001,536+pad, 160]`, shard-major (shard k → rows `[k·2,500,012, …)`, exact-size anon mmap + `cudaHostRegister`); gathers index flat 160-wide rows (`weight_ptr + local_idx*embedding_dim`) and a token's 2560-vector is 16 head rows re-viewed (`view(t, ngram_heads, -1)`). **Phase A spike must use `.view(rows, 160)`** — the earlier `frombuffer(...).view(rows, 2560)` zero-copy premise was WRONG (a 2560-row view silently mixes heads across "rows"). The image's own PLE extractor must preserve this layout. |
| other PLE cfg | `ngram_size 3`, `heads_per_ngram 8`, `split_ngram_parts 128`, `ple_conv_kernel_size 4`, `vocab_size 248,320`, `hidden_size 2560` |
| effective dtype | `float8_e4m3fn` — forced at runtime by `--json-model-override-args {"text_config":{…,"ple_embedding_dtype":"float8_e4m3fn"}}`, **not** by the config string |
| ⚠ draft defect (2026-10-03) | This served checkpoint is **byte-identical to `nvidia/Qwen3.8-Flash-Next-NVFP4` main** (LFS sha256: fp8-mtp-ple `3525520c…78823`, index `660414e8…1c3c0d`) — the community's **known-defective artifact**: draft MoE experts fp8-serialized (`mtp.layers.0.mlp.experts` in both quant maps, `ignore: []`) and the fork loads them unscaled (draft built bf16-unquantized, all 1,536 `weight_scale_inv` silently dropped — `qwen3_5_mtp.py:50-67,421-437`) → accept floors ~1.46 (community defect fingerprint 1.50/124.8). The `--speculative-draft-model-quantization unquant` launch flag is INERT. **FIXED 2026-10-03 07:32Z** — dir swapped to `…-nvfp4-mtpfix` + the flag removed (verified: L2 EQUAL `mtpfix-equality.json`; L3v2 IDENTITY-PASS, dequant-mult max relerr 2.7%, `weight_scale_inv` mean ≈3.2e-4 ⇒ the raw-cast was loading draft experts ≈3,686× off in magnitude — the defect, quantified; Pearson corr is scale-invariant and mis-passed v1). Post-swap C1 probe (thinking off, 300 W): accept **2.34–3.61**, decode **185–345 tok/s**; counting accept holds the **4.0 window cap at 8k/32k** contexts. FR-Spec keep/drop A/B measured here → **map KEPT** (8/9 cells, +5–12% tps vs full-vocab; fv accepts higher but steps slower — community's opposite EN-prose result does not replicate on this box). Run artifacts (`swap-mtpfix`, `fv-window`) stayed on the serving host and are not published; full record: our internal research notes §11 |

## Fork code path (what `shared_ram` must replace)

`/opt/pennyroyal/python/sglang/srt/models/qwen4_exp.py` — the relevant span was captured verbatim to a private contract file, sha256 `6ae2e3cdd305d88d5c7939d1c8487f4f9649ef8a35f9a6048f6df4c2eac5dde4` (the file itself is not in this repo; the digest lets anyone holding the image check our reading against their copy).

- `_allocate_pinned_table(shape, dtype)` → `mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)` then `torch.cuda.cudart().cudaHostRegister(...)`, raising on failure.
- `class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding)` → binds storage to that table.
- Confirms the plan's premise: **private anon mmap, no file/MAP_SHARED backend exists** in this image. Phase C replaces this path only.

## Live launch args

Full argv (API key redacted) in `live-argv-redacted.txt`, sha256 `02df9d4937c64f22c7e881a0a9b231313375697be68aa5e7cdcad2a500de8183` (regenerated 2026-10-03 post-swap from the promoted server; the pre-swap capture `af865f43…` is superseded).
Load-bearing subset: `--tp 1 --load-format safetensors --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.981 --max-total-tokens 824384 --context-length 524288 --page-size 64 --max-running-requests 4 --chunked-prefill-size 4096 --ple-offload-embedding --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 24 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --speculative-token-map …/frspec/flash-next-64k.pt --sleep-on-idle --enable-metrics --enable-request-time-stats-logging --reasoning-parser qwen3 --tool-call-parser qwen3_coder --watchdog-timeout 1800`.
**Amended 2026-10-03:** the flag is proven inert and removal + model-dir swap = the mtp-fix. **Unresolved discrepancy, recorded rather than smoothed:** the promoted server's captured argv has no `--speculative-draft-model-quantization`, yet the recipe copy whose digest `LICENSING.md` pins (FR-Spec `37e43d6f…`, 10,930 B) still carries it at line 189. Both statements are in this repo; we have not established which recipe the measured boot actually executed, so do not treat either as confirmed; `live-argv-redacted.txt` regenerated post-swap (sha `02df9d49…`: token-map present, `draft-model-quantization` absent, `ple-offload-embedding` present). FR-Spec token-map flag RETAINED (measured A/B vs full-vocab, map won 8/9 cells).
Server default sampling: `--default-chat-template-kwargs {"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}` — **benches must pin these explicitly** (the interactive client overrides `reasoning_effort` to `max`, so unpinned legs are not comparable).
Env: `SGLANG_SM120_ONLINE_MXFP8=true`, `PENNY_HICACHE_SIZE_GB=1`, `shm_size: 32gb`, `cpus: 16.0` (no cpuset).

## Accept-length source (corrects our internal research notes §6)

`/metrics` exposes **no** `sglang:spec_*` / accept-length series in this build — only `gen_throughput`, `num_running_reqs`, `num_queue_reqs`, queue/latency histograms. Accept-length is therefore **log-only**, sampled ~every 40 decode steps:

```
[2026-10-03 00:58:54] Decode batch, #running-req: 1, #full token: 103744, full token usage: 0.13, mamba num: 4, mamba usage: 0.17, accept len: 1.90, accept rate: 0.30, cuda graph: True, gen throughput (token/s): 164.51, #queue-req: 0
```

Aggregate per window: mean `accept len` over `Decode batch` lines whose timestamps fall inside clean request windows (`#running-req: 1`, `#queue-req: 0`, `cuda graph: True`). `/get_server_info`'s cumulative ratio is not leg-subtractable — do not use it.

## Power-cap ownership (found during 0.0, corrects the premise recorded in our internal plan)

- `/etc/systemd/system/gpu-power-limit.service` — **enabled**, `Type=oneshot RemainAfterExit=yes`, `ExecStart=/usr/bin/nvidia-smi -pl 300` (no `-i` ⇒ all GPUs), `Requires=nvidia-persistenced`, `Before=docker.service`. This host unit is not published here; its content is quoted above, which is what the bench rules depend on.
- Sibling `cpu-power-limit.service` — RAPL PL1 100 W / PL2 120 W.
- **Live state deviates:** both GPUs read `power.limit = 450.00 W` (`default 600, min 150, max 600`), set by hand — root history holds `sudo nvidia-smi -pl 450` twice. ⇒ 450 W is **non-persistent**; a reboot returns 300 W.
- Boot at 2026-10-03 00:10:15 followed a **clean** poweroff (prev boot ends `systemd-poweroff.service: Deactivated successfully`, fsync+unmount complete) — **not** a PSU trip. No `NVRM: Xid`.
- Throttle counters at capture (GPU0, cumulative since boot ≈ 50 min): `sw_power_cap = 265,882,919 µs` (≈ 266 s), `hw_power_brake = hw_thermal = sw_thermal = sync_boost = 0`, `clocks_event_reasons.active = 0x0`, `power.draw 351.8 W (instant 221 W)`, `clocks.sm 2812/3090 MHz`, 67 °C, fan 41%.
- A privileged homelab management-agent container (ours; `NVIDIA_VISIBLE_DEVICES=all`, power writes enabled) is a *possible* competing cap-writer — its identity is not part of this recipe and is not published. No power writes in its last 60 min at capture. Any bench leg must sample `enforced.power.limit` and void itself on deviation.

## Memory / swap state at capture (the 0.2 defect)

| field | value |
|---|---|
| `memory.max` | 115,964,116,992 (108 GiB) |
| `memory.swap.max` | 115,964,116,992 (**+108 GiB swap allowed**) |
| `memory.peak` | 115,964,116,992 — already **hit the ceiling** |
| `memory.current` | 115,265,638,400 |
| `memory.stat` | `anon 55,539,826,688` (51.7 GiB) / `file 59,209,621,504` (55.1 GiB, reclaimable loader cache) / `shmem 44,965,888` / **`unevictable 0`** |
| swap in use | `/mnt/fast/swapfile` 3,787,124 KiB ≈ **3.62 GiB** (pri 10, same NVMe as weights+PLE); `/dev/dm-9` 0 |
| host | 244 GiB total, 81 used, 163 available |

`unevictable 0` alongside a genuinely registered 47.7 GiB table confirms the plan's measurement rule: **driver-pinned pages are invisible to the cgroup `unevictable` counter** — never size RAM from it; use `smaps_rollup` Pss + `memory.stat` anon/file.

## Governance notes

- `ps` on the serve argv prints the live API key in plaintext, so every argv dump must go through a redactor before it is recorded anywhere (`bench/baseline/live-argv-redacted.txt` is the shape). The key lives in a bind-mounted file, never in compose.
- Bench client lineage: a private predecessor script produced the documented 119.8 tok/s (old port 30074, no key). Phase 0.1 reuses that prompt/greedy contract, retargeted to 30071 with auth.

## Post-swap concurrency×context matrix (complete 2026-10-03 16:53Z) — 19/19 cells

Artifacts published: `bench/baseline/matrix-post-swap.csv` only — the raw `matrix.jsonl` journal and `run3.log` are machine-specific and stay private. Harness `bench/bench-matrix.py` (compare against its git blob at this commit — a content md5 written into prose goes stale the moment the file is edited, which is why the value previously quoted here no longer matched), gated wrapper `bench/matrix-gated-rerun.py` (foreign-chat quiet gate reads **both** docker-log streams — the stderr-only first version was blind to container chat-probes). Post-swap mtpfix checkpoint, 300 W, thinking off, greedy.

```
C  ctx      pref_cold  cached_prefix  dec/stream  agg     accept  achv  note
1  1 024      8 173        7 998        226.1      226.1   2.37    1.0
1  4 096      9 384       28 924        223.1      223.1   2.68    1.0
1  16 384     8 940       79 478        221.4      221.4   2.97    1.0
1  65 536     8 651      197 826        240.2      240.2   2.76    1.0
1  131 072    8 232      189 342        147.5      147.5   2.71    1.0   noisy attempt
1  262 144    7 711      226 033        239.6      239.6   2.60    0.0*  kept-flag (sampler miss, plausible)
1  393 216    6 952      258 690        206.2      206.2   2.42    1.0
1  520 192    6 194      263 323        192.9      192.9   2.01    1.0
2  1 024      7 902        8 010        186.0      367.1   2.59    2.0
2  4 096      9 213       30 209        190.0      375.0   2.62    2.0
2  16 384     8 831       80 622        139.1      184.5   2.66    2.0
2  65 536     8 714      135 948        112.3       62.3   2.67    2.0   kept-flag
2  131 072    8 412      160 817        119.4       16.6   3.01    1.0   kept-flag, agg NOT trustworthy
2  262 144    7 865      248 262         92.7        7.7   2.68    2.0   kept-flag
4  1 024      5 537        6 725        133.7      292.1   2.48    4.0
4  4 096      8 163       26 989        133.7      241.9   2.50    4.0
4  16 384     9 152       80 101         83.4      137.1   2.66    3.0   VERDICT cell
4  65 536     8 845      176 541         22.7       42.4   2.52    3.0   kept-flag
4  131 072    8 523      240 872         89.8       11.0   2.75    3.0   kept-flag
```

**Verdicts / rules for future comparisons:**
1. **Admission caps at 3 for ≥16k contexts** — C4 long-context reached `achv=3.0` on a *fresh* server with the fixed gate (zero foreign chats in-window on several attempts): the 4th long-prompt stream stays queued behind chunked-prefill (4096) admission at 300 W. Short contexts reach 4-up. C6/C8 experiments MUST raise `--max-running-requests` first and expect prefill serialization to dominate.
2. C2/131072 `achv=1.0` (one stream never admitted) — its `agg=16.6` is void; `dec/stream` and `accept` remain usable references.
3. Accept is 2.0–3.0 across the whole grid, **context-flat to 520k** (healthy draft, post-mtpfix).
4. Client-SSE decode reads **~13% below** engine `decode_throughput` (native `/generate` meta_info is the contract for 2-GPU comparisons).
5. Long-context cells carry ±30% per-attempt noise (128-token decode windows, 0.4 s sampler) — compare medians across attempts, never single cells.
6. `contaminated_accepted=true` rows (7 — verified against `matrix-post-swap.csv`) were kept under container-probe chatter (health-check chats from co-tenant management agents); treat their rates as best-case and re-run only if a decision hinges on them.

## Proxy-pool matrix (llmproxy C1–C8, both replicas, 2026-10-03 ~21:2xZ)

Artifacts published: `bench/baseline/matrix-proxy.csv` (journal and run log stay private); harness `bench/bench-proxy.py` (BASE :8787, model `qwen38-flash-next`, accept sampled from both replicas via ssh; `/metrics` not proxied → no achv — purity via llmproxy-log attribution join). Box state: post-power-cycle reboot 20:53:35, caps re-verified 300 W on both cards, both replicas container-restarted before the clean run (fresh radix — `/flush_cache` is broken, restart is the only reset), GPU1 link verified **32GT/s x8 under load**. The run died once (HTTP 503 — the power event) and once degraded (assistant self-load); those journals are kept aside (`matrix.pre-outage-2cells.jsonl`, `matrix.attempt1-contaminated.jsonl`), never mixed.

Headlines (vs the single-card contract above):
- **Short context scales ≈2×: C4/1k agg 690.9 (×2.37 vs single-card 292.1), C6/1k 876.8, C7/1k 925.3, C8/1k 1138.8 = 3.9× single-card peak — with per-stream still ~157 tok/s** (≈ single-card C4 feel): the sticky pool spreads by content key and each replica serves its share at native admission.
- Long context: the per-replica admission cap (rule 1) becomes a **pool cap ≈6 running ≥16k streams**; C5/131k (8.5/stream) and C6–C8/65k show the queue wall (agg 21.6–98.9). Raising `--max-running-requests` gates anything higher.
- ttft worst at 520k C1 through the pool = 78.9 s (single-card parity — no proxy tax).
- **CAVEAT (systematic):** this CLI on `approvalMode: auto` fires an **auto-mode-classifier chat per tool round** through the same proxy model — each a distinct session key and burst of load. Per-cell `+N-extra-keys` in the run record marks affected rows; their aggs are **lower bounds**. Proxy-pool benches must run fully detached with zero tool rounds in the window (single long blocking call, empty-boundary turns) or pin the approval mode.
- Sticky contract verified from the proxy's own source (private; not vendored here): `Derive` = sha256(model+tools+user-field+system/developer msgs+first user msg); `X-Session-Id` header wins; per-wire-API namespace; raw-hash fallback is visible as `SourceRaw`. Same markers rebind the same card (cross-attempt cold passes can hit stale radix — restart before re-runs).

## Spec-off ablation on replica B (2026-10-03 21:37–21:5xZ) — the draft, priced

Detached driver `bench/spec-ab-run.sh`: anchor-strips the four `--speculative-*` lines into `serve-flash-next-nospec.sh`, compose-swaps B's serve-script bind per arm with force-recreate, health-gates each boot, runs the same C1 harness (`bench-b2.py` = bench-matrix retargeted to :30073 + B container for accept), audits every cell for foreign chat in-window (zero culls needed — pass 1 clean both arms), restores production compose unconditionally. Artifacts: `matrix.jsonl` under the driver's output dir per arm (`spec`, `nospec`); not published (run-local).

```
ctx      spec-OFF dec   spec-ON dec   uplift   accept(on)
1k       129.8          221.4         ×1.71    2.63
4k       128.8          236.1         ×1.83    2.59
65k      128.0          242.9         ×1.90    2.84
262k     122.1          209.1         ×1.71    2.52
```

**Verdicts:** the healthy NEXTN draft buys **×1.7–1.9 decode** at C1 (accept 2.5–2.8 × zero draft-overhead ≈ ceiling 2.5–2.8 → draft+verify overhead eats ~30%). Spec-off decode is **context-flat ~128 tok/s** — and that is precisely the defect-era spec-ON rate (124–131): the broken fp8-MTP draft had been buying literally nothing. Prefill arms equal between arms (sanity ✓; first two cells of each boot read low — 2.7k/0.9k — early-boot JIT warmup artifact, ignore for prefill contract, use the ≥65k steady cells). Residual (accepted, not proven): whether spec-off captures bs=1 CUDA graphs — this build logs `Decode batch` (with its `cuda graph:` field) only in spec mode; flat 128 tok/s (~7.8 ms/step) is consistent with graphs on; re-ask via MicroProfiler only if the number ever becomes load-bearing.
