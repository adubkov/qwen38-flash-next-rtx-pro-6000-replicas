# Licensing and third-party material

This file exists because the repo sits on top of things we do not own. The MIT
license in `LICENSE` covers **only our own material** — it does not license the
model, the inference engine, or the vendor configuration we quote. Read this
before you reuse anything.

## What is ours (MIT)

| Path | What it is |
|---|---|
| `plugin/sglang_shared_ram_ple/` | the shared-RAM PLE table plugin (plugin entry, mmap backend, config loader) |
| `plugin/{deploy-shared-ram.sh,ple-shared-ram-check.sh,selftest.py}` | build/verify tooling and the live-mapping integrity check |
| `deploy/` | the compose definition, staging script, boot and integrity units, cutover driver |
| `bench/` | the benchmark harnesses, analyzers and instrument probes |
| `README.md`, `docs/` | our write-up, except the quoted material identified below |

## What is not ours

### The checkpoint — reference only, zero bytes shipped

Two things sit under the numbers in this repo, and only one of them is ours to
name loosely. The upstream checkpoint is **`nvidia/Qwen3.8-Flash-Next-NVFP4`**
(NVFP4). What the measurements actually ran on is a **third-party community
reshard of it** — `dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix`, a
BF16-MTP repack in 141 shards — recorded as the served directory in
`bench/baseline/BASELINE-PIN.md`. The pre-swap directory (the vendor bytes as
published, carrying the fp8-serialised MTP draft that is the closed root cause of
the ~1.42 accept floor; `124 G`, 26 files) was **deleted after promotion was
verified**, so §1 of `docs/RESULTS.md` cannot be re-derived from anything we still
hold. No local re-pack is claimed or shipped: this repo carries the two verifiers
(`bench/verify-mtpfix.py`, `bench/mtpfix-stats.py`), not a repacker.

We publish the checkpoint by **name and digest only**. No weight file, index, or
tokenizer asset is in this repository, and nothing here lets you reconstruct one.
What you may redistribute is our own material, under the MIT licence in `LICENSE`;
what you may not is the weight bytes, the vendor's or the community's repack, or
our engine image. You pull each yourself and accept its own terms; those terms are
yours to read at the source, not ours to restate.

Pinned digests we measured against, for reproducibility:

| Component | Value |
|---|---|
| PLE table, layer 0 (as staged and served) | 51,200,245,760 B · sha256 `b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3` |
| `plugin/sglang_shared_ram_ple/backend.py` | sha256 `bd23762510fc74c102e739d337764b6759141334f6ee4a45e749db5ced7fe055` |
| `plugin/sglang_shared_ram_ple/config.py` | sha256 `b2e6634c01598bda56710f3d91dd236384749d075689979339c46f5f7d7f25b6` |
| `plugin/sglang_shared_ram_ple/__init__.py` | sha256 `fd5f8b49d2c21b9052e3bb69c8177c673b1a16521b9f9aa7d30e634f24f186ef` |

The table digest above is the value recorded in the manifest the replicas verify
against at bring-up, and the value the hourly integrity check recomputes from the
live mapping. The three plugin digests are the ones the serving host pins in its
own deployed-source baseline, which is how we know the files in this repo are the
files that produced the measurements.

> **One claim we cannot source.** The vocabulary figures the repo's central
> technical story turns on — the mtpfix recipe reading `ple_packed_vocab=157312`
> and its shared-index key names the tensor `embed_tokens.0.weight`, with **155,691** PLE rows in the
> checkpoint, against the vendor recipe's under-read of **151,936**
> (`docs/architecture.md` §3) — are carried by **nothing shipped here**: no file,
> log, manifest or CSV in this tree contains them. The artifact that would, the
> serve recipe, is deliberately not published. They are
> also unreconciled with the one PLE geometry this repo *does* pin:
> `bench/baseline/BASELINE-PIN.md` records `[20,000,096, 2560]` with **96 pad rows**
> over a 20,000,000 n-gram base — a different axis from the 155,691/157,312 pair,
> and no artifact maps one to the other. **No controlled measurement of the vocab
> delta is on file**: `docs/RESULTS.md` §2 attributes the recorded gain to the
> draft fix and says the vocab line was never isolated by A/B. Read these as a
> claim we assert from our own notes, not as digest-backed facts.

### The inference engine — you must supply it

Both replicas run an SM120 (Blackwell) SGLang-class engine, self-reported as
`sglang 0.5.19+g<git-rev>` on `torch 2.13.0+cu130` / CUDA 13.0, driver
`595.91.07`. It is a **third-party build we do not redistribute**.

`deploy/compose.yaml` defaults to the **open-source** engine image at its own
registry — Docker Hub `lmsysorg/sglang`, digest-pinned
(`sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`,
tag `v0.5.19-cu130`; re-resolve the tag before trusting the digest — a tag move
is not a contradiction). It is the version and CUDA lane our measurements were
taken on, but **not** the vendor fork the numbers came from: it will not accept
this checkpoint's serve flags, and the shared-ram guard `plugin/deploy-shared-ram.sh`
generates refuses to patch an engine whose internal hashes differ
(`docs/deploy.md` §1). Override `ENGINE_IMAGE` with a build you are entitled to
use (`deploy/.env.example`). Nothing in this repo works without such a build, and
no license we grant here implies you may fetch ours. The engine's own upstream
license terms apply to it; we make no claim about a downstream fork's compliance.

The plugin does import engine internals (`sglang.srt.plugins`,
`sglang.srt.server_args`). That is an API dependency at a pinned version — the
engine registry API moved once during 2025 and may move again. Expect to touch
`plugin/sglang_shared_ram_ple/plugin.py` on a major bump.

### Vendor configuration we quote, not ship

The pair does not run a command line of our own. It runs two **config fragments**
that bind over the ones baked into the engine image
(`configs/<vendor>/serve-flash-next.sh` and `…-frspec.sh`). The live file's own
header credits its lineage to **`gabrielolympie/sglang-flashnext-sm120`**
("Qwen3.8-Flash-Next 180B NVFP4 @ 231 tok/s on one RTX PRO 6000 Blackwell"), and
describes itself as a *guarded variant* of that project's `serve-flash-next-frspec.sh`
plus three host-fit guards.

**That repository states no license.** Absent a license, default copyright
applies, so we ship **no file from it**. We quote the handful of lines that carry
the measurable result, with attribution, and pin the files by digest so you can
compare against your own copy:

| File | Size | sha256 |
|---|---|---|
| `serve-flash-next.sh` | 10,976 B | `50832edfab81e7464c39eb390a582aef6556ec0c84a00a3fdeda800415777312` |
| `serve-flash-next-frspec.sh` | 10,930 B | `37e43d6f944dc7b952f68a5da53ae50db1a2280163016fae38a7e26320f551c2` |

The quoted lines appear in `README.md` with attribution. The full files are not
in this repo.

### Third-party code we depend on at runtime

Python standard library (`mmap`, `hashlib`, `ctypes`, `json`, `urllib`, …) plus
four packages that are **not** stdlib, each named where shipped code imports it:
**`torch`** (`plugin/sglang_shared_ram_ple/backend.py`, `…/plugin.py`,
`plugin/selftest.py`, `bench/mtpfix-stats.py` — the plugin needs the CUDA build),
**`numpy`** (`…/backend.py`, only on the `torch.frombuffer` fallback path),
**`safetensors`** (`bench/mtpfix-stats.py`, bench-only) and **PyYAML**
(`bench/trial-boot.sh`). Outside Python: `docker`,
`systemd`, `curl`, `python3`. No vendored source from any of them.

## Attribution and names

Internal identifiers belonging to the engine build or to our deployment tooling
have been kept only where they are functionally required — a bind-mount target
path inside the image, for example — and are not presented as part of this
project's name or branding.

## If you think something here is mislicensed

Open an issue. The provenance that matters (exact digests above) is documented
specifically so a claim can be checked rather than argued.
