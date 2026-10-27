# Deploy

Order of operations and failure modes for the two-replica pair. Every path and flag
below is quoted from the files in this repo (`deploy/`, `plugin/`); nothing here is a
paraphrase of a flag that does not exist in them. Performance numbers are pointers to
[RESULTS.md](RESULTS.md). The reasoning behind the design also lives in working notes we do not vendor here,
cited inline as `(note: <file>:<line>)`. Those records are our internal research and
plan documents; **they are not in this repo, so a claim resting only on a `(note: …)`
citation is not independently checkable here.** `(repo: <path>:<line>)` means a file in
this repo and can be followed. One document carries both labels: `BASELINE-PIN.md` was
written in our notes tree and is published here as `bench/baseline/BASELINE-PIN.md` —
cite that path, and note that its line numbers are the published copy's, not the
private original's. Full list in [Sources](#sources).

What the repo ships:

| file | role |
|---|---|
| `deploy/compose.yaml` | one project: `ple-stage` (one-shot stager) → `flash-next-a` → `flash-next-b` |
| `deploy/ple-stage.sh` | verify-then-publish the 47.68 GiB table into tmpfs (called by both the container and the boot unit) |
| `deploy/ple-stage-onboot.sh` + `.service` | boot-time staging, ordered `Before=docker.service` |
| `plugin/ple-shared-ram-check.sh` + `deploy/ple-shared-ram-check.service` + `.timer` | hourly integrity witness for the **live** mapping |
| `deploy/flash-next-pair-boot.service` | boot-start with compose ordering while `restart: "no"` keeps crashes visible |
| `deploy/cutover.sh` | the phased one-replica-at-a-time swap, with its rollback paths |
| `plugin/{__init__,plugin,backend,config}.py` | the `shared_ram` SGLang plugin (one REPLACE hook) |
| `plugin/deploy-shared-ram.sh` | builds the dist on the host, generates the engine guard, runs the three verification legs |
| `plugin/selftest.py` | exercises `backend.allocate_shared_table` at the served size, with a 128-shard layout check |

---

## 1. Prerequisites — what you must supply

The repo does not contain the engine image, the checkpoint, the prepared payload, or the
third-party serve recipe. You bring all four, then the vendored files wire them together.

**Engine image.** Both the replicas and the one-shot stager read
`image: ${ENGINE_IMAGE:-lmsysorg/sglang@sha256:d6e728…}` — the open-source image at its own registry
(Docker Hub `lmsysorg/sglang`, tag `v0.5.19-cu130`), which is the version and CUDA lane we measured on
but **not** the vendor fork our numbers came from; override `ENGINE_IMAGE` (see `deploy/.env.example`
and `LICENSING.md`). Our measured build was
`sha256:294953f40d847b5ff78e89fab41e8657ca925709cb78893d51607792bf51a0d3`,
`sglang 0.5.19+g5cd0818abf27`, driver `595.91.07` (repo:
`bench/baseline/BASELINE-PIN.md:12`, `:13`, `:15`, `:17`). This is not a
pull-and-go image: `plugin/deploy-shared-ram.sh` generates `shared-ram-guard.json` by
hashing `sglang.srt.models.qwen4_exp`, `sglang.srt.plugins` and
`sglang.srt.plugins.hook_registry` **inside that image**, and `plugin.py::_check_guard()`
refuses to patch an engine whose hashes differ. Re-generate the guard (`--stage`) for your
build; do not edit the recorded hashes to make a drifted engine pass.

**Checkpoint + prepared payload.** The replicas serve a *prepared overlay*, not the raw
checkpoint: `/model` is the `prepare_ple_nvme.py` output, and the real checkpoint directory
is bound at its own path so the overlay's symlinks resolve. The image ships the preparer —
that is how we made ours (note: `flashnext-shared-ram-ple-plan.md:170`):

```
docker run --rm --entrypoint python3 <image> /opt/pennyroyal/scripts/pennyroyal/prepare_ple_nvme.py --source /mnt/fast/models/qwen38-flash-next-nvfp4-mtpfix --output /mnt/fast/ple-shared/prepared-mtpfix
```

exit 0 in **33 s**, 48 G, and its `ple/layer-0.bin` is **51,200,245,760 B** with
`ssd-stream.json` recording `rows=320,001,536 columns=160 dtype=float8_e4m3fn` and the
payload `sha256 b070f9644adf9379…` (note: `flashnext-shared-ram-ple-plan.md:173`,
`:175`, `:176`). Two plugin guards
make this ordering mandatory rather than stylistic: `plugin.py::_check_prepared_checkpoint_is_overlay`
aborts if the served index still contains `ngram_embedding.shard_*` (the loader would write
into a mapping we protect read-only), and `plugin.py::_bind_manifest_to_checkpoint` hashes
the served `config.json`/index against the manifest's recorded `source.path`,
`config_sha256`, `index_sha256` — without that binding a stale-but-self-consistent
(payload, manifest) pair passes every other check, because every checkpoint of this
architecture has the same table geometry.

Use the **mtpfix** reshard, not the pre-fix artifact: the pre-fix checkpoint's draft MoE
experts are fp8-serialized and loaded un-scaled, which floors NEXTN accept at ~1.46
(repo: `bench/baseline/BASELINE-PIN.md:35`; the directory we serve is named at
`:26`).

**Serve recipe.** Third-party, not in this repo — see §6.

**Host.** Two cards, ~95.6 GiB each, and enough RAM to hold one pinned 47.68 GiB table
plus both engines. Our box is **244 GiB** (note: `flashnext-shared-ram-ple-plan.md:40`;
`bench/baseline/BASELINE-PIN.md:83`). What the change bought, measured on the pair
(note: `flashnext-shared-ram-ple-plan.md:283`):

| quantity | before (private `ram`) | after (`shared_ram`) |
|---|---|---|
| host `used` | 179 GiB | **89 GiB** |
| host `available` | 64 GiB | **155 GiB** |
| replica anon (own cgroup) | 53.35 GiB, of which 47.68 GiB was the private table (note: `flashnext-shared-ram-ple-plan.md:291`) | **6.04 GB / 6.01 GB** (note: `flashnext-shared-ram-ple-plan.md:283`) |
| `cudaHostRegister` of the table | — | **2.14 s / 2.08 s** at boot (note: `flashnext-shared-ram-ple-plan.md:283`); 2.3–2.4 s measured at Phase B (note: `flashnext-shared-ram-ple-plan.md:181`) |
| sha256 recomputed **through the read-only mapping** | — | **12.9 s / 12.6 s** (note: `flashnext-shared-ram-ple-plan.md:283`; the host-side hash at idle priority is ~69 s, note: `flashnext-shared-ram-ple-plan.md:327`) |
| boot at the 24 GiB caps, cold page cache | — | `memory.max=25769803776`, `memory.peak` exactly **24.00 GiB**, `oom_kill 0`, both replicas reached `The server is fired up and ready to roll!` (note: `flashnext-shared-ram-ple-plan.md:301`) |

Retired figure, kept so nobody re-quote it: the single-replica trial's `available 62 →
111 GiB` was three copies becoming two, not a saving (note: `flashnext-shared-ram-ple-plan.md:262`).

Sizing consequences for your box:

- `/dev/shm` (or your dedicated tmpfs) must hold the table **and** a transient second copy:
  `ple-stage.sh` copies to `.staging.<ts>.$$`, hashes it, then `mv -T` publishes. Its own
  header notes that a unit killed between `cp` and `mv -T` strands 47.68 GiB of charged
  tmpfs and "two of those exceed /dev/shm and the next copy dies with ENOSPC"; the script
  sweeps `.staging.*` older than `ORPHAN_MAX_AGE_MIN=180` minutes.
- Do **not** give `ple-stage` a memory limit. The compose header says it plainly: "Never
  given a memory limit: while it runs, the 47.68 GiB of tmpfs pages it writes are charged
  to this container's own cgroup" — which is also why the replicas read as ~6 GiB.
- The 24 GiB replica cap is only valid while the table is shared. compose is explicit:
  "unset `PENNY_SHARED_PLE` under this cap OOMs the replica rather than reverting it —
  rollback is the old 108 GiB projects, not flipping the knob."
- Keep ~53 GiB free per old single replica you might want to resurrect: the compose header
  says each old replica "needs ~53 GiB of free RAM to come back in ram mode", and this file
  *is* the rollback path.
- The payload's mode and the bind's mode are load-bearing. `plugin/sglang_shared_ram_ple/backend.py` records why:
  the substrate must be shmem (a block-device page-cache file is rejected with
  `cudaErrorInvalidValue`) and the VMA must be CPU-writable
  (`cudaDevAttrHostRegisterReadOnlySupported=0`), so `/ple` is bound **rw** and the file is
  `root:975 0660` with `group_add: ["975"]` on the replicas. Read-only safety is restored by
  the unconditional, fatal `mprotect(PROT_READ)` after registration.
- The plugin dist must be on `PYTHONPATH` as a real distribution directory. The package
  files live in this repo at `plugin/sglang_shared_ram_ple/` — the layout the tooling
  expects (see `SRC_PKG` / `DIST` / the generated `RECORD` in `deploy-shared-ram.sh`, and
  the import in `selftest.py`): a directory named `sglang_shared_ram_ple` containing
  `__init__.py`, `plugin.py`, `backend.py`, `config.py`, with `selftest.py` as its sibling
  at `plugin/`. Note the dist that `deploy-shared-ram.sh` builds renames the package to
  `shared_ram_ple` inside `$DIST` — that is the *installed* name the engine's plugin
  registry loads; the source name is the one above.
  `SRC_PKG` defaults to `/var/tmp/ple-shared-ram/sglang_shared_ram_ple`, and the deploy
  script's failure message warns that `/var/tmp` is aged at 30 days by tmpfiles.d — a
  missing `SRC_PKG` after a quiet month is expected, not a missing dependency.

---

## 2. Boot order, and why it is the first thing that breaks

Two different mechanisms publish the table, because two different things start the pair:

1. **`sudo docker compose -f compose.yaml up -d`** — `depends_on` does the ordering:
   `ple-stage` must reach `service_completed_successfully` before either replica starts,
   and `flash-next-b` additionally waits on `flash-next-a: condition: service_healthy`.
   The compose header explains why A and B are one project: "`depends_on` is evaluated
   inside a project only" — with A and B separate they cold-started together and raced
   writing the Triton/JIT cache (note: `flashnext-shared-ram-ple-plan.md:277`, proven live at `:281`).
2. **Boot** — the Docker daemon restarts containers from their own policies and honours no
   `depends_on`, so `ple-stage-onboot.service` (`Before=docker.service`,
   `DefaultDependencies=no`, `RequiresMountsFor=/mnt/fast`, `Type=oneshot`,
   `RemainAfterExit=yes`, `TimeoutStartSec=420`) stages the payload before dockerd exists.
   Alongside it, `flash-next-pair-boot.service` runs `docker compose … up -d` at boot so the
   ordering above is what actually applies. (See §7.3 for why the pair's `restart` policy is
   `"no"` and how those two units interlock.)

The table must be staged **before** any replica starts. Nothing gets staged against a live
replica by accident, and the vendored code refuses the related operations explicitly:

- `ple-stage.sh` rule, quoted: "never truncate a file a live replica may have mapped+pinned:
  copy to a temp name, verify sha256 against the manifest, publish with an atomic rename
  (unlinking a mapped file is safe -- its pages live under the existing mapping; truncating
  one is not)". A right-size/wrong-bytes payload is *replaced* under a new inode, and the log
  line names the old inode so `ple-shared-ram-check.sh` can alarm on the split.
- `backend.py` aborts a boot whose payload was swapped underneath it (`"shared table was
  replaced under us"`), comparing the `fstat` of the descriptor it mapped against a later
  pathname re-lookup.
- `deploy-shared-ram.sh --clean` refuses outright: `REFUSING to retire the dist while $c
  exists:` — removing the dist under a live pair means the next `up` boots with
  `SGLANG_PLUGINS=shared_ram` resolving to nothing and serving a **private** table until the
  cap OOM-kills it.
- `ple-shared-ram-check.sh` fails with `maps inode … but the host file is … (republished
  under a running replica)`.

`ple-stage-onboot.sh` stages by calling the same `ple-stage.sh` that the compose one-shot
runs, so a boot run inherits its distrust of tmpfs contents. Because tmpfs is wiped at
boot, a boot run normally takes the fresh-copy path — `cp`, `sha256sum` the temp file
against `tables[0].sha256` from the manifest, `chown root:$GID` / `chmod 0660`, `mv -T` —
and if a payload *is* already present the stager re-hashes it before any replica maps it,
accepting the existing bytes only on a stamp that is root-owned `0600` and keyed on the
file's current `(inode,size,mtime)`. Why a stale payload is worse than none, from the
stager's own header: the old fast path trusted a size match alone, and since "every
checkpoint of this architecture has the same table geometry", a right-size/wrong-bytes
file "would be published again and only met a hash once a replica had already pinned it,
where the failure is an aborted boot, not a repair." A missing payload, by contrast, fails
closed immediately with a named cause (note: `flashnext-shared-ram-ple-plan.md:330`).

Two operational notes on that unit:

- It is **conditional**. Before copying anything it reads Docker's own on-disk intent from
  `<data-root>/containers/<id>/hostconfig.json` (restart policy) and `config.v2.json`
  (`HasBeenManuallyStopped`, `HasBeenStartedBefore`) and stages only when dockerd itself
  intends to start the pair — `PLE_STAGE_ONBOOT_DRY_RUN=1` prints that verdict. Run it after
  any restart-policy change: while the policy was briefly `on-failure:3` the predicate did not
  recognise the policy and would have skipped staging while Docker still started the
  replicas — a crash-loop caused by "a unit that silently disables itself"
  (note: `flashnext-shared-ram-ple-plan.md:289`).
- Its cost is at boot, not in the serving window: the measured boot published
  `bytes=51200245760 sha=b070f964… inode=3 seconds=82` and both replicas started one second
  after the unit finished (note: `flashnext-shared-ram-ple-plan.md:297`); total boot 4 min 23.9 s, both
  replicas serving ~4 min after power-on (note: `flashnext-shared-ram-ple-plan.md:303`).

The hourly witness is a separate unit and it checks the **live** artifact, not the source
tree: `ple-shared-ram-check.service` runs `/usr/local/bin/ple-shared-ram-check.sh --hash`,
and the timer fires `OnBootSec=15min`, `OnUnitActiveSec=1h`, `RandomizedDelaySec=900`. The
payload it re-hashes is `$PLE_SHARED_DST` (default `/dev/shm/ple-shared/layer-0.bin`) and the
mappings it inspects are those of the running processes; the `/mnt/fast` source tree is never
read. `Nice=19` + `IOSchedulingClass=idle` + `TimeoutStartSec=900` exist because that is a
47.68 GiB read next to live traffic — ~69 s idle-class versus ~13 s at normal priority
(`ple-shared-ram-check.sh` header).

---

## 3. The sharing gate — one physical table, proven

Do not trust the compose file, the env vars, or a healthy port. The only proof that two
replicas share one table is that the host file's `(dev, inode)` equals the
`dev=… inode=…` in each replica's `/ple/layer-0.bin` mapping, and that each such mapping is
`r--s`. Everything else is configuration; this is the observation.

Run the shipped checker (this is the command the timer runs):

```
/usr/local/bin/ple-shared-ram-check.sh            # identity + perms, ~1 s, safe at any time
/usr/local/bin/ple-shared-ram-check.sh --hash     # also re-hashes 47.68 GiB
```

Exit codes, from its header: `Exit 0 = PASS, 1 = FAIL (reason on the last line), 2 = cannot
determine.`

The manual form, assembled from the two lines the checker itself uses (`docker top -o pid`,
then a `grep` of that pid's maps) — verbatim from `plugin/ple-shared-ram-check.sh`:

```
pids=$($SUDO docker top "$c" -o pid 2>/dev/null | tail -n +2)
m=$($SUDO grep -h "$(basename "$DST")" /proc/$p/maps 2>/dev/null | head -1)
read -r host_ino host_dev host_size <<<"$($SUDO stat -c '%i %D %s' "$DST")"
```

so, expanded for one container:

```
c=flash-next-pair-flash-next-a-1
DST=/dev/shm/ple-shared/layer-0.bin
for p in $(sudo docker top "$c" -o pid | tail -n +2); do sudo grep -h "$(basename "$DST")" /proc/$p/maps | head -1; done
sudo stat -c '%i %D %s' "$DST"
```

Compare field 2 (perms) = `r--s`, field 4 (`dev`), field 5 (`inode`) against `stat`. The dev
fields are not in the same notation and a naive string compare fails — the checker's
`norm_dev` exists for exactly this, quoted: "/proc/PID/maps prints the device as
"major:minor" in hex ("00:1b"); stat %D prints the raw device number in hex with no padding
or colon ("1b"). Compare one form." Copy the checker's `stat -c '%i %D %s'` rather than
inventing a format of your own; the common `%d:%i` shorthand is not what it compares.

Two independent corroborations, both cheap:

- The plugin's own boot line (`plugin/sglang_shared_ram_ple/backend.py`):
  `shared_ram PLE: inode=%s %s bytes=%.2f GiB registered in %.2fs pin_pages=%d/%d devptr_eq_host=True mprotect=ok sha=%s`.
  The acceptance test is that both replicas print the **same** inode — `deploy/cutover.sh`
  greps exactly that: `grep -oE "shared_ram PLE: inode=[0-9]+"` on each container's logs and
  says `(must be the SAME inode: one physical table)`.
- `pin_pages=` is a *ratio*, not a truthiness test. `nr_foll_pin_acquired` is box-wide, so
  two replicas pinning the same pages read **2× expected** — that doubling is the sharing
  signature (note: `flashnext-shared-ram-ple-plan.md:299`); `deploy-shared-ram.sh` therefore requires
  `>=`, never exactness.

**Do not gate on mtime.** The payload's `mtime`/`ctime` post-date both boots on our box, and
that is expected rather than alarming: measured on tmpfs, neither `open(O_RDWR)` nor a
writable `MAP_SHARED` mapping moves those fields — only a **store through the mapping**
does, and the live payload accepted one during a replica's `cudaHostRegister` window with
its bytes verifying unchanged afterwards (note: `flashnext-shared-ram-ple-plan.md:337`; same reasoning in
`plugin/sglang_shared_ram_ple/backend.py`'s module docstring and
`plugin/ple-shared-ram-check.sh`: "timestamps are
an alarm, never an identity: identity is (dev, inode, size) from fstat, content is the
sha256"). A health check or alert that watches mtime will therefore fire on normal boots and
stay quiet on the failure that matters. Identity is the `fstat` of the descriptor that was
mapped (`identity_from_fd` in `backend.diagnostics()`), and content is the sha256 recomputed
through the read-only mapping — never a second lookup of the pathname, which a republish
between `open` and check could make describe a file the process is not reading.

---

## 4. Names to edit for your box

Everything below is a host path, an identifier, or a device choice that we happened to have.
Grep each value in `deploy/` and `plugin/` and replace it consistently; the failure mode of
half-editing is a service that starts and does the wrong thing.

| item | declared in | our value | notes |
|---|---|---|---|
| compose project name | `compose.yaml` → `name:` | `flash-next-pair` | **containers are derived from it**: `<name>-<service>-1` = `flash-next-pair-flash-next-a-1`, `-b-1` (as hardcoded in `cutover.sh` and `ple-shared-ram-check.sh`) |
| pair directory on the host | paths inside the two `.service` files | `/mnt/apps/compose/sglang-flash-next-pair` | `Documentation=`, `ConditionPathExists=`, and `ExecStart=/usr/bin/docker compose -f …` all carry it |
| boot-start sentinel | `flash-next-pair-boot.service` | `/mnt/apps/compose/sglang-flash-next-pair/boot-start.enabled` | delete it to keep the pair down across a reboot (deliberate when the GPUs/RAM go to another model) |
| tmpfs payload | `ple-stage.sh` `DST`/`PUB`; `ple-stage-onboot.sh` `PLE_STAGE_DST`/`PLE_STAGE_PUB`; checker `PLE_SHARED_DST`/`PLE_SHARED_MANIFEST` | `/dev/shm/ple-shared/{layer-0.bin,ssd-stream.json}` | container side is `/ple` (compose) |
| prepared payload (source of truth) | compose `/prep` + `/model` binds; `PLE_STAGE_SRC`, `PLE_STAGE_MAN` | `/mnt/fast/ple-shared/prepared-mtpfix` | the manifest's `bytes`/`sha256` are what everything verifies against |
| checkpoint | compose bind (source == target, so overlay symlinks resolve) | `/mnt/fast/models/qwen38-flash-next-nvfp4-mtpfix` | must be the mtpfix reshard (§1) |
| API key file | compose bind → `/api-key` | `/mnt/fast/models/llama-api-key` | read from inside the container; never printed, never in argv (`cutover.sh` rule) |
| plugin dist (host) | `deploy-shared-ram.sh` `DIST`; compose `/plugins` bind | `/mnt/apps/sglang-flashnext/shared-ram` | served read-only; the parent dir must not be group/world-writable (note: `flashnext-shared-ram-ple-plan.md:331`) |
| plugin sources (host) | `deploy-shared-ram.sh` `SRC_PKG` | `/var/tmp/ple-shared-ram/sglang_shared_ram_ple` | tmpfiles ages `/var/tmp` at 30 d — re-materialise from the repo copy |
| per-replica caches | compose `/cache`, `/nixl` binds | `/mnt/apps/sglang-flashnext/cache`, `…/nixl` and `/mnt/apps/sglang-flashnext-b/{cache,nixl}` | A and B must not share them; that is the JIT-race fix |
| serve recipes | the two `/opt/pennyroyal/configs/pennyroyal/…` binds | `/mnt/apps/sglang-flashnext/serve-flash-next.sh`, `…/serve-flash-next-frspec.sh` | §6 |
| payload group id | compose `group_add: ["975"]`; `GID`/`PLE_STAGE_GID` in both scripts | `975` (our host group is `pledata`, note: `flashnext-shared-ram-ple-plan.md:297`) | must be the group that gets `0660` on the payload |
| engine image | compose `image:` (replicas **and** `ple-stage`), `deploy-shared-ram.sh` `IMG` | `ENGINE_IMAGE` (one value for all three; defaults to the upstream digest in `deploy/.env.example`) | re-run `--stage` per image so the guard matches |
| GPU selection | compose `deploy.resources.reservations.devices[].device_ids` | `GPU_UUID_A`, `GPU_UUID_B` (required; compose refuses to resolve without them) | one UUID per replica, never `--gpus all`; get yours from `nvidia-smi -L`. `deploy-shared-ram.sh --gates` needs `LEG_GPU_UUID` + `CONFIRM_GPU_LEG=1` precisely because there is no safe default card |
| published ports | compose `ports:` | A `30071:8001`, B `30073:8001` | `cutover.sh` hardcodes `30071`/`30073` in its `health()`/`wait_healthy()` calls |
| host install of the scripts | the three `.service` files | `/usr/local/bin/{ple-stage.sh,ple-stage-onboot.sh,ple-shared-ram-check.sh}` | `PLE_STAGE_SCRIPT` overrides the stager path |
| plugin baseline manifest | `deploy-shared-ram.sh` `BASELINE`; checker `PLE_SHARED_SHA_BASELINE` | `/usr/local/share/ple/deployed-sha256.txt` | without it the checker prints `baseline: none recorded … skipped`, i.e. deployed bytes unattested |
| Docker data-root for the boot predicate | `ple-stage-onboot.service` → `Environment=DATA_ROOT=` | `/mnt/fast/docker` | otherwise the script probes `/etc/docker/daemon.json`, then `/mnt/fast/docker`, `/var/lib/docker` |

---

## 5. Rollback

`deploy/cutover.sh` is the record of how the swap was actually done, and its rules are the
rollback contract (all quoted from its header):

- "never take both replicas down at once" — one replica always stays healthy so the pool
  keeps serving; `gate 0` refuses to touch A unless B is already `200` and smoke-tested.
- "each replacement must pass health + smoke before the next old replica is touched, and any
  failure restores the old project it displaced" — `wait_healthy` then `evidence` (plugin log
  lines, `memory.stat` anon/file/shmem, the `/ple` bind) then `smoke`, and `restore_old` on
  failure, including after a replica that starts but never becomes healthy.
- "old projects are `stop`ped, never `down`ed, so rollback is `docker start` (or `up -d`) on
  a file that was never edited." The pre-shared_ram single-replica projects were **preserved
  untouched, not edited in place**; they are stopped, not deleted.
- "the API key is read from the file mounted inside the container; it is never printed and
  never appears in argv."
- run it detached (`systemd-run`) "so the window contains no per-tool-round model calls of
  ours, which are pool members too" — and remember that measurement said neither replica was
  spare (note: `flashnext-shared-ram-ple-plan.md:285`, A=33/B=10 req per 10 min).

To roll back by hand, free the pair's slot one replica at a time (compose header):

```
cd /mnt/apps/compose/sglang-flash-next-pair
sudo docker compose stop flash-next-a                     # frees :30071 + GPU0
(cd ../sglang-qwen38-flash-next   && sudo docker compose up -d flash-next-opt)
sudo docker compose stop flash-next-b                     # frees :30073 + GPU1
(cd ../sglang-qwen38-flash-next-b && sudo docker compose up -d flash-next-opt-b)
```

Going forward again is the same dance in reverse: stop the old projects, then
`sudo docker compose -f compose.yaml up -d` (stager → A healthy → B). Reserve the RAM: each
old replica wants ~53 GiB to come back in `ram` mode (§1). And note what `cutover.sh` step 3
does that the shipped compose no longer says: it finishes with
`docker update --restart=unless-stopped "$CTA_NEW" "$CTB_NEW"`. If you re-run it today, the
pair returns to crash-restart and `ple-stage-onboot.service` starts staging again — see §7.3
and the compose header's "CURRENT POLICY (2026-10-06): `no`".

Retiring the plugin itself is `deploy-shared-ram.sh --clean`, which moves the dist to
`${DIST}.disabled-<ts>` rather than deleting it, and refuses while either pair container still
exists (§2).

---

## 6. The serve recipe: fragments over the image's config dir

The pair does not pass engine argv in compose. `command: ["next"]` selects a recipe *inside
the image*, and compose then bind-mounts two host files over the image's own copies:

```
- {type: bind, read_only: false, source: /mnt/apps/sglang-flashnext/serve-flash-next.sh, target: /opt/pennyroyal/configs/pennyroyal/serve-flash-next.sh}
- {type: bind, read_only: false, source: /mnt/apps/sglang-flashnext/serve-flash-next-frspec.sh, target: /opt/pennyroyal/configs/pennyroyal/serve-flash-next-frspec.sh}
```

So **changing argv means editing the bind-mounted host file and recreating the container —
not editing the image.** Both replicas mount the *same* two host files (the B-only fork was
closed once `MEM_FRACTION_STATIC` became env-settable inside the shared copies), which keeps
a single recipe source for the pair.

The recipe is third-party. It is quoted with attribution in the README and deliberately **not
shipped here**; the copies we run are identified by hash:

| file | size | sha256 |
|---|---|---|
| `serve-flash-next.sh` | 10,976 B | `50832edfab81e7464c39eb390a582aef6556ec0c84a00a3fdeda800415777312` |
| `serve-flash-next-frspec.sh` | 10,930 B | `37e43d6f944dc7b952f68a5da53ae50db1a2280163016fae38a7e26320f551c2` |

Verify what is deployed against those values before comparing any behaviour claim:
`sha256sum /mnt/apps/sglang-flashnext/serve-flash-next*.sh`.

Two recipe-adjacent facts that cost us time and are worth knowing before you touch argv:

- `PENNY_PLE_BACKEND` is **not ours to set**. The image's
  `/opt/pennyroyal/configs/pennyroyal/ple-backend.sh` owns it, validates a closed
  `{ram, nvme}` domain and exits before the plugin loader runs — which is exactly how our
  first trial boot died (`PENNY_PLE_BACKEND must be ram or nvme`). The plugin therefore
  selects itself with `PENNY_SHARED_PLE=1` and leaves the engine on `ram`, keeping
  `--ple-offload-embedding` and `PLE_OFFLOAD_EMBEDDING=true` and swapping only the allocator;
  `config.check_engine_backend()` aborts if the two are ever combined with `nvme`
  (note: `flashnext-shared-ram-ple-plan.md:209`).
- The pool-size knobs are env-readable, so no recipe fork is needed: `MAX_TOTAL_TOKENS`
  (via `ple-backend.sh`, positive and 64-page aligned) and `MEM_FRACTION_STATIC` (default
  `0.981` in the recipe, set to `0.998` from compose). Read the resulting pool from the boot
  log, not from the ask — compose's comment explains why the ask is inert on this card.

---

## 7. Sharp edges we already hit

### 7.1 A single-file bind mount is invisible in a *running* container

Editing a bind-mounted file on the host does not reach a live container, and compose will
not tell you: it hashes service config, not bind contents. The warning shipped at the end of
`deploy-shared-ram.sh`, verbatim:

```
NOTE: the running replicas still execute the code they imported at boot.
      Bind-mounted file changes do NOT reach a live container, and compose
      hashes service config, not bind contents -- so 'up -d' reports Running.
      Apply with: docker compose -f compose.yaml up -d --force-recreate flash-next-a
      one replica at a time, gating on health between them.
```

This bit us on the recipe and on the plugin (same shape as the proxy-side story in our notes:
a service reads stale config not because the file is wrong but because the running container
never re-opened it). `--force-recreate`, one replica at a time, health-gated between them
(note: `flashnext-shared-ram-ple-plan.md:325`, where the hardening batch was landed: A healthy in 133 s, B in
147 s, still `inode=3`, still `perms=r--s`).

### 7.2 The 24 GiB caps were not load-bearing at deploy time

The pair came up at the old **108 GiB** limits and the cap was applied later the same night
with `docker update`, no recreate (note: `flashnext-shared-ram-ple-plan.md:279`, `:287`):

```
docker update --memory 24g --memory-swap 24g
```

Two lessons in that sequence. (a) `deploy.resources.limits.memory` alone leaves cgroup
`MemorySwap` at 2× memory, i.e. swap allowed; it is the **service-level** `memswap_limit`
equal to `memory` that yields `memory.swap.max=0` (note: `flashnext-shared-ram-ple-plan.md:46` — the 0.2 fix
records the empirical syntax test). (b) A cap only protects you once it is in force: verify
it against the live cgroup (`memory.max`, `memory.events` `oom_kill`) and mirror it into the
compose file, because the next plain `up -d` reconciles drift **by recreating** (note:
`flashnext-shared-ram-ple-plan.md:287`, and the compose header's "keep file == live"). The cap's purpose is the
fail-open backstop the hook enforcement cannot be — a silently-fallback private allocation
dies at boot instead of serving; it is explicitly *not* a working-set limit, and page cache
from reading ~60 GB of weights legitimately presses it (note: `flashnext-shared-ram-ple-plan.md:291`: peak
54.28 GiB in shared mode, resolved by reclaim, `oom_kill 0`).

### 7.3 The boot-time race between the staging unit and the replicas

Three moving parts have to agree, and each disagreement has already happened once:

- The daemon restarts containers from their policies with no stager and no ordering, so
  `ple-stage-onboot.service` must exist and must be `Before=docker.service` (its header:
  "Without this unit the pair reboot-crash-loops with serving down").
- The predicate in `ple-stage-onboot.sh` must recognise whatever restart policy you are
  actually using. When the policy was briefly `on-failure:3` the accepted set was
  `always|unless-stopped` only, so the stager would have skipped while Docker still started
  the replicas. It now accepts `always|unless-stopped|on-failure`; re-check with
  `PLE_STAGE_ONBOOT_DRY_RUN=1` after **any** policy change (note: `flashnext-shared-ram-ple-plan.md:289`).
- The shipped steady state is `restart: "no"` plus `flash-next-pair-boot.service` plus the
  `boot-start.enabled` sentinel, because no Docker policy expresses "start at boot but NOT
  after a crash" and this pair runs past the margin the stock config was proven at — a crash
  must survive as an `Exited` container rather than being auto-healed out of sight. With the
  policy at `no`, the onboot unit correctly **SKIPPS** and the boot unit's `compose up -d`
  runs the stager in order, so nothing can start unstaged. That split is deliberate, not
  drift (note: `flashnext-shared-ram-ple-plan.md:337`). If you set the policy back to `unless-stopped` you
  make the onboot stager load-bearing again and reintroduce crash-restart.
- Corollary, said plainly in the compose header: `docker start <container>` alone is **not**
  a valid recovery after a host reboot — tmpfs does not survive, `/ple` would be empty, and
  the plugin fails closed. Use `sudo docker compose -f compose.yaml up -d`.

### 7.4 The power-cap drift trap

A systemd unit's `ExecStart` is a statement of intent, not a reading. Our power-limit unit
declares `ExecStart=/usr/bin/nvidia-smi -pl 300` with no `-i` (so, all GPUs),
`Requires=nvidia-persistenced`, `Before=docker.service` — and at capture both cards actually
read `power.limit = 450.00 W`, set by hand from root history; the hand value is
non-persistent and a reboot returns 300 W (repo:
`bench/baseline/BASELINE-PIN.md:66`, `:68`). Anything you measure while the live cap
disagrees with the unit file is a different configuration than you think, and the throttle
counters will not save you — `clocks_event_reasons.active` read `0x0` on a genuinely capped
card, so use draw-vs-limit proximity plus clock depression instead (note:
`flashnext-shared-ram-ple-plan.md:38`). Read the limit, every time, before a measurement or a comparison:

```
nvidia-smi --query-gpu=index,power.limit,power.default_limit --format=csv
```

The vendored bench harness samples the same thing its own way and voids a leg whose enforced
limit deviates — `bench/bench-matrix.py:109`:

```
"--query-gpu=power.draw,clocks.sm,temperature.gpu,enforced.power.limit,utilization.gpu",
```

(note: `flashnext-shared-ram-ple-plan.md:42` for the void-on-deviation rule.)

### 7.5 Other things worth knowing before they bite

- **`--clean` and the plugin dist.** See §2/§5: retire it only with the pair down.
- **Memory is judged from the cgroup, never from the GPU.** `nvidia-smi memory.used` sits at
  ~94 GB because the KV pool is preallocated, and `memory.current` is page-cache dominated;
  size from `memory.stat anon`/`file` and host `MemAvailable`. Driver-pinned pages are
  invisible to cgroup `unevictable` — it reads 0 with 47.7 GiB registered
  (note: `flashnext-shared-ram-ple-plan.md:13`, `:291`).
- **`/var/tmp` ages.** `SRC_PKG` under `/var/tmp` disappears after ~30 quiet days and
  `deploy-shared-ram.sh` tells you so rather than failing obscurely.
- **Fail-closed only survives as `SystemExit`.** `load_plugins()` runs each plugin's
  `register()` under `try/except Exception`, so an ordinary exception is logged and the boot
  continues into a private table. Every active-config failure in `plugin.py` is inside the
  wrapper that converts `BaseException` → `SystemExit`. If you edit `register()`, nothing may
  raise before that wrapper (note: `flashnext-shared-ram-ple-plan.md:211`).
- **Sharing is a single point of failure, by construction.** One physical copy means one
  corruption event takes out both replicas, and because read-only registration is unsupported
  on this driver the pinned pages are device-writable. The cover is the hourly checker, not
  the code (quoted in `plugin/sglang_shared_ram_ple/backend.py`'s docstring as "Residual risk inherent to sharing,
  not to this code").
- **`selftest.py` is the pre-flight for a new box.** It calls the exact function the plugin
  installs at the served size (320,001,536 × 160 fp8) and independently re-derives the
  shard-major layout from the live checkpoint across all 128 shards; `SELFTEST|layout|…
  "mismatch": 0` is the line to look for. Its docstring shows the `docker run` form.
- **The `--gates` legs are not idle-safe.** They open a CUDA context and re-hash the table
  twice; `deploy-shared-ram.sh` refuses to guess a card (`LEG_GPU_UUID` + `CONFIRM_GPU_LEG=1`)
  and deliberately omits `--ulimit memlock` so the gate runs with the same privileges
  production has.

---

## Sources

- This repo: `deploy/compose.yaml`, `deploy/ple-stage.sh`, `deploy/ple-stage-onboot.sh`,
  `deploy/ple-stage-onboot.service`, `deploy/ple-shared-ram-check.service`,
  `deploy/ple-shared-ram-check.timer`, `deploy/flash-next-pair-boot.service`,
  `deploy/cutover.sh`, `plugin/{__init__,plugin,backend,config,selftest}.py`,
  `plugin/deploy-shared-ram.sh`, `plugin/ple-shared-ram-check.sh`,
  `bench/bench-matrix.py`.
- `models/qwen3.8-flash-next/flashnext-shared-ram-ple-plan.md` in the private working-notes
  repo — the reasoning and the measurements cited above: `:13` (measurement rule, pinned
  pages invisible to `unevictable`), `:40` (host RAM 244 GiB), `:42` (void-on-deviation),
  `:46` (0.2 swap fix and the `memswap_limit` syntax test), `:48` (power-cap ownership),
  `:96` (rw bind / writable-VMA correction), `:170`–`:181` (Phase B: preparer invocation,
  payload bytes, manifest fields, copy/hash/register timings), `:209` (the Phase C selector
  correction), `:211` (the `load_plugins` exception-swallowing fact), `:262` (retired `62 → 111 GiB`
  figure), `:275`–`:291` (Phase D executed: `Shape`, ordering, acceptance, the memory-cap
  backstop landing via `docker update`, sizing rationale), `:293`–`:303` (boot verified by an
  actual reboot: ordering, sharing across a cold boot, cap-at-boot), `:321`–`:337` (four-lens
  review and the hardening that shipped, incl. the integrity checker, the `/mnt/apps` mode
  fix, and the mtime-is-an-alarm finding).
- `bench/baseline/BASELINE-PIN.md` in this repo — the pinned identity every later phase
  diffs against: `:12`–`:19` (image reference and digest, sglang rev, driver, per-card
  link state; the cards are named by variable, not by UUID), `:26` (the
  served mtpfix checkpoint), `:30`/`:32` (the qualified 51,200,245,760 B PLE figure and its
  128-shard layout), `:35` (the pre-fix draft defect and its fix), `:66`–`:68` (power-cap
  ownership vs live state), `:83` (host memory at capture).
- [RESULTS.md](RESULTS.md) for what any of this is worth in tokens/s, and
  [benchmarking.md](benchmarking.md) before comparing numbers.
