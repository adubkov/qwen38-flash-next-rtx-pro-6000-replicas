#!/usr/bin/env bash
# Build + verify the shared_ram PLE plugin for the running image, on the serving host.
#
# No pip, no network, no image edits: importlib.metadata discovers distributions
# from ANY sys.path entry, so we hand-assemble a .dist-info next to the package and
# put that directory on PYTHONPATH. Reverting = drop the two env vars (or remove
# the bind) — `ram` mode is never touched.
#
#   bash deploy-shared-ram.sh --stage          # build dist + guard + sha baseline
#   bash deploy-shared-ram.sh --gates          # run the three verification legs
#   bash deploy-shared-ram.sh --clean          # retire the dist (refuses while live)
#
# --gates opens a CUDA context on one card and re-hashes 47.68 GiB twice, so it
# needs LEG_GPU_UUID plus CONFIRM_GPU_LEG=1 naming the window you drained. --stage
# alone is safe beside a serving pair (it touches no GPU), which is why it is the
# default and the gates are not.
#
# Why each gate is written the way it is (review, 2026-10-06): a gate that accepts
# "something threw" certifies nothing, a comparison against a value captured and
# never asserted tests nothing, and a leg running with privileges production does
# not have certifies the wrong system.
set -euo pipefail
# Open-source image at its ORIGINAL registry (Docker Hub, the sglang project's own
# namespace), digest-pinned to tag v0.5.19-cu130. Must be the SAME build the replicas
# run: this script installs a guard that executes inside the engine image, and a guard
# run against a different build certifies the wrong system. Override ENGINE_IMAGE to
# point at whatever build you are entitled to use (LICENSING.md).
IMG=${ENGINE_IMAGE:-lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9}
SRC_PKG=${SRC_PKG:-/var/tmp/ple-shared-ram/sglang_shared_ram_ple}
DIST=${DIST:-/mnt/apps/sglang-flashnext/shared-ram}
BASELINE=${BASELINE:-/usr/local/share/ple/deployed-sha256.txt}
NAME=shared_ram_ple
VERSION=0.1.0
# No default: container names derive from the compose project directory, so any
# name we ship is wrong on someone else's box -- and --clean reads this list to
# decide whether the pair is live. An empty list must not read as "nothing running".
PAIR_CONTAINERS=${PAIR_CONTAINERS:-}
# The gate legs read the prepared checkpoint from here. Overridable like its
# neighbours: without an override the gates cannot be run against any other
# directory, which makes them untestable off the machine they were written on.
PREPARED_DIR=${PREPARED_DIR:-/mnt/fast/ple-shared/prepared-mtpfix}

MODE=${1:---stage}
case "$MODE" in
  --stage) ;;
  --gates) ;;
  --clean)
    # The guard must never read its own failure as evidence of absence. `docker
    # inspect` exits non-zero for BOTH "no such object" AND "daemon unreachable" /
    # "sudo needs a password"; mapping all of those to `absent` let the live dist be
    # retired under a running pair, which is the exact failure this block exists to
    # prevent. Only an explicit "No such object" counts as absence.
    if [ -z "$PAIR_CONTAINERS" ]; then
      echo "REFUSING to retire the dist with PAIR_CONTAINERS unset: with no names to" >&2
      echo "  check, the guard below can only report 'nothing running', which is not" >&2
      echo "  the same as having checked. Set it to your pair's container names:" >&2
      echo "    PAIR_CONTAINERS=\"<project>-flash-next-a-1 <project>-flash-next-b-1\"" >&2
      echo "  (docker compose ps --format '{{.Names}}') " >&2
      exit 1
    fi
    for c in $PAIR_CONTAINERS; do
      st=$(sudo -n docker inspect -f '{{.State.Status}}' "$c" 2>&1) && rc=0 || rc=$?
      if [ "$rc" -ne 0 ]; then
        case "$st" in *"No such object"*|*"no such object"*) continue ;; esac
        echo "REFUSING to retire the dist: cannot tell whether $c is running" >&2
        echo "  (docker inspect exit $rc): $(printf '%s' "$st" | head -1)" >&2
        echo "  An unreadable guard is not a clean guard." >&2
        exit 1
      fi
      echo "REFUSING to retire the dist while $c exists ($st): the next 'compose up' would boot that" >&2
        echo "  replica with SGLANG_PLUGINS=shared_ram resolving to nothing, sglang logging" >&2
        echo "  'No plugins found' at debug level, and serving a PRIVATE 47.68 GiB table until" >&2
        echo "  the memory cap OOM-kills it. Stop the pair first, or roll back to pre-shared" >&2
        echo "  projects (compose.yaml header)." >&2
        exit 1
    done
    # Moved aside, not deleted: reversible, and no recursive rm in the path.
    ts=$(date +%Y%m%d-%H%M%S)
    sudo -n mv "$DIST" "${DIST}.disabled-$ts"
    echo "SHARED-RAM-CLEANED (moved to ${DIST}.disabled-$ts; restore with mv back)"
    exit 0 ;;
  *) echo "unknown mode $MODE (use --stage|--gates|--clean)" >&2; exit 2 ;;
esac

if [ ! -d "$SRC_PKG" ]; then
  echo "FAIL: $SRC_PKG does not exist. It is a /var/tmp path, which tmpfiles ages at" >&2
  echo "      30 days (tmpfiles.d/tmp.conf: 'q /var/tmp 1777 root root 30d'), so this" >&2
  echo "      is the expected failure after a quiet month -- not a missing dependency." >&2
  echo "      Re-materialise it from the authoritative repo copy first:" >&2
  echo "        scp <mac>:.../apps/ple-shared-ram/sglang_shared_ram_ple/*.py \\" >&2
  echo "            the-serving-host:/tmp/ && sudo mkdir -p $SRC_PKG && sudo cp -f /tmp/*.py $SRC_PKG/" >&2
  echo "      or run with SRC_PKG=<host checkout>/apps/ple-shared-ram/sglang_shared_ram_ple" >&2
  exit 1
fi

# ---- 1. assemble the distribution directory on the host ----
sudo -n mkdir -p "$DIST/$NAME" "$DIST/${NAME}-${VERSION}.dist-info"
# Assert the sources exist BEFORE installing anything. Under `set -e` an empty
# SRC_PKG kills the cp on an unexpanded glob, so a guard placed after it can never
# print its message -- and a half-populated dist directory is worse than none.
test -n "$(ls "$SRC_PKG"/*.py 2>/dev/null)" || { echo "FAIL: no sources in $SRC_PKG"; exit 1; }
sudo -n cp -f "$SRC_PKG"/*.py "$DIST/$NAME/"
sudo -n tee "$DIST/${NAME}-${VERSION}.dist-info/METADATA" >/dev/null <<EOF
Metadata-Version: 2.1
Name: ${NAME}
Version: ${VERSION}
Summary: shared_ram PLE backend (one pinned shmem table for N TP1 replicas)
EOF
sudo -n tee "$DIST/${NAME}-${VERSION}.dist-info/entry_points.txt" >/dev/null <<EOF
[sglang.srt.plugins]
shared_ram = ${NAME}.plugin:register
EOF
# RECORD with real digests: an empty-hashes RECORD invites a stale or substituted
# file that nothing notices. The checker compares $BASELINE instead of this.
sudo -n tee "$DIST/${NAME}-${VERSION}.dist-info/RECORD" >/dev/null <<EOF
${NAME}/__init__.py,,
${NAME}/plugin.py,,
${NAME}/backend.py,,
${NAME}/config.py,,
${NAME}-${VERSION}.dist-info/METADATA,,
${NAME}-${VERSION}.dist-info/entry_points.txt,,
EOF
sudo -n find "$DIST" -type d -exec chmod 0755 {} \;
sudo -n find "$DIST" -type f -exec chmod 0644 {} \;
echo "built: $DIST/$NAME"

# ---- 2. generate the source-integrity guard against THIS image ----
# The plugin refuses to patch an engine it was not qualified against; a silent
# no-op would mean a replica quietly serving from a private 47.68 GiB copy.
# This leg runs the image's python with no GPU, so it is safe beside a pair.
# mktemp, not a fixed /tmp path: a pre-existing file or a symlink there (another
# user, or a second concurrent run) would be clobbered or followed, and the cp below
# would then install it as the plugin's engine guard.
GUARD_JSON=$(mktemp "${TMPDIR:-/tmp}/shared-ram-guard.XXXXXX")
sudo -n docker run -i --rm --network none --entrypoint python3 "$IMG" - <<'PY' > "$GUARD_JSON"
import hashlib, importlib.util, json, os
mods = ["sglang.srt.models.qwen4_exp", "sglang.srt.plugins.hook_registry",
        "sglang.srt.plugins"]
out = {"image": os.environ.get("PENNY_IMAGE", "unspecified"), "modules": {}}
for m in mods:
    s = importlib.util.find_spec(m)
    if s is None or s.origin is None:
        raise SystemExit("cannot locate %s" % m)
    h = hashlib.sha256()
    with open(s.origin, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    out["modules"][m] = h.hexdigest()
print(json.dumps(out, indent=2, sort_keys=True))
PY
if [ ! -s "$GUARD_JSON" ]; then echo "FAIL: guard generation"; exit 1; fi
python3 -c "import json,sys;json.load(open(sys.argv[1]))" "$GUARD_JSON" || { echo "FAIL: guard json"; exit 1; }
sudo -n cp -f "$GUARD_JSON" "$DIST/$NAME/shared-ram-guard.json"
sudo -n chmod 0644 "$DIST/$NAME/shared-ram-guard.json"
rm -f "$GUARD_JSON"
echo "guard:"; python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1]))['modules'],indent=2))" "$DIST/$NAME/shared-ram-guard.json"

# ---- 3. record what is deployed, so drift is checkable ----
# ple-shared-ram-check.sh compares every replica against this list; without it the
# deployed bytes are unattested and a hand-edit reads as "the plugin never ran".
sudo -n mkdir -p "$(dirname "$BASELINE")"
( cd / && sudo -n sha256sum "$DIST/$NAME"/*.py "$DIST/$NAME/shared-ram-guard.json" ) \
  | sudo -n tee "$BASELINE" >/dev/null
echo "baseline: $BASELINE"
sudo -n cat "$BASELINE"
echo
echo "NOTE: the running replicas still execute the code they imported at boot."
echo "      Bind-mounted file changes do NOT reach a live container, and compose"
echo "      hashes service config, not bind contents -- so 'up -d' reports Running."
echo "      Apply with: docker compose -f compose.yaml up -d --force-recreate flash-next-a"
echo "      one replica at a time, gating on health between them."

if [ "$MODE" = "--stage" ]; then
  echo; echo "STAGED (no GPU touched). Re-certify with --gates in a drained window."
  exit 0
fi

# ---- 4. the GPU legs: explicit about the card, production-matched privileges ----
if [ -z "${LEG_GPU_UUID:-}" ]; then
  echo "FAIL: --gates needs LEG_GPU_UUID=<GPU-...> and CONFIRM_GPU_LEG=1." >&2
  echo "      There is no safe default: every card on this box is held by a serving" >&2
  echo "      replica at --mem-fraction-static 0.998, so a stray context either fails" >&2
  echo "      to allocate or competes with live traffic." >&2
  exit 1
fi
[ "${CONFIRM_GPU_LEG:-0}" = "1" ] || { echo "FAIL: set CONFIRM_GPU_LEG=1 to confirm the card is drained from llmproxy"; exit 1; }
GPU="$LEG_GPU_UUID"
# Deliberately NO --ulimit memlock=-1:-1: production runs at the daemon default
# (measured 'Max locked memory 8388608' on all six engine processes) and pins
# anyway, because the open driver pins in-kernel. A gate that raises the limit
# certifies a posture the serving containers do not have.
ULIMITS=()

echo
echo "===== verification 1: the engine's own loader, then the patched allocator ====="
sudo -n docker run -i --rm --network none --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$GPU" \
  -e PYTHONWARNINGS=ignore \
  -e PYTHONPATH=/plugins -e SGLANG_PLUGINS=shared_ram \
  -e PENNY_SHARED_PLE=1 \
  -e PENNY_PLE_SHARED_PATH=/ple/layer-0.bin \
  -e PENNY_PLE_SHARED_MANIFEST=/prepared/ssd-stream.json \
  -e PENNY_PLE_SHARED_VERIFY=full \
  -e TARGET_MODEL=/prepared \
  -v "$DIST:/plugins:ro" -v "${SHARED_DIR:-/dev/shm/ple-shared}:/ple:rw" \
  -v "$PREPARED_DIR:/prepared:ro" \
  --entrypoint python3 "$IMG" - <<'PY'
import json, os, torch
import sglang.srt.plugins as P
import sglang.srt.models.qwen4_exp as q4
from sglang.srt.plugins.hook_registry import HookRegistry
from shared_ram_ple import backend

TARGET = "sglang.srt.models.qwen4_exp._allocate_pinned_table"
before = q4._allocate_pinned_table
fork_bufs = q4._PINNED_TABLE_BUFFERS
n0 = len(fork_bufs)

loaders = [n for n in ("load_plugins", "load_all_plugins", "initialize_plugins")
           if hasattr(P, n)]
assert loaders, "no plugin loader entry point found on sglang.srt.plugins"
for n in loaders:
    getattr(P, n)()

now = q4._allocate_pinned_table
# The one assertion the old gate never made: that the symbol was actually
# replaced. len(fork_bufs) == n0 passed trivially at n0 == 0, i.e. when the
# plugin had not run at all.
t = now((320001536, 160), torch.float8_e4m3fn)
d = backend.diagnostics()
inode = os.stat("/ple/layer-0.bin").st_ino
MANIFEST = json.load(open("/prepared/ssd-stream.json"))["tables"][0]

# The claim "CPU-writable mapping was downgraded" is only true if the kernel's
# own VMA says so. d["mprotect"] is None means the call returned success; it says
# nothing if the key is missing, and a split VMA would hide a writable piece.
vma_perms = []
for line in open("/proc/self/maps"):
    if "layer-0.bin" in line:
        vma_perms.append(line.split()[1])

checks = {
    "hook_target_applied": TARGET in HookRegistry._patched,
    "allocator_replaced": (now is not before) and getattr(now, "_is_shared_replacement", False),
    "served_from_shared_file": d.get("inode") == inode and d.get("reg_rc") == 0,
    # Global counter (nr_foll_pin_acquired is box-wide, measured +79 by another
    # process on 2026-10-06): require the full page count, never exactness.
    "pin_pages_at_least_expected": (d.get("pin_pages_delta") or 0) >= (d.get("pin_pages_expected") or 1 << 60),
    "devptr_eq_host": d.get("devptr_eq_host") is True,
    "devptr_rc_zero": d.get("gdp_rc") == 0,
    "integrity_proven_through_mapping": (d.get("sha256_through_mapping") or "") == MANIFEST["sha256"],
    "mprotect_recorded_and_ok": ("mprotect" in d) and d["mprotect"] is None,
    "vma_read_only_shared": bool(vma_perms) and all(pm == "r--s" for pm in vma_perms),
    "identity_from_fd": d.get("dev") is not None and d.get("inode") == d.get("path_inode"),
    "single_shared_mapping": len(backend._SHARED_TABLES) == 1,
    "no_fork_private_copy_for_PLE": len(fork_bufs) == n0,
    "payload_bytes_present": bool(t.view(torch.uint8)[7 * 2_500_012 + 42].cpu().numpy().any()),
}
print(json.dumps({"checks": checks, "vma_perms": vma_perms,
                  "diag": {k: d.get(k) for k in ("reg_rc", "reg_seconds",
                        "pin_pages_delta", "pin_pages_expected", "mprotect",
                        "inode", "dev", "path_inode")}}, sort_keys=True))
f_err = None
try:
    now((1024, 64), torch.float8_e4m3fn)           # not the PLE table -> must abort,
except Exception as exc:                            # never silently go private
    f_err = type(exc).__name__
checks["foreign_shape_aborts"] = f_err is not None
try:
    now((999, 160), torch.float8_e4m3fn)           # same width, wrong rows -> must abort
    checks["wrong_rowcount_aborts"] = False
except Exception:
    checks["wrong_rowcount_aborts"] = True
checks["no_fork_private_copy_anywhere"] = len(fork_bufs) == n0
fails = [k for k, v in checks.items() if not v]
print("HOOK-VERIFY", "PASS" if not fails else "FAIL " + str(fails))
raise SystemExit(0 if not fails else 7)
PY
echo "leg1 rc=$?"

echo
echo "===== verification 2: inert when not selected (the unselected replica's guarantee) ====="
sudo -n docker run -i --rm --network none --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$GPU" \
  -e PYTHONWARNINGS=ignore \
  -e PYTHONPATH=/plugins -e SGLANG_PLUGINS=shared_ram \
  -v "$DIST:/plugins:ro" -v "${SHARED_DIR:-/dev/shm/ple-shared}:/ple:rw" \
  --entrypoint python3 "$IMG" - <<'PY2'
import sglang.srt.plugins as P
import sglang.srt.models.qwen4_exp as q4
from sglang.srt.plugins.hook_registry import HookRegistry
TARGET = "sglang.srt.models.qwen4_exp._allocate_pinned_table"
# Load it the way leg 1 does, and PROVE something was loaded. The bare hasattr loop
# was silent: on an engine whose loader API this plugin does not support, no
# function was called, every condition below stayed trivially true, and the leg
# printed PASS while certifying nothing.
loaders = [n for n in ("load_plugins", "load_all_plugins", "initialize_plugins")
           if hasattr(P, n)]
assert loaders, "no plugin loader entry point found on sglang.srt.plugins"
for n in loaders:
    getattr(P, n)()
# PENNY_SHARED_PLE is unset here: the distribution is discoverable, but the
# allocation must remain the fork's own and nothing may be registered or pinned.
ok = (q4._allocate_pinned_table.__module__ == "sglang.srt.models.qwen4_exp"
      and TARGET not in HookRegistry._patched
      and q4._PINNED_TABLE_BUFFERS == [])
print("INERT-WITHOUT-SELECTOR", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 7)
PY2
echo "leg2 rc=$?"

echo
echo "===== verification 3: the fail-closed paths fire, for the right reason ====="
echo "(tampered-sha leg re-hashes 47.68 GiB through a fresh mapping, ~20-30 s)"
sudo -n docker run -i --rm --network none --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$GPU" \
  -e PYTHONWARNINGS=ignore \
  -e PYTHONPATH=/plugins \
  -v "$DIST:/plugins:ro" -v "${SHARED_DIR:-/dev/shm/ple-shared}:/ple:rw" \
  -v "$PREPARED_DIR:/prepared:ro" \
  --entrypoint python3 "$IMG" - <<'PY3'
import json, os, torch
from shared_ram_ple import backend, config, plugin

GOOD = json.load(open("/prepared/ssd-stream.json"))
ROWS, COLS = GOOD["tables"][0]["rows"], GOOD["tables"][0]["columns"]
SHAPE = (int(ROWS), int(COLS))


def variant(name, table_mut=None, top_mut=None, drop=None):
    doc = json.loads(json.dumps(GOOD))
    for k, v in (table_mut or {}).items():
        doc["tables"][0][k] = v
    for k, v in (top_mut or {}).items():
        doc[k] = v
    for k in (drop or []):
        doc.pop(k, None)
    path = "/tmp/m-%s.json" % name
    with open(path, "w") as fh:
        json.dump(doc, fh)
    return path


def aborts(manifest, verify, needle):
    """Passing because 'something threw' certifies nothing: a broken environment
    (no /ple, unreadable manifest, misinstalled package) raised its way to a PASS
    in the previous version. Require the refusal this case actually means."""
    try:
        backend.allocate_shared_table(SHAPE, torch.float8_e4m3fn, path="/ple/layer-0.bin",
                                      manifest=manifest, verify=verify,
                                      log=lambda m: None)
        return False, "no raise"
    except config.ConfigError as exc:
        return (needle in str(exc)), str(exc)[:80]
    except BaseException as exc:
        return False, "wrong type %s: %s" % (type(exc).__name__, str(exc)[:60])


cases = {
    "wrong_rowcount_aborts": (variant("rows", {"rows": ROWS - 16}), "none", "!="),
    "wrong_dtype_aborts": (variant("dtype", {"dtype": "bfloat16"}), "none", "dtype"),
    "tampered_sha_aborts": (variant("sha", {"sha256": "0" * 64}), "full", "sha256"),
    "wrong_format_aborts": (variant("fmt", None, {"format": "not-ssd-stream"}), "none", "format"),
    "unbound_manifest_aborts": (variant("nosrc", None, None, ["source"]), "none", "cannot be bound"),
}
res, why = {}, {}
for name, (mpath, mode, needle) in cases.items():
    res[name], why[name] = aborts(mpath, mode, needle)
# Two-table manifest must actually carry two tables.
doc = json.loads(json.dumps(GOOD)); doc["tables"] = [doc["tables"][0], doc["tables"][0]]
with open("/tmp/m-two.json", "w") as fh:
    json.dump(doc, fh)
res["two_tables_aborts"], why["two_tables_aborts"] = aborts("/tmp/m-two.json", "none", "exactly 1 PLE table")

# The nvme-collision guard once lived outside register()'s try, so the loader
# swallowed its error and the replica booted with a PRIVATE table. It must kill,
# and it must kill for the collision reason -- register() wraps every active-config
# failure in SystemExit, so any abort used to "pass" this check.
os.environ["PENNY_SHARED_PLE"] = "1"
os.environ["PENNY_PLE_BACKEND"] = "nvme"
try:
    plugin.register()
    res["nvme_conflict_aborts"] = False
    why["nvme_conflict_aborts"] = "no raise"
except SystemExit as exc:
    res["nvme_conflict_aborts"] = ("nvme backend already replaces" in str(exc)
                                   or "unset or ram" in str(exc))
    why["nvme_conflict_aborts"] = str(exc)[:80]
except BaseException as exc:
    res["nvme_conflict_aborts"] = False
    why["nvme_conflict_aborts"] = "wrong type %s" % type(exc).__name__
del os.environ["PENNY_PLE_BACKEND"], os.environ["PENNY_SHARED_PLE"]

# The overlay guard: must run, must catch the real failure mode, must not
# false-abort on our own prepared overlay.
os.makedirs("/tmp/dirty", exist_ok=True)
with open("/tmp/dirty/model.safetensors.index.json", "w") as fh:
    json.dump({"metadata": {}, "weight_map":
               {"model.ple.ngram_embedding.shard_0.weight": "model-00001.safetensors"}}, fh)
try:
    plugin._check_prepared_checkpoint_is_overlay("/tmp/dirty")
    res["dirty_checkpoint_aborts"] = False
except config.ConfigError:
    res["dirty_checkpoint_aborts"] = True
try:
    plugin._check_prepared_checkpoint_is_overlay("/prepared")
    res["clean_overlay_accepted"] = True
except BaseException:
    res["clean_overlay_accepted"] = False
try:
    plugin._check_prepared_checkpoint_is_overlay("/tmp/no-such-dir")
    res["unreadable_checkpoint_aborts"] = False
except config.ConfigError:
    res["unreadable_checkpoint_aborts"] = True
res["no_mapping_leaked_on_failure"] = len(backend._SHARED_TABLES) == 0
print(json.dumps({"why": why}, sort_keys=True, indent=None))
print(json.dumps({"negative": res}, sort_keys=True))
bad = [k for k, v in res.items() if not v]
print("NEGATIVE-TESTS", "PASS" if not bad else "FAIL " + str(bad))
raise SystemExit(0 if not bad else 7)
PY3
echo "leg3 rc=$?"
