#!/usr/bin/sh
# Boot-time gate for the shared PLE payload.
#
# Why this exists: tmpfs does not survive reboot, and `depends_on` is a
# `docker compose up` feature — the daemon restarts the pair from their
# `unless-stopped` policies at boot WITHOUT running `ple-stage` and WITHOUT any
# ordering. Without this unit the pair reboot-crash-loops with serving down.
#
# Why it is conditional (the important part): this box runs other models too
# (ninfer 27B, and the pair can be deliberately stopped to free the GPUs/RAM), so
# staging 47.68 GiB of tmpfs on EVERY boot would strand ~19% of a 244 GiB host for
# a model that is not starting. So we publish only when dockerd itself intends to
# start the pair, using the daemon's own predicate (CanBeRestarted):
#     restart policy in {always, unless-stopped}  AND  HasBeenManuallyStopped == false
# Both fields are read from the on-disk container config, because dockerd is not
# running yet when this executes (Before=docker.service).
#
# Verified on this box 2026-10-04: running pair container -> HasBeenManuallyStopped
# False; the two old projects we `compose stop`ped -> True. That is the same field
# the daemon consults, so "wanted" here means "docker will start it".
#
# Ops:
#  PLE_STAGE_ONBOOT_DRY_RUN=1  -> evaluate intent and exit 0 without copying
#  DATA_ROOT=/mnt/fast/docker   -> explicit docker data-root (else discovered)
set -eu

NAME_RE=${NAME_RE:-'^/flash-next-pair-flash-next-[ab]-1$'}
SCRIPT=${PLE_STAGE_SCRIPT:-/usr/local/bin/ple-stage.sh}
SRC=${PLE_STAGE_SRC:-/mnt/fast/ple-shared/prepared-mtpfix/ple/layer-0.bin}
MAN=${PLE_STAGE_MAN:-/mnt/fast/ple-shared/prepared-mtpfix/ssd-stream.json}
DST=${PLE_STAGE_DST:-/dev/shm/ple-shared/layer-0.bin}
PUB=${PLE_STAGE_PUB:-/dev/shm/ple-shared/ssd-stream.json}
GID=${PLE_STAGE_GID:-975}

discover_root() {
  if [ -n "${DATA_ROOT:-}" ]; then echo "$DATA_ROOT"; return; fi
  r=$(python3 - <<'PY' 2>/dev/null || true
import json
try:
    print(json.load(open("/etc/docker/daemon.json")).get("data-root", ""))
except Exception:
    pass
PY
)
  [ -n "$r" ] && { echo "$r"; return; }
  for cand in /mnt/fast/docker /var/lib/docker; do
    [ -d "$cand/containers" ] && { echo "$cand"; return; }
  done
  echo /var/lib/docker
}

ROOT=$(discover_root)
if [ ! -d "$ROOT/containers" ]; then
  echo "PLE-STAGE-ONBOOT|SKIP|no docker data-root at $ROOT (nothing to start anyway)"
  exit 0
fi

# Ask the on-disk config, not the daemon: which pair containers does docker intend
# to start at boot?
wanted=$(DATA_ROOT="$ROOT" NAME_RE="$NAME_RE" python3 - <<'PY'
# Persistence layout on this box (verified 2026-10-04, docker data-root
# /mnt/fast/docker from /etc/docker/daemon.json):
#   containers/<id>/hostconfig.json  -> RestartPolicy.Name, Memory, MemorySwap
#   containers/<id>/config.v2.json   -> Name, HasBeenManuallyStopped, HasBeenStartedBefore
# The policy is NOT in config.v2.json (no HostConfig key there in this version) --
# reading it from the wrong file silently yields "no" and would skip every boot.
import glob, json, os, re
root, pat = os.environ["DATA_ROOT"], os.environ["NAME_RE"]
rx = re.compile(pat)
wanted, seen, unreadable = [], [], 0
for cfg in glob.glob(os.path.join(root, "containers", "*", "config.v2.json")):
    try:
        d = json.load(open(cfg))
    except Exception:
        unreadable += 1          # dockerd may be mid-rewrite; never crash the boot
        continue
    name = d.get("Name") or ""
    if not rx.match(name):
        continue
    rp = "no"
    try:
        h = json.load(open(os.path.join(os.path.dirname(cfg), "hostconfig.json")))
        rp = ((h.get("RestartPolicy") or {}) if isinstance(h.get("RestartPolicy"), dict)
              else {}).get("Name") or "no"
    except Exception:
        unreadable += 1
    manual = bool(d.get("HasBeenManuallyStopped"))
    started = bool(d.get("HasBeenStartedBefore"))
    seen.append("%s(%s%s)" % (name, rp, ",manually-stopped" if manual else ""))
    # docker's own CanBeRestarted predicate: restartable policy AND not hand-stopped.
    # Accept every auto-restart policy docker knows. If a policy ever drops out of
    # this tuple, the stager silently skips while docker still starts the replicas
    # -> crash-loop. That mismatch (comment said it was handled, code disagreed) was
    # caught by preflight on 2026-10-04 when the policy was briefly on-failure:3.
    if rp in ("always", "unless-stopped", "on-failure") and not manual:
        wanted.append(name)
print(json.dumps({"seen": seen, "wanted": wanted, "unreadable": unreadable}))
PY
)
echo "PLE-STAGE-ONBOOT|intent=$wanted"

n=$(printf '%s' "$wanted" | python3 -c 'import json,sys;print(len(json.load(sys.stdin)["wanted"]))')
if [ "$n" = "0" ]; then
  echo "PLE-STAGE-ONBOOT|SKIP|pair is not wanted at boot (stopped, or another model owns the GPUs) -- leaving $(printf '%s' "${DST}") unpublished and its RAM free"
  exit 0
fi

if [ "${PLE_STAGE_ONBOOT_DRY_RUN:-0}" = "1" ]; then
  echo "PLE-STAGE-ONBOOT|DRY-RUN|would stage src=$SRC dst=$DST"
  exit 0
fi

[ -f "$SRC" ] || { echo "PLE-STAGE-ONBOOT|FAIL|prepared payload missing: $SRC" >&2; exit 1; }
SRC="$SRC" MAN="$MAN" DST="$DST" PUB="$PUB" GID="$GID" "$SCRIPT"
echo "PLE-STAGE-ONBOOT|OK|staged for ${n} wanted replica(s)"
