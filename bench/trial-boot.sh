#!/usr/bin/env bash
# shared_ram PLE — TRIAL BOOT of replica B (on the serving host). Self-restoring.
#
# Safety properties, all learned the hard way on this box:
#  * pristine compose is NEVER edited: the trial runs from a scratch project copy
#    in /var/tmp/ple-trial/proj, and the last act (trap EXIT) restores production
#    compose + force-recreates, whatever happened above it.
#  * restart: "no" during the trial, so a failed boot cannot crash-loop and thrash
#    RAM the way the 2026-09-27 first launch did on the NAS.
#  * replica A is never referenced: it serves this assistant's own tokens.
#  * the API key is read from the file mounted inside the container; it is never
#    printed and never appears in argv.
#  * run this DETACHED (systemd-run) so the measurement window contains no
#    per-tool-round model calls of ours, which are pool members too.
set -uo pipefail

PROJ_PROD=/mnt/apps/compose/sglang-qwen38-flash-next-b
PROJ_TRIAL=/var/tmp/ple-trial/proj
CT=flash-next-pair-flash-next-b-1
PREP=/mnt/fast/ple-shared/prepared-mtpfix
CKPT=/mnt/fast/models/qwen38-flash-next-nvfp4-mtpfix
SHM_DIR=/dev/shm/ple-shared
PLUGIN=/mnt/apps/sglang-flashnext/shared-ram
D=/var/tmp/ple-trial
SUDO="sudo -n"
mkdir -p "$D"
: > "$D/trial.log"
say() { echo "[$(date -u +%T)] $*" | tee -a "$D/trial.log"; }

restore() {
  say "RESTORE: bringing production compose back"
  ( cd "$PROJ_PROD" && $SUDO docker compose -f compose.yaml up -d --force-recreate ) >>"$D/trial.log" 2>&1
  for i in $(seq 1 40); do
    sleep 8
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:30073/health || echo 000)
    [ "$code" = "200" ] && { say "RESTORE: production healthy after $((i*8))s"; break; }
  done
  say "RESTORE: argv spec flags -> $(docker inspect -f '{{join .Config.Cmd \" \"}}' $CT 2>/dev/null | grep -o 'speculative-algorithm [A-Z]*' || echo 'n/a(cmd empty)')"
  # The control arm runs only once production is healthy again, so the pool is
  # never without a second replica while we measure. Fresh boot = fair control.
  if [ "${RUN_CONTROL:-0}" = "1" ]; then
    say "control: same text ladder against restored production B (ram mode)"
    # Fresh outdir, never the captured C1 control: bench-b2.py resumes per
    # outdir, and reusing cells from an earlier boot would silently compare a
    # shared-mode arm against stale numbers.
    OUTC="$D/bench-B-prod-cs124"
    rm -rf "$OUTC"; mkdir -p "$OUTC"
    if ( cd /var/tmp/phase0 && timeout 2400 python3 bench-b2.py --cs "${BENCH_CS:-1,2,4}" \
           --ctxs "${BENCH_CTXS:-1024,4096,65536,262144}" --outdir "$OUTC" ) >>"$D/trial.log" 2>&1; then
      say "control: done"
    else
      say "control: FAILED (see trial.log)"
    fi
    P=$(docker inspect -f '{{.State.Pid}}' "$CT")
    SUB=$(awk -F: '{print $3; exit}' /proc/$P/cgroup)
    say "control: anon for the ram-mode arm (expect ~47 GiB, the thing we remove)"
    grep -E '^(anon|file|shmem) ' "/sys/fs/cgroup$SUB/memory.stat" | tee -a "$D/trial.log"
  fi
  say "TRIAL-DONE"
}
# bash only runs the EXIT trap for a signal it handles, and systemd-run's
# RuntimeMaxSec sends SIGTERM: without this the unit could be killed with B left
# in trial config. Untrapped re-entry guard: restore is idempotent.
trap restore EXIT
trap 'trap - EXIT; restore; exit 143' TERM INT

# ---------- 1. stage the shared table in tmpfs (it does not survive reboot) ----------
say "stage: ensuring the shared table is present in $SHM_DIR"
$SUDO mkdir -p "$SHM_DIR" && $SUDO chmod 0755 "$SHM_DIR"
# NEVER overwrite the payload in place: `cp -f` truncates, and if any replica has
# the file mapped and pinned, the new bytes land in fresh pages while the running
# mapping keeps reading the old ones -- two physical copies and a silently stale
# server. So: copy to a temp name, verify its sha256 against the manifest, then
# rename atomically. A file that merely *looks* right (correct size, wrong bytes)
# still cannot slip through: the plugin re-hashes through its own mapping after
# registering, and aborts the boot on mismatch.
want_sha=$(python3 -c "import json;print(json.load(open('$PREP/ssd-stream.json'))['tables'][0]['sha256'])")
if [ ! -s "$SHM_DIR/layer-0.bin" ] || \
   [ "$(stat -c %s "$SHM_DIR/layer-0.bin")" != "$(stat -c %s "$PREP/ple/layer-0.bin")" ]; then
  t0=$SECONDS
  tmp="$SHM_DIR/.staging.$(date +%s).tmp"
  $SUDO cp "$PREP/ple/layer-0.bin" "$tmp" || { say "FAIL: stage copy"; $SUDO rm -f "$tmp"; exit 1; }
  got_sha=$($SUDO sha256sum "$tmp" | cut -d' ' -f1)
  if [ "$got_sha" != "$want_sha" ]; then
    say "FAIL: staged sha256 $got_sha != manifest $want_sha -- refusing to publish"
    $SUDO rm -f "$tmp"; exit 1
  fi
  $SUDO chmod 0666 "$tmp"                 # 0666: the driver needs a CPU-writable VMA,
                                          # so uid 1000 must open(O_RDWR) it
  $SUDO mv -T "$tmp" "$SHM_DIR/layer-0.bin"
  say "stage: copied, hashed and published in $((SECONDS-t0))s"
fi
$SUDO cp "$PREP/ssd-stream.json" "$SHM_DIR/.ssd-stream.json.tmp"
$SUDO mv -T "$SHM_DIR/.ssd-stream.json.tmp" "$SHM_DIR/ssd-stream.json"
$SUDO chmod 0644 "$SHM_DIR/ssd-stream.json"
ls -l "$SHM_DIR" | tee -a "$D/trial.log"
free -g | head -2 | tee -a "$D/trial.log"

# ---------- 2. scratch project with the shared_ram deltas ----------
rm -rf "$PROJ_TRIAL"; mkdir -p "$PROJ_TRIAL"
$SUDO cat "$PROJ_PROD/compose.yaml" > "$PROJ_TRIAL/compose.pristine.copy" || { say "FAIL: read compose"; exit 1; }
cp "$PROJ_TRIAL/compose.pristine.copy" "$PROJ_TRIAL/compose.yaml"

python3 - "$PROJ_TRIAL/compose.yaml" "$PREP" "$CKPT" "$PLUGIN" "$SHM_DIR" <<'PY' || { say "FAIL: yaml edit"; exit 1; }
import sys
import yaml

path, prep, ckpt, plugin, shm = sys.argv[1:6]
doc = yaml.safe_load(open(path))
svc_name, svc = next(iter(doc["services"].items()))

# volumes: point /model at the prepared overlay (its index drops the n-gram
# shards) and expose the real checkpoint at its own host path so the overlay's
# symlinks still resolve inside the container.
vols = svc.get("volumes", [])
seen_model = False
out = []
for v in vols:
    tgt = v.get("target") if isinstance(v, dict) else str(v).split(":")[1:2][0] if ":" in str(v) else None
    if isinstance(v, dict) and v.get("target") == "/model":
        v = dict(v); v["source"] = prep; v["read_only"] = True
        seen_model = True
    out.append(v)
if not seen_model:
    out.append({"type": "bind", "source": prep, "target": "/model", "read_only": True})
out.append({"type": "bind", "source": ckpt, "target": ckpt, "read_only": True})
out.append({"type": "bind", "source": plugin, "target": "/plugins", "read_only": True})
out.append({"type": "bind", "source": shm, "target": "/ple", "read_only": False})
svc["volumes"] = out

# environment: activate the backend. PYTHONPATH is unset in this image (verified),
# so /plugins cannot shadow the venv.
def add_env(**kv):
    env = svc.get("environment") or {}
    if isinstance(env, list):                      # - KEY=VAL form
        d = {}
        for e in env:
            k, _, val = str(e).partition("=")
            d[k] = val
        d.update({k: str(v) for k, v in kv.items()})
        svc["environment"] = ["%s=%s" % (k, v) for k, v in d.items()]
    else:
        env.update({k: str(v) for k, v in kv.items()})
        svc["environment"] = env

# PENNY_PLE_BACKEND is deliberately NOT set: the image's
# configs/pennyroyal/ple-backend.sh validates a closed {ram,nvme} domain and
# exits before the plugin loader runs. Stay on its `ram` path (same
# --ple-offload-embedding arguments); select the plugin with our own knob.
add_env(PYTHONPATH="/plugins", SGLANG_PLUGINS="shared_ram", PENNY_SHARED_PLE="1",
        PENNY_PLE_SHARED_PATH="/ple/layer-0.bin",
        PENNY_PLE_SHARED_MANIFEST="/ple/ssd-stream.json")

# no crash-looping during a trial
svc["restart"] = "no"
yaml.safe_dump(doc, open(path, "w"), sort_keys=False)
print("compose trial written; service=%s; restart=%s; env has backend=%s" %
      (svc_name, svc["restart"], "PENNY_SHARED_PLE=1" in str(svc.get("environment"))))
PY

# ---------- 3. boot the trial ----------
say "boot: force-recreate from scratch project"
( cd "$PROJ_TRIAL" && $SUDO docker compose -f compose.yaml up -d --force-recreate ) >>"$D/trial.log" 2>&1 \
  || { say "FAIL: compose up (see trial.log)"; exit 1; }

HEALTH=000
for i in $(seq 1 30); do
  sleep 8
  st=$(docker inspect -f '{{.State.Status}}/{{.State.Health.Status}}' "$CT" 2>/dev/null || echo gone)
  HEALTH=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:30073/health || echo 000)
  say "boot: t+$((i*8))s state=$st health_http=$HEALTH"
  [ "$HEALTH" = "200" ] && break
  case "$st" in *exited*|*dead*|gone) say "FAIL: container died"; break ;; esac
done
if [ "$HEALTH" != "200" ]; then
  say "FAIL: trial never became healthy; last 40 log lines:"
  docker logs --tail 40 "$CT" 2>&1 | tee -a "$D/trial.log"
  exit 1
fi

# ---------- 4. evidence ----------
say "evidence: plugin lines from this boot"
docker logs "$CT" 2>&1 | grep -E "shared_ram_ple|register-failed|HOOK" | tail -8 | tee -a "$D/trial.log"

say "evidence: host memory with B in shared mode (expect B's anon to drop ~47.7 GiB)"
free -g | head -2 | tee -a "$D/trial.log"
P=$(docker inspect -f '{{.State.Pid}}' "$CT")
SUB=$(awk -F: '{print $3; exit}' /proc/$P/cgroup)
grep -E '^(anon|file|shmem) ' "/sys/fs/cgroup$SUB/memory.stat" | tee -a "$D/trial.log"
echo "memory.current=$(cat /sys/fs/cgroup$SUB/memory.current)" | tee -a "$D/trial.log"
echo "tmpfs used: $(df -h /dev/shm | tail -1)" | tee -a "$D/trial.log"

say "evidence: smoke + decode sample (key read inside container, never printed)"
docker exec "$CT" sh -c 'K=$(cat /api-key 2>/dev/null || true); curl -s -m 120 -o /tmp/smoke.json -w "http=%{http_code} ttft=%{time_starttransfer} total=%{time_total}\n" \
  -H "Content-Type: application/json" -H "Authorization: Bearer $K" \
  http://127.0.0.1:8001/v1/chat/completions \
  -d "{\"model\":\"pennyroyal\",\"messages\":[{\"role\":\"user\",\"content\":\"TRIAL-SMOKE: list three properties of a quorum system\"}],\"max_tokens\":128,\"temperature\":0.0,\"stream\":false,\"chat_template_kwargs\":{\"enable_thinking\":false,\"preserve_thinking\":false}}"; \
  python3 -c "import json;d=json.load(open(\"/tmp/smoke.json\"));u=d.get(\"usage\",{});print(\"completion_tokens\",u.get(\"completion_tokens\"),\"prompt_tokens\",u.get(\"prompt_tokens\"));print(\"head:\",(d[\"choices\"][0][\"message\"][\"content\"][:80] if d.get(\"choices\") else d))"' \
  2>&1 | tee -a "$D/trial.log"

sleep 20
say "evidence: engine decode lines (accept len / running / queue)"
docker logs --since 2m "$CT" 2>&1 | grep -E "Decode batch" | tail -5 | tee -a "$D/trial.log"
# ---------- 5. optional: bench while B is still in shared mode ----------
# The bench must run inside the trial window, before the EXIT trap restores
# production ram mode -- a post-hoc bench would silently measure the control arm.
if [ "${RUN_BENCH:-0}" = "1" ]; then
  say "bench: text ladder cs=${BENCH_CS:-1,2,4} against B in shared mode"
  OUT="$D/bench-B-shared"
  rm -rf "$OUT"; mkdir -p "$OUT"
  if ( cd /var/tmp/phase0 && timeout 2400 python3 bench-b2.py --cs "${BENCH_CS:-1,2,4}" \
         --ctxs "${BENCH_CTXS:-1024,4096,65536,262144}" --outdir "$OUT" ) >>"$D/trial.log" 2>&1; then
    say "bench: done"
  else
    say "bench: FAILED (see trial.log)"
  fi
  grep -E "^\[[0-9]/4\]|MATRIX-DONE" "$D/trial.log" | tail -6 | tee -a /dev/null
  say "bench: memory after load (anon must stay low)"
  P=$(docker inspect -f '{{.State.Pid}}' "$CT")
  SUB=$(awk -F: '{print $3; exit}' /proc/$P/cgroup)
  grep -E '^(anon|file|shmem) ' "/sys/fs/cgroup$SUB/memory.stat" | tee -a "$D/trial.log"
  free -g | head -2 | tee -a "$D/trial.log"
fi

say "evidence: replica A untouched"
docker ps --format '{{.Names}} | {{.Status}}' | grep -E 'flash-next-pair-flash-next-a-1' | tee -a "$D/trial.log"
say "trial complete; restoring"
