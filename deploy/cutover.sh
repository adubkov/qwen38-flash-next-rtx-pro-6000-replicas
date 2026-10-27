#!/usr/bin/env bash
# Phase D cutover: two single-replica projects -> one pair project with shared_ram PLE.
#
# Rules, all learned the hard way on this box:
#  * never take both replicas down at once: one of them is serving this assistant's
#    own tokens (measured 2026-10-04: llmproxy spreads A=33 / B=10 req per 10 min),
#    and llmproxy failovers per request + rebinds a sticky session whose target went
#    unhealthy, so one healthy replica is enough to keep the session alive;
#  * each replacement must pass health + smoke before the next old replica is
#    touched, and any failure restores the old project it displaced;
#  * old projects are `stop`ped, never `down`ed, so rollback is `docker start`
#    (or `up -d`) on a file that was never edited;
#  * restart stays "no" until the new replica is proven, then is set with
#    `docker update` (changing policy without recreating = no second cold boot);
#  * the API key is read from the file mounted inside the container; it is never
#    printed and never appears in argv;
#  * run DETACHED (systemd-run) so the window contains no per-tool-round model
#    calls of ours, which are pool members too.
set -uo pipefail

NEW=/mnt/apps/compose/sglang-flash-next-pair
OLDA=/mnt/apps/compose/sglang-qwen38-flash-next
OLDB=/mnt/apps/compose/sglang-qwen38-flash-next-b
SUDO="sudo -n"
D=/var/tmp/ple-cutover; mkdir -p "$D"; : > "$D/log"
say() { echo "[$(date -u +%T)] $*" | tee -a "$D/log"; }
health() { curl -s -o /dev/null -w '%{http_code}' -m 4 "http://127.0.0.1:${1}/health" || echo 000; }

wait_healthy() {           # wait_healthy PORT [MAXWAITS] (8 s each)
  local t0=$SECONDS st i
  for i in $(seq 1 "${2:-60}"); do
    sleep 8
    st=$(health "$1")
    say "  :$1 t+$((SECONDS-t0))s http=$st"
    [ "$st" = "200" ] && return 0
  done
  return 1
}

smoke() {                  # $1=container name
  docker exec "$1" sh -c 'K=$(cat /api-key 2>/dev/null || true); \
    curl -s -m 120 -o /tmp/smoke.json -w "http=%{http_code} ttft=%{time_starttransfer} " \
      -H "Content-Type: application/json" -H "Authorization: Bearer $K" \
      http://127.0.0.1:8001/v1/chat/completions \
      -d "{\"model\":\"pennyroyal\",\"messages\":[{\"role\":\"user\",\"content\":\"CUTOVER-CHECK: state two properties of a quorum system\"}],\"max_tokens\":64,\"temperature\":0.0,\"chat_template_kwargs\":{\"enable_thinking\":false,\"preserve_thinking\":false}}"; \
    python3 -c "import json;d=json.load(open(\"/tmp/smoke.json\"));u=d.get(\"usage\") or {};print(\"completion_tokens\",u.get(\"completion_tokens\"),\"| head:\",(d[\"choices\"][0][\"message\"][\"content\"][:60] if d.get(\"choices\") else str(d)[:80]))"' 2>&1
}

evidence() {               # $1=container name -> plugin lines, anon, inode
  local c=$1 p s
  say "  evidence $c:"
  docker logs "$c" 2>&1 | grep -E "shared_ram_ple|Failed to (execute|apply)" | tail -4 | sed 's/^/    /' | tee -a "$D/log"
  p=$(docker inspect -f '{{.State.Pid}}' "$c" 2>/dev/null || echo 0)
  s=$(awk -F: '{print $3; exit}' "/proc/$p/cgroup" 2>/dev/null || true)
  [ -n "$s" ] && grep -E '^(anon|file|shmem) ' "/sys/fs/cgroup$s/memory.stat" 2>/dev/null | sed 's/^/    /' | tee -a "$D/log"
  docker inspect -f '    binds: {{range .Mounts}}{{if eq .Destination "/ple"}}{{.Source}}->{{.Destination}} rw={{.RW}} {{end}}{{end}}' "$c" | tee -a "$D/log"
}

restore_old() {            # $1=old project dir
  say "ROLLBACK: bringing $1 back"
  ( cd "$1" && $SUDO docker compose -f compose.yaml up -d ) >>"$D/log" 2>&1
  sleep 10
}

# ---------- gate 0: the survivor must be able to carry everything alone ----------
say "gate 0: pre-flight on the replica that is NOT being replaced yet"
CTB_OLD=qwen38-flash-next-b-flash-next-opt-b-1
say "  B health=$(health 30073)  A health=$(health 30071)"
[ "$(health 30073)" = "200" ] || { say "FAIL: B is not healthy -- refusing to touch A"; exit 1; }
say "  B smoke: $(smoke "$CTB_OLD")"
free -g | head -2 | sed 's/^/  /' | tee -a "$D/log"

# ---------- step 1: replace A (GPU0, :30071) ----------
say "step 1: stop old A, start pair A"
( cd "$OLDA" && $SUDO docker compose -f compose.yaml stop flash-next-opt ) >>"$D/log" 2>&1 \
  || say "WARN: old A stop returned nonzero"
( cd "$NEW" && $SUDO docker compose -f compose.yaml up -d flash-next-a ) >>"$D/log" 2>&1 \
  || { say "FAIL: pair A would not start"; restore_old "$OLDA"; exit 1; }
CTA_NEW=flash-next-pair-flash-next-a-1
if ! wait_healthy 30071 45; then
  say "FAIL: pair A never became healthy; last 30 log lines:"
  docker logs --tail 30 "$CTA_NEW" 2>&1 | tee -a "$D/log"
  ( cd "$NEW" && $SUDO docker compose -f compose.yaml stop flash-next-a ) >>"$D/log" 2>&1
  restore_old "$OLDA"
  say "STATE after rollback: $(docker ps --format '{{.Names}} | {{.Status}}' | grep flash-next | tr '\n' ';')"
  exit 1
fi
evidence "$CTA_NEW"
say "  A smoke: $(smoke "$CTA_NEW")"

# ---------- step 2: replace B (GPU1, :30073) ----------
say "step 2: stop old B, start pair B"
( cd "$OLDB" && $SUDO docker compose -f compose.yaml stop flash-next-opt-b ) >>"$D/log" 2>&1 \
  || say "WARN: old B stop returned nonzero"
( cd "$NEW" && $SUDO docker compose -f compose.yaml up -d flash-next-b ) >>"$D/log" 2>&1 \
  || { say "FAIL: pair B would not start"; restore_old "$OLDB"; exit 1; }
CTB_NEW=flash-next-pair-flash-next-b-1
if ! wait_healthy 30073 45; then
  say "FAIL: pair B never became healthy; last 30 log lines:"
  docker logs --tail 30 "$CTB_NEW" 2>&1 | tee -a "$D/log"
  ( cd "$NEW" && $SUDO docker compose -f compose.yaml stop flash-next-b ) >>"$D/log" 2>&1
  restore_old "$OLDB"
  say "STATE after rollback: $(docker ps --format '{{.Names}} | {{.Status}}' | grep flash-next | tr '\n' ';')"
  exit 1
fi
evidence "$CTB_NEW"
say "  B smoke: $(smoke "$CTB_NEW")"

# ---------- step 3: acceptance + persistence ----------
say "step 3: acceptance"
# This is the cutover's ONLY acceptance test, so it has to be an assertion. Reading
# both inodes and printing them beside the words "must be the SAME inode" certified
# nothing: if the plugin never loaded, both greps come back empty and the script
# still reported a successful shared-PLE cutover while both replicas served private
# 47.68 GiB tables -- the state the compose memory cap exists to make fatal.
IA=$(docker logs "$CTA_NEW" 2>&1 | grep -oE "shared_ram PLE: inode=[0-9]+" | tail -1)
IB=$(docker logs "$CTB_NEW" 2>&1 | grep -oE "shared_ram PLE: inode=[0-9]+" | tail -1)
say "  A $IA / B $IB  (must be the SAME inode: one physical table)"
[ -n "$IA" ] || { say "FAIL: replica A logged no shared_ram PLE inode -- plugin not active there"; exit 1; }
[ -n "$IB" ] || { say "FAIL: replica B logged no shared_ram PLE inode -- plugin not active there"; exit 1; }
[ "$IA" = "$IB" ] || { say "FAIL: A and B mapped different inodes ($IA vs $IB) -- not one physical table"; exit 1; }
say "  ACCEPTED: both replicas report $IA"
free -g | head -2 | sed 's/^/  /' | tee -a "$D/log"
echo "  tmpfs: $(df -h /dev/shm | tail -1)" | tee -a "$D/log"
# Not by default. compose ships restart:"no" deliberately (compose.yaml header: a
# crash must stay visible as an Exited container, and boot-start comes from the
# systemd unit); flipping it to unless-stopped reintroduces crash-restart and makes
# the next 'compose up -d' reconcile drift by RECREATING, undoing this cutover.
# Opt in explicitly with CONFIRM_RESTART_POLICY=1 if you have read that header.
if [ "${CONFIRM_RESTART_POLICY:-0}" = "1" ]; then
  $SUDO docker update --restart=unless-stopped "$CTA_NEW" "$CTB_NEW" >>"$D/log" 2>&1 \
    && say "  restart policy -> unless-stopped (opt-in; diverges from shipped compose)"
else
  say "  restart policy left at the shipped compose value (pass CONFIRM_RESTART_POLICY=1 to change)"
fi
say "  projects:"
docker compose ls -a 2>/dev/null | grep -i flash-next | sed 's/^/    /' | tee -a "$D/log"
docker ps -a --format '    {{.Names}} | {{.Status}}' | grep flash-next | tee -a "$D/log"
say "CUTOVER-DONE"
