#!/usr/bin/bash
# Is the shared PLE table still ONE physical, read-only, correctly-hashed artifact?
#
# The plugin proves all of this once, at allocation, and nothing re-proves it while
# the pair serves. This script is that missing witness: it reads the kernel's own
# VMA table for every serving process, compares it against the host file, and
# (with --hash) recomputes the sha256 the engines are actually reading from.
#
#   ple-shared-ram-check.sh            # identity + perms, ~1 s, safe at any time
#   ple-shared-ram-check.sh --hash     # also re-hashes 47.68 GiB (~13 s normal prio,
#                                      # ~69 s at nice 19 / idle io class)
#
# Exit 0 = PASS, 1 = FAIL (reason on the last line), 2 = cannot determine.
# Installed on the serving host at /usr/local/bin/ple-shared-ram-check.sh with an hourly timer.
#
# Why each check is here:
#  * perms must be exactly r--s: the plugin mprotects its mapping read-only after
#    cudaHostRegister, so a writable VMA in a serving process means one stray store
#    corrupts every replica at once. Measured on tmpfs, only a store through the
#    mapping moves mtime/ctime -- the payload took one at 02:35:38Z on 2026-10-06.
#  * all replicas must share one (dev, inode): two replicas on two inodes is two
#    pinned 47.68 GiB tables and 96 GiB of tmpfs, which is the failure this whole
#    tier exists to prevent; nothing else in the system compares them.
#  * size must equal the manifest: the stager's fast path checks size only.
#  * the expected sha is read from the manifest, not from a container's memory.
set -uo pipefail

DST=${PLE_SHARED_DST:-/dev/shm/ple-shared/layer-0.bin}
MAN=${PLE_SHARED_MANIFEST:-/dev/shm/ple-shared/ssd-stream.json}
CONTAINERS=${PLE_SHARED_CONTAINERS:-flash-next-pair-flash-next-a-1 flash-next-pair-flash-next-b-1}
DO_HASH=${1:-}
SUDO=${SUDO:-sudo -n}
BASELINE=${PLE_SHARED_SHA_BASELINE:-/usr/local/share/ple/deployed-sha256.txt}

fail() { echo "PLE-CHECK|FAIL|$1"; exit 1; }
unable() { echo "PLE-CHECK|UNABLE|$1"; exit 2; }

# /proc/PID/maps prints the device as "major:minor" in hex ("00:1b"); stat %D prints
# the raw device number in hex with no padding or colon ("1b"). Compare one form.
norm_dev() { echo "$1" | tr 'A-Z' 'a-z' | tr -d ':' | sed 's/^0*//'; }

if [ "$(id -u)" = 0 ]; then SUDO=""; else exec $SUDO "$0" "$DO_HASH"; fi

[ -f "$DST" ] || fail "payload $DST missing (was the tmpfs staged?)"
[ -f "$MAN" ] || fail "manifest $MAN missing"

want_size=$(python3 -c "import json;print(json.load(open('$MAN'))['tables'][0]['bytes'])") \
  || unable "cannot read manifest bytes"
want_sha=$(python3 -c "import json;print(json.load(open('$MAN'))['tables'][0]['sha256'])") \
  || unable "cannot read manifest sha256"
[ -n "$want_size" ] && [ -n "$want_sha" ] || fail "manifest carries no bytes/sha256"

read -r host_ino host_dev host_size <<<"$($SUDO stat -c '%i %D %s' "$DST")"
[ -n "$host_ino" ] || unable "cannot stat payload"
echo "host:      file=$DST inode=$host_ino dev=$host_dev size=$host_size"
[ "$host_size" = "$want_size" ] || fail "payload size $host_size != manifest bytes $want_size"

# 1. Per-process VMA witness: exactly one mapping per serving process, read-only.
seen_inodes=""
processes=0
for c in $CONTAINERS; do
  state=$($SUDO docker inspect -f '{{.State.Status}}/{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$c" 2>/dev/null) \
    || fail "container $c not inspectable"
  pids=$($SUDO docker top "$c" -o pid 2>/dev/null | tail -n +2)
  [ -n "$pids" ] || fail "no processes for $c"
  found=""
  for p in $pids; do
    mlines=$($SUDO grep -h "$(basename "$DST")" /proc/$p/maps 2>/dev/null)
    [ -n "$mlines" ] || continue
    processes=$((processes + 1))          # one per serving PROCESS, not per VMA
    # EVERY VMA, not the first. A split mapping -- a surviving `rw-s` piece beside
    # the `r--s` one -- passed a `head -1` check while leaving CPU-writable pages in
    # the table every replica gathers from. deploy-shared-ram.sh reads all matches
    # for exactly this reason; the two witnesses have to agree.
    while IFS= read -r m; do
      [ -n "$m" ] || continue
      perms=$(echo "$m" | awk '{print $2}')
      dev=$(echo "$m" | awk '{print $4}')
      ino=$(echo "$m" | awk '{print $5}')
      found="$found $c:pid=$p:perms=$perms:dev=$dev:inode=$ino"
      [ "$perms" = "r--s" ] || fail "$c pid=$p maps the shared table as '$perms', expected r--s (a CPU-writable shared table corrupts every replica)"
      [ "$ino" = "$host_ino" ] || fail "$c pid=$p maps inode $ino but the host file is $host_ino (republished under a running replica)"
      [ "$(norm_dev "$dev")" = "$(norm_dev "$host_dev")" ] || fail "$c pid=$p maps dev $dev but the host file is on $host_dev"
    done <<< "$mlines"
  done
  [ -n "$found" ] || fail "$c has no mapping of $(basename "$DST") -- it is serving from a PRIVATE table (state $state); check for zero shared_ram_ple lines in 'docker logs $c'"
  echo "mapped:   $found"
  seen_inodes="$seen_inodes$(echo "$found" | tr ' ' '\n' | sed -n 's/.*:inode=//p' | sort -u | tr '\n' ' ')"
done

# 2. One physical table: every process must agree on one inode.
uniq_inodes=$(echo "$seen_inodes" | tr ' ' '\n' | sort -u | grep -c . || true)
[ "$uniq_inodes" = "1" ] || fail "$processes serving processes map $uniq_inodes distinct inodes -- expected 1 physical table"
echo "sharing:   OK ($processes processes, 1 inode)"

# 3. Deployed plugin bytes vs the baseline recorded by deploy-shared-ram.sh.
if [ -f "$BASELINE" ]; then
  if ! ( cd "$(dirname "$(awk 'NR==1{print $2}' "$BASELINE")")" && sha256sum -c --quiet "$BASELINE" ) 2>/dev/null; then
    echo "baseline:  $BASELINE"
    fail "deployed plugin bytes differ from the recorded baseline (drift or tampering; re-run deploy-shared-ram.sh to re-certify)"
  fi
  echo "baseline:  OK ($(wc -l < "$BASELINE") files match $BASELINE)"
else
  # Not a skip. The hourly unit must never report PASS on the one property it did
  # not check: with no recorded baseline the deployed plugin bytes are unattested,
  # so a hand-edit or a stale dist reads as "the plugin never ran" while the timer
  # still shows green. UNABLE (exit 2) makes the gap visible to systemd.
  unable "no baseline recorded at $BASELINE - deployed plugin bytes unattested (run deploy-shared-ram.sh to certify)"
fi

# 4. Optional content proof. Off the serving path, but it is a 47.68 GiB read.
if [ "$DO_HASH" = "--hash" ]; then
  t0=$(date +%s)
  got=$($SUDO nice -n 19 ionice -c3 sha256sum "$DST" | awk '{print $1}')
  t1=$(date +%s)
  [ -n "$got" ] || unable "hash produced no output"
  echo "content:   sha256=$got ($((t1 - t0))s)"
  [ "$got" = "$want_sha" ] || fail "payload sha256 $got != manifest $want_sha -- BOTH replicas are serving corrupted embeddings; drain them from llmproxy"
fi

echo "PLE-CHECK|PASS|inode=$host_ino size=$host_size processes=$processes hash=$([ "$DO_HASH" = "--hash" ] && echo verified || echo not-run)"
exit 0
