#!/usr/bin/sh
# Publish the shared PLE table into tmpfs so the pair can pin ONE copy.
#
# Two callers, one script (paths come from env so the container and the host boot
# unit share the exact same verify-then-publish logic):
#   * compose service `ple-stage`  -> SRC=/prep/... DST=/shm/...   (this file bind-mounted ro)
#   * host boot unit               -> ple-stage-onboot.sh sets host paths and calls it
#
# Rules:
#  * never truncate a file a live replica may have mapped+pinned: copy to a temp
#    name, verify sha256 against the manifest, publish with an atomic rename
#    (unlinking a mapped file is safe -- its pages live under the existing mapping;
#    truncating one is not);
#  * publish root:GID 0660 so the serving uid 1000 can open(O_RDWR) — the driver
#    only registers a CPU-writable VMA — while mprotect(PROT_READ) after
#    registration remains the plugin's job;
#  * idempotent and instant on the happy path;
#  * PLE_STAGE_SKIP_HASH=1 skips the sha256 of a fresh copy (only for boot-time
#    latency experiments — the plugin re-hashes through its mapping regardless).
#
# Verifying what is already there (added 2026-10-06 after review):
#  * the old fast path trusted a size match alone. Every checkpoint of this
#    architecture has the same table geometry, so a right-size/wrong-bytes payload --
#    stale tmpfs, an interrupted earlier copy, a re-prepared checkpoint whose bytes
#    never made it here — would be published again and only met a hash once a replica
#    had already pinned it, where the failure is an aborted boot, not a repair.
#  * so: hash an existing payload before ANY replica maps it, and cache the verdict in
#    a stamp that is only trusted when it is root-owned 0600 and matches the file's
#    current (inode,size,mtime). tmpfs is wiped at boot, so the stamp expires exactly
#    when the answer stops being valid: one 47.68 GiB hash per boot, and repeat
#    `compose up -d` invocations stay instant.
#  * a right-size/wrong-bytes payload is REPLACED, not accepted. That can leave a
#    still-running sibling on the old inode; ple-shared-ram-check.sh is what alarms on
#    that divergence, and the log line below names both inodes.
set -eu

SRC=${SRC:-/prep/ple/layer-0.bin}
MAN=${MAN:-/prep/ssd-stream.json}
DST=${DST:-/shm/layer-0.bin}
PUB=${PUB:-/shm/ssd-stream.json}
GID=${GID:-975}

[ -f "$SRC" ] || { echo "PLE-STAGE|FAIL|missing source $SRC" >&2; exit 1; }
[ -f "$MAN" ] || { echo "PLE-STAGE|FAIL|missing manifest $MAN" >&2; exit 1; }
mkdir -p "$(dirname "$DST")"

want=$(python3 -c "import json;print(json.load(open('$MAN'))['tables'][0]['sha256'])")
size=$(stat -c %s "$SRC")
STAMP="$(dirname "$DST")/.staged-sha"
ORPHAN_MAX_AGE_MIN=180

# A unit killed between `cp` and `mv -T` strands 47.68 GiB of charged tmpfs that no
# other code path ever looks at; two of those exceed /dev/shm and the next copy dies
# with ENOSPC. Age-filtered so a concurrent run's temp file is never touched.
find "$(dirname "$DST")" -maxdepth 1 -name '.staging.*' -type f \
     -mmin +"$ORPHAN_MAX_AGE_MIN" -exec rm -f {} + 2>/dev/null || true

publish_manifest() {
  cp "$MAN" "$PUB.tmp.$$"
  mv -T "$PUB.tmp.$$" "$PUB"
  chmod 0644 "$PUB"
}

write_stamp() {
  # root:root 0600, keyed on the identity of the bytes it certifies.
  st_ino=$(stat -c %i "$DST"); st_sz=$(stat -c %s "$DST"); st_mt=$(stat -c %Y "$DST")
  printf '%s %s %s %s\n' "$st_ino" "$st_sz" "$st_mt" "$1" > "$STAMP.tmp.$$"
  chown root:root "$STAMP.tmp.$$" 2>/dev/null || true
  chmod 0600 "$STAMP.tmp.$$"
  mv -T "$STAMP.tmp.$$" "$STAMP"
}

stamp_certifies() {
  # args: inode size mtime. A stamp anyone can write certifies nothing.
  [ -f "$STAMP" ] || return 1
  [ "$(stat -c %u "$STAMP" 2>/dev/null)" = "0" ] || return 1
  [ "$(stat -c %a "$STAMP" 2>/dev/null)" = "600" ] || return 1
  read -r s_ino s_size s_mtime s_sha < "$STAMP" || return 1
  [ "$s_ino" = "$1" ] && [ "$s_size" = "$2" ] && [ "$s_mtime" = "$3" ] && [ "$s_sha" = "$want" ]
}

housekeeping() {
  chown "root:$GID" "$DST" 2>/dev/null || true
  chmod 0660 "$DST" 2>/dev/null || true
  publish_manifest
}

if [ -f "$DST" ]; then
  cur_ino=$(stat -c %i "$DST"); cur_sz=$(stat -c %s "$DST"); cur_mt=$(stat -c %Y "$DST")
  if [ "$cur_sz" = "$size" ] && stamp_certifies "$cur_ino" "$cur_sz" "$cur_mt"; then
    housekeeping
    echo "PLE-STAGE|skip|size=$cur_sz|inode=$cur_ino|mode=$(stat -c %a "$DST")|owner=$(stat -c %U:%G "$DST")|stamp=verified"
    exit 0
  fi
  if [ "$cur_sz" = "$size" ]; then
    # No trustworthy stamp for these exact bytes: prove them before a replica maps them.
    echo "PLE-STAGE|note|verifying existing payload (no valid stamp)" >&2
    t0=$(date +%s)
    got=$(sha256sum "$DST" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
      housekeeping
      write_stamp "$got"
      echo "PLE-STAGE|skip|size=$cur_sz|inode=$cur_ino|mode=$(stat -c %a "$DST")|owner=$(stat -c %U:%G "$DST")|stamp=fresh|verified=$(( $(date +%s) - t0 ))s"
      exit 0
    fi
    echo "PLE-STAGE|note|existing payload is right-size/wrong-bytes ($got != $want); replacing under old_inode=$cur_ino -- if a replica is still mapped it keeps the old pages, and ple-shared-ram-check.sh will report the inode split" >&2
  fi
fi

tmp=$(dirname "$DST")/.staging.$(date +%s).$$
trap 'rm -f "$tmp"' EXIT
t0=$(date +%s)
cp "$SRC" "$tmp"
if [ "${PLE_STAGE_SKIP_HASH:-0}" = "1" ]; then
  got="$want"
  echo "PLE-STAGE|note|hash skipped by request" >&2
else
  got=$(sha256sum "$tmp" | cut -d' ' -f1)
fi
if [ "$got" != "$want" ]; then
  echo "PLE-STAGE|FAIL|staged sha256 $got != manifest $want" >&2
  exit 1
fi
chown "root:$GID" "$tmp"
chmod 0660 "$tmp"
mv -T "$tmp" "$DST"
publish_manifest
# Only stamp bytes we actually hashed.
if [ "${PLE_STAGE_SKIP_HASH:-0}" != "1" ]; then
  write_stamp "$got"
fi
echo "PLE-STAGE|ok|bytes=$size|sha=$got|inode=$(stat -c %i "$DST")|seconds=$(( $(date +%s) - t0 ))"
