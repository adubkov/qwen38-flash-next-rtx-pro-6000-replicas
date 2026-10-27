#!/usr/bin/env bash
# SPEC-OFF vs SPEC-ON ablation on replica B (serving host, 2026-10-03).
# Detached design: this script is self-contained (compose swap, health gates,
# cell purity audits, restore) so the orchestrating session makes ZERO model
# calls during measurement windows (auto-mode classifier chats land on a
# random replica via the sticky pool — see memory self-serving-gpu-benchmarks).
set -uo pipefail
CT=flash-next-pair-flash-next-b-1
PROJ=/mnt/apps/compose/sglang-qwen38-flash-next-b
SD=/mnt/apps/sglang-flashnext
D=/var/tmp/phase0/spec-ab
mkdir -p "$D"

# ---- 1. no-spec variant of the serve recipe (exact single-anchor edit) ----
if [ ! -f "$SD/serve-flash-next-nospec.sh" ]; then
python3 - <<'EOF'
src = open("/mnt/apps/sglang-flashnext/serve-flash-next-frspec.sh").read()
old = '''  --speculative-algorithm NEXTN --speculative-num-steps 3 \\
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \\
   \\
  --speculative-token-map "$TOKEN_MAP" --watchdog-timeout 1800)'''
new = '''  --watchdog-timeout 1800)'''
assert src.count(old) == 1, "spec anchor not found exactly once"
out = src.replace(old, new)
assert "--speculative-" not in out.split("launch_args")[1] if "launch_args" in out else "--speculative-" not in out
open("/mnt/apps/sglang-flashnext/serve-flash-next-nospec.sh", "w").write(out)
print("nospec-written")
EOF
  [ $? -eq 0 ] || { echo DRIVER-FAIL-nospecgen; exit 1; }
  chmod 755 "$SD/serve-flash-next-nospec.sh"
fi
bash -n "$SD/serve-flash-next-nospec.sh" || { echo DRIVER-FAIL-parse; exit 1; }
grep -c -- "--speculative-algorithm" "$SD/serve-flash-next-nospec.sh" | xargs echo "nospec-launch-flag-count:"

# ---- 2. pristine compose backup (restore target) ----
cp -f "$PROJ/compose.yaml" "$D/compose.pristine"

# This driver rewrites the LIVE production compose and force-recreates a serving
# replica. Without a trap, a Ctrl-C or a kill between the sed and the restore at the
# bottom leaves production bound to the generated nospec recipe, silently serving a
# different engine configuration than the file says. Idempotent, so the trap and the
# explicit call at the end can both run.
restore_production() {
  [ -f "$D/compose.pristine" ] || return 0
  cp -f "$D/compose.pristine" "$PROJ/compose.yaml"
  ( cd "$PROJ" && docker compose up -d --force-recreate >/dev/null 2>&1 ) || true
  for i in $(seq 1 45); do
    sleep 8
    [ "$(curl -s -o /dev/null -w %{http_code} -m 3 http://127.0.0.1:30073/health)" = "200" ] && break
  done
  echo "restored-to-production $(date -u +%FT%TZ)"
}
trap 'restore_production' EXIT
trap 'restore_production; exit 130' INT TERM

arm() {  # $1 = nospec|spec
  local SRC="$SD/serve-flash-next-frspec.sh"
  [ "$1" = "nospec" ] && SRC="$SD/serve-flash-next-nospec.sh"
  cp -f "$D/compose.pristine" "$PROJ/compose.yaml"
  sed -i "s|source: $SD/serve-flash-next-frspec.sh|source: $SRC|" "$PROJ/compose.yaml"
  grep -q "source: $SRC" "$PROJ/compose.yaml" || { echo DRIVER-FAIL-compose; exit 1; }
  ( cd "$PROJ" && docker compose up -d --force-recreate >/dev/null 2>&1 )
  for i in $(seq 1 45); do
    sleep 8
    [ "$(curl -s -o /dev/null -w %{http_code} -m 3 http://127.0.0.1:30073/health)" = "200" ] \
      && { echo "$1 healthy ~$((i*8))s $(date -u +%T)"; return 0; }
  done
  echo "$1 HEALTH-TIMEOUT"; return 1
}

audit() {  # $1 = outdir — cull cells with foreign chat or achv>1.5 within their windows
python3 - "$1" <<'EOF'
import json, subprocess, sys, re, datetime as dt, os
out = sys.argv[1]; jf = os.path.join(out, "matrix.jsonl")
if not os.path.exists(jf):
    print("audit %s: NO JOURNAL - nothing to certify" % out, file=sys.stderr); sys.exit(1)
rows = [json.loads(l) for l in open(jf)]
ctr = os.environ.get("CT", "")
if not ctr:
    print("audit: CT (container name) unset - the log read below can only return empty,"
          " which is not the same as no foreign chats", file=sys.stderr); sys.exit(1)
log = subprocess.run(["docker", "logs", ctr], capture_output=True, text=True)
if log.returncode != 0:
    print("audit: docker logs %s failed rc=%d: %s" % (ctr, log.returncode,
          (log.stderr or "").strip()[:200]), file=sys.stderr); sys.exit(1)
txt = (log.stdout or "") + (log.stderr or "")
foreign = []
for line in txt.splitlines():
    if "POST /v1/chat" in line and "127.0.0.1" not in line:
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", line)
        if m: foreign.append(dt.datetime.strptime(m.group(1),"%Y-%m-%d %H:%M:%S").timestamp())
keep, dropped = [], []
for r in rows:
    w0 = r.get("ts",0) - 200; w1 = r.get("ts",0) + max(r.get("cell_s",30),30) + 15
    fhit = sum(1 for t in foreign if w0 <= t <= w1)
    achv = r.get("achieved_concurrency")
    ok = fhit == 0 and (achv is None or achv <= 1.5)
    (keep if ok else dropped).append(r)
if dropped:
    with open(jf, "w") as fh:
        for r in keep: fh.write(json.dumps(r) + "\n")
    print("audit %s: kept=%d culled=%d" % (out, len(keep), len(dropped)))
else:
    print("audit %s: kept=%d clean" % (out, len(keep)))
EOF
}

cells() {  # $1 = arm; C1 ladder; audit-cull loop up to 4 passes
  local OUT="$D/$1"; mkdir -p "$OUT"
  cp -n /var/tmp/phase0/matrix/ctxcal.json "$OUT/" 2>/dev/null  # seed calibration (same prompts as baseline)
  for pass in 1 2 3 4; do
    CT=$CT timeout 500 python3 /var/tmp/phase0/bench-b2.py --cs 1 --ctxs 1024,4096,65536,262144 --outdir "$OUT" >> "$D/harness.log" 2>&1
    local n; n=$( [ -f "$OUT/matrix.jsonl" ] && wc -l < "$OUT/matrix.jsonl" || echo 0 )
    # The audit's verdict decides whether these cells may be published at all, so
    # its exit status has to reach the caller; `set -e` is not on in this file.
    if [ "$n" -ge 4 ]; then
      CT=$CT audit "$OUT" || { echo "$1 audit FAILED - cells cannot be certified clean"; return 1; }
    fi
    n=$( [ -f "$OUT/matrix.jsonl" ] && wc -l < "$OUT/matrix.jsonl" || echo 0 )
    [ "$n" -ge 4 ] && { echo "$1 cells complete pass=$pass"; return 0; }
    echo "$1 pass=$pass rows=$n retrying"; sleep 5
  done
  echo "$1 cells INCOMPLETE after 4 passes"
  return 1   # `echo` returned 0 here, so the caller could not tell this from success
}

RC=0
for ARM in nospec spec; do
  echo "== ARM $ARM start $(date -u +%FT%TZ)"
  arm "$ARM" || { echo "ARM $ARM never became healthy"; RC=1; break; }
  docker logs --since 6m "$CT" 2>&1 | grep -iE "cuda graph|capturing|EAGLE|draft|speculative|PLE:" | head -10 > "$D/$ARM-bootgrep.txt"
  cells "$ARM" || RC=1
done

# ---- 3. unconditional restore to production (spec-on recipe) ----
restore_production
# SPEC-AB-DONE used to print even when an arm never came up and the second arm was
# skipped entirely, because `set -e` is off in this file and the statuses were dropped.
if [ "$RC" = 0 ]; then
  echo "SPEC-AB-DONE"
else
  echo "SPEC-AB-FAILED (an arm did not boot or its cells stayed incomplete — no comparison was made)"
  exit 1
fi
