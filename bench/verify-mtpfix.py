#!/usr/bin/env python3
"""Layer-2 equality proof: our nvidia checkpoint vs the mtp-fix artifact.
Compares EVERY non-mtp-expert tensor (main NVFP4, BF16 attention/gate/shared-expert,
all 132 PLE keys, 29 non-expert mtp tensors) for name+dtype+shape+BYTE equality via the
safetensors header data_offsets. Stdlib only; run detached on the serving host; ~260 GB read total.
Excluded by construction (different representation, handled by mtpfix-stats.py):
  ours: mtp.layers.0.mlp.experts.<i>.<proj>.{weight,weight_scale_inv}  (3,072 keys)
  fix:  mtp.layers.0.mlp.experts.<gate_up_proj|down_proj>               (2 fused keys)"""
import argparse, hashlib, json, os, re, sys, time

OUR_EXP = re.compile(r"^mtp\.layers\.0\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(weight|weight_scale_inv)$")
FIX_EXP = re.compile(r"^mtp\.layers\.0\.mlp\.experts\.(gate_up_proj|down_proj)$")

def headers(dirpath, files):
    out = {}
    for f in files:
        with open(os.path.join(dirpath, f), "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(n))
        base = 8 + n  # safetensors data_offsets are relative to END OF HEADER, not
        # file start — hashing at raw offsets compares garbage (the first version
        # of this script did exactly that: 295k/296k false mismatches + ~1.1k false
        # "matches" from header-region overlap. All-meta-clean + ~all-hash-differ
        # = constant alignment error, not different weights.)
        out[f] = {k: dict(v, _base=base) for k, v in hdr.items() if k != "__metadata__"}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--fix", required=True)
    ap.add_argument("--out", default="/var/tmp/phase0/mtpfix-equality.json")
    a = ap.parse_args()
    wi = json.load(open(os.path.join(a.ours, "model.safetensors.index.json")))["weight_map"]
    wf = json.load(open(os.path.join(a.fix, "model.safetensors.index.json")))["weight_map"]

    ours_exp = {k for k in wi if OUR_EXP.match(k)}
    fix_exp = {k for k in wf if FIX_EXP.match(k)}
    print(f"mtp-expert keys excluded: ours={len(ours_exp)} fix={len(fix_exp)} (expect 3072/2)")
    wk = {k for k in wi if k not in ours_exp}
    fk = {k for k in wf if k not in fix_exp}
    only_o, only_f = sorted(wk - fk), sorted(fk - wk)
    if only_o or only_f:
        print("KEYSET MISMATCH:", len(only_o), len(only_f), "sample:", (only_o + only_f)[:5])
    common = sorted(wk & fk)
    print(f"common non-mtp-expert keys: {len(common)}")

    ho = headers(a.ours, {wi[k] for k in common})
    hf = headers(a.fix, {wf[k] for k in common})

    mism = []
    byt = 0
    next_report = 10 << 30
    t0 = time.time()
    for k in common:
        o_file, f_file = wi[k], wf[k]
        oh = ho[o_file].get(k)
        fh = hf[f_file].get(k)
        if oh is None or fh is None:
            mism.append((k, "no-header")); continue
        o_dt, o_sh, o_a, o_b = oh["dtype"], oh["shape"], oh["data_offsets"][0] + oh["_base"], oh["data_offsets"][1] + oh["_base"]
        f_dt, f_sh, f_a, f_b = fh["dtype"], fh["shape"], fh["data_offsets"][0] + fh["_base"], fh["data_offsets"][1] + fh["_base"]
        if (o_dt, o_sh) != (f_dt, f_sh):
            mism.append((k, f"meta {o_dt}{o_sh} vs {f_dt}{f_sh}")); continue
        if o_b - o_a != f_b - f_a:
            mism.append((k, "size")); continue
        if o_b == o_a:  # empty tensor
            continue
        d_o = hashlib.blake2b(digest_size=16)
        d_f = hashlib.blake2b(digest_size=16)
        bad = False
        with open(os.path.join(a.ours, o_file), "rb") as go, open(os.path.join(a.fix, f_file), "rb") as gf:
            go.seek(o_a); gf.seek(f_a)
            left = o_b - o_a
            while left > 0:
                n = min(1 << 23, left)
                c1, c2 = go.read(n), gf.read(n)
                if len(c1) != n or len(c2) != n:
                    bad = True; break
                d_o.update(c1); d_f.update(c2)
                left -= n
        byt += o_b - o_a
        if bad:
            mism.append((k, "short-read"))
        elif d_o.digest() != d_f.digest():
            mism.append((k, f"hash {d_o.hexdigest()[:8]} vs {d_f.hexdigest()[:8]}"))
        if byt >= next_report:
            print(f"  {byt >> 30} GiB/side hashed, {len(mism)} mismatches, {time.time()-t0:.0f}s", flush=True)
            next_report += 10 << 30
    res = {"keys_compared": len(common), "bytes_per_side": byt, "mismatch_count": len(mism),
           "keyset_only_o": len(only_o), "keyset_only_f": len(only_f),
           "mismatches": mism[:50], "seconds": round(time.time() - t0, 1),
           "verdict": "EQUAL" if not mism and not only_o and not only_f else "NOT-EQUAL"}
    print(json.dumps(res, indent=1))
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    sys.exit(0 if res["verdict"] == "EQUAL" else 1)

if __name__ == "__main__":
    main()
