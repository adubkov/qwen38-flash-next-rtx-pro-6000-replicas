#!/usr/bin/env python3
"""Layer-3 mtp-expert identity proof, v2 (magnitude-based).
Question: is the fix's fused BF16 draft the SAME weights as our checkpoint's fp8 experts?
v1 used Pearson corr as the discriminator — WRONG: corr is invariant to uniform
rescaling, so it passes even when our dequant has the scale direction inverted
(10x off still corr'd 0.9996; relerr 9.9 = 1/0.091 was the tell).
v2 tests all three dequant semantics and lets MAGNITUDE pick:
  mult: w*s   |   div: w/s   |   raw: w        (raw = what the fork's raw-cast loads today)
PASS: one of (mult,div) achieves mean relerr <= 0.10 and max <= 0.25 (e4m3 block-quant
noise floor), and raw relerr > 0.5 — which simultaneously (a) proves the fix carries OUR
weights, (b) identifies nvidia's scale convention, and (c) quantifies how far today's
scale-dropping raw-cast is from truth (the defect, in one number).
CPU-only, runs inside a throwaway container with /ours /fix /tools bind-mounted :ro."""
import json, random, sys
import torch
from safetensors import safe_open

OURS = "/ours"; FIX = "/fix"
SAMPLE = 16
random.seed(20261003)
experts = sorted(random.sample(range(512), SAMPLE))

wi = json.load(open(f"{OURS}/model.safetensors.index.json"))["weight_map"]
wf = json.load(open(f"{FIX}/model.safetensors.index.json"))["weight_map"]
_ocs, _fcs = {}, {}
def ours_t(name):
    f = wi[name]
    if f not in _ocs: _ocs[f] = safe_open(f"{OURS}/{f}", framework="pt")
    return _ocs[f].get_tensor(name)
def fix_t(name):
    f = wf[name]
    if f not in _fcs: _fcs[f] = safe_open(f"{FIX}/{f}", framework="pt")
    return _fcs[f].get_tensor(name)

def variants(name):
    w = ours_t(name).to(torch.float32)
    s = ours_t(name + "_scale_inv").to(torch.float32)
    r, c = w.shape
    want = ((r + 127) // 128, (c + 127) // 128)
    if tuple(s.shape) != want:
        s = s.T
        assert tuple(s.shape) == want, f"scale shape {tuple(s.shape)} != {want} for {name}"
    sx = s.repeat_interleave(128, 0)[:r, :].repeat_interleave(128, 1)[:, :c]
    return {"mult": w * sx, "div": w / sx, "raw": w}, float(sx.mean())

gu = fix_t("mtp.layers.0.mlp.experts.gate_up_proj")
dn = fix_t("mtp.layers.0.mlp.experts.down_proj")
E, A, H = gu.shape; I = A // 2
print(f"fused shapes: gate_up=({E},{A},{H}) down={tuple(dn.shape)} -> inter={I}")
assert dn.shape == (E, H, I)

def relerr(a, b):   # ||a-b||/||b||, b = reference = fix/true
    return float((a - b).norm() / (b.norm() + 1e-30))

acc = {"mult": [], "div": [], "raw": []}
corr_best = 2.0
scale_means = []
for e in experts:
    for proj in ("gate", "up", "down"):
        name = f"mtp.layers.0.mlp.experts.{e}.{proj}_proj.weight"
        va, sm = variants(name)
        scale_means.append(sm)
        if proj == "down":
            ref = dn[e].float()
            for k in acc: acc[k].append(relerr(va[k], ref))
            b = va[min(va, key=lambda k: acc[k][-1])]
        else:
            primary = (gu[e, :I, :] if proj == "gate" else gu[e, I:, :]).float()
            alt     = (gu[e, I:, :] if proj == "gate" else gu[e, :I, :]).float()
            r1 = {k: relerr(v, primary) for k, v in va.items()}
            r2 = {k: relerr(v, alt)     for k, v in va.items()}
            if min(r1["mult"], r1["div"]) > min(r2["mult"], r2["div"]):
                primary, r1 = alt, r2
            for k in acc: acc[k].append(r1[k])
            b = va[min(r1, key=r1.get)]
        a1 = b - b.mean(); r0 = ref_alt = primary if proj != "down" else dn[e].float()
        r0 = r0.float(); r0 = r0 - r0.mean()
        corr_best = min(corr_best, float((a1 * r0).sum() / (a1.norm() * r0.norm() + 1e-30)))
    print(f"  expert {e:3d} done", flush=True)

means = {k: sum(v) / len(v) for k, v in acc.items()}
mx = {k: max(v) for k, v in acc.items()}
winner = min(("mult", "div"), key=lambda k: means[k])
ok = means[winner] <= 0.10 and mx[winner] <= 0.25 and means["raw"] > 0.5
res = {
    "sampled": SAMPLE, "pairs": len(acc["mult"]),
    "relerr_mean": {k: round(means[k], 4) for k in means},
    "relerr_max": {k: round(mx[k], 4) for k in mx},
    "scale_inv_mean_of_sx": round(sum(scale_means) / len(scale_means), 5),
    "best_mode": winner, "best_corr": round(corr_best, 5),
    "fork_today_rawcast_relerr_mean": round(means["raw"], 4),
    "verdict": "IDENTITY-PASS" if ok else "IDENTITY-FAIL",
}
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
