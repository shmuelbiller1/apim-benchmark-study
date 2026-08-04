#!/usr/bin/env python3
"""
APIM CLOUD STUDY PACK
=====================
Three studies that are compute-bound, not insight-bound. Each answers a
question the program currently cannot answer, and none needs hardware.

  STUDY 1  scaling   Extend the N-scan to N=256. Decides polynomial vs
                     exponential growth -- the single most consequential
                     open number in the program (Appendix K).
  STUDY 2  reference How optimistic is the "best-found" reference? Every
                     p_success above N=20 is measured against it, so if it
                     is loose, the scaling exponent is understated.
  STUDY 3  baseline  Twin vs discrete Simulated Bifurcation at MATCHED
                     compute. The program has never once compared its own
                     algorithm against a competing one on the same
                     instances -- Ch 9.2 cites literature numbers only.

Vectorized over instances (I,B,N tensors) so one Python step advances every
instance at once. Saves incrementally: killing the job keeps finished sizes.

USAGE
  pip install numpy
  python3 apim_cloud_study.py --study scaling  --sizes 8,16,24,32,48,64,96,128,192,256 \
                              --instances 25 --batch 100 --out scaling.json
  python3 apim_cloud_study.py --study reference --sizes 12,16,20 --instances 20 --out ref.json
  python3 apim_cloud_study.py --study baseline  --sizes 32,64,128 --instances 25 --out base.json

COST GUIDE (single modern CPU core, numpy w/ good BLAS)
  N=64   25 inst x 100 batch x 30k steps  ~  3 min
  N=128  same                             ~ 10 min
  N=256  same                             ~ 45 min
  Full scaling study to N=256             ~  3-5 core-hours
Use --threads to let BLAS parallelize, or run sizes as separate jobs.
"""
import numpy as np, json, os, argparse, time

# ----------------------------------------------------------------------
# Instances
# ----------------------------------------------------------------------
def make_instances(N, n_inst, seed=777):
    rng = np.random.default_rng(seed)
    J = np.zeros((n_inst, N, N))
    for i in range(n_inst):
        U = np.triu(rng.choice([-1.0, 1.0], size=(N, N)), 1)
        J[i] = U + U.T
    return J

def normalize(J):
    """Row-sum normalization, per instance (matches the frozen twin)."""
    return J / np.abs(J).sum(axis=2).max(axis=1)[:, None, None]

def energies(s, J):
    """s:(I,B,N) J:(I,N,N) -> (I,B)"""
    return -0.5 * np.einsum('ibn,inm,ibm->ib', s, J, s, optimize=True)

# ----------------------------------------------------------------------
# ARM B — the APIM digital twin, frozen schedule (Ch 5.3)
# ----------------------------------------------------------------------
def twin(J, batch, steps, seed, eps=1.0, beta=0.8, p_max=1.5,
         sigma=0.03, dt=0.01, update_every=10, lag=1, dtype=np.float64):
    I, N, _ = J.shape
    rng = np.random.default_rng(seed)
    Jn = normalize(J).astype(dtype)
    x = (1e-3 * rng.standard_normal((I, batch, N))).astype(dtype)
    e = np.ones((I, batch, N), dtype=dtype)
    f = np.zeros((I, batch, N), dtype=dtype)
    buf = [x.copy() for _ in range(lag + 1)]
    sq = dtype(sigma * np.sqrt(dt))
    for t in range(steps):
        p = p_max * t / steps
        if t % update_every == 0:
            buf.append(x.copy())
            xm = buf.pop(0)
            f = e * np.einsum('ibn,inm->ibm', xm, Jn, optimize=True)
        a = max(p - 1.0, 0.05)
        x += dt * ((p - 1.0 - x * x) * x + eps * f) + sq * rng.standard_normal(x.shape)
        e = np.clip(e + dt * (-beta * e * (x * x - a)), 0.2, 5.0)
    s = np.sign(x); s[s == 0] = 1.0
    return energies(s, J), steps // update_every       # (energies, matmul count)

# ----------------------------------------------------------------------
# COMPETITOR — discrete Simulated Bifurcation (Goto et al.), reference impl.
# ----------------------------------------------------------------------
def dsbm(J, batch, steps, seed, a0=1.0, dt=0.5, dtype=np.float64):
    I, N, _ = J.shape
    rng = np.random.default_rng(seed)
    c0 = 0.5 / (np.sqrt((J ** 2).sum(axis=(1, 2)) / (N * (N - 1))) * np.sqrt(N))
    c0 = c0[:, None, None].astype(dtype)
    Jd = J.astype(dtype)
    x = (0.1 * rng.standard_normal((I, batch, N))).astype(dtype)
    y = (0.1 * rng.standard_normal((I, batch, N))).astype(dtype)
    for t in range(steps):
        a = a0 * t / steps
        y += dt * (-(a0 - a) * x + c0 * np.einsum('ibn,inm->ibm',
                                                  np.sign(x), Jd, optimize=True))
        x += dt * a0 * y
        over = np.abs(x) > 1.0
        y[over] = 0.0
        x[over] = np.sign(x[over])
    s = np.sign(x); s[s == 0] = 1.0
    return energies(s, J), steps                        # one matmul per step

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def tts(p, cost):
    if p <= 0: return float('inf')
    if p >= 1: return float(cost)
    return float(cost * np.log(0.01) / np.log(1 - p))

def psucc(E, ref, tol=0.5):
    return (E <= ref[:, None] + tol).mean(axis=1)        # (I,)

def fit_and_bootstrap(sizes, per_inst_tts, n_boot=2000, seed=0):
    """per_inst_tts: dict N -> array of per-instance TTS. Returns fits + CIs."""
    rng = np.random.default_rng(seed)
    Ns = np.array(sorted(sizes), dtype=float)
    med = np.array([np.median(per_inst_tts[int(N)]) for N in Ns])
    ok = np.isfinite(med) & (med > 0)
    lp = np.polyfit(np.log(Ns[ok]), np.log(med[ok]), 1)
    le = np.polyfit(Ns[ok], np.log(med[ok]), 1)
    rp = np.log(med[ok]) - np.polyval(lp, np.log(Ns[ok]))
    re = np.log(med[ok]) - np.polyval(le, Ns[ok])
    n = ok.sum()
    aic = lambda r: n * np.log(max((r ** 2).sum() / n, 1e-300)) + 2 * 2
    sl, ex = [], []
    for _ in range(n_boot):
        mb = []
        for N in Ns:
            v = np.asarray(per_inst_tts[int(N)], dtype=float)
            v = v[np.isfinite(v)]
            mb.append(np.median(rng.choice(v, len(v), replace=True)) if len(v) else np.nan)
        mb = np.array(mb); m = np.isfinite(mb) & (mb > 0)
        if m.sum() >= 3:
            sl.append(np.polyfit(np.log(Ns[m]), np.log(mb[m]), 1)[0])
            ex.append(np.polyfit(Ns[m], np.log(mb[m]), 1)[0])
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float('nan')
    return {
        "power_exponent": float(lp[0]), "power_resid_ss": float((rp ** 2).sum()),
        "power_AIC": float(aic(rp)),
        "power_CI95": [q(sl, 2.5), q(sl, 97.5)],
        "exp_rate": float(le[0]), "exp_resid_ss": float((re ** 2).sum()),
        "exp_AIC": float(aic(re)),
        "exp_CI95": [q(ex, 2.5), q(ex, 97.5)],
        "verdict": "POWER-LAW" if (rp ** 2).sum() < (re ** 2).sum() else "EXPONENTIAL",
        "median_tts": {int(N): float(m) for N, m in zip(Ns, med)},
    }

def save(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)

# ----------------------------------------------------------------------
# STUDY 1 — scaling
# ----------------------------------------------------------------------
def study_scaling(args):
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    grids = {}
    for N in args.sizes:
        base = [1000, 3000, 10000, 30000] if N <= 16 else [3000, 10000, 30000, 60000]
        if N >= 128: base = [10000, 30000, 60000, 120000]
        grids[N] = base
    for N in args.sizes:
        key = str(N)
        if key in out and not args.force:
            print(f"N={N}: cached, skipping"); continue
        t0 = time.time()
        J = make_instances(N, args.instances, seed=args.seed)
        runs, allE = {}, []
        for st in grids[N]:
            E, cost = twin(J, args.batch, st, seed=args.seed + st, dtype=args.dtype)
            runs[st] = (E, cost); allE.append(E)
        ref = np.min(np.concatenate(allE, axis=1), axis=1)
        rec = {}
        for st, (E, cost) in runs.items():
            p = psucc(E, ref)
            rec[str(st)] = {"p_per_instance": p.tolist(),
                            "p_median": float(np.median(p)),
                            "matmuls": int(cost),
                            "tts_per_instance": [tts(pi, cost) for pi in p]}
        best = min(rec.items(), key=lambda kv: tts(kv[1]["p_median"], kv[1]["matmuls"]))
        rec["_optimum"] = {"steps": int(best[0]), "p_median": best[1]["p_median"],
                           "tts": tts(best[1]["p_median"], best[1]["matmuls"])}
        out[key] = rec; save(args.out, out)
        print(f"N={N:>4} ({time.time()-t0:6.0f}s) opt_steps={best[0]:>6} "
              f"p={best[1]['p_median']:.4f} TTS={rec['_optimum']['tts']:.0f}")
    per = {int(k): np.array(v["_optimum"] and
             v[str(v["_optimum"]["steps"])]["tts_per_instance"], dtype=float)
           for k, v in out.items() if "_optimum" in v}
    if len(per) >= 3:
        fit = fit_and_bootstrap(list(per), per)
        out["_fit"] = fit; save(args.out, out)
        print("\n" + "=" * 70)
        print(f"POWER-LAW   TTS ~ N^{fit['power_exponent']:.3f}  "
              f"CI95 {fit['power_CI95']}  AIC {fit['power_AIC']:.2f}")
        print(f"EXPONENTIAL TTS ~ exp({fit['exp_rate']:.4f} N)  "
              f"CI95 {fit['exp_CI95']}  AIC {fit['exp_AIC']:.2f}")
        print(f"VERDICT: {fit['verdict']} (lower AIC wins; "
              f"ΔAIC = {abs(fit['power_AIC']-fit['exp_AIC']):.2f})")

# ----------------------------------------------------------------------
# STUDY 2 — reference quality
# ----------------------------------------------------------------------
def study_reference(args):
    """Compare best-found against EXACT ground state (feasible to N~22).
    Quantifies how much p_success is overestimated above the exact range."""
    out = {}
    for N in args.sizes:
        if N > 24:
            print(f"N={N}: skipped (exact enumeration infeasible)"); continue
        J = make_instances(N, args.instances, seed=args.seed)
        st = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) * 2 - 1
        exact = np.array([(-0.5 * np.einsum('si,ij,sj->s', st, J[i], st)).min()
                          for i in range(args.instances)])
        E, cost = twin(J, args.batch, args.steps, seed=args.seed, dtype=args.dtype)
        bf = E.min(axis=1)
        p_exact = psucc(E, exact); p_bf = psucc(E, bf)
        gap = (bf - exact)
        out[str(N)] = {"p_vs_exact_median": float(np.median(p_exact)),
                       "p_vs_bestfound_median": float(np.median(p_bf)),
                       "inflation_factor": float(np.median(p_bf) / max(np.median(p_exact), 1e-9)),
                       "instances_where_bestfound_missed_GS": int((gap > 0.5).sum())}
        save(args.out, out)
        print(f"N={N:>3}: p_vs_exact={out[str(N)]['p_vs_exact_median']:.4f}  "
              f"p_vs_bestfound={out[str(N)]['p_vs_bestfound_median']:.4f}  "
              f"inflation={out[str(N)]['inflation_factor']:.2f}x  "
              f"missed GS on {out[str(N)]['instances_where_bestfound_missed_GS']}/{args.instances}")
    print("\nIf inflation grows with N, the Appendix K exponent is UNDERSTATED.")

# ----------------------------------------------------------------------
# STUDY 3 — competitive baseline at matched compute
# ----------------------------------------------------------------------
def study_baseline(args):
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for N in args.sizes:
        key = str(N)
        if key in out and not args.force: print(f"N={N}: cached"); continue
        t0 = time.time()
        J = make_instances(N, args.instances, seed=args.seed)
        twin_steps = args.steps
        Et, mm = twin(J, args.batch, twin_steps, seed=args.seed, dtype=args.dtype)
        # dSBM gets the SAME number of matrix multiplies
        Eb, mmb = dsbm(J, args.batch, mm, seed=args.seed + 1, dtype=args.dtype)
        ref = np.minimum(Et.min(axis=1), Eb.min(axis=1))
        pt, pb = psucc(Et, ref), psucc(Eb, ref)
        ttst = np.array([tts(p, mm) for p in pt])
        ttsb = np.array([tts(p, mmb) for p in pb])
        both = np.isfinite(ttst) & np.isfinite(ttsb)
        out[key] = {"matmuls_each": int(mm),
                    "twin_p_median": float(np.median(pt)),
                    "dsbm_p_median": float(np.median(pb)),
                    "twin_tts_median": float(np.median(ttst[both])) if both.any() else None,
                    "dsbm_tts_median": float(np.median(ttsb[both])) if both.any() else None,
                    "twin_wins_on_instances": int((ttst < ttsb).sum()),
                    "n_instances": int(args.instances)}
        save(args.out, out)
        r = out[key]
        ratio = (r["dsbm_tts_median"] / r["twin_tts_median"]) if (r["twin_tts_median"]) else float('nan')
        print(f"N={N:>4} ({time.time()-t0:5.0f}s) twin p={r['twin_p_median']:.3f} "
              f"dSBM p={r['dsbm_p_median']:.3f} | TTS ratio (dSBM/twin) = {ratio:.2f} "
              f"| twin wins {r['twin_wins_on_instances']}/{args.instances}")
    print("\nRatio > 1 means the APIM algorithm beats dSBM at equal compute.")
    print("Ratio < 1 means the hardware must make up the gap. Either way it is")
    print("the first head-to-head this program has ever run.")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=["scaling", "reference", "baseline"])
    ap.add_argument("--sizes", default="8,16,24,32,48,64")
    ap.add_argument("--instances", type=int, default=25)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--steps", type=int, default=30000, help="studies 2 and 3")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--float32", action="store_true", help="~2x faster, verify agreement first")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.sizes = [int(s) for s in a.sizes.split(",")]
    a.dtype = np.float32 if a.float32 else np.float64
    print(f"# APIM cloud study: {a.study} | sizes={a.sizes} | inst={a.instances} "
          f"| batch={a.batch} | dtype={a.dtype.__name__}")
    {"scaling": study_scaling, "reference": study_reference, "baseline": study_baseline}[a.study](a)
