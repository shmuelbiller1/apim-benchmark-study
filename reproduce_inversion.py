#!/usr/bin/env python3
"""
reproduce_inversion.py
======================
Reproduces the paper's central claim: the competitive verdict between an analog
Ising machine and discrete Simulated Bifurcation INVERTS depending on which
solver is tuned.

This is the shortest path to checking the main result. Runtime ~3-6 minutes on
one CPU core.

    pip install numpy
    python3 reproduce_inversion.py

Expected output: three verdicts from the same instances at matched compute --
dSBM ahead, then APIM ahead, then dSBM ahead by a small margin. Only the third
(both solvers tuned) is a defensible comparison.
"""
import numpy as np
import time


# ----------------------------------------------------------------------
# Instances
# ----------------------------------------------------------------------
def make_instances(N, n_inst, seed=4242):
    rng = np.random.default_rng(seed)
    J = np.zeros((n_inst, N, N))
    for k in range(n_inst):
        U = np.triu(rng.choice([-1.0, 1.0], size=(N, N)), 1)
        J[k] = U + U.T
    return J


def normalize(J):
    return J / np.abs(J).sum(axis=2).max(axis=1)[:, None, None]


def energies(s, J):
    return -0.5 * np.einsum('ibn,inm,ibm->ib', s, J, s, optimize=True)


# ----------------------------------------------------------------------
# Solver A: parametron / measurement-feedback Ising machine
# ----------------------------------------------------------------------
def apim(J, batch, steps, seed, update_every=10, eps=1.0, sigma=0.03,
         beta=0.8, p_max=1.5, dt=0.01, clamp=None, gamma=1.0):
    """update_every, clamp and gamma are the tunable parameters. The paper's
    'untuned' configuration is the schedule frozen at N=8: update_every=10,
    no clamp, gamma=1 (linear ramp)."""
    I, N, _ = J.shape
    rng = np.random.default_rng(seed)
    Jn = normalize(J)
    x = 1e-3 * rng.standard_normal((I, batch, N))
    e = np.ones((I, batch, N))
    f = np.zeros((I, batch, N))
    prev = x.copy()
    macs = 0
    for t in range(steps):
        p = p_max * ((t / steps) ** gamma)
        if t % update_every == 0:
            f = e * np.einsum('ibn,inm->ibm', prev, Jn, optimize=True)
            prev = x.copy()
            macs += N * N
        a = max(p - 1.0, 0.05)
        x = x + dt * ((p - 1.0 - x * x) * x + eps * f) \
              + sigma * np.sqrt(dt) * rng.standard_normal(x.shape)
        if clamp is not None:
            over = np.abs(x) > clamp
            x = np.where(over, np.sign(x) * clamp, x)
        e = np.clip(e + dt * (-beta * e * (x * x - a)), 0.2, 5.0)
    s = np.sign(x); s[s == 0] = 1.0
    return energies(s, J), macs


# ----------------------------------------------------------------------
# Solver B: discrete Simulated Bifurcation (reference implementation)
# ----------------------------------------------------------------------
def dsbm(J, batch, steps, seed, a0=1.0, dt=0.5):
    I, N, _ = J.shape
    rng = np.random.default_rng(seed)
    c0 = 0.5 / (np.sqrt((J ** 2).sum(axis=(1, 2)) / (N * (N - 1))) * np.sqrt(N))
    c0 = c0[:, None, None]
    x = 0.1 * rng.standard_normal((I, batch, N))
    y = 0.1 * rng.standard_normal((I, batch, N))
    for t in range(steps):
        a = a0 * t / steps
        y += dt * (-(a0 - a) * x
                   + c0 * np.einsum('ibn,inm->ibm', np.sign(x), J, optimize=True))
        x += dt * a0 * y
        over = np.abs(x) > 1.0
        y[over] = 0.0
        x[over] = np.sign(x[over])
    s = np.sign(x); s[s == 0] = 1.0
    return energies(s, J), steps * N * N


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def tts(p, cost):
    if p <= 0:
        return float('inf')
    return float(cost) if p >= 1 else float(cost * np.log(0.01) / np.log(1 - p))


def p_success(E, ref, tol=0.5):
    return float(np.median((E <= ref[:, None] + tol).mean(axis=1)))


# ----------------------------------------------------------------------
def main():
    N_LIST = [32, 64]
    N_INST = 12
    BATCH = 80

    print("=" * 78)
    print("REPRODUCING THE PAPER'S CENTRAL CLAIM")
    print("Same instances, matched compute, three different tuning regimes.")
    print("=" * 78)

    for N in N_LIST:
        t0 = time.time()
        J = make_instances(N, N_INST)
        runs = {}

        # --- (1) UNTUNED analog: the schedule frozen at N=8
        runs['apim_untuned'] = apim(J, BATCH, 30000, seed=5,
                                    update_every=10, eps=1.0, sigma=0.03,
                                    clamp=None, gamma=1.0)

        # --- (2) TUNED analog: update_every, inelastic wall, reshaped ramp
        best_apim = None
        for ue in (100, 200):
            for st in (10000, 20000):
                r = apim(J, BATCH, st, seed=5, update_every=ue, eps=2.0,
                         sigma=0.01, clamp=0.3, gamma=5.0)
                if best_apim is None or r[1] < best_apim[1]:
                    pass
                runs[f'apim_tuned_{ue}_{st}'] = r

        # --- (3) dSBM at several run lengths (untuned = the long one)
        for st in (50, 100, 200, 300):
            runs[f'dsbm_{st}'] = dsbm(J, BATCH, st, seed=6)

        # shared reference: best energy any configuration found
        ref = np.min(np.concatenate([E for E, _ in runs.values()], axis=1), axis=1)

        def best_of(prefix):
            out = None
            for k, (E, m) in runs.items():
                if not k.startswith(prefix):
                    continue
                T = tts(p_success(E, ref), m)
                if np.isfinite(T) and (out is None or T < out):
                    out = T
            return out

        apim_untuned = tts(p_success(runs['apim_untuned'][0], ref),
                           runs['apim_untuned'][1])
        apim_tuned = best_of('apim_tuned')
        dsbm_untuned = tts(p_success(runs['dsbm_300'][0], ref), runs['dsbm_300'][1])
        dsbm_tuned = best_of('dsbm_')

        def verdict(a, d):
            return (f"APIM ahead {d/a:.2f}x" if a < d else f"dSBM ahead {a/d:.2f}x")

        print(f"\nN = {N}   ({N_INST} instances, {time.time()-t0:.0f}s)")
        print(f"  {'comparison':<38} {'APIM TTS':>12} {'dSBM TTS':>12}  verdict")
        print(f"  {'-'*38} {'-'*12} {'-'*12}  {'-'*22}")
        print(f"  {'untuned APIM vs untuned dSBM':<38} {apim_untuned:>12,.0f} "
              f"{dsbm_untuned:>12,.0f}  {verdict(apim_untuned, dsbm_untuned)}")
        print(f"  {'TUNED APIM vs untuned dSBM':<38} {apim_tuned:>12,.0f} "
              f"{dsbm_untuned:>12,.0f}  {verdict(apim_tuned, dsbm_untuned)}")
        print(f"  {'TUNED APIM vs TUNED dSBM  <- fair':<38} {apim_tuned:>12,.0f} "
              f"{dsbm_tuned:>12,.0f}  {verdict(apim_tuned, dsbm_tuned)}")

    print("\n" + "=" * 78)
    print("Only the third row of each block is a defensible comparison.")
    print("The first two would support opposite conclusions in a paper abstract.")
    print("Rule: no comparison is reportable unless every algorithm in it")
    print("received the same tuning budget, with settings stated.")
    print("=" * 78)


if __name__ == "__main__":
    main()
