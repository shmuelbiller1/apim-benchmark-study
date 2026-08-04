#!/usr/bin/env python3
"""
APIM v-1 -- Digital Twin & Arm B Reference Implementation
=========================================================
Three jobs, one file:
  1. TWIN: simulate the exact bench system (parametron SDEs + v0.3 feedback
     law + measurement DELAY) to sweep schedules and predict p_success.
  2. ARM B: this very code, ported to the FPGA, is the digital arm of the
     self-simulation kill test (KC2). Arm A replaces simulate_batch() with
     bench I/O; every other line stays identical. That IS the pairing.
  3. ANALYSIS: p_success / TTS@99% / bootstrap-CI functions reused verbatim
     on bench data, so twin predictions and bench results share one pipeline.

Honesty labels:
  - The oscillator model is the standard normalized parametron/CIM envelope
    equation [LIT - verify]. Parameter values below are STARTING POINTS for
    the sweep, not claims.
  - Twin output PREDICTS. It never PROVES. (See v-1 hazard note in the doc.)

Model per spin (rotating-frame envelope, Euler-Maruyama):
  dx_i = [ (p(t) - 1 - x_i^2) x_i + eps * f_i ] dt + sigma dW
  f_i  = e_i * sum_j Jn_ij x_j(t - lag)        <- DELAYED analog feedback
  de_i = -beta * e_i * (x_i^2 - a(t)) dt       <- amplitude-error correction
                                                  (Leleu-style [LIT - verify])
Run:  python3 apim_twin.py
"""
import numpy as np


# ----------------------------------------------------------------------
# Instances
# ----------------------------------------------------------------------
def make_instance(N, rng):
    """Dense random +/-1 Ising instance (frustrated with overwhelming prob.)."""
    J = np.triu(rng.choice([-1.0, 1.0], size=(N, N)), 1)
    return J + J.T


def brute_min_energy(J):
    """Exact ground energy by enumeration. Feasible to N ~ 20."""
    N = J.shape[0]
    states = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) * 2 - 1
    E = -0.5 * np.einsum("si,ij,sj->s", states, J, states)
    return E.min()


# ----------------------------------------------------------------------
# The twin / Arm B dynamics
# ----------------------------------------------------------------------
def simulate_batch(J, batch=300, steps=4000, dt=0.01, p_max=2.0,
                   eps=0.30, sigma=0.06, beta=0.20,
                   update_every=10, lag_updates=1, seed=0):
    """
    One call = `batch` independent anneals of one instance, vectorized.

    ARM A NOTE: on the bench, this function is replaced by DMA I/O to the
    physical cells (measure -> same f computation -> inject). Everything
    upstream and downstream of this function is shared by both arms.
    """
    rng = np.random.default_rng(seed)
    N = J.shape[0]
    Jn = J / np.abs(J).sum(axis=1).max()          # row-sum normalize
    x = 1e-3 * rng.standard_normal((batch, N))    # seed noise
    e = np.ones((batch, N))                       # error-correction vars
    buf = [x.copy() for _ in range(lag_updates + 1)]   # measurement delay line
    f = np.zeros((batch, N))
    for t in range(steps):
        p = p_max * t / steps                     # linear pump ramp (sweep me)
        if t % update_every == 0:                 # sampled, DELAYED feedback
            buf.append(x.copy())
            x_meas = buf.pop(0)                   # measurement lag_updates old
            f = e * (x_meas @ Jn)                 # analog-amplitude feedback
        a = max(p - 1.0, 0.05)                    # target amplitude^2
        x = x + dt * ((p - 1.0 - x * x) * x + eps * f) \
              + sigma * np.sqrt(dt) * rng.standard_normal((batch, N))
        e = np.clip(e + dt * (-beta * e * (x * x - a)), 0.2, 5.0)
    s = np.sign(x)
    s[s == 0] = 1.0
    E = -0.5 * np.einsum("bi,ij,bj->b", s, J, s)
    return s, E


# ----------------------------------------------------------------------
# Analysis pipeline (REUSED VERBATIM ON BENCH DATA)
# ----------------------------------------------------------------------
def p_success(E, E_min, tol=0.5):
    return float(np.mean(E <= E_min + tol))


def tts99(p, t_anneal):
    if p <= 0.0:
        return np.inf
    if p >= 1.0:
        return t_anneal
    return t_anneal * np.log(0.01) / np.log(1.0 - p)


def bootstrap_median_tts(p_list, t_anneal, B=2000, seed=1):
    """Median TTS across instances with bootstrap 95% CI. A result
    reported without this CI is not a result (v0.3 protocol)."""
    rng = np.random.default_rng(seed)
    arr = np.array([tts99(p, t_anneal) for p in p_list])
    meds = [np.median(rng.choice(arr, size=len(arr), replace=True))
            for _ in range(B)]
    return np.median(arr), np.percentile(meds, 2.5), np.percentile(meds, 97.5)


def energy_histogram(E, E_min):
    """Free third result: terminal-state distribution. Comparing this to a
    Boltzmann fit characterizes the machine as a SAMPLER at zero marginal
    cost -- same data, second market question."""
    vals, counts = np.unique(E - E_min, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


# ----------------------------------------------------------------------
# Demo run = the v-1 deliverable
# ----------------------------------------------------------------------
if __name__ == "__main__":
    N, N_INST, BATCH = 8, 12, 300
    T_ANNEAL = 1.0                       # model units; bench maps via t_update
    rng = np.random.default_rng(42)
    print("=" * 66)
    print(f"APIM v-1 twin: N={N}, {N_INST} instances x {BATCH} anneals")
    print("=" * 66)
    ps = []
    for k in range(N_INST):
        J = make_instance(N, rng)
        E_min = brute_min_energy(J)
        s, E = simulate_batch(J, batch=BATCH, seed=100 + k)
        p = p_success(E, E_min)
        ps.append(p)
        hist = energy_histogram(E, E_min)
        print(f"  inst {k:02d}: E_min={E_min:+.0f}  p_success={p:.3f}  "
              f"residual-E histogram={hist}")
    med, lo, hi = bootstrap_median_tts(ps, T_ANNEAL)
    print("-" * 66)
    print(f"  median p_success = {np.median(ps):.3f}   "
          f"(min {min(ps):.3f}, max {max(ps):.3f})")
    print(f"  median TTS@99%   = {med:.2f} anneals  [95% CI {lo:.2f}, {hi:.2f}]")
    print("-" * 66)
    print("  PREDICTION, not proof. Sweep eps/sigma/beta/p_max/lag, then")
    print("  freeze the schedule and take it to the board. (KC3 applies.)")
