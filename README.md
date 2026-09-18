# Benchmark Verdicts for Analog Ising Machines Invert Under Tuning

Code, benchmark instances and results for the paper:

> **Benchmark Verdicts for Analog Ising Machines Invert Under Tuning:
> A Negative Result and a Reproducible Protocol**
> Shmuel Y. Biller, Independent researcher

## The result in one table

We benchmarked a measurement-feedback parametron Ising machine against discrete
Simulated Bifurcation (dSBM) on identical instances at matched compute. **The
verdict inverts depending purely on which solver is tuned:**

| Comparison | N=32 | N=64 |
|---|---|---|
| Untuned analog vs untuned dSBM | dSBM ahead | dSBM ahead |
| Tuned analog vs untuned dSBM | **analog ahead** | **analog ahead** |
| Both tuned (the only fair comparison) | near-parity | near-parity |

The first two rows would support opposite conclusions in a paper abstract.

**Proposed rule:** no comparison is reportable unless the same tuning budget has
been applied to every algorithm in it, and the settings used for each are stated
alongside the result. *An untuned baseline is not a baseline.*

## Reproduce the central claim in ~4 minutes

```bash
pip install numpy
python3 reproduce_inversion.py
```

This runs all three tuning regimes on the same instances at matched compute and
prints the three verdicts.

## An important caveat about the "fair" row

While preparing this repository we found that **even the both-tuned comparison is
sensitive to how large a tuning grid each side receives.** Our paper reports
dSBM ahead by 1.13× at N=32 using one tuning grid; `reproduce_inversion.py`,
which searches a different grid over 12 instances, reports the analog machine
ahead by 1.72× at the same size, and an exact tie at N=64.

We consider this a strengthening of the paper's argument rather than a
contradiction of it. The defensible claim is **near-parity, with the sign of the
residual difference dependent on tuning budget** — which is precisely why a
single reported ratio from a single tuning effort should not be trusted. Readers
reproducing this work should expect the third row to land near 1× and should not
expect to recover our exact ratios.

Anyone extending this work should report the tuning grid searched, not only the
final settings.

## Other results in the paper

- **Scaling:** TTS ~ N^2.115, 95% CI [1.93, 2.30]; power law strongly favoured
  over exponential (ΔAIC = 9.97).
- **Algorithmic gains:** ~24× compounded from four parameters that had been fixed
  by inheritance rather than measurement — update interval (8×), inelastic wall
  (2×), ramp exponent (2×), confidence-gated feedback (1.5×).
- **No energy advantage** in measurement-feedback form: the coupling multiply is
  digital in both machines, so there is neither an operation-count surplus nor a
  per-operation advantage.
- **Reference quality verified:** on 80 instances at N=12–22, the best-found
  energy equalled the exact enumerated ground state in every case.

## Contents

| File | Description |
|---|---|
| `reproduce_inversion.py` | **Start here.** Reproduces the paper's central claim |
| `apim_twin.py` | Simulation model (parametron dynamics + measurement feedback + AEC) |
| `apim_cloud_study.py` | Three larger studies: scaling, reference quality, competitive baseline |
| `apim_instances_v1.json` | 100 dense ±1 instances, N=8, with brute-forced ground states |
| `paper.pdf` | The paper (compiled) |
| `main.tex` | LaTeX source for the paper |

## Running the larger studies

```bash
# Scaling scan (~20 min on 4 vCPU)
python3 apim_cloud_study.py --study scaling \
    --sizes 8,16,24,32,48,64 --instances 25 --batch 100 --out scaling.json

# Head-to-head at matched compute
python3 apim_cloud_study.py --study baseline \
    --sizes 32,64,128 --instances 25 --batch 100 --out baseline.json

# Reference quality: best-found vs exact ground state
python3 apim_cloud_study.py --study reference \
    --sizes 12,16,20,22 --instances 20 --out reference.json
```

All studies run on free-tier cloud notebooks. The scaling scan completes on
Kaggle (4 vCPU) in about 20 minutes.

## Benchmark integrity

Results quoted against this instance library should cite the hash:

```bash
python3 -c "import json,hashlib; d=json.load(open('apim_instances_v1.json')); \
print(hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest())"
```

Expected:
```
b8c6050b2edf1ed40d02ae97e0e37b39ea60661822ccc5afa3c40cfdf3c48812
```

Note the exact serialization: `json.dumps(d, sort_keys=True)` with **default**
separators. Compact separators produce a different hash.

## Limitations

- **All results are simulation.** No hardware was built. Every statement about
  physical implementations is a projection from the model.
- **The central hypothesis is untested.** Whether physical analog dynamics
  outperforms a digital simulation of the same dynamics cannot be evaluated
  without hardware.
- **Instance class.** All results use dense random ±1 instances; structured and
  sparse instances may rank algorithms differently.
- **Comparator.** Our dSBM is a reference implementation, not the authors'
  optimised version, which makes the near-parity result conservative.

## License

MIT — see [LICENSE](LICENSE).
