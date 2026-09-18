# The Machine This Study Was Built to Evaluate

This is the hardware design the benchmark in [`README.md`](README.md) and the
paper were built to test — written as it stood *before* the result came in, and
left unedited by it.

The idea and the finding are two different things. One is an engineering
proposal; the other is a test of it. Nothing below is disproven by the
benchmark. It is simply shown to have no measurable edge over software at the
problem sizes tested — which is the whole reason the design is published here
rather than fabricated.

---

## 1. The core idea, in one paragraph

Build a network of small electronic oscillators, each physically capable of
settling into exactly one of two stable states (a "parametron," a real 1954
concept). Wire or digitally couple them so that the state the *whole network*
settles into represents the answer to a hard optimization problem — the same
class of problem D-Wave's quantum annealer and NTT's laser-based Coherent Ising
Machine are built to solve. Do it with ordinary electronic parts, at room
temperature, with no refrigeration and no lasers.

## 2. The physical trick

A parametron is created by pumping a resonant circuit at exactly **twice** its
natural frequency — not once per cycle like pushing a swing, but twice. Do this
to an ordinary LC tank circuit (an inductor and a capacitor) or a quartz
crystal, using a varactor (a capacitor whose value changes with voltage) as the
pump input, and above a threshold pump strength the circuit spontaneously locks
into oscillating in one of exactly two phases — 0° or 180° apart. That is a
physical bit, built entirely out of motion rather than a transistor switch.

**Threshold condition:** the pump modulation depth must exceed `2/Q`, where Q is
the circuit's quality factor — how many oscillations it sustains before losing
its energy to resistance.

## 3. Two cell designs

| | Goto LC cell | Quartz crystal cell |
|---|---|---|
| Frequency | 1.7–2.35 MHz | 8–15 MHz |
| Q (quality factor) | ~250 (guaranteed to oscillate) | 30,000–300,000 (higher precision, riskier) |
| Threshold margin | 5.6× — safe | 2.4×–23.6× — depends on real component tolerance |
| Role | Reliable fallback | High-Q upgrade, if the threshold margin holds on real parts |

## 4. LC cell schematic

```
+Vbias ──[R1 100k]──┬──[R2 100k]──── PUMP IN (2f_i) ──[C4 1nF]──┐
                     │                                            │
                (center tap) ◄────────────────────────────────────┘
                     │
             ┌───────┴────────┐
             │ D1 ►|──┴──|◄ D2 │   anti-series hyperabrupt varactor pair
             └───┬─────────┬───┘   (~80 pF each at Vbias; pair ≈ 40 pF)
                 │         │
  TANK ──────────┴────┬────┴──────┬────────────┬──────────────┬──────
                      │           │            │              │
                [L1 33 µH]  [C1 150 pF] [C2 padder      [Q1 FET
                 toroid,     C0G ±5%    0–22 pF,         follower,
                 ±5%, Q≥100]            fit-on-test]     1 MΩ gate]
                      │           │            │              │
                     GND         GND          GND      └──► SUM BUS ──► LPF ──► ADC
  INJECTION IN (f_i) ──[C3 3.3 pF]──► TANK node
```

- **Anti-series varactor pair** cancels even-harmonic distortion.
- **Padder capacitor (C2)** — a hand-fit 0–22 pF trim — absorbs ±5% component
  tolerance so each cell lands exactly on its assigned frequency slot.
- **FET follower (Q1)** reads the cell's state without loading down the tank and
  killing its Q.
- **Low-pass filter** ahead of the shared ADC keeps pump tones from one cell
  from leaking into another's measurement.

## 5. Bill of materials, smallest working version (≈$550)

| Item | Quantity | Est. cost |
|---|---|---|
| Goto LC parametron cells | 8 | $40 |
| Quartz resonator cells | 8 | $20 |
| Summing / injection network | 1 bus | $40 |
| ADC/DAC/FPGA dev board (2ch, 125 MS/s) | 1 | $400 |
| Custom PCB + passives | 1 | $50 |
| **Total** | | **≈$550** |

## 6. How the cells "talk" without individual wiring

Wiring every pair of cells directly does not scale — 500 spins would need over
124,000 physical connections. Instead: every cell's output feeds one shared bus;
an FPGA continuously measures every cell's state, computes what each pairwise
coupling would contribute, and injects the sum back in as a drive signal,
thousands of times per second. The couplings live in the FPGA's memory as
numbers, not as copper.

**The binding constraint this creates:** the FPGA's loop (measure → compute →
inject) must complete well inside the cell's own memory time, or the correction
arrives too late to mean anything. This is why the carrier frequency has to stay
low — MHz, not GHz — in this architecture. It sets how much time the digital
loop has to work with.

This is also exactly the architecture the paper's Section 7 analyses, and the
reason it finds no energy advantage: in this form the coupling multiply `Jx` is
computed **digitally**, on the FPGA, precisely as dSBM computes it on a GPU.

## 7. The build ladder (cheapest-first)

| Stage | What | Cost | Tests |
|---|---|---|---|
| v−1 | Pure software simulation | $0 | Predict everything before touching hardware |
| v0⁻ | Two analog cells, scope, signal generator | $40–190 | Does a single cell actually oscillate and lock into two states? |
| v0 | 8 cells + FPGA board (Section 5's BOM) | ≈$550 | Full anneal, first real success-rate numbers |
| v1 | 64 cells, same board class, no custom fabrication | ≈$700 | Does it scale to a useful size? |
| v2 | Custom chip-scale array, higher frequency | Foundry run | Product-scale test |

**This project stopped at stage v−1**, which is the point of publishing it.

## 8. What the benchmark found when this was tested

Before spending money on hardware, the design above was simulated and
benchmarked against Toshiba's Simulated Bifurcation — pure software, no exotic
hardware — at matched computational cost. The result: **near-parity.** Properly
tested, with equal tuning effort on both sides, this design and a piece of
software land within roughly 15–50% of each other, with neither reliably ahead.

That finding, and the benchmarking mistake that nearly hid it, is the subject of
the paper and the README.

**The one part that survives intact.** If the couplings are *physically wired* —
fixed problem structure — rather than digitally computed, and the problem is
very large (thousands of variables, not the hundreds tested here), the physics
can in principle outrun any digital chip, because digital work grows with
problem size and physical settling time does not. Under that assumption the
paper computes a 150–180× reduction in digital operations on lattice instances,
and verifies in simulation that such an array anneals successfully *without*
measurement feedback or amplitude error correction (p=0.292 vs. 0.254 at 64
spins on lattice instances) — which the wired configuration could not support
anyway.

That is a narrower and different machine than the general-purpose one described
above. It remains unbuilt and untested, and the 150–180× figure is an
architectural projection, not a measurement.
