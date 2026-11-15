# Which event timing measures a kernel best on the GB10?

`event_vs_nsys.cu` (2026-11-14, before the final runs) times a read-only kernel over a
4 MB, a 20 MB and a 100 MB buffer, 200 launches each, two ways: **cold** (events around
one launch on an idle GPU, synchronized after each) and **spin** (the same with a 100 µs
`%globaltimer` spin kernel queued before the start event, meant to hide the host's launch
latency). It was then run under Nsight Systems, whose kernel durations are the ground
truth. Output (medians, µs):

| buffer | Nsight Systems kernel | events, cold | events, spin |
| --- | --- | --- | --- |
| 4 MB | 2.816 | 4.544 (+1.7) | 5.952 (+3.1) |
| 20 MB | 22.752 | 24.032 (+1.3) | 25.408 (+2.6) |
| 100 MB | 439.168 / 438.096 | 438.800 | 439.072 |

A companion sweep over 4096–5120 KB in 128 KB steps gave spin-mode medians of exactly
5.952 µs at every size while cold mode moved in ~0.1 µs steps: the spin variant adds
about 1.3 µs and lands on a ~1.024 µs lattice, hiding real differences. Cold events add
a fixed ~1.3–1.7 µs of event overhead (not the host's launch latency, which the driver
evidently submits together with the start event).

Decision: the pre-registered isolated measure is **cold** events around one launch (the
proposal's original method); the spin variant is kept as a recorded diagnostic column
(`spin_med_us`); Nsight Systems kernel durations of the final set are the ground-truth
cross-check (`results/nsys/`). The tuning run was repeated with this measure.
