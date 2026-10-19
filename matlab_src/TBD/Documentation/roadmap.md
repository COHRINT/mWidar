# TBD roadmap

A working list of features and fixes for the TBD particle filter. The main
direction is replacing the Gaussian-PSF likelihood with one built from the
mWidar forward model (derivation in `forward_model_likelihood.tex`). Each item
says **why**, **what**, and **which files** it touches.

File shorthand: everything lives in `TBD/Includes/` unless a path is given.

---

## Where things stand (2026-10-11)

**Filter.** Single-target PF-TBD, state `[px vx py vy I E]` in meters. Two
likelihoods, chosen by `run(..., 'Likelihood', ...)`: `'model'` (forward model,
the default) and `'gauss'` (windowed Gaussian PSF, the baseline). With `'model'`
the intensity is **fitted per cell**, `I_hat = clamp(C/E, I_min, I_max)` (item 21),
and written back into the state for display. Weights are log-domain within a
step (`normalize_log`). Resampling is SIS with an adaptive ESS threshold.

**Tracking now works** on the `env_test` S-curve once the target is acquired:
~2 cm RMSE at `Var` = 5–150, pE correct, median ESS/N ≈ 0.15. What made it work:
- **Fitted `I` (item 21).** With `I` sampled at birth, the likelihood was a
  narrow ridge in (position, I): the curvature in `I` is `E/sigma^2` ≈ 7e7, so
  a 1% error in `I` cost ~3,600 nats, more than the ~2,800 separating the truth
  cell from its x-mirror. The newborn with the best-matched `I` won wherever it
  was, and the cloud then locked there (x mirrored, error 1.75 m). Fitting `I`
  per cell makes the likelihood position-only: the truth cell ranks 1st of
  15,360 on every frame checked.
- **`round` placement in the simulator (item 9).** With `ceil`, the truth
  position's own cell scored ~3.8e4 nats below the cell the simulator imaged.

**Still failing: acquisition and recovery at low `Var`** (e.g. `Var` = 0.1 on
moving targets). See the new section "P0 — acquisition and exploration" below.

**Why the target is hard to see in raw frames (not a bug).** The target is a
small bump on a large pedestal (at k = 40: pedestal ~72, peak 6.35 above it).
Blur made per-pixel contrast worse (peak 2.52), and normalization only rescaled
each frame. The filter still sees the target because it sums over the whole
template.

**New likelihood.** For a unit-intensity template `h(x)` (the clean
forward-model image of a target in cell x) and i.i.d. Gaussian image noise:

```
log LR(x, I) = ( I*C(x) - 0.5*I^2*E(x) ) / sigma^2
C(x) = <h(x), z>   per frame, all cells at once: c = M'*(G*z)   (adjoint)
E(x) = ||h(x)||^2  static 128 x 128 lookup table
```

Each particle then needs two table lookups. Measured facts this rests on:
- The adjoint matches the direct dot product to all printed digits, and takes
  0.043 s per frame.
- Along x = 0, `E` runs from 5.4e11 (y = 0.3 m) to 1.4e7 (y = 3.7 m). Over the
  whole table it spans 6.2e6 to 2.5e13.
- The forward-model template is nonzero everywhere. Only 1% / 3% / 8–10% of
  its energy falls inside `pDist` = 5 / 10 / 20, so the window factorization
  does not carry over. The adjoint makes the full-image sum cheap instead.
- Min-max normalization breaks the model: 92% superposition error, and the
  normalized image doesn't depend on `I`.

**Open bugs found 2026-10-11.**
- `logBirthCorr` (`p_birth/q = nKeep/nValid`) is computed in `timestep` but
  never added to the newborns' `logL` (item 7).
- `debug_print` passes its string to `fprintf` as the format, so the `%` in
  "top 30% of valid cells" truncates the `timestep` debug line and drops its
  newline (item 18).
- Weights are carried between frames in the linear domain, so any weight that
  underflows becomes `log(0) = -Inf` next frame and can never recover (item 11).

---

## P0 — acquisition and exploration (found 2026-10-11)

### What goes wrong
At low `Var` the filter often locks onto the wrong cell when the target
appears and never recovers. Three causes combine:

1. **Existence is detected from the pedestal, not from position.** With fitted
   `I`, almost every cell explains the frame's pedestal energy: cells anywhere
   score within ~0.01% of the truth cell's log LR (~1e7–1e9). So the first
   newborn *anywhere* drives pE to exactly 1, and whichever of the ~120
   newborns scores best wins. Birth cells are the top `(1-gamma)` of raw pixels:
   at `gamma` = 0.70 that is 4,609 cells, only 25 within 0.1 m of the target,
   so ~0.6 newborns are expected near it. Measured k = 10 winners: 0.03–0.49 m
   from the truth.
2. **No births once a track is declared.** Births only come from particles
   that were dead (E = 0) on the previous frame (Ristic's regime transition).
   Particles do die every frame (~100), but with live particles at log LR
   ~1e9 their weight is exactly 0, and the resample at the end of the *same*
   timestep removes all of them. Next frame: 0 dead particles, 0 births. Births
   only resume on frames that skip the resample (v = 15: 5 births over
   k = 11–89; v = 0.1: none).
   - Births can't fix a wrong lock even in principle: a newborn's prior mass is
     ~(1 − pE)·Pb ≈ exp(−1e9). Landing exactly on the peak gains ~1e4 nats and
     still loses. The fix has to act inside the E = 1 hypothesis.
3. **At low `Var` the cloud can't climb out.** Away from the target the
   log-LR map has ripples (PSF whiskers, pedestal arcs) whose depth scales as
   1/sigma². The resample keeps the best ripple and `Q` only diffuses
   ~0.03 m/frame. Forced-offset test (cloud started a fixed distance from truth,
   40 frames):

   | target | Var | 0.15 m | 0.30 m | 0.50 m | median neighbour dLL |
   |---|---|---|---|---|---|
   | static | 0.1 | recovers | stuck (0.38 m) | stuck (0.38 m) | ~2,600 |
   | static | 15 | recovers | recovers | recovers | ~22 |
   | S-curve | 0.1 | recovers | recovers | recovers | ~16,000 |
   | S-curve | 15 | recovers | recovers | recovers | ~116 |

   The S-curve runs were started with the *true* velocity. Real newborns draw
   `v` from ±0.05 m/s while the target moves ~0.3 m/s; very likely why moving
   targets fail where static ones sometimes work (not isolated yet).

**Low ESS and a tight cloud are not bugs in themselves.** At `Var` = 0.1,
neighbouring cells differ by thousands of nats, so the true posterior sits on
about one cell. The problem is the lack of any proposal that tries other cells
once the cloud has collapsed on the wrong one.

### Workarounds, in recommended order
The existing items this section promotes are **19** (pedestal projection) and
**7** (likelihood-map birth proposal). New items 23–27 below.

1. **Item 19, pedestal projection.** Attacks the cause: a newborn in the wrong
   place no longer scores high, so pE stays low, dead particles survive and
   births continue until one lands on the target. Also matters on real data,
   where the pedestal won't match the model exactly.
2. **Item 7, birth proposal from the likelihood map.** Truth cell was rank 1
   of 15,360 on the profile log-LR map on every frame checked, so acquisition
   becomes near-certain. Doesn't help if a track is lost later.
3. **Item 24 or 25** if mid-track recovery is needed (24 is less work, 25 more
   principled).
4. **Item 27** is a scope decision, not a fix: it removes sampling entirely.

### 23. Birth velocity range
**Why.** Newborns draw `v` from `[v_min, v_max]` = ±0.05 m/s; the S-curve target
moves ~0.3 m/s. A newborn on the right cell with the wrong velocity drifts off.
**What.** Widen the birth velocity prior to the scenario's speed range, or
draw it from a prior centered on 0 with std ~ max target speed. Cheap; a
tuning change, not a fix for items 1–3 above.
**Files.** `TBD.m` (defaults), `env_test.m`.

### 24. Jump component in the dynamics
**Why.** Lets the model itself say "the current position may be wrong".
**What.** With small probability ε per frame, a live particle's position is
redrawn from the birth proposal instead of moved by NCV. Weights stay
standard bootstrap weights (it is part of the transition model). ε is a tuning
knob and should be reported as a modelling choice.
**Files.** `TBD_PF.m` (`dynamics` / `timestep`), `TBD.m` (parser).

### 25. Resample-move (MCMC) step
**Why.** Restores diversity while keeping the correct target distribution.
**What.** After resampling, one Metropolis–Hastings step per particle, proposing
from a mixture of a small local random walk and an independent jump drawn from
the likelihood map. A jump to the true peak is accepted because it scores far
better. The acceptance ratio needs `p(x_k | x_{k-1})`, so ancestors (item 14)
must be kept. Reference: Gilks & Berzuini 2001 (resample-move); I believe
Ristic ch. 3 also covers an MCMC move step, unverified.
**Files.** `TBD_PF.m` (`resample` returns ancestors, new `move` method).

### 26. Tempering / progressive correction
**Why.** Stops the cloud collapsing to one cell before it has explored.
**What.** Apply the likelihood in stages, `exp(beta * logL)` with beta rising to
1, moving particles between stages (progressive correction, Oudjane & Musso
2000; annealed PF, Deutscher et al. 2000). A permanent beta < 1 is the same as
assuming noise variance sigma²/beta: only defensible as robustness to model
mismatch. "High `Var` works" is effectively this.
**Files.** `TBD_PF.m` (`timestep`).

### 27. Grid-based Bernoulli filter (alternative to particles)
**Why.** The likelihood is defined per cell and `llr_map` already scores all
16,384 cells every frame, so an exact grid filter is nearly free and has no
sampling or exploration problem at all.
**What.** Bayes filter over cells (plus existence), with motion as a blur
kernel; velocity needs a discretized velocity dimension or a diffusion model.
A natural baseline against the PF, and a coherent thesis contribution in its
own right. Davey et al. 2008 discusses grid-based Bayesian TBD (details not
checked). Unclear how Vo et al. 2010's formulation carries over. Bigger
detour: weigh against thesis scope before starting.
**Files.** new `TBD/Includes/TBD_Grid.m` (subclass of `TBD`), new test script.

### Related, lower value here
- **Item 15 (roughening / regularized PF, jitter).** Local only: can't cross
  0.3–0.5 m, and jitter within a cell changes nothing under the cell-constant
  likelihood.
- **More births.** Can't fix a wrong lock (see cause 2 above).

---

## P0 — forward-model likelihood

### 1. Template-energy table `E` as library code
**Why.** The table is the static half of the likelihood. Right now it's a
one-off script whose output silently depends on simulator defaults (`Var`,
`Blur`), and nothing records which model it was built for.

**What.**
- Move the computation into `Includes/`, e.g. a `simulator` method
  `E = template_energy(obj)`. Pass `'Var', 0` explicitly, and use whatever
  blur setting the run uses.
- Save the settings alongside the table (`blur`, `sigma`, `npx`, date). The
  filter refuses a table whose settings don't match its own.
- Rename it to what it is: `E` / `template_energy.mat`, not `L` /
  `TargetContribution.mat`.
- Optional: compute it as column norms of `G'*M` in blocks of columns. That
  takes seconds instead of the ~20 min of one forward run per cell.

**Files.** `simulator.m`, `TBD/supp/target_contribution.m` (becomes a thin
wrapper or is deleted), new `TBD/test/template_energy_test.m`.

**Status (2026-10-11).** Open. Still the one-off script; the filter loads
`TargetContribution.mat` (variable `L`) into `obj.TC`.

### 2. Adjoint correlation map `C`
**Why.** The data-dependent half of the likelihood, for every cell at once.

**What.**
- A method returning the 128 x 128 map for one frame:
  `c = M' * (G * vec(K'(z)))`, where `K` is the blur (skip it if blur is off).
- Use the simulator's row-major ordering: `vec(Z) = reshape(Z.', [], 1)` and
  back with `reshape(c, 128, 128).'`. Mixing this up with `sub2ind`
  (column-major) mirrors the map across the diagonal, and nothing errors.
- The filter needs `M` and `G` (about 1 GB). Share the matrices already loaded
  by `simulator` rather than loading a second copy (item 13).

**Files.** `simulator.m` (method next to the forward model), `TBD_PF.m`
(called once per frame in `timestep`).

**Status (2026-10-11).** Partly done. `C` is computed once per frame, inline
in `timestep`, with the correct row-major vec. Not yet a `simulator` method, and
`run` loads its own copy of `M` and `G` (item 13).

### 4. One consistent image pipeline
**Why.** `E`, `C` and the frame `z` must all come from the same `h(x)`. Today
`simulator` defaults to no blur or normalization, but `environment` overrides
both to `true`.

**What.**
- `environment` defaults `Normalize = false`, `Blur = false`.
- With `'forward'`, error if a scenario's `meta.normalize` is true, or if
  `meta.blur` doesn't match the blur `E` was built with.
- Normalization stays available only for the `'gauss'` baseline.

**Files.** `environment.m`, `TBD.m` / `TBD_PF.m` (the check in `run`).

**Status (2026-10-11).** Defaults done (`environment` now `false`/`false`). The
`meta.normalize` / `meta.blur` check in `run` is still open. Better: read the
noise level from `scenario.meta.noise * scenario.meta.var` instead of the
separate `'Var'` argument.

### 5. Validate the likelihood before tracking
**Why.** Confirm the forward-model likelihood is sharper and less biased than
the Gaussian before reading anything into tracking results.

**What.**
- On clean frames at near, mid and far range, plot `log LR(x, I = 1)` over a
  window of cells around the truth, for `'gauss'` and `'forward'`.
- Report peak location vs truth, and the width of the peak in x and y.
- Then rerun the `env_test.m` scenario with both settings and the same seed.
- Expect `I` near 1: the simulator target is `S = 1` and the template has unit
  intensity.

**Files.** new `TBD/test/likelihood_surface_test.m`, `env_test.m`.

**Status (2026-10-11).** Partly checked by hand, not as a test script: on the
profile log-LR map the truth cell ranks 1st of 15,360 (static frame) and is the
best cell or a neighbour on every `env_test` frame checked; `I_hat` ≈ 1.00 at
the best cell. Neighbouring cells differ by ~500 nats at `Var` = 5, so the
`'model'` peak is about one cell wide. The `R.llr` animation panel shows the
same map per frame. A test script and the `'gauss'` comparison are still open.

### 6. Noise model (placeholder)
**Why.** The likelihood assumes a specific noise model, and you are reworking
the additive noise. This item records what the likelihood needs, not the
design of the rework.

**What the likelihood requires.**
- Noise added in the image domain: zero mean, Gaussian, i.i.d. across pixels.
- `NoiseSTD` is the noise standard deviation in **raw** (unnormalized) image
  units. With the current simulator that is `sqrt(Var)`.
- No per-frame normalization after the noise is added.

**What would break it, and the consequence.**
- Noise with nonzero mean (CLAUDE.md says "always positive") adds an offset
  the model doesn't have.
- Noise injected before reconstruction becomes correlated in the image domain,
  so the likelihood would need a noise covariance.

Either one needs a revision to the likelihood derivation.

**Files.** `simulator.m`, `forward_model_likelihood.tex` (assumptions
section).

---

## P1 — correctness

### 7. Birth proposal: drop `gamma`, sample from the likelihood map, correct the weight
**Why.**
- `sample_new` draws newborns from pixels above `gamma`. On raw forward-model
  frames `gamma` has no meaning: the pedestal alone is 30–6000 depending on
  target range.
- Separately, newborn weights are missing the factor `p_birth(x) / q(x | z)`,
  which over-weights newborns on noise spikes.

**What.**
- For `'forward'`, propose birth cells from a density built from the
  likelihood map, e.g. `q(r,c) ∝ exp(GLRT(r,c))` with `GLRT = C^2/(2 sigma^2 E)`
  for `C > 0`, tempered if it is too peaked. There is no threshold
  (principle 1), and `q` is known exactly.
- Draw a uniform position within the chosen cell, so the proposal is a proper
  density.
- Multiply newborn likelihoods by `p_birth / q`. `I` and `v` drawn from their
  priors contribute no extra factor.
- Unverified: that the GLRT map peaks at the target given the pedestal
  dominance. Check it in item 5 first.
- Keep the `gamma` proposal for `'gauss'`, adding the same weight correction
  (`N_above / npx^2` for pixel births).

**Files.** `TBD_PF.m` (`sample_new`, `timestep`).

**Status (2026-10-11).** Partly done; now **P0** (see "acquisition and
exploration").
- `gamma` is a fraction now: `birth_cells` takes the brightest `(1-gamma)` of
  the raw pixels with y ≥ 0.25 m (`maxk`, never empty), and `sample_new` picks
  one uniformly. Still raw-pixel based, so on raw frames it mostly selects
  noise or pedestal pixels.
- The correction `logBirthCorr = log(nKeep/nValid)` is computed but **not
  applied** to newborn `logL`.
- The "unverified" point above is resolved: the profile log-LR map peaks at
  the target (truth rank 1 of 15,360). Ranking that map instead of `z` is a
  one-line change in `birth_cells`, once `obj.C` is set.
- Principle 1: a top-fraction cut in the birth *proposal* is not a threshold
  in the measurement update, as long as the `p_b/q` correction is applied.

### 8. One valid region for births, likelihood and tables
**Why.**
- `importance_weights` zeroes particles with `py < 0.25`, but births can land
  there.
- The forward model is unreliable near the array: at y = 0.3 m the image peak
  is 7 rows from the target cell.

**What.** One property (e.g. `y_min_valid = 0.25`, i.e. rows 1–8 excluded)
used by the birth proposal, `importance_weights`, and as a mask on `C` / `E`.

**Files.** `TBD.m` (property, parser), `TBD_PF.m` (`sample_new`,
`importance_weights`).

**Status (2026-10-11).** Births now respect it (`birth_cells` excludes
y < 0.25 m), but `0.25` is hard-coded in three places: `importance_weights`,
`birth_cells`, `llr_map`. Still one property to add.

### 10. Collapse the duplicated simulator paths
**Why.** `generate_mWidar_image_meters` and `_pixels` duplicate the forward
model, SNR and noise code. Past fixes landed in one path and not the other.

**What.** The meters path converts to pixel indices (item 9) and calls the
pixels path, or both call one `forward_model(S)` helper. The adjoint (item 2)
belongs next to that helper.

**Files.** `simulator.m`.

### 11. Process noise, intensity bounds, log-domain weights
**Why.**
- `Q` is added raw each step: no `dt` scaling, no position–velocity coupling.
  It spreads a static target's cloud every step.
- `dynamics` clamps `I` to `[I_min, I_max]`, which piles posterior mass on the
  bounds.
- With `E` up to 2.5e13, `exp(log LR)` overflows. Today `normalize` patches
  that with `realmax`.

**What.**
- Build `Q` from acceleration and intensity noise. Keep `'Q'` as an override.
  ```
  Qcv = q_a * [dt^3/3  dt^2/2;
               dt^2/2  dt    ];
  Q   = blkdiag(Qcv, Qcv, q_I * dt);   % [px vx py vy I]
  ```
- Reflect `I` at the bounds instead of clamping. For `'forward'`, center
  `[I_min, I_max]` on 1.
- Carry log weights throughout and normalize with log-sum-exp. Remove the
  `realmax` clamp.

**Files.** `TBD.m` (parser, `Q`, `I` defaults), `TBD_PF.m` (`dynamics`,
`timestep`, `normalize`).

**Status (2026-10-11).** Partly done. Within a step the weights are log-domain
(`log(w_prior) + logL` → `normalize_log`, log-sum-exp). Open:
- Weights are still carried between frames in the linear domain, so an
  underflowed weight is `-Inf` forever (e.g. the ~1,000 zero-weight live
  particles that pile up before a target appears). Carry `logw` through `run`.
- `Q` construction and the `I` clamp are unchanged. With fitted `I` the clamp
  only matters for the `'gauss'` baseline.
- The old linear `normalize` (with `realmax`) is still there, unused by
  `timestep`, and still declared abstract in `TBD.m`.

---

## P2 — performance and robustness

### 12. Vectorize the per-particle loop
**Why.** With `'forward'` the likelihood itself is two lookups, so the
per-particle `for` loop in `timestep` (`sample_new`, `dynamics`,
`importance_weights` one particle at a time) and the loop in `RT` become the
cost. `'gauss'` still makes `N*(2*pDist+1)^2` scalar calls per frame.

**What.**
- Vectorize `RT` (`pOn = Ps.*Eprev + Pb.*(1-Eprev)`), `dynamics`, births, and
  the lookups (`sub2ind` on the `(r, c)` vectors).
- For `'gauss'`, compute the windowed log LR with array ops over precomputed
  window offsets.

**Files.** `TBD.m` (`RT`), `TBD_PF.m`.

### 13. Load `M` and `G` once
**Why.** The matrices are ~512 MB each. `simulator` loads them, and the
`'forward'` filter needs them too.

**What.** Pass the filter a handle to an already-constructed `simulator` (or
the matrices) instead of loading again. Let `environment` hand over its `sim`.

**Files.** `TBD.m` (parser), `environment.m`.

### 14. Impoverishment diagnostics
**Why.** Resampling frequency alone doesn't say whether the filter is healthy.
The question is how many distinct particles survive.

**What.** Have `resample` return ancestor indices. Record `R.nUnique(k)` and
plot it on the ESS row.

**Files.** `TBD_PF.m` (`resample`, `timestep`, `run`), `visualize.m`
(`ess_axes`, `plot_TBD`), `visualize.md`.

**Status (2026-10-11).** Partly done in the plot only: the dashboard's
"Occupied cells" tile counts distinct cells holding E = 1 particles with
nonzero weight (1 to ~10 while tracking). Ancestor indices are still not
returned; item 25 needs them.

### 15. Post-resample move step
**Why.** With small process noise, resampled duplicates stay stacked.
Diversity in `I` in particular never recovers.

**What.** Optional `'Jitter'`: after a resample, perturb with a small kernel
scaled to the weighted posterior covariance, or do one Metropolis–Hastings
move per particle. Only worth it if item 14 shows collapse. Note that with the
nearest-cell likelihood, jitter within a cell doesn't change the weight.

**Files.** `TBD_PF.m` (`run` options, `timestep`, `resample`).

**Status (2026-10-11).** Collapse is confirmed (ESS/N ≈ 0.02 at `Var` = 0.1),
but plain jitter is local and can't fix a wrong lock. The MCMC variant is now
item 25 with a likelihood-map jump proposal.

---

## P3 — simulation, testing, cleanup

### 16. Specify scenarios by SNR
**Why.** What `Var` means depends on range. Peak amplitude falls about 200×
from near to far.

**What.**
- Pick an SNR definition consistent with the likelihood. Candidates: the
  current peak-above-median, or matched-filter SNR `I^2 * E(x) / sigma^2`,
  which is what sets detectability for this likelihood.
- Add an `environment` option `'SNR'` (dB) that sets `Var` from it, per frame
  or per scenario (record which).
- Depends on item 6.

**Files.** `environment.m`, `simulator.m` (`get_SNR`).

**Status (2026-10-11).** Definition chosen and implemented: `get_SNR` is
matched-filter SNR, `10*log10(sum(h.^2)/Var)`. It counts the pedestal, so it is
very high (58–74 dB at `Var` = 15; ~80–96 dB at `Var` = 0.1) and not a
measure of how visible the target is per pixel. After item 19 the projected
energy `E~ = E - a.^2` is the better basis. The `'SNR'` option is still open.

### 17. Fill in `TBD_PF_test.m`
**Why.** It's empty, and several past bugs would have been caught by a couple
of numbers.

**What.** Fixed-seed checks, small `N`:
- **Adjoint:** `C(r,c)` equals `<h(r,c), z>` at a few off-diagonal cells.
- **`E` table:** off-diagonal spot values, which catch transposition.
- **Target present, moderate SNR:** declared fraction above a floor, for both
  `'gauss'` and `'forward'`.
- **No target ever:** no declarations.
- **Static target, `Var = 0`:** ESS stays high with matched `Q`.

**Files.** `TBD/test/TBD_PF_test.m`.

### 18. Naming and leftovers
- **Filter `Sigma` vs simulator `Sigma`:** they mean different things. Rename
  the filter's to `PSFSigma`; it is only used by `'gauss'`.
- **`'STD'` vs `NoiseSTD`:** the constructor argument and the property should
  share one name.
- **Typos:** `guass_likelihood` → `gauss_likelihood`; "varaince" in
  `simulator.m`.
- **`Ip` in `TBD.m`:** unused. Remove it.
- **`particle_estimate` in `visualize.m`:** reads the last row as `E` whenever
  there are ≥ 5 rows, so a 5-row set without `E` would be misread.
- **`target_contribution.m`:** the header still describes the old window
  product, and `p_window` and `var(L(:))` are leftovers. Superseded by item 1.
- **Gaussian-only tools:** `fit_gauss.m` and anisotropic `sigma_x/sigma_y` only
  matter for the `'gauss'` baseline. Calibrate `PSFSigma` with `fit_gauss.m`
  before the comparison in item 5, so the baseline is fair.
- **`visualize.md`:** paths say `TDB/`, and the particle layout is the old
  5-row one.
- **Empty folders:** `test/config/` and `test/TBD/test/figs/`.
- **`debug_print` and `%`:** both `TBD.debug_print` and `simulator.debug_print`
  pass the message to `fprintf` as its format string. Any `%` in the text (e.g.
  "top 30% of valid cells") truncates the line and drops the newline. Use
  `fprintf('%s\n', str)`.
- **Duplicated formula:** `llr_map` (plotting) repeats `model_likelihood`'s
  profile log-LR expression. Change both together, or have `model_likelihood`
  index the map.

**Files.** `TBD.m`, `TBD_PF.m`, `simulator.m`, `visualize.m`,
`TBD/supp/`, `TBD/Documentation/visualize.md`.

---

## Later — experiments

### 19. Common-mode (pedestal) projection
**Why.**
- 99.999% of every template's energy is one shared pattern: the pedestal plus
  the array dips, scaled by range.
- Without projection, the likelihood mostly measures `I * a(x)`, which mixes
  intensity and range. It is also very sensitive to any pedestal mismatch.
- For two targets, the cross term `<h(x1), h(x2)>` is large however far apart
  they are.

**What.** Project the shared pattern `u` out of data and templates:
`C~ = C - (u'z)*a`, `E~ = E - a.^2`, with `a(x) = u'h(x)` as a third table.
Compare with and without projection, on likelihood surfaces (item 5) and on
tracking.

**Files.** `simulator.m` (`u`, `a`), `TBD_PF.m`.

**Status (2026-10-11).** Promoted to **P0**: it is the fix for pE locking to 1
on a wrong cell (see "acquisition and exploration"). Check first that, after
projection, a wrong-place newborn's log LR is near zero or negative and the
truth cell still ranks first.

### 20. Sub-cell positions
**Why.** The forward model only defines `h` on cells, so the likelihood is
constant within a cell.

**What.** Bilinear interpolation of `C` and `E`. It is an approximation; only
worth it if position error is floored at about half a cell.

### 22. Toward multiple targets
**What.**
- With superposition, two targets add a cross term
  `-I1*I2*<h(x1), h(x2)>/sigma^2`.
- Computing `<h(x1), h(x2)>` needs two forward runs, or a precomputed `B'B`
  restricted to nearby cells.
- This is the superpositional-sensor problem (Davies & García-Fernández).
  Scope it after the single-target forward-model filter works.

---

## Suggested order (revised 2026-10-11)

1. **Acquisition and exploration:** item 19 (pedestal projection), then item 7
   (likelihood-map birth proposal, apply `logBirthCorr`), then item 23 (birth
   velocity range). Re-run the low-`Var` cases (`Var` = 0.1, S-curve and
   static) after each.
2. **Lock in a baseline:** item 17 (tests), item 5 as a script, and carry
   `logw` across frames (item 11).
3. **Recovery mid-track,** only if 1 isn't enough: item 24 or 25.
4. **Scope decision:** item 27 (grid filter) vs. continuing with the PF.
5. Items 1, 8, 10, 13, 18: library and cleanup work, as convenient.
6. Item 6 when the noise rework starts, then the rest of item 16.
7. Items 20, 22 once single-target `'model'` tracking is robust.
