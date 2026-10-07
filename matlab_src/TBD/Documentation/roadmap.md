# TBD roadmap — recommended changes

A working list of fixes and features for the TBD particle filter, from the
ESS / existence / SNR troubleshooting sessions. Grouped by priority. Each item
says **why**, **what**, and **which files** it touches.

File shorthand: everything lives in `TBD/Includes/` unless a path is given.

---

## Where things stand

Baseline used throughout: `env_test.m` scenario (parabola, `Var = 5`,
seed `123456`), `N = 1000`, `pDist = 3`, `gamma = 0.85`, Gaussian noise.

| Background | STD | median pE, target alive | median pE, no target | declared alive | declared dead |
|---|---|---|---|---|---|
| `median` (preprocess on) | 2    | 0.30 | 0.29 | 0/81  | 0/20  |
| `median`                 | 0.5  | 0.34 | 0.16 | 36/81 | 0/20  |
| `median`                 | 0.25 | 1.00 | 0.85 | 81/81 | 20/20 |
| `none` (preprocess off)  | 0.5  | 1.00 | 0.99 | 81/81 | 19/20 |

- Reported scenario SNR at `Var = 5` is **~2 dB** (peak above median), and
  that matches the contrast measured on the preprocessed frames.
- A well-calibrated `STD` (≈ the measured noise, 0.25) declares everything,
  including empty frames. Items 1 and 3 below are the reasons.
- Resampling frequency was driven by cloud spread (velocity process noise,
  wide birth velocities) times likelihood steepness, not by pixel noise.

---

## P0 — correctness

### 1. Birth proposal weight correction
**Why.** `sample_new` draws newborns from pixels above `gamma` (a
data-driven proposal), but the weight is only the likelihood ratio. The
missing factor `p_birth(x) / q(x | z)` is about `N_above / N_pixels`
(≈ 5 / 16384 ≈ −8 nats). Without it, a newborn on a noise spike is
over-weighted, which drives pE up on empty frames. This is the main
blocker to using a calibrated `NoiseSTD`.

**What.**
- `sample_new` returns the proposal density it used (or the ratio
  directly): `N_above / npx^2` for a pixel-proposal birth, `1` for the
  uniform fallback.
- `timestep` multiplies newborn `L(n)` by that ratio. `I` and `v` are drawn
  from their priors, so they contribute no extra factor.
- Pixel-proposal births should also pick a uniform position *within* the
  pixel, not the pixel center, so the proposal is a proper density.

**Files.** `TBD_PF.m` (`sample_new`, `timestep`).

### 2. Births and the likelihood disagree on the valid region
**Why.** `importance_weights` hard-zeroes any particle with `py < 0.25`,
but `sample_new` can place births there (above-gamma clutter near the
array is common), and the uniform fallback draws from `[min_y, max_y]`.
Those births are dead on arrival.

**What.** One property for the valid region (e.g. `y_min_valid = 0.25`)
used by both: mask pixels below it out of the birth proposal, and draw the
uniform fallback from the valid region only.

**Files.** `TBD.m` (property + parser), `TBD_PF.m` (`sample_new`,
`importance_weights`).

### 3. Noise-scale (MAD) preprocessing
**Why.** `preprocess` rescales every frame so its peak is 1. On a
target-free frame the biggest noise spike becomes an `I = 1` target, so
empty and occupied frames look alike (noise level 0.25 vs 0.27 after
preprocessing). Peak scaling throws away exactly the contrast existence
detection needs.

**What.** Add `Background = 'mad'`:
`z = (z - median(z)) / (1.4826 * mad(z(:), 1))`.
- Invariant to any per-frame affine map, so it undoes the simulator's
  min-max normalization automatically.
- `NoiseSTD` becomes ≈ 1 by construction.
- `I` becomes "target SNR" (peak / sigma); `I_min`, `I_max` and `gamma`
  need rescaling to that unit (e.g. `gamma ≈ 3`).
- Needs a floor on the MAD: with `Var = 0` a blank frame has MAD = 0.

**Files.** `TBD_PF.m` (`preprocess`), `TBD.m` (`Background` validator,
defaults for `STD`, `I_min`, `I_max`, `gamma`), `env_test.m`.

### 4. Truth-to-pixel mapping: ceil vs round
**Why.** `generate_mWidar_image_meters` places the target with
`find(px <= obj.xgrid, 1, 'first')` (rounds up). The filter maps meters to
pixels with `round`. Up to a 1 px (mean ½ px) bias between truth and
image, which shows up as a position error floor.

**What.** Use `round((px - xgrid(1)) / dx) + 1` in the simulator, same as
the filter.

**Files.** `simulator.m` (`generate_mWidar_image_meters`).

### 5. Collapse the duplicated simulator paths
**Why.** `generate_mWidar_image_meters` and `_pixels` duplicate the
forward model, SNR and noise code. The half-Gaussian noise bug lived on in
the meters path (the one `environment` actually uses) after it was fixed
in the pixels path.

**What.** Meters path converts to pixel indices (item 4) and calls the
pixels path, or both call a shared `forward_model(S)` helper.

**Files.** `simulator.m`.

---

## P1 — model calibration

### 6. Measure `Sigma` from the simulator instead of guessing
**Why.** Default `Sigma = 0.5 px` is far narrower than the real blob (the
simulator blur alone is 2 px, on top of a broad reconstruction). Symptoms:
`I` piles up at `I_max`, each particle effectively reads one pixel.

**What.**
- Run `TBD/supp/fit_gauss.m` at several ranges (near / mid / far from the
  array) and record `sigma_x`, `sigma_y`.
- Check the fitted center lands on the true pixel (peak bias).
- Set `pDist ≈ ceil(3 * Sigma)`; anything wider is wasted compute.
- Once `Sigma` matches, `I` should settle near 1 (or near the true SNR
  after item 3). Then drop `I_max` back down.

**Files.** `TBD/supp/fit_gauss.m` (loop over positions), `TBD.m` defaults,
`env_test.m`.

### 7. Anisotropic PSF (`sigma_x`, `sigma_y`)
**Why.** mWidar blobs are elongated; the `TODO` in `TBD.m` already notes
it. One isotropic `Sigma` is a compromise whose error ends up in
`NoiseSTD`.

**What.** Replace `Sigma` with `SigmaX`, `SigmaY` (keep `Sigma` as a
shortcut that sets both). In the likelihood use
`exp(-dx^2/(2 sx^2) - dy^2/(2 sy^2))`. If the fit shows width varies with
range, make them a function of `py`.

**Files.** `TBD.m` (properties, parser), `TBD_PF.m` (`guass_likelihood`,
`importance_weights` window size), `fit_gauss.m`.

### 8. Calibrate `NoiseSTD` from data
**Why.** `NoiseSTD` is the per-pixel residual after preprocessing:
sensor noise + sidelobes / clutter + PSF mismatch. It should be measured,
not tuned to make ESS look good.

**What.** A supp script that, for a scenario with known truth, computes
`z - I_hat * g` around the target and `1.4826 * MAD` over the background.
Report both; the larger is the floor for `NoiseSTD`. With item 3 this
becomes a sanity check (should be ≈ 1).

**Files.** new `TBD/supp/fit_noise.m`.

### 9. Proper constant-velocity process noise
**Why.** `Q` is added directly each step: no `dt` scaling, no
position–velocity coupling, and the velocity random walk is undamped. For
a static target this spreads the cloud every step, which was the main
cause of frequent resampling.

**What.** Parameterize by acceleration and intensity noise instead of a
raw 5×5:
```
Qcv = q_a * [dt^3/3  dt^2/2;
             dt^2/2  dt    ];
Q   = blkdiag(Qcv, Qcv, q_I * dt);   % [px vx py vy I]
```
Keep `'Q'` as an override for experiments. Pick `q_a` from expected
accelerations in the scenarios; keep `v_min`/`v_max` near scenario speeds.

**Files.** `TBD.m` (parser, build `Q`), `env_test.m`.

### 10. Intensity bound handling
**Why.** `dynamics` clamps `I` to `[I_min, I_max]`, so every
out-of-range proposal lands exactly on the bound and creates a spike in
the posterior.

**What.** Reflect at the bounds (`I = 2*I_max - I` above, likewise
below), or give out-of-range particles zero weight. Revisit `I_max = 3`
once `Sigma` is matched.

**Files.** `TBD_PF.m` (`dynamics`).

---

## P2 — performance and robustness

### 11. Vectorize the likelihood and work in log domain
**Why.** `importance_weights` calls `guass_likelihood` once per pixel per
particle: `N * (2*pDist+1)^2` scalar calls per frame (≈ 2.2M at
`N = 5000`, `pDist = 10`). The product of exponentials also overflows,
which `normalize` patches with `realmax`.

**What.**
- Precompute the window offsets once per frame. For all alive particles
  at once, compute
  `logL = sum((2*h.*z - h.^2) / (2*NoiseSTD^2))` over the window with
  array ops.
- Carry log-weights; normalize with log-sum-exp
  (`w = exp(lw - max(lw)); w = w / sum(w)`).
- Removes the overflow clamp and should make `N = 5000` cheap enough for
  parameter sweeps.

**Files.** `TBD_PF.m` (`importance_weights`, `guass_likelihood`,
`timestep`, `normalize`).

### 12. Impoverishment diagnostics
**Why.** Resampling frequency alone doesn't say whether the filter is
healthy. Once `Sigma` is matched it *should* resample often; the real
question is how many distinct particles survive.

**What.** Have `resample` return the ancestor indices; record
`R.nUnique(k)` (distinct ancestors after resample) and plot it on the ESS
row.

**Files.** `TBD_PF.m` (`resample`, `timestep`, `run`), `visualize.m`
(`ess_axes`, `plot_TBD`), `TBD/Documentation/visualize.md`.

### 13. Post-resample move step
**Why.** With small process noise, resampled duplicates stay stacked.
Diversity in `I` in particular never recovers (random-walk std 0.01).

**What.** Optional `'Jitter'` option: after a resample, perturb with a
small kernel scaled to the weighted posterior covariance (regularized
PF), or one Metropolis–Hastings move per particle against the current
frame (resample-move). Only worth it if item 12 shows collapse.

**Files.** `TBD_PF.m` (`run` options, `timestep`, `resample`).

---

## P3 — simulation, testing, cleanup

### 14. Specify scenarios by SNR, not `Var`
**Why.** What `Var` means depends on the reconstruction's amplitude at
that position (`Var = 5` turned out to be ~2 dB). Experiments are easier to
compare across trajectories by SNR.

**What.** `environment` option `'SNR'` (dB): compute the noiseless peak
above median for the scenario, set `var = (peak / 10^(SNR/20))^2`.
Either per frame or one value per scenario (mean peak); record which.

**Files.** `environment.m` (parser, `simulate`), `simulator.m` (let `var`
be set per call).

### 15. Fill in `TBD_PF_test.m`
**Why.** It's empty, and most of the bugs above would have been caught by
a couple of numbers.

**What.** Fixed-seed regression checks, small `N`:
- Target present, moderate SNR: declared fraction above a floor.
- No target ever (`Var > 0`): no declarations, pE stays below threshold.
- `Var = 0`, static target: ESS stays high with matched `Q`.
- `get_SNR` matches measured contrast within ~1 dB.

Also promote the pE comparison sweep (Background × STD table above) to
`TBD/test/tune_pE.m` so the baseline table can be regenerated.

**Files.** `TBD/test/TBD_PF_test.m`, new `TBD/test/tune_pE.m`.

### 16. Naming and leftovers
- Filter `Sigma` vs simulator `Sigma` mean different things; rename the
  filter's to `PSFSigma` (or `SigmaX/SigmaY`, item 7).
- Constructor arg `'STD'` vs property `NoiseSTD`; pick one name.
- `guass_likelihood` → `gauss_likelihood`; "varaince" typos in
  `simulator.m`.
- `Ip` in `TBD.m` is stored but unused ("Do more research on this term").
  Remove, or define it as the expected target intensity used as the birth
  prior mean for `I`.
- `particle_estimate` in `visualize.m` treats the last row as `E`
  whenever there are ≥ 5 rows. A 5-row `[px vx py vy I]` set (no `E`)
  would read `I` as existence.

**Files.** `TBD.m`, `TBD_PF.m`, `simulator.m`, `visualize.m`.

---

## Suggested order

1. Item 11 first if runs are slow: everything else needs sweeps.
2. Items 4, 5: cheap, remove a bias and the duplicate code.
3. Item 6 (measure `Sigma`), then 9 (process noise).
4. Items 1, 2, 3 together: they're what makes a calibrated `NoiseSTD`
   (item 8) usable without false tracks.
5. Item 15 to lock in the baseline, then 12/13 if impoverishment shows up.
6. Item 14 and 16 whenever.
