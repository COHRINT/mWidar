# `visualize` — plotting for TBD / mWidar

`TDB/Includes/visualize.m`. Subclass of `mWidar`, so it carries the scene geometry
(`Lscene = 4 m`, `npx = 128`, `xgrid` over `[-2, 2]`, `ygrid` over `[0, 4]`).

All methods are **stateless**: the object holds scene + styling only, data is always
passed in. Every method returns a figure handle and accepts `'Save', <path>`.

```matlab
addpath("TDB/Includes")
v = visualize();                 % defaults: meters, gray colormap
fig = v.signal_frame(sig, 'Truth', [x; 0; y; 0]);
```

---

## Conventions

| Thing | Shape / meaning |
|---|---|
| Signal image | `npx x npx`, indexed `signal(row, col) = signal(y, x)` — same as `simulator.m`'s `S(Gy,Gx)`. Drawn with `axis xy`. |
| Signal stack | `npx x npx x K` |
| State vector | `[px; vx; py; vy]` — position rows set by the `StateIdx` property (default `[1 3]`). A 2-row state is read directly as `[px; py]`. |
| Track | `4 x K`, `4 x K x T` (T targets), or `1 x T` cell of `4 x K`. `NaN` means "not present at this k". |
| Particle history | `5 x N x K` (`[px;vx;py;vy;E]`), `5 x N x K x T`, or a cell of `5 x N x K` |
| Weights | `N x K`, normalized, and paired with the particle set they
  were stored beside. With an ESS-triggered resampler that column is flat
  `1/N` on the frames that resampled and non-uniform on the ones that
  didn't, so anything read off the cloud has to be weighted. |
| ESS | `1 x K`, absolute (`1..N`), measured **before** each resample decision |
| Existence prob | `T x K` (a `1 x K` vector is fine for one target) |


### Units

Two independent knobs on every method:

- `'Units'` — what the **axes** show, `'meters'` (default) or `'pixels'`.
- `'DataUnits'` — what the **data you passed** is already in. Defaults to `'Units'`,
  so nothing is converted unless you say so.

The filter typically works in pixels while the scene is defined in meters:

```matlab
v.trajectories(est, 'Truth', truth, 'DataUnits','pixels', 'Units','meters')
```

Only positions are converted. Velocity rows are passed through untouched.

---

## Constructor

```matlab
v = visualize('Units','pixels', 'Debug', true);
```

| Option | Default | Meaning |
|---|---|---|
| `'Debug'` | `false` | Print `[DEBUG][VISUALIZE]` messages |
| `'Units'` | `'meters'` | Default axis units for every plot |
| `'StateIdx'` | `[1 3]` | Rows of the state vector holding `[x; y]` |
| `'Colormap'` | `'gray'` | Colormap for signal images |
| `'FontSize'` | `11` | |
| `'LineWidth'` | `1.5` | |
| `'MarkerSize'` | `6` | |
| `'DPI'` | `200` | Resolution used by `exportgraphics` |
| `'FPS'` | `15` | Default animation frame rate |

### Options accepted by every method

`'Units'`, `'DataUnits'`, `'Axes'` (draw into existing axes instead of a new figure),
`'Title'`, `'Save'` (path; no extension → `.png`), `'Colormap'`.

---

## Scene / signal plots

### `trajectories(tracks, ...)`
Tracks in the XY plane. `tracks` may be `[]` if you only want truth.

| Option | Meaning |
|---|---|
| `'Truth'` | Ground truth tracks, drawn black |
| `'Background'` | An `npx x npx` image, or a full `npx x npx x K` stack — a max-intensity projection over `k` is drawn, so the track lies on the accumulated energy |
| `'ColorByTime'` | Scatter the estimate colored by frame index |
| `'Mask'` | `1 x K` or `T x K` logical; only plot where true (e.g. `pE > thresh`) |
| `'Labels'` | Cellstr of per-target legend names |

```matlab
v.trajectories(est, 'Truth', truth, 'Background', signals, 'Mask', pE > 0.5)
```

### `signal_frame(signal, ...)`
A single mWidar frame, optionally with overlays.

| Option | Meaning |
|---|---|
| `'Truth'`, `'Est'` | `4 x 1` or `4 x T` states to mark |
| `'Particles'` | `5 x N` particle set for that frame |
| `'Weights'` | `1 x N`; shades and sizes the particles |
| `'CLim'` | Color limits, `[]` = auto |
| `'Colorbar'` | Default `true` |

### `signal_montage(signals, ...)`
Grid of frames from a stack — the whole run at a glance, no animation.

| Option | Meaning |
|---|---|
| `'Frames'` | Explicit frame indices (overrides `'Count'`) |
| `'Count'` | Number of evenly spaced frames, default `9` |
| `'Truth'`, `'Est'` | Tracks; marked on the frame they belong to |
| `'CLim'` | Shared color limits |

### `compare_signals(A, B, ...)`
Two frames side by side plus their difference, titled with RMSE and correlation.
For validating the forward model against measured data, or one setting against another.

| Option | Meaning |
|---|---|
| `'Names'` | `1 x 2` cellstr of panel titles, default `{'A','B'}` |

### `animate_time_history(signals, ...)`
Signal behind, targets moving through it.

| Option | Meaning |
|---|---|
| `'Truth'`, `'Est'` | Tracks |
| `'Particles'`, `'Weights'` | Particle cloud per frame. `'Weights'` shades and sizes the particles and adds a weight colorbar beside the scene |
| `'WeightColormap'` | Colormap for that shading, default `'gray'`. `''` draws the flat particle color instead |
| `'WeightScale'` | `'log'` (default) or `'linear'`. The likelihood ratio is a product over a pixel window, so on a linear scale one particle is white and the other `N-1` are black |
| `'WeightCLim'` | Fix the weight color limits, in the scaled domain (`log10` units under `'log'`). Default: computed once over the whole run, floored at the 2nd percentile and at most 6 decades wide |
| `'pE'` | `T x K`, shown in the title |
| `'Trail'` | Past samples kept drawn, default `15`, `Inf` for the full track |
| `'FPS'` | Playback / export rate |
| `'Format'` | `'gif'`, `'mp4'`, `'png'` (frame dump). Inferred from the `'Save'` extension |
| `'CLim'` | Fixed color limits so brightness doesn't flicker (defaults to the stack min/max) |
| `'Pause'` | Extra pause per frame during live playback |

```matlab
v.animate_time_history(signals, 'Truth', truth, 'Particles', Y, 'pE', pE, ...
                       'Save', 'figs/run.gif', 'FPS', 20)
```

---

## TBD plots

### `plot_TBD(res, ...)`
The results dashboard:

```
[ track over energy map | existence P(E) ]
[         p_x           |      p_y       ]
[    position error     | existing parts ]
[         ESS / N       (full width)     ]   <- when res.ess or res.weights is set
[      measurement SNR  (full width)     ]   <- when res.snr is set
```

`res` is a struct — see `visualize.results_template()`. **Every field is optional**;
anything missing is skipped, so partial results still plot. If `est` / `pE` are
absent but `particles` are present, the MMSE estimate and existence probability
are derived from the particles (weighted by `weights` if given).

```matlab
res = visualize.results_template();
res.signals   = signals;    % npx x npx x K
res.truth     = truth;      % 4 x K x T
res.Etruth    = Etruth;     % T x K
res.est       = est;        % 4 x K x T      (optional)
res.pE        = pE;         % T x K          (optional)
res.particles = Y;          % 5 x N x K      (optional)
res.weights   = W;          % N x K          (optional)
res.ess       = ess;        % 1 x K, absolute (optional)
res.resampled = didResample;% 1 x K logical   (optional)
res.t         = [];         % 1 x K seconds; [] -> x axis is k
res.pEthresh  = 0.5;

v.plot_TBD(res, 'Title', 'Run 3', 'Save', 'figs/tbd.png')
```

`'pEthresh'` overrides `res.pEthresh`; the track panel shows declared frames only.
`'ESSThresh'` is the `ESS/N` the filter resamples at, drawn as a line on the ESS row.

### `existence(pE, ...)`
`P(E_k)` vs truth with the declaration threshold. Split out of `plot_TBD` for
tuning `Pb` / `Ps` / the threshold.

| Option | Meaning |
|---|---|
| `'Etruth'` | `T x K` or `1 x K` true existence flags |
| `'pEthresh'` | Default `0.5` |
| `'Time'` | `1 x K`, defaults to `1:K` |

### `ess_vs_time(ess, ...)`
Effective sample size vs time, with the resampling threshold and the frames that
actually resampled. The plot that says whether an ESS-triggered resampler is
earning its keep: a curve that never approaches the threshold means the weights
are healthy and resampling every step was throwing information away; a curve
pinned to the floor means the likelihood is too peaked for `N` particles.

| Option | Meaning |
|---|---|
| `'N'` | Particle count, used to turn an absolute ESS into a ratio. Required unless `ess` is already in `[0, 1]` |
| `'Threshold'` | `ESS/N` the filter resamples at |
| `'Resampled'` | `1 x K` logical, marks the frames that resampled |
| `'Time'` | `1 x K`, defaults to `1:K` |

```matlab
R = pf.run(scenario);                      % adaptive resampling
v.ess_vs_time(R.ess, 'N', R.N, 'Threshold', R.essThresh, 'Resampled', R.resampled)
```

### `cardinality(estCard, ...)`
Estimated vs true target count. `estCard` may be a `1 x K` count, or a `T x K`
matrix of existence probabilities (summed to give expected cardinality).

| Option | Meaning |
|---|---|
| `'Truth'` | A `1 x K` count, or truth tracks (non-NaN targets are counted per `k`) |
| `'Time'` | |

### `particle_cloud(Yk, ...)`
One frame, three panels: positions colored by weight, the velocity cloud, and the
weight histogram (with ESS/N). The plot to reach for when the filter loses a target.

| Option | Meaning |
|---|---|
| `'Weights'` | `1 x N` |
| `'Signal'` | Frame to draw underneath |
| `'Truth'`, `'Est'` | States for that frame |
| `'ShowDead'` | Also draw the `E = 0` particles in grey |

### `particle_density(Yk, ...)`
Weighted 2-D histogram of the particles next to the measurement. Shows whether the
cloud sits on the energy, and exposes multi-modality a mean estimate hides.

| Option | Meaning |
|---|---|
| `'Weights'` | `1 x N` |
| `'Signal'` | Measurement to compare against |
| `'Bins'` | Grid coarsening factor, default `2` → 64×64 bins |
| `'Truth'` | |

---

## Performance / validation

### `[fig, err] = position_error(truth, est, ...)`
Per-axis and total position error vs time, with the RMSE line. `err` comes back
`T x K` so you can stack it across Monte Carlo runs and feed `rmse_mc`.

| Option | Meaning |
|---|---|
| `'Mask'` | Blank the error where the track isn't declared |
| `'Time'` | |

### `rmse_mc(errStack, ...)`
Monte Carlo RMSE with a spread band.

`errStack` is `K x M` (K frames, M runs) or a `1 x M` cell of `1 x K` traces.

| Option | Meaning |
|---|---|
| `'Band'` | `'quantile'` (default, IQR), `'std'`, or `'none'` |
| `'Label'` | Legend entry |
| `'Color'` | Line/band color |
| `'Time'` | |

Overlay several configurations by passing the same `'Axes'`:

```matlab
[~, e1] = v.position_error(truth, estA);
f = v.rmse_mc(e1', 'Label', 'N = 1000');
v.rmse_mc(e2', 'Axes', gca, 'Label', 'N = 5000', 'Color', [0 0.45 0.85]);
```

### `[fig, d] = ospa(truth, est, ...)`
OSPA distance vs time, split into localization and cardinality components. The
right metric once births/deaths and multiple targets are in play, because it
penalizes both position error and the wrong number of tracks on one scale.
`d` is a struct with `.total`, `.loc`, `.card`.

| Option | Meaning |
|---|---|
| `'c'` | Cutoff in plotting units; default `Lscene/4` (m) or `npx/4` (px) |
| `'p'` | Order, default `2` |
| `'Mask'` | Declaration mask applied to the estimates |
| `'Time'` | |

Uses `matchpairs` for the optimal assignment, with a greedy fallback if unavailable.

### `pf_diagnostics(W, ...)`
Filter health over the run: ESS/N and max normalized weight vs `k`, plus weight
histograms at selected frames. Flat ESS or a max weight near 1 means the cloud has
collapsed and the track output is luck.

| Option | Meaning |
|---|---|
| `'Particles'` | `5 x N x K`; adds a unique-particle count to the debug log |
| `'Frames'` | Frames to histogram, default 3 evenly spaced |
| `'ESS'` | `1 x K` pre-resample ESS. `W` is the *posterior* weight, so on a resampled frame its ESS reads `N`; pass `R.ess` to plot what actually drove the decision |
| `'Time'` | |

---

## Static

### `visualize.results_template()`
Empty `res` struct documenting what `plot_TBD` accepts. Fill in what the run
produced, leave the rest empty.

---

## Notes

- Requires no toolboxes beyond base MATLAB — percentiles and the OSPA assignment
  fall back to hand-rolled implementations if the usual functions aren't there.
- Running headless (`matlab -batch`) prints
  `String scalar or character vector must have valid interpreter syntax` warnings
  for any TeX subscript in a label (`p_x`). That's the batch renderer missing the
  math font; the labels render fine on screen. Suppress with `warning('off','all')`
  if you're batch-generating figures.
- `'Save'` creates the parent directory if it doesn't exist.
