# TBD_PF Test Framework

A modular config → test pipeline. Parameter sweeps, Monte Carlo repeats, scenario sets and multiple filters all compose.

```matlab
out = run_test('T5');                                % unique prefix of config/T5_*.m
out = run_test(1);                                   % or a number
out = run_test('T1', 'perTrial', false, 'save', false);  % override any t.output field
run TBD_PF_test                                      % interactive: list tests and prompt
```

`run_test` can be called from any working directory. It switches to `matlab_src` for the run, because the filter loads `supplemental/*.mat` by relative path, and switches back afterwards.

## Core idea

1. **Test definition** (`config/T*.m`): declarative, says *what varies*. Lists overrides plus optional axes. Never lists every parameter.
2. **Trial**: one fully resolved `cfg`. `expand_trials` builds them:

```
trials = scenarios × seeds × sweep points × variants
```

Every axis is optional; a missing axis contributes one no-op entry, so a test with no axes is a single trial.

## Layout

```
TBD_PF_test/
  run_test.m              entry point
  TBD_PF_test.m           interactive wrapper around run_test()
  core/
    default_cfg.m         every trial parameter + default (single source of truth)
    default_test.m        every test-definition field + default
    load_test.m           find config/T*.m by name/prefix, merge onto default_test
    apply_overrides.m     strict recursive merge (errors on unknown field)
    set_path.m            strict set of a dotted path, e.g. 'TBD.Pb'
    resolve_cfg.m         derived fields + cross-block checks
    expand_trials.m       test definition -> trial struct array
    name_value.m          struct -> {'Field', val, ...}
    build_filter.m        factory: cfg.filter -> filter object
    compute_metrics.m     scalar metrics per trial
    summarize.m           per-trial / grouped tables + comparison plot
  config/
    scenarios.m           named scenario library
    variants.m            named filter-variant library
    T1_...m … T5_...m     test definitions
  results/<name>_<yyyyMMdd_HHmmss>/   (created by run_test)
```

## 1. `default_cfg`: the trial cfg

```
cfg.filter   "TBD_PF"
cfg.env      -> environment(...)          forwarded verbatim as name/value
cfg.TBD      -> TBD_PF(...)
cfg.run      -> f.run(scenario, ...)
cfg.show     -> f.show(R, scenario, ...)
```

Rule: **every field in env/TBD/run/show is a real argument of that call.** That is what lets `name_value` forward whole blocks with no argument lists in the runner. Test-level switches (plot or not, save or not) live in `t.output`, never in `cfg`.

`[]` / `''` marks a derived field:

| Field | Filled from | By |
|---|---|---|
| `TBD.Seed` | `env.Seed` | `resolve_cfg` |
| `TBD.dt` | `env.dt` | `resolve_cfg` |
| `TBD.K` | `scenario.K` (generated frame count, `kEnd + 1`) | `run_test` |
| `run.Var` | `env.Var` | `resolve_cfg` |
| `show.Save` | `<outDir>/figures/<trial file label>` | `run_test` |
| `run.DebugDir`, `run.DebugName` | `<outDir>/debug`, trial label | `run_test` |

Set any of these explicitly in a config to break the link.

**Adding a parameter**: add it to `default_cfg` (in the block of the call it belongs to). Nothing else changes.

## 2. Strict merging

```matlab
s = apply_overrides(s, ov)        % recurse into scalar sub-structs; replace everything else
s = set_path(s, 'TBD.Pb', 0.2)    % path must already exist
```

Unknown names error with the list of valid fields:

```
Unknown field 'cfg.TBD.pEthres'. Valid fields: Debug, Units, Seed, ...
```

The test definition itself is checked the same way (`load_test` against `default_test`, `t.output` and `t.mc` strictly).

## 3. `resolve_cfg`

```
fill derived links (table above)
assert Q is 4x4
assert v_min <= v_max
assert numel(env.trajectories) == env.objs
assert env.Noise || env.Var == 0
assert run.Likelihood in {gauss, model}
```

Add cross-block checks here; single-argument validation already happens in the class input parsers.

## 4. Libraries

`config/scenarios.m` and `config/variants.m` return structs of named partial cfgs.

```matlab
S.static_noisy = struct('env', struct('trajectories', {{'static'}}, 'Noise', true, 'Var', 15));
V.pf_gauss     = struct('filter', "TBD_PF", 'run', struct('Likelihood', 'gauss'));
```

Tests reference them **by name** (the name becomes the trial label). One-off entries can be inline as a name followed by an overrides struct:

```matlab
t.scenarios = {"static", "line", "custom", struct('env', struct('kEnd', 50))};
```

Note `{{'static'}}`: `struct()` expands one level of cell, so a cell-valued field needs double braces.

## 5. Test definitions

All fields (from `default_test`):

| Field | Default | Meaning |
|---|---|---|
| `name` | file name | used for the results folder |
| `description` | `""` | printed at the start |
| `base` | `struct()` | overrides on `default_cfg`, every trial |
| `scenarios` | `{}` | axis 1, names / inline pairs |
| `variants` | `{}` | axis 4, names / inline pairs |
| `sweep` | empty | axis 3, struct array `('path', {...}, 'values', {...})` |
| `sweep_mode` | `"grid"` | `"grid"` = cross product, `"zip"` = paired by index |
| `mc` | `[]` | axis 2, `struct('n', 10, 'seed0', 1)` → seeds `seed0 : seed0+n-1` |
| `output.perTrial` | `true` | call `f.show` for every trial |
| `output.closeFigs` | `false` | close per-trial figures after saving |
| `output.summaryPlot` | `true` | comparison plot (only if > 1 group or MC) |
| `output.save` | `true` | write the results folder |
| `output.saveR` | `false` | put filter output `R` in `trials.mat` (~20 MB per trial at N = 5000) |
| `output.keepR` | `true` | keep `R` in the returned struct; set false for big MC runs |
| `output.rootDir` | `""` | `""` → `TBD_PF_test/results` |

Minimal (T1):

```matlab
function t = T1_PF_Static_N5000_v0_s1()
    t.description = "...";
    t.scenarios = {"static"};
    t.base.env  = struct('Debug', true);
    t.base.TBD  = struct('Debug', true, 'Pb', 0.13, 'v_max', 0.05, 'v_min', -0.05);
    t.base.run  = struct('ESS', 0.0005, 'DebugFrames', [9 11 20 40 60 80 91]);
    t.base.show = struct('Animate', true);
end
```

Everything composed (2 scenarios × 10 seeds × 3 Pb × 2 likelihoods = 120 trials):

```matlab
function t = T6_likelihood_compare()
    t.description = "model vs gauss likelihood, Pb sweep, 10 MC seeds";
    t.scenarios = {"static_noisy", "line_noisy"};
    t.variants  = {"pf_model", "pf_gauss"};
    t.sweep     = struct('path', {'TBD.Pb'}, 'values', {[0.03 0.1 0.2]});
    t.mc        = struct('n', 10, 'seed0', 1);
    t.output    = struct('perTrial', false, 'keepR', false);
end
```

Multiple sweep parameters: `struct('path', {'TBD.Pb', 'TBD.N'}, 'values', {[0.05 0.1], [1000 5000]})`. Values may be a cell for non-numeric sweeps: `'values', {{'gauss', 'model'}}`.

## 6. `expand_trials`

```
base = apply_overrides(default_cfg(), t.base)
for each scenario i
  for each seed m                       (NaN = keep cfg seed when no mc)
    for each sweep point k
      for each variant j                (innermost: consecutive trials share env)
        cfg = apply_overrides(base, scenario_i)
        cfg = apply_overrides(cfg,  variant_j)
        for (path, value) in point_k:  cfg = set_path(cfg, path, value)
        if seed: cfg.env.Seed = seed_m
        cfg = resolve_cfg(cfg)
        append trial {cfg, scenario, variant, sweep, seed, group, label, file}
```

Precedence (later wins): `default_cfg → base → scenario → variant → sweep → seed`.

- `group` = `scenario|variant|sweep`. MC seeds are averaged within a group. It is `"default"` when there are no names.
- `label` = `group|s<seed>`, e.g. `static|pf_gauss|Pb=0.1|s3`.
- `file` = label made file-name safe, e.g. `static_pf_gauss_Pb=0.1_s3`.

Seeds: MC sets `env.Seed`; `TBD.Seed` follows it via `resolve_cfg`. Every variant on the same seed therefore gets the same filter RNG stream too, which is what a paired comparison wants.

## 7. `run_test`

```
cd matlab_src (restored on exit), addpath Includes/core/config
t      = load_test(name);  apply varargin to t.output
trials = expand_trials(t)
outDir = results/<name>_<timestamp>   (if output.save)

for each trial
  fill show.Save / run.DebugDir / run.DebugName if empty
  if jsonencode(cfg.env) differs from the previous trial's:
      scenario = environment(env args).setup()       % else reuse
  f = build_filter(cfg)
  R = f.run(scenario, run args)      (timed)
  metrics = compute_metrics(R, scenario, runtime)
  if output.perTrial: f.show(R, scenario, show args)
  keep R if output.keepR / saveR

summary = summarize(trials, t)
save: trials.csv, summary.csv, summary.png, trials.mat {test, saved, tables}
return out = {test, trials, summary, dir}
```

Scenario reuse only keeps the latest scenario in memory. Variants are the innermost loop, so all variants and non-env sweep points of a scenario+seed reuse it. Sweeping an `env.*` parameter changes the key and regenerates, which is correct.

MATLAB cannot index a call result, so forwarding is always two lines:

```matlab
args = name_value(cfg.run);
R = f.run(scenario, args{:});
```

## 8. Multiple filters

`build_filter` returns an object with `run(scenario, ...)` and `show(R, scenario, ...)`. To add one:

1. Add its parameter block to `default_cfg` (e.g. `cfg.EKF`).
2. Add a `case` in `build_filter` that builds from that block.
3. Add a variant in `variants.m` (`struct('filter', "TBD_EKF", 'EKF', struct(...))`).

If its `run`/`show` signature differs, wrap it in a small adapter class rather than special-casing the runner.

## 9. Metrics and outputs

`compute_metrics(R, scenario, runtime)` returns scalars only:

| Metric | Definition |
|---|---|
| `rmse` | position RMSE [m] over frames where a target exists and a track is declared. Estimate = weighted mean of E = 1 particles; error to nearest existing truth |
| `detect` | fraction of target-present frames with a declared track |
| `false_alarm` | fraction of target-absent frames with a declared track |
| `delay` | frames from first birth to first declaration (NaN = never) |
| `ess_min`, `ess_med` | min / median ESS ÷ N |
| `resamples` | number of resampling frames |
| `runtime` | `f.run` wall time [s] |

To add a metric, add a scalar field in `compute_metrics`. The tables and the summary plot pick it up automatically (the plot shows a fixed subset; edit `plot_summary` to change it).

`summarize` returns `perTrial` (one row per trial) and `grouped` (`groupsummary` mean/std per group, `GroupCount` = number of seeds). It prints the group means and draws box + swarm charts per metric.

Results folder:

```
results/T5_..._20261019_160000/
  figures/<file label>_tbd.png, _history.gif     (per trial, if perTrial)
  debug/<file label>/...                          (if run.DebugPlots)
  trials.csv, summary.csv, summary.png
  trials.mat    test (merged definition), saved (trials: cfg, labels, metrics[, R]), tables
```

Every trial's resolved `cfg` is stored, so any result can be re-run exactly.

## Open points

- **Parallelism**: trials are independent. A `parfor` version needs scenarios generated up front per unique env key, and per-trial plotting turned off.
- **Units**: metrics assume scenario truth is in meters (it is for `Units = 'meters'`). `rmse` is skipped with a warning otherwise.
- **Precedence**: variants override scenarios. A variant that touches `env.*` defeats scenario reuse (still correct, just slower).
