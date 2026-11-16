%%% Per-run metrics for run_compare: RMSE, OSPA and runtime, plus the
%%% per-frame series the time-history plots need.
%%%
%%%   [T, TS] = compare_metrics(results, scenarios, 'OSPACutoff', 0.5, 'OSPAOrder', 2)
%%%
%%% Inputs
%%%   results     1 x J struct array from run_compare (fields filter, traj,
%%%               trial, seed, N, envIdx, runtime, R)
%%%   scenarios   cell of scenarios, results(j).envIdx indexes into it
%%%
%%% Estimate per frame (a set: 2 x 0 or 2 x 1, meters)
%%%   TBD-PF   weighted mean of the E = 1 particles on frames where the
%%%            filter declared a track (R.exist), empty otherwise. Same rule
%%%            as compute_metrics.
%%%   KF_RBPF  x_est([1 2], k) on every frame (state is [x y vx vy ax ay]).
%%%
%%% Metrics (one row of T per run)
%%%   rmse        position RMSE over frames where the target exists AND the
%%%               filter reported an estimate, error to the nearest truth
%%%   ospa        mean OSPA(c, p) over all K frames, with the
%%%   ospa_loc    localization and
%%%   ospa_card   cardinality components (mean of each per-frame component)
%%%   detect      fraction of target-present frames with an estimate
%%%   false_alarm fraction of target-absent frames with an estimate
%%%   runtime     filtering wall time [s] from the job
%%%
%%% TS(j) per-frame series: err, ospa, ospa_loc, ospa_card [1 x K],
%%% reported, present [1 x K logical]

function [T, TS] = compare_metrics(results, scenarios, varargin)

    p = inputParser;
    addParameter(p, 'OSPACutoff', 0.5, @(x) isscalar(x) && x > 0);
    addParameter(p, 'OSPAOrder', 2, @(x) isscalar(x) && x >= 1);
    parse(p, varargin{:});
    c = p.Results.OSPACutoff;
    po = p.Results.OSPAOrder;

    J = numel(results);
    rows = cell(J, 1);
    TS = repmat(struct('err', [], 'ospa', [], 'ospa_loc', [], 'ospa_card', [], ...
                       'reported', [], 'present', []), 1, J);

    for j = 1:J
        r = results(j);
        scenario = scenarios{r.envIdx};
        if ~ismember(lower(string(scenario.units)), ["meters", "m"])
            error("compare_metrics:units", "scenario units '%s' are not meters", scenario.units);
        end

        K = scenario.K;
        est = estimates(r, K);
        present = any(scenario.exist(:, 1:K), 1);

        err  = nan(1, K);
        ospa = nan(1, K);
        oloc = nan(1, K);
        ocrd = nan(1, K);
        reported = false(1, K);
        for k = 1:K
            alive = find(scenario.exist(:, k))';
            gt = zeros(2, numel(alive));
            for a = 1:numel(alive)
                gt(:, a) = scenario.truth{alive(a)}([1 3], k);
            end
            x = est{k};
            reported(k) = ~isempty(x);

            if reported(k) && ~isempty(gt)
                err(k) = min(vecnorm(gt - x(:, 1)));
            end
            [ospa(k), oloc(k), ocrd(k)] = compute_ospa(x, gt, c, po);
        end

        m = struct();
        m.filter = string(r.filter);
        m.traj   = string(r.traj);
        m.trial  = r.trial;
        m.seed   = r.seed;
        m.N      = r.N;
        m.rmse        = sqrt(mean(err.^2, 'omitnan'));
        m.ospa        = mean(ospa);
        m.ospa_loc    = mean(oloc);
        m.ospa_card   = mean(ocrd);
        m.detect      = mean(reported(present));
        m.false_alarm = mean(reported(~present));
        m.runtime     = r.runtime;
        if ~any(present), m.detect = NaN; end
        if all(present),  m.false_alarm = NaN; end
        rows{j} = m;

        TS(j) = struct('err', err, 'ospa', ospa, 'ospa_loc', oloc, 'ospa_card', ocrd, ...
                       'reported', reported, 'present', present);
    end

    T = struct2table([rows{:}]', 'AsArray', true);

end


%%% 1 x K cell of 2 x 0 / 2 x 1 position estimates for one run
function est = estimates(r, K)

    est = repmat({zeros(2, 0)}, 1, K);
    R = r.R;

    if r.filter == "KF_RBPF"
        for k = 1:K
            x = R.x_est([1 2], k);
            if all(isfinite(x)), est{k} = x; end
        end
        return
    end

    % TBD-PF: particles are [px; vx; py; vy; E] x N x K
    for k = find(R.exist(1:K))
        P = R.particles(:, :, k);
        w = R.weights(:, k)';
        alive = P(5, :) ~= 0 & ~isnan(P(1, :));
        if ~any(alive) || sum(w(alive)) <= 0, continue; end
        wa = w(alive) / sum(w(alive));
        est{k} = [P(1, alive) * wa'; P(3, alive) * wa'];
    end

end
