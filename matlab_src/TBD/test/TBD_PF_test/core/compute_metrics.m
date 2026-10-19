%%% Per-trial scalar metrics from a filter run. Every field is a scalar so
%%% trials stack straight into a table (see summarize).
%%%
%%%   rmse        position RMSE [m] over frames where a target exists AND the
%%%               track is declared. Estimate = weighted mean of the E = 1
%%%               particles; error is to the nearest existing truth.
%%%   detect      fraction of target-present frames with a declared track
%%%   false_alarm fraction of target-absent frames with a declared track
%%%   delay       frames from first birth to first declaration (NaN = never)
%%%   ess_min     min ESS / N over the run
%%%   ess_med     median ESS / N
%%%   resamples   # of frames that resampled
%%%   runtime     filter run() wall time [s]
%%%
%%% Assumes scenario truth is in meters (particle state always is).

function m = compute_metrics(R, scenario, runtime)

    K = R.K;
    present = any(scenario.exist(:, 1:K), 1);
    declared = R.exist(1:K);

    %% Position error
    err = nan(1, K);
    if ~ismember(lower(string(scenario.units)), ["meters", "m"])
        warning("compute_metrics:units", "scenario units '%s' are not meters, rmse skipped", scenario.units);
    else
        for k = find(present & declared)
            P = R.particles(:, :, k);
            w = R.weights(:, k)';
            alive = P(5, :) ~= 0 & ~isnan(P(1, :));
            if ~any(alive) || sum(w(alive)) <= 0, continue; end
            wa = w(alive) / sum(w(alive));
            est = [P(1, alive) * wa'; P(3, alive) * wa'];

            d = inf;
            for i = find(scenario.exist(:, k))'
                tru = scenario.truth{i}([1 3], k);
                d = min(d, norm(est - tru));
            end
            err(k) = d;
        end
    end

    m = struct();
    m.rmse        = sqrt(mean(err.^2, 'omitnan'));
    m.detect      = mean(declared(present));
    m.false_alarm = mean(declared(~present));
    kBirth = find(present, 1);
    kDecl  = find(declared & (1:K) >= kBirth, 1);
    if isempty(kBirth) || isempty(kDecl)
        m.delay = NaN;
    else
        m.delay = kDecl - kBirth;
    end
    m.ess_min   = min(R.ess) / R.N;
    m.ess_med   = median(R.ess, 'omitnan') / R.N;
    m.resamples = nnz(R.resampled);
    m.runtime   = runtime;

    % Empty selections give NaN from mean, which is what we want, but make it explicit
    if ~any(present),  m.detect = NaN;      end
    if all(present),   m.false_alarm = NaN; end

end
