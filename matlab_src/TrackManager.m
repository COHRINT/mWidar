classdef TrackManager < handle
% TRACKMANAGER  Orchestrates a multi-target filter and a count estimator.
%
% Each call to step(signal_k, z_k):
%   1. Updates the count estimator.
%   2. Compares estimated N to filter's current N_t with hysteresis.
%   3. Adds new targets (seeded from estimator's cluster centroids)
%      or removes targets (per filter-specific worst-target rule).
%   4. Runs one filter timestep on z_k.
%
% USAGE:
%   filt = PDA_PF_multi(x0_cell, 500, F, Q, H, R, ...);
%   est  = HeuristicEstimator(cfg);
%   tm   = TrackManager(filt, est);
%   for k = 1:K
%       tm.step(signal{k}, z{k});
%       [x_est, P_est] = filt.getGaussianEstimate();
%   end
%
% PARAMETERS:
%   K_hyst       - frames N_est must remain at a new value before the
%                  manager applies the change (default 3).
%   P_init_diag  - diag entries of new-target initial covariance
%                  (default [0.5 0.5 1.0 1.0 0.5 0.5]^2 for 6-state CV/CA).
%   N_min, N_max - clamp range for filter's N_t (default 1, 5).

    properties
        filt                  % handle to the multi-target filter
        estimator             % handle to a TargetCountEstimator
        K_hyst        = 3
        P_init_diag           % length matches filter's N_x
        N_min         = 1
        N_max         = 5
        debug         = false

        % --- internal state ---
        candidate_N           % last requested N_est that's pending confirmation
        candidate_count = 0   % consecutive frames at candidate_N
        history               % struct array per call to step()
    end

    methods
        function obj = TrackManager(filt, estimator, varargin)
            p = inputParser;
            addParameter(p, 'KHyst', 3, @(x) x >= 1);
            addParameter(p, 'PInitDiag', [], @isnumeric);
            addParameter(p, 'NMin', 1, @(x) x >= 1);
            addParameter(p, 'NMax', 5, @(x) x >= 1);
            addParameter(p, 'Debug', false, @islogical);
            parse(p, varargin{:});

            obj.filt      = filt;
            obj.estimator = estimator;
            obj.K_hyst    = p.Results.KHyst;
            obj.N_min     = p.Results.NMin;
            obj.N_max     = p.Results.NMax;
            obj.debug     = p.Results.Debug;

            N_x = filt.N_x;
            if isempty(p.Results.PInitDiag)
                base = [0.5, 0.5, 1.0, 1.0, 0.5, 0.5].^2;
                if N_x <= numel(base)
                    obj.P_init_diag = base(1:N_x);
                else
                    obj.P_init_diag = [base, ones(1, N_x - numel(base))];
                end
            else
                if numel(p.Results.PInitDiag) ~= N_x
                    error('TrackManager:PInitDiag', ...
                        'PInitDiag must have %d entries (filter N_x)', N_x);
                end
                obj.P_init_diag = p.Results.PInitDiag(:).';
            end

            obj.candidate_N = NaN;
            obj.history = struct('k', {}, 'N_est', {}, 'N_t_before', {}, ...
                'N_t_after', {}, 'action', {});
        end

        function step(obj, signal_k, z_k)
            % --- 1. Update count estimator ---
            [N_est, seed_xy, ~] = obj.estimator.update(signal_k, z_k);

            N_t_before = obj.filt.N_t;
            action = 'none';

            % --- 2. Hysteresis check & cardinality update ---
            if ~isnan(N_est)
                N_target = max(obj.N_min, min(obj.N_max, N_est));
                if N_target == N_t_before
                    obj.candidate_N = NaN;
                    obj.candidate_count = 0;
                else
                    if N_target == obj.candidate_N
                        obj.candidate_count = obj.candidate_count + 1;
                    else
                        obj.candidate_N = N_target;
                        obj.candidate_count = 1;
                    end

                    if obj.candidate_count >= obj.K_hyst
                        if N_target > N_t_before
                            obj.add_targets(N_target - N_t_before, seed_xy);
                            action = sprintf('add x%d', N_target - N_t_before);
                        elseif N_target < N_t_before
                            obj.remove_targets(N_t_before - N_target);
                            action = sprintf('remove x%d', N_t_before - N_target);
                        end
                        obj.candidate_N = NaN;
                        obj.candidate_count = 0;
                    end
                end
            end

            % --- 3. Run filter timestep ---
            obj.filt.timestep(z_k);

            % --- 4. Log ---
            entry.k          = numel(obj.history) + 1;
            entry.N_est      = N_est;
            entry.N_t_before = N_t_before;
            entry.N_t_after  = obj.filt.N_t;
            entry.action     = action;
            obj.history(end+1) = entry;

            if obj.debug && ~strcmp(action, 'none')
                fprintf('TrackManager k=%d: N_est=%g, %s, N_t=%d -> %d\n', ...
                    entry.k, N_est, action, N_t_before, obj.filt.N_t);
            end
        end
    end

    methods (Access = private)
        function add_targets(obj, n_to_add, seed_xy)
            % Pick `n_to_add` seeds from seed_xy that are farthest from
            % existing tracks. If we have fewer seeds than requested, add
            % what we can and pad with the fallback (centred at scene).

            if isempty(seed_xy), seed_xy = zeros(2, 0); end

            % Existing track positions (xy only)
            existing_xy = obj.collect_existing_xy();

            % Greedily pick seeds farthest from existing + previously-picked
            picked = zeros(2, 0);
            available = seed_xy;
            ref = existing_xy;

            for k = 1:n_to_add
                if isempty(available)
                    % Fallback seed: centre of typical scene
                    new_xy = [0; 2.0];
                else
                    if isempty(ref)
                        sel = 1;
                    else
                        d = inf(1, size(available, 2));
                        for j = 1:size(available, 2)
                            d(j) = min(vecnorm(ref - available(:, j), 2, 1));
                        end
                        [~, sel] = max(d);
                    end
                    new_xy = available(:, sel);
                    available(:, sel) = [];
                end
                picked(:, end+1) = new_xy; %#ok<AGROW>
                ref = [ref, new_xy]; %#ok<AGROW>

                x_init = obj.xy_to_state(new_xy);
                P_init = diag(obj.P_init_diag);
                obj.filt.add_target(x_init, P_init);
            end
        end

        function remove_targets(obj, n_to_remove)
            % Pick the n_to_remove worst targets via filter-specific rule
            % and drop them. Removal indices are recomputed after each
            % drop so they always refer to the current (shrinking) state.

            for k = 1:n_to_remove
                if obj.filt.N_t <= obj.N_min, return; end
                t_idx = obj.pick_worst_target();
                if isempty(t_idx) || isnan(t_idx), return; end
                obj.filt.remove_target(t_idx);
            end
        end

        function t_idx = pick_worst_target(obj)
            % Filter-specific worst-target selection.  Returns a
            % SLOT INDEX (column in the filter's slot pool), not a
            % local active-index, so it can be passed straight to
            % filter.remove_target().
            if isa(obj.filt, 'KF_RBPF_multi')
                rate = obj.filt.getClutterRate(10);   % [N_max x 1], NaN for inactive
                if all(isnan(rate))
                    t_idx = obj.pick_by_det();
                else
                    [~, t_idx] = max(rate, [], 'omitnan');
                end
            else
                t_idx = obj.pick_by_det();
            end
        end

        function t_idx = pick_by_det(obj)
            % Falls back to largest-position-covariance among active
            % slots. getGaussianEstimate returns active-only cells in
            % slot-index order; the third return is the slot index map.
            [~, P_est_cell, active_idx] = obj.filt.getGaussianEstimate();
            if isempty(active_idx), t_idx = []; return; end
            dets = zeros(1, numel(active_idx));
            for a = 1:numel(active_idx)
                P_pos = P_est_cell{a}(1:2, 1:2);
                dets(a) = det(P_pos);
            end
            [~, a_max] = max(dets);
            t_idx = active_idx(a_max);
        end

        function existing_xy = collect_existing_xy(obj)
            % Gather current per-target xy estimates over active slots.
            x_cell = obj.filt.getGaussianEstimate();
            N_active = numel(x_cell);
            existing_xy = zeros(2, N_active);
            for a = 1:N_active
                existing_xy(:, a) = x_cell{a}(1:2);
            end
        end

        function x = xy_to_state(obj, xy)
            % Promote a 2D position seed to the filter's full state vector
            % with zero velocity / acceleration.
            x = zeros(obj.filt.N_x, 1);
            x(1:2) = xy(:);
        end
    end
end
