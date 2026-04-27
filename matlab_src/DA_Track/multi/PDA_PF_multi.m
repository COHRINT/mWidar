% PDA_PF_multi.m
% Multi-Target Probabilistic Data Association Particle Filter
% (slot-pool / variable-cardinality variant)
%
% DESCRIPTION:
%   Joint-state SIR particle filter for multi-target tracking. Each
%   particle carries state for an N_max-slot pool of targets in a
%   [N_x x N_max] matrix, where only the active slots (obj.active mask)
%   are processed. obj.N_t = sum(obj.active) is the live target count.
%
%   This shape lets the TrackManager add and remove targets at runtime
%   (via add_target / remove_target) WITHOUT discarding history or
%   fabricating zero-padded entries: dormant slots simply have
%   active(t)=false and are skipped by every per-target loop.
%
%   ALGORITHM (per timestep), unchanged from the fixed-N variant aside
%   from iterating over find(obj.active) instead of 1:obj.N_t:
%     1. PREDICTION  : propagate active target states through F + Q noise
%     2. VALIDATION  : per-active-target measurement gating
%     3. WEIGHT UPDATE: JPDA sum over feasible joint hypotheses over the
%                       N_active = sum(active) targets, with measurement
%                       exclusivity enforced
%     4. NORMALIZE; compute ESS; resample if ESS < threshold
%
% PARTICLE STRUCTURE:
%   particles{i}.states  [N_x x N_max]  joint state for slot pool;
%                                       inactive columns are dormant
%                                       (carry stale state, ignored).
%   particles{i}.weight  scalar          importance weight
%
% BACK-COMPAT:
%   When constructed with x0_cell of length N_t and no NMax option, the
%   filter defaults to N_max = max(N_t, 5). The first N_t slots are
%   active. With no add_target / remove_target calls, obj.N_t stays at
%   the constructor value and behaviour is bit-identical to the
%   fixed-N filter.

classdef PDA_PF_multi < handle

    properties
        % Dimensions
        N_p     % Number of particles
        N_x     % State dimension (per target)
        N_z     % Measurement dimension
        N_t     % Number of ACTIVE targets   (= sum(obj.active))
        N_max   % Slot-pool capacity

        % Active mask: logical [1 x N_max]; obj.active(t) == true means
        % slot t is currently a tracked target.
        active

        % Particle state
        particles  % {1 x N_p} cell of structs: .states [N_x x N_max], .weight

        % System matrices
        F   % State transition [N_x x N_x]
        Q   % Process noise covariance [N_x x N_x]
        H   % Measurement matrix [N_z x N_x]
        R   % Measurement noise covariance [N_z x N_z]

        % mWidar likelihood lookup table (optional)
        pointlikelihood_image

        % Detection model
        PD
        PFA
        lambda_clutter

        % Gating
        validation_sigma_bounds

        % Resampling
        ESS_threshold_percentage

        % Diagnostics
        current_ESS
        timestep_counter

        % History
        history

        % Flags
        debug
        store_full_history

        % Init params (kept so add_target can re-use the same defaults)
        init_sigma_pos
        init_sigma_vel
    end

    methods

        function obj = PDA_PF_multi(x0_cell, N_particles, F, Q, H, R, varargin)
            % Constructor — see file header for option list.
            p = inputParser;
            addParameter(p, 'PD',              0.95,  @(x) x > 0 && x <= 1);
            addParameter(p, 'PFA',             0.05,  @(x) x >= 0);
            addParameter(p, 'LambdaClutter',   2.5,   @isnumeric);
            addParameter(p, 'ValidationSigma', 5,     @(x) x > 0);
            addParameter(p, 'ESSThreshold',    0.2,   @(x) x > 0 && x <= 1);
            addParameter(p, 'Debug',           false, @islogical);
            addParameter(p, 'UniformInit',     false, @islogical);
            addParameter(p, 'store_full_history', true, @islogical);
            addParameter(p, 'PointlikelihoodImage', [], @(x) isempty(x) || ismatrix(x));
            addParameter(p, 'InitSigmaPos', 0.30, @(x) x >= 0);
            addParameter(p, 'InitSigmaVel', 0.50, @(x) x >= 0);
            addParameter(p, 'NMax', [], @(x) isempty(x) || (isnumeric(x) && x >= 1));
            parse(p, varargin{:});
            opt = p.Results;

            N_t_init   = numel(x0_cell);
            if isempty(opt.NMax)
                obj.N_max = max(N_t_init, 5);
            else
                obj.N_max = max(opt.NMax, N_t_init);
            end
            obj.active = false(1, obj.N_max);
            obj.active(1:N_t_init) = true;
            obj.N_t = N_t_init;

            obj.N_p = N_particles;
            obj.N_x = size(F, 1);
            obj.N_z = size(H, 1);

            obj.F = F;  obj.Q = Q;  obj.H = H;  obj.R = R;

            obj.PD             = opt.PD;
            obj.PFA            = opt.PFA;
            obj.lambda_clutter = opt.LambdaClutter;
            obj.validation_sigma_bounds = opt.ValidationSigma;
            obj.ESS_threshold_percentage = opt.ESSThreshold;
            obj.debug          = opt.Debug;
            obj.store_full_history = opt.store_full_history;
            obj.pointlikelihood_image = opt.PointlikelihoodImage;
            obj.init_sigma_pos = opt.InitSigmaPos;
            obj.init_sigma_vel = opt.InitSigmaVel;

            obj.timestep_counter = 0;
            obj.current_ESS      = NaN;

            obj.particles = PDA_PF_multi.init_particles( ...
                N_particles, x0_cell, obj.N_x, obj.N_max, opt.UniformInit, ...
                opt.InitSigmaPos, opt.InitSigmaVel);

            obj.history = struct('measurements', {}, 'estimate', {}, ...
                'covariance', {}, 'ESS', {}, 'active_indices', {});

            if obj.debug
                fprintf('PDA_PF_multi: N_p=%d, N_t=%d, N_max=%d, N_x=%d\n', ...
                    obj.N_p, obj.N_t, obj.N_max, obj.N_x);
            end
        end

        % ------------------------------------------------------------------
        function timestep(obj, z, varargin)
            true_state = [];
            if ~isempty(varargin), true_state = varargin{1}; end

            obj.prediction();

            valid_z = obj.validation(z);
            obj.weight_update(z, valid_z);

            total_w = sum(cellfun(@(p) p.weight, obj.particles));
            if total_w < eps
                for i = 1:obj.N_p
                    obj.particles{i}.weight = 1/obj.N_p;
                end
                total_w = 1;
                if obj.debug
                    warning('PDA_PF_multi: weight collapse at step %d', ...
                        obj.timestep_counter+1);
                end
            end
            for i = 1:obj.N_p
                obj.particles{i}.weight = obj.particles{i}.weight / total_w;
            end

            w_vec = cellfun(@(p) p.weight, obj.particles);
            obj.current_ESS = 1 / sum(w_vec .^ 2);

            obj.resample();

            obj.timestep_counter = obj.timestep_counter + 1;
            if obj.store_full_history
                obj.storeHistory(z, true_state);
            end
        end

        % ------------------------------------------------------------------
        function prediction(obj)
            sqrtQ = chol(obj.Q, 'lower');
            active_slots = find(obj.active);
            for i = 1:obj.N_p
                for t = active_slots
                    noise = sqrtQ * randn(obj.N_x, 1);
                    obj.particles{i}.states(:, t) = ...
                        obj.F * obj.particles{i}.states(:, t) + noise;
                end
            end
        end

        % ------------------------------------------------------------------
        function valid_z = validation(obj, z)
            % Returns a {1 x N_max} cell. Inactive slots get [].
            valid_z = cell(1, obj.N_max);
            if isempty(z), return; end

            w_vec = cellfun(@(p) p.weight, obj.particles);
            for t = find(obj.active)
                states_t = zeros(obj.N_x, obj.N_p);
                for i = 1:obj.N_p
                    states_t(:, i) = obj.particles{i}.states(:, t);
                end
                x_emp = states_t * w_vec';
                diff  = states_t - x_emp;
                P_emp = diff * diag(w_vec) * diff';
                P_emp = 0.5*(P_emp + P_emp') + 1e-8*eye(obj.N_x);

                S_t = obj.H * P_emp * obj.H' + obj.R;
                S_inv = inv(S_t); %#ok<MINV>
                z_pred = obj.H * x_emp;
                gate_sq = obj.validation_sigma_bounds^2;

                mask = false(1, size(z, 2));
                for j = 1:size(z, 2)
                    innov = z(:, j) - z_pred;
                    NIS   = innov' * S_inv * innov;
                    mask(j) = NIS < gate_sq;
                end
                valid_z{t} = z(:, mask);
            end
        end

        % ------------------------------------------------------------------
        function weight_update(obj, z, ~)
            % JPDA likelihood with exclusivity, over the ACTIVE slot set.
            if isempty(z), return; end

            active_slots = find(obj.active);
            N_active = numel(active_slots);
            if N_active == 0, return; end

            N_m         = size(z, 2);
            tensor_size = (N_m + 1) * ones(1, N_active);
            num_hyp     = (N_m + 1)^N_active;

            % L_all is indexed by (particle, active_slot_index, m+1)
            L_all = zeros(obj.N_p, N_active, N_m + 1);
            L_all(:, :, 1) = 1 - obj.PD;

            if ~isempty(obj.pointlikelihood_image)
                npx   = 128;
                xgrid = linspace(-2, 2, npx);
                ygrid = linspace(0, 4, npx);
                sf    = 0.15;

                meas_lin = zeros(N_m, 1);
                for m = 1:N_m
                    [~, mx] = min(abs(xgrid - z(1, m)));
                    [~, my] = min(abs(ygrid - z(2, m)));
                    meas_lin(m) = sub2ind([npx, npx], my, mx);
                end

                for ts = 1:N_active
                    t = active_slots(ts);
                    px = cellfun(@(p) p.states(1, t), obj.particles)';
                    py = cellfun(@(p) p.states(2, t), obj.particles)';

                    [~, xi] = min(abs(px - xgrid), [], 2);
                    [~, yi] = min(abs(py - ygrid), [], 2);
                    xi = max(1, min(npx, xi));
                    yi = max(1, min(npx, yi));
                    part_lin = sub2ind([npx, npx], yi, xi);

                    for m = 1:N_m
                        lk      = obj.pointlikelihood_image(meas_lin(m), part_lin)';
                        dx      = px - z(1, m);
                        dy      = py - z(2, m);
                        gauss_w = exp(-(dx.^2 + dy.^2) / (2 * sf^2));
                        L_all(:, ts, m + 1) = obj.PD * lk .* gauss_w + eps;
                    end
                end
            else
                R_inv      = inv(obj.R); %#ok<MINV>
                norm_const = sqrt((2*pi)^obj.N_z * det(obj.R));

                for ts = 1:N_active
                    t = active_slots(ts);
                    states_t = cell2mat(cellfun(@(p) p.states(:, t), obj.particles, ...
                                                'UniformOutput', false));
                    z_preds  = obj.H * states_t;
                    for m = 1:N_m
                        innov    = z(:, m) - z_preds;
                        expvals  = -0.5 * sum((R_inv * innov) .* innov, 1);
                        L_all(:, ts, m + 1) = obj.PD * exp(expvals)' / norm_const;
                    end
                end
            end

            sub_table = zeros(num_hyp, N_active);
            if N_active == 1
                % ind2sub requires size vector with >=2 elements; the 1D
                % case is trivial -> linear index IS the subscript.
                sub_table(:, 1) = (1:num_hyp).';
            else
                for h = 1:num_hyp
                    sub_idx = cell(1, N_active);
                    [sub_idx{:}] = ind2sub(tensor_size, h);
                    sub_table(h, :) = cell2mat(sub_idx);
                end
            end

            valid_mask = true(num_hyp, 1);
            for h = 1:num_hyp
                c = sub_table(h, :) - 1;
                meas_used = c(c > 0);
                if length(meas_used) ~= length(unique(meas_used))
                    valid_mask(h) = false;
                end
            end
            valid_hyp_idx = find(valid_mask);

            for i = 1:obj.N_p
                Z_i = 0;
                for h_pos = 1:length(valid_hyp_idx)
                    h       = valid_hyp_idx(h_pos);
                    c_plus1 = sub_table(h, :);
                    hyp_w   = 1;
                    for ts = 1:N_active
                        hyp_w = hyp_w * L_all(i, ts, c_plus1(ts));
                    end
                    Z_i = Z_i + hyp_w;
                end
                if Z_i < eps, Z_i = eps; end
                obj.particles{i}.weight = obj.particles{i}.weight * Z_i;
            end
        end

        % ------------------------------------------------------------------
        function resample(obj)
            w_vec = cellfun(@(p) p.weight, obj.particles);
            ESS_thresh = obj.ESS_threshold_percentage * obj.N_p;

            if obj.current_ESS < ESS_thresh
                cumW  = cumsum(w_vec);
                cumW(end) = 1.0;
                step  = 1 / obj.N_p;
                start = rand() * step;
                positions = start + (0:obj.N_p-1) * step;

                new_particles = cell(1, obj.N_p);
                idx = 1;
                for i = 1:obj.N_p
                    while positions(i) > cumW(idx) && idx < obj.N_p
                        idx = idx + 1;
                    end
                    new_particles{i} = struct( ...
                        'states', obj.particles{idx}.states, ...
                        'weight', 1/obj.N_p);
                end
                obj.particles = new_particles;

                if obj.debug
                    fprintf('  PDA_PF_multi: resampled (ESS=%.1f < %.1f)\n', ...
                        obj.current_ESS, ESS_thresh);
                end
            end
        end

        % ------------------------------------------------------------------
        function [x_est_cell, P_est_cell, active_idx] = getGaussianEstimate(obj)
            % Returns one entry per ACTIVE slot, in slot-index order.
            %   x_est_cell{a} - [N_x x 1] weighted mean for active_idx(a)
            %   P_est_cell{a} - [N_x x N_x] weighted covariance
            %   active_idx    - [1 x N_t] slot indices (in obj.active)

            active_idx = find(obj.active);
            N_active = numel(active_idx);
            x_est_cell = cell(1, N_active);
            P_est_cell = cell(1, N_active);

            w_vec = cellfun(@(p) p.weight, obj.particles);

            for a = 1:N_active
                t = active_idx(a);
                states_t = zeros(obj.N_x, obj.N_p);
                for i = 1:obj.N_p
                    states_t(:, i) = obj.particles{i}.states(:, t);
                end
                x_est = states_t * w_vec';
                P_est = zeros(obj.N_x, obj.N_x);
                for i = 1:obj.N_p
                    d = states_t(:, i) - x_est;
                    P_est = P_est + w_vec(i) * (d * d');
                end
                P_est = 0.5*(P_est + P_est') + 1e-10*eye(obj.N_x);

                x_est_cell{a} = x_est;
                P_est_cell{a} = P_est;
            end
        end

        % ------------------------------------------------------------------
        function storeHistory(obj, measurements, true_state)
            k = obj.timestep_counter;
            [x_est, P_est, active_idx] = obj.getGaussianEstimate();

            obj.history(k).measurements   = measurements;
            obj.history(k).true_state     = true_state;
            obj.history(k).estimate       = x_est;
            obj.history(k).covariance     = P_est;
            obj.history(k).ESS            = obj.current_ESS;
            obj.history(k).active_indices = active_idx;
        end

        % ------------------------------------------------------------------
        function add_target(obj, x_init, P_init)
            % Activate the first inactive slot and seed it.
            if all(obj.active)
                error('PDA_PF_multi:add_target:full', ...
                    'Slot pool full (N_max=%d). Construct with larger NMax.', ...
                    obj.N_max);
            end
            if numel(x_init) ~= obj.N_x
                error('PDA_PF_multi:add_target:dim', ...
                    'x_init must be [%d x 1], got %d', obj.N_x, numel(x_init));
            end
            x_init = x_init(:);
            sqrtP = chol(P_init + 1e-9*eye(obj.N_x), 'lower');
            t_new = find(~obj.active, 1, 'first');
            for i = 1:obj.N_p
                obj.particles{i}.states(:, t_new) = ...
                    x_init + sqrtP * randn(obj.N_x, 1);
            end
            obj.active(t_new) = true;
            obj.N_t = sum(obj.active);

            if obj.debug
                fprintf('PDA_PF_multi: activated slot %d -> N_t=%d\n', ...
                    t_new, obj.N_t);
            end
        end

        % ------------------------------------------------------------------
        function remove_target(obj, t_idx)
            % Deactivate slot t_idx (history preserved by being absent
            % from active iteration; state column kept dormant).
            if t_idx < 1 || t_idx > obj.N_max
                error('PDA_PF_multi:remove_target:bounds', ...
                    't_idx=%d outside [1, %d]', t_idx, obj.N_max);
            end
            if ~obj.active(t_idx)
                warning('PDA_PF_multi:remove_target:inactive', ...
                    'Slot %d already inactive', t_idx);
                return
            end
            if sum(obj.active) <= 1
                warning('PDA_PF_multi:remove_target:lastTarget', ...
                    'Refusing to remove last active target');
                return
            end
            obj.active(t_idx) = false;
            obj.N_t = sum(obj.active);

            if obj.debug
                fprintf('PDA_PF_multi: deactivated slot %d -> N_t=%d\n', ...
                    t_idx, obj.N_t);
            end
        end

        % ------------------------------------------------------------------
        function idx = activeIndices(obj)
            idx = find(obj.active);
        end

    end % methods

    methods (Static)

        function particles = init_particles(N_p, x0_cell, N_x, N_max, uniform_init, ...
                                              init_sigma_pos, init_sigma_vel)
            % Allocate [N_x x N_max] state matrices per particle. The first
            % numel(x0_cell) columns are seeded from x0_cell; the rest are
            % zeros (dormant — never read until a slot activates).

            if nargin < 6, init_sigma_pos = 0.30; end
            if nargin < 7, init_sigma_vel = 0.50; end

            N_t = numel(x0_cell);
            particles = cell(1, N_p);

            if uniform_init
                pos_lo  = [-2.0;  0.5]; pos_hi = [2.0; 4.0];
                vel_lo  = [-2.0; -2.0]; vel_hi = [2.0; 2.0];
                acc_lo  = [-2.0; -2.0]; acc_hi = [2.0; 2.0];

                for i = 1:N_p
                    states = zeros(N_x, N_max);
                    for t = 1:N_t
                        s = zeros(N_x, 1);
                        s(1:2) = pos_lo + rand(2,1) .* (pos_hi - pos_lo);
                        if N_x >= 4, s(3:4) = vel_lo + rand(2,1) .* (vel_hi - vel_lo); end
                        if N_x >= 6, s(5:6) = acc_lo + rand(2,1) .* (acc_hi - acc_lo); end
                        states(:, t) = s;
                    end
                    particles{i} = struct('states', states, 'weight', 1/N_p);
                end

            else
                acc_std = 1.0;
                std_vec = ones(N_x, 1);
                std_vec(1:min(2,N_x)) = init_sigma_pos;
                if N_x >= 4, std_vec(3:4) = init_sigma_vel; end
                if N_x >= 6, std_vec(5:6) = acc_std; end

                for i = 1:N_p
                    states = zeros(N_x, N_max);
                    for t = 1:N_t
                        states(:, t) = x0_cell{t} + std_vec .* randn(N_x, 1);
                    end
                    particles{i} = struct('states', states, 'weight', 1/N_p);
                end
            end
        end

    end % static methods

end
