% PDA_PF_multi.m
% Multi-Target Probabilistic Data Association Particle Filter
%
% DESCRIPTION:
%   Joint-state SIR particle filter for multi-target tracking.
%   Each particle carries state for ALL targets simultaneously in a
%   [N_x x N_t] matrix.  Weights are updated as the product of
%   independent PDA marginal likelihoods, one per target.
%
%   "Independent PDA" means measurement exclusivity is NOT enforced:
%   the same measurement can softly contribute to multiple targets.
%   This works well when targets are spatially separated.  For targets
%   that merge or closely cross, KF_RBPF_multi (which enforces
%   exclusivity via the joint association hypothesis) is preferable.
%
%   ALGORITHM (per timestep):
%     1. PREDICTION  : propagate all target states through F + Q noise
%     2. VALIDATION  : per-target measurement gating (empirical cov)
%     3. WEIGHT UPDATE: multiply each particle weight by
%                       prod_t [ lambda*(1-PD) + PD * sum_{j valid} L_{i,j,t} ]
%                       where L_{i,j,t} = mvnpdf(z_j ; H*state_i_t, R)
%     4. NORMALIZE weights; compute ESS; RESAMPLE if ESS < threshold
%
% PARTICLE STRUCTURE:
%   particles{i}.states  - [N_x x N_t] joint state for all targets
%   particles{i}.weight  - scalar importance weight
%
% DETECTION LIKELIHOOD:
%   Uses Gaussian likelihood N(z_j; H*x_i_t, R) where R = obj.R.
%   No magnitude likelihood (detection only).
%
% EXAMPLE:
%   F = ...; Q = ...; H = ...; R = 0.1^2*eye(2);
%   x0_cell = {[x1;y1;vx1;vy1], [x2;y2;vx2;vy2], [x3;y3;vx3;vy3]};
%   filt = PDA_PF_multi(x0_cell, 500, F, Q, H, R);
%   for k = 1:T
%       filt.timestep(z_k);
%       [x_est, P_est] = filt.getGaussianEstimate();
%   end
%
% SEE ALSO: KF_RBPF_multi, HMM_RBPF_multi, PDA_PF, FilterHyperParams

classdef PDA_PF_multi < handle

    properties
        % Dimensions
        N_p     % Number of particles
        N_x     % State dimension (per target)
        N_z     % Measurement dimension
        N_t     % Number of targets

        % Particle state
        particles  % {1 x N_p} cell of structs: .states [N_x x N_t], .weight

        % System matrices
        F   % State transition [N_x x N_x]
        Q   % Process noise covariance [N_x x N_x]
        H   % Measurement matrix [N_z x N_x]
        R   % Measurement noise covariance [N_z x N_z]

        % Detection model
        PD              % Detection probability
        PFA             % False alarm rate
        lambda_clutter  % Clutter density (returns per unit volume)

        % Gating
        validation_sigma_bounds  % Gate size in sigma units

        % Resampling
        ESS_threshold_percentage % Fraction of N_p below which to resample

        % Diagnostics
        current_ESS      % ESS computed before resampling (written each step)
        timestep_counter

        % History
        history  % Struct array: one entry per timestep

        % Flags
        debug
        store_full_history
    end

    methods

        function obj = PDA_PF_multi(x0_cell, N_particles, F, Q, H, R, varargin)
            % PDA_PF_MULTI  Constructor
            %
            % INPUTS:
            %   x0_cell     - {1 x N_t} cell of initial state estimates [N_x x 1]
            %   N_particles - Number of particles
            %   F, Q, H, R  - System / noise matrices
            %   varargin    - Name-value pairs (see below)
            %
            % OPTIONS:
            %   'PD'              Detection probability    (default 0.95)
            %   'PFA'             False alarm rate         (default 0.05)
            %   'LambdaClutter'   Clutter density          (default 2.5)
            %   'ValidationSigma' Gate size (sigma)        (default 5)
            %   'ESSThreshold'    Resampling trigger frac. (default 0.2)
            %   'Debug'           Verbose output           (default false)
            %   'UniformInit'     Uniform particle spread  (default false)
            %   'store_full_history' Save history          (default true)

            p = inputParser;
            addParameter(p, 'PD',              0.95,  @(x) x > 0 && x <= 1);
            addParameter(p, 'PFA',             0.05,  @(x) x >= 0);
            addParameter(p, 'LambdaClutter',   2.5,   @isnumeric);
            addParameter(p, 'ValidationSigma', 5,     @(x) x > 0);
            addParameter(p, 'ESSThreshold',    0.2,   @(x) x > 0 && x <= 1);
            addParameter(p, 'Debug',           false, @islogical);
            addParameter(p, 'UniformInit',     false, @islogical);
            addParameter(p, 'store_full_history', true, @islogical);
            % InitSigmaPos / InitSigmaVel: Gaussian particle init std-devs.
            % Must be comparable to sigma_z so that particles cover the likelihood peak.
            % Default 0.3 m / 0.5 m/s — reduce significantly when initializing from GT.
            addParameter(p, 'InitSigmaPos', 0.30, @(x) x >= 0);
            addParameter(p, 'InitSigmaVel', 0.50, @(x) x >= 0);
            parse(p, varargin{:});
            opt = p.Results;

            obj.N_t = numel(x0_cell);
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

            obj.timestep_counter = 0;
            obj.current_ESS      = NaN;

            % Initialise particles
            obj.particles = PDA_PF_multi.init_particles( ...
                N_particles, x0_cell, obj.N_x, obj.N_t, opt.UniformInit, ...
                opt.InitSigmaPos, opt.InitSigmaVel);

            % Pre-allocate history as empty struct array
            obj.history = struct('measurements', {}, 'estimate', {}, ...
                'covariance', {}, 'ESS', {});

            if obj.debug
                fprintf('PDA_PF_multi: N_p=%d, N_t=%d, N_x=%d\n', ...
                    obj.N_p, obj.N_t, obj.N_x);
            end
        end

        % ------------------------------------------------------------------
        function timestep(obj, z, varargin)
            % TIMESTEP  Run one predict-update cycle.
            %
            % INPUTS:
            %   z          - Measurements [N_z x N_meas]  (can be empty)
            %   varargin{1} - (optional) true state cell {1xN_t} for history

            true_state = [];
            if ~isempty(varargin), true_state = varargin{1}; end

            obj.prediction();

            valid_z = obj.validation(z);       % {1 x N_t} cell of valid meas per target
            obj.weight_update(z, valid_z);      % update weights using PDA likelihoods

            % Normalize
            total_w = sum(cellfun(@(p) p.weight, obj.particles));
            if total_w < eps
                % Weight collapse: reset to uniform
                for i = 1:obj.N_p
                    obj.particles{i}.weight = 1/obj.N_p;
                end
                total_w = 1;
                if obj.debug
                    warning('PDA_PF_multi: all weights collapsed at step %d — reset to uniform', ...
                        obj.timestep_counter+1);
                end
            end
            for i = 1:obj.N_p
                obj.particles{i}.weight = obj.particles{i}.weight / total_w;
            end

            % ESS before resampling
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
            % PREDICTION  Propagate each particle's target states through dynamics.
            sqrtQ = chol(obj.Q, 'lower');

            for i = 1:obj.N_p
                for t = 1:obj.N_t
                    noise = sqrtQ * randn(obj.N_x, 1);
                    obj.particles{i}.states(:, t) = ...
                        obj.F * obj.particles{i}.states(:, t) + noise;
                end
            end
        end

        % ------------------------------------------------------------------
        function valid_z = validation(obj, z)
            % VALIDATION  Gate measurements per target using empirical particle cov.
            %
            % OUTPUTS:
            %   valid_z - {1 x N_t} cell; each cell is [N_z x N_valid_j]

            valid_z = cell(1, obj.N_t);
            if isempty(z)
                for t = 1:obj.N_t, valid_z{t} = []; end
                return
            end

            w_vec = cellfun(@(p) p.weight, obj.particles);

            for t = 1:obj.N_t
                % Empirical weighted mean and covariance of target t
                states_t = zeros(obj.N_x, obj.N_p);
                for i = 1:obj.N_p
                    states_t(:, i) = obj.particles{i}.states(:, t);
                end
                x_emp = states_t * w_vec';
                diff  = states_t - x_emp;
                P_emp = diff * diag(w_vec) * diff';
                P_emp = 0.5*(P_emp + P_emp') + 1e-8*eye(obj.N_x);

                % Innovation covariance for gating
                S_t = obj.H * P_emp * obj.H' + obj.R;
                S_inv = inv(S_t);

                z_pred = obj.H * x_emp;
                gate_sq = obj.validation_sigma_bounds^2;

                mask = false(1, size(z,2));
                for j = 1:size(z, 2)
                    innov = z(:,j) - z_pred;
                    NIS   = innov' * S_inv * innov;
                    mask(j) = NIS < gate_sq;
                end
                valid_z{t} = z(:, mask);
            end
        end

        % ------------------------------------------------------------------
        function weight_update(obj, z, ~)
            % WEIGHT_UPDATE  Joint JPDA likelihood with exclusivity constraint (per Schulz 2001).
            %
            % For each particle i, compute the joint likelihood:
            %   Z^i = sum_{theta in Theta} prod_t L(t, theta(t))
            %
            % where Theta = set of feasible joint associations (each non-zero
            % measurement assigned to at most one target) and:
            %   L(t, 0)   = (1 - PD)                         (missed detection)
            %   L(t, m)   = PD * N(z_m ; H*x_t^i, R)        (m = 1..N_m)
            %
            % Weight update: w^i *= Z^i  (bootstrap SIR: importance weight = likelihood)

            if isempty(z)
                return
            end

            N_m = size(z, 2);

            % Precompute constants (avoids repeated det/inv inside loops)
            R_inv     = inv(obj.R);
            norm_const = sqrt((2*pi)^obj.N_z * det(obj.R));
            tensor_size = (N_m + 1) * ones(1, obj.N_t);
            num_hyp     = (N_m + 1)^obj.N_t;

            for i = 1:obj.N_p
                % L(t, 1)     = missed detection = (1-PD)
                % L(t, m+1)   = PD * N(z_m ; H*x_t^i, R)  for m = 1..N_m
                L = zeros(obj.N_t, N_m + 1);
                for t = 1:obj.N_t
                    z_pred    = obj.H * obj.particles{i}.states(:, t);
                    L(t, 1)   = 1 - obj.PD;  % missed detection
                    for m = 1:N_m
                        innov      = z(:, m) - z_pred;
                        exponent   = -0.5 * (innov' * R_inv * innov);
                        L(t, m+1)  = obj.PD * exp(exponent) / norm_const;
                    end
                end

                % Enumerate all (N_m+1)^N_t joint hypotheses; sum valid ones (exclusivity)
                Z_i = 0;
                for h = 1:num_hyp
                    sub_idx = cell(1, obj.N_t);
                    [sub_idx{:}] = ind2sub(tensor_size, h);
                    c_plus1 = cell2mat(sub_idx);   % 1-indexed into L (1=miss, 2..N_m+1 = meas)
                    c       = c_plus1 - 1;         % 0-indexed (0=miss, 1..N_m = measurement)

                    % Exclusivity: each non-zero measurement to at most one target
                    meas_used = c(c > 0);
                    if length(meas_used) ~= length(unique(meas_used))
                        continue  % Invalid — skip
                    end

                    hyp_weight = 1;
                    for t = 1:obj.N_t
                        hyp_weight = hyp_weight * L(t, c_plus1(t));
                    end
                    Z_i = Z_i + hyp_weight;
                end

                if Z_i < eps, Z_i = eps; end
                obj.particles{i}.weight = obj.particles{i}.weight * Z_i;
            end
        end

        % ------------------------------------------------------------------
        function resample(obj)
            % RESAMPLE  Systematic resampling when ESS < threshold.
            w_vec = cellfun(@(p) p.weight, obj.particles);
            ESS_thresh = obj.ESS_threshold_percentage * obj.N_p;

            if obj.current_ESS < ESS_thresh
                % Systematic resampling
                cumW  = cumsum(w_vec);
                cumW(end) = 1.0;        % Force exact 1 for floating-point safety
                step  = 1 / obj.N_p;
                start = rand() * step;
                positions = start + (0:obj.N_p-1) * step;  % Always exactly N_p elements

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
        function [x_est_cell, P_est_cell] = getGaussianEstimate(obj)
            % GETGAUSSIANESTIMATE  Weighted mean and covariance per target.
            %
            % OUTPUTS:
            %   x_est_cell - {1 x N_t} of [N_x x 1] weighted means
            %   P_est_cell - {1 x N_t} of [N_x x N_x] weighted covariances

            x_est_cell = cell(1, obj.N_t);
            P_est_cell = cell(1, obj.N_t);

            w_vec = cellfun(@(p) p.weight, obj.particles);

            for t = 1:obj.N_t
                states_t = zeros(obj.N_x, obj.N_p);
                for i = 1:obj.N_p
                    states_t(:, i) = obj.particles{i}.states(:, t);
                end

                x_est = states_t * w_vec';

                P_est = zeros(obj.N_x, obj.N_x);
                for i = 1:obj.N_p
                    d = states_t(:,i) - x_est;
                    P_est = P_est + w_vec(i) * (d * d');
                end
                P_est = 0.5*(P_est + P_est') + 1e-10*eye(obj.N_x);

                x_est_cell{t} = x_est;
                P_est_cell{t} = P_est;
            end
        end

        % ------------------------------------------------------------------
        function storeHistory(obj, measurements, true_state)
            k = obj.timestep_counter;
            [x_est, P_est] = obj.getGaussianEstimate();

            obj.history(k).measurements = measurements;
            obj.history(k).true_state   = true_state;
            obj.history(k).estimate     = x_est;
            obj.history(k).covariance   = P_est;
            obj.history(k).ESS          = obj.current_ESS;
        end

    end % methods

    methods (Static)

        function particles = init_particles(N_p, x0_cell, N_x, N_t, uniform_init, ...
                                              init_sigma_pos, init_sigma_vel)
            % INIT_PARTICLES  Initialise joint-state particle array.
            %
            % CRITICAL: init_sigma_pos must be comparable to sigma_z (measurement noise).
            % If pos_std >> sigma_z, the joint-state likelihood peak covers only a
            % fraction (sigma_z/pos_std)^(2*N_t) of particles — leads to weight collapse.

            if nargin < 6, init_sigma_pos = 0.30; end
            if nargin < 7, init_sigma_vel = 0.50; end

            particles = cell(1, N_p);

            if uniform_init
                % Uniform spread over scene bounds (mWidar typical scene)
                pos_lo  = [-2.0;  0.5]; pos_hi = [2.0; 4.0];
                vel_lo  = [-2.0; -2.0]; vel_hi = [2.0; 2.0];
                acc_lo  = [-2.0; -2.0]; acc_hi = [2.0; 2.0];

                for i = 1:N_p
                    states = zeros(N_x, N_t);
                    for t = 1:N_t
                        s = zeros(N_x, 1);
                        s(1:2) = pos_lo + rand(2,1) .* (pos_hi - pos_lo);
                        if N_x >= 4
                            s(3:4) = vel_lo + rand(2,1) .* (vel_hi - vel_lo);
                        end
                        if N_x >= 6
                            s(5:6) = acc_lo + rand(2,1) .* (acc_hi - acc_lo);
                        end
                        states(:, t) = s;
                    end
                    particles{i} = struct('states', states, 'weight', 1/N_p);
                end

            else
                % Gaussian spread around each target's initial state
                acc_std = 1.0;
                std_vec = ones(N_x, 1);
                std_vec(1:min(2,N_x)) = init_sigma_pos;
                if N_x >= 4, std_vec(3:4) = init_sigma_vel; end
                if N_x >= 6, std_vec(5:6) = acc_std; end

                for i = 1:N_p
                    states = zeros(N_x, N_t);
                    for t = 1:N_t
                        states(:, t) = x0_cell{t} + std_vec .* randn(N_x, 1);
                    end
                    particles{i} = struct('states', states, 'weight', 1/N_p);
                end
            end
        end

    end % static methods

end
