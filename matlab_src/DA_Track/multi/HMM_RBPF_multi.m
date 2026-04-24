% HMM_RBPF_multi.m
% Multi-Target Rao-Blackwellized Particle Filter with HMM Inner Filters
%
% DESCRIPTION:
%   Extends HMM_RBPF to multi-target tracking.  Each particle maintains
%   N_t independent HMM objects (one 128x128 occupancy grid per target)
%   and a joint association hypothesis vector [N_t x 1].
%
%   All inner HMMs share the same A_transition and pointlikelihood_image
%   lookup tables (the environment is the same for all targets).  Only
%   the ptarget_prob grid differs per-particle per-target.
%
%   ALGORITHM (per timestep):
%     1. PREDICTION  : call hmm.prediction() for every target in every particle
%     2. ASSOCIATION : sample joint hypothesis from optimal importance dist.
%                      Uses HMM normalization constants (dot products) instead
%                      of KF innovation likelihoods.  The measurement likelihood
%                      vectors are precomputed once per measurement (O(N_meas)
%                      calls to likelihoodLookup, not O(N_p * N_t * N_meas)).
%     3. UPDATE      : call hmm.measurement_update(z_assoc) per target per particle
%     4. WEIGHTING   : weight = product over targets of normalization constants
%     5. RESAMPLING  : systematic resampling with HMM.copyHMM() deep copies
%
% PARTICLE STRUCTURE:
%   particles{i}.hmms              - {1 x N_t}  HMM objects, one per target
%   particles{i}.associations      - [N_t x 1]  joint association (0=clutter)
%   particles{i}.weight            - scalar
%   particles{i}.association_history - [N_t x k] growing history
%
% PERFORMANCE NOTE:
%   likelihoodLookup is O(npx2) per call.  This class precomputes the
%   likelihood vector for each measurement once and reuses it across all
%   particles — see generateAssociations() for details.
%
% EXAMPLE:
%   load('precalc_imagegridHMMEmLike.mat',  'pointlikelihood_image');
%   load('precalc_imagegridHMMSTMn15.mat', 'A');
%
%   x0_cell = {[0.5; 2.0], [1.5; 2.5], [0.0; 1.5]};  % 3 targets, [x;y]
%   filt = HMM_RBPF_multi(x0_cell, 50, A, pointlikelihood_image);
%   for k = 1:T
%       filt.timestep(z_k);
%       [x_est, P_est] = filt.getGaussianEstimate();
%   end
%
% SEE ALSO: HMM, HMM_RBPF, KF_RBPF_multi, PDA_PF_multi, FilterHyperParams

classdef HMM_RBPF_multi < handle

    properties
        % Dimensions
        N_p     % Number of particles
        N_t     % Number of targets
        N_x     % State dim per target (always 2 for HMM: [x, y])
        N_z     % Measurement dim (always 2: [x, y])

        % Particle state
        particles   % {1 x N_p} cell of structs (see class header)

        % Shared HMM lookup tables (read-only across all particles/targets)
        A_transition            % [npx2 x npx2] state transition matrix
        pointlikelihood_image   % [npx2 x npx2] detection likelihood lookup

        % Detection model
        PD              % Detection probability
        PFA             % False alarm rate

        % Association strategy
        association_strategy  % 'uniform' | 'optimal'

        % Resampling
        ESS_threshold_percentage

        % Diagnostics
        current_ESS
        timestep_counter

        % History
        history
        store_full_history

        % Control flags
        debug
    end

    methods

        function obj = HMM_RBPF_multi(x0_cell, N_particles, A_transition, ...
                                       pointlikelihood_image, varargin)
            % HMM_RBPF_MULTI  Constructor
            %
            % INPUTS:
            %   x0_cell               - {1 x N_t} initial positions [2 x 1] per target
            %   N_particles           - Number of particles
            %   A_transition          - HMM transition matrix [npx2 x npx2]
            %   pointlikelihood_image - Detection likelihood lookup [npx2 x npx2]
            %
            % OPTIONS:
            %   'PD'                Detection probability       (default 0.95)
            %   'PFA'               False alarm probability     (default 0.05)
            %   'ESSThreshold'      Resampling fraction trigger (default 0.5)
            %   'AssociationStrategy' 'uniform' or 'optimal'   (default 'optimal')
            %   'Debug'             Verbose output              (default false)
            %   'UniformInit'       Uniform grid prior          (default false)
            %   'store_full_history' Save history               (default true)

            p = inputParser;
            addParameter(p, 'PD',                0.95,    @(x) x >= 0 && x <= 1);
            addParameter(p, 'PFA',               0.05,    @(x) x >= 0 && x <= 1);
            addParameter(p, 'ESSThreshold',      0.5,     @(x) x > 0 && x <= 1);
            addParameter(p, 'AssociationStrategy','optimal', ...
                @(x) ismember(x, {'uniform', 'optimal'}));
            addParameter(p, 'Debug',             false,   @islogical);
            addParameter(p, 'UniformInit',       false,   @islogical);
            addParameter(p, 'store_full_history',true,    @islogical);
            parse(p, varargin{:});
            opt = p.Results;

            obj.N_t  = numel(x0_cell);
            obj.N_p  = N_particles;
            obj.N_x  = 2;   % HMM is always 2D position
            obj.N_z  = 2;

            obj.A_transition          = A_transition;
            obj.pointlikelihood_image = pointlikelihood_image;

            obj.PD                       = opt.PD;
            obj.PFA                      = opt.PFA;
            obj.ESS_threshold_percentage = opt.ESSThreshold;
            obj.association_strategy     = opt.AssociationStrategy;
            obj.debug                    = opt.Debug;
            obj.store_full_history       = opt.store_full_history;

            obj.timestep_counter = 0;
            obj.current_ESS      = NaN;

            % Initialise particles
            obj.particles = HMM_RBPF_multi.init_particles( ...
                N_particles, x0_cell, A_transition, pointlikelihood_image, ...
                opt.UniformInit);

            obj.history = struct('measurements', {}, 'estimate', {}, ...
                'covariance', {}, 'ESS', {});

            if obj.debug
                fprintf('HMM_RBPF_multi: N_p=%d, N_t=%d\n', N_particles, obj.N_t);
                fprintf('  Grid size: %d cells\n', size(A_transition,1));
                fprintf('  PD=%.2f, PFA=%.2f, strategy=%s\n', ...
                    obj.PD, obj.PFA, obj.association_strategy);
            end
        end

        % ------------------------------------------------------------------
        function timestep(obj, z, varargin)
            % TIMESTEP  One predict-associate-update cycle.
            %
            % INPUTS:
            %   z          - [2 x N_meas] measurements ([] for missed detection)
            %   varargin{1}- (optional) true state cell {1xN_t} for history

            true_state = [];
            if ~isempty(varargin), true_state = varargin{1}; end

            obj.prediction();
            obj.generateAssociations(z);
            obj.measurement_update(z);

            % ESS before resampling
            w_vec = cellfun(@(p) p.weight, obj.particles);
            total_w = sum(w_vec);
            if total_w < eps
                for i = 1:obj.N_p
                    obj.particles{i}.weight = 1/obj.N_p;
                end
                w_vec = ones(1, obj.N_p) / obj.N_p;
                if obj.debug
                    warning('HMM_RBPF_multi: weight collapse at step %d', ...
                        obj.timestep_counter+1);
                end
            else
                for i = 1:obj.N_p
                    obj.particles{i}.weight = obj.particles{i}.weight / total_w;
                end
                w_vec = w_vec / total_w;
            end

            obj.current_ESS = 1 / sum(w_vec .^ 2);

            obj.resample();
            obj.timestep_counter = obj.timestep_counter + 1;

            if obj.store_full_history
                obj.storeHistory(z, true_state);
            end
        end

        % ------------------------------------------------------------------
        function prediction(obj)
            % PREDICTION  Propagate each particle's HMMs via the transition matrix.
            for i = 1:obj.N_p
                for t = 1:obj.N_t
                    obj.particles{i}.hmms{t}.prediction();
                end
            end
        end

        % ------------------------------------------------------------------
        function generateAssociations(obj, z)
            % GENERATEASSOCIATIONS  Sample joint association hypotheses.
            %
            % OPTIMIZATION: likelihoodLookup(z_m) depends only on z_m and the
            % shared pointlikelihood_image — not on any particle's state.
            % So we precompute the [npx2 x 1] likelihood vector for each
            % measurement once (using the first particle's first HMM), then
            % compute per-particle normalization constants via dot products.

            switch obj.association_strategy
                case 'optimal'
                    obj.generateAssociations_optimal(z);
                case 'uniform'
                    obj.generateAssociations_uniform(z);
                otherwise
                    obj.generateAssociations_optimal(z);
            end
        end

        % ------------------------------------------------------------------
        function generateAssociations_optimal(obj, z)
            % GENERATEASSOCIATIONS_OPTIMAL  Optimal importance distribution.
            %
            % Association weight for (target t, measurement m) in particle i:
            %   w(t, m) = PD * dot(ptarget_prob_t, L_m)   (m > 0 = real meas)
            %   w(t, 0) = (1 - PD)                        (clutter/miss)
            % Subject to exclusivity: same meas cannot be used for two targets.
            % (Same algorithm as KF_RBPF_multi but using HMM norm. constants.)

            if isempty(z)
                % No measurements — all targets assigned to clutter
                for i = 1:obj.N_p
                    obj.particles{i}.associations = zeros(obj.N_t, 1);
                    obj.particles{i}.association_history = ...
                        [obj.particles{i}.association_history, zeros(obj.N_t,1)];
                end
                return
            end

            N_meas = size(z, 2);
            p_clutter = 1 - obj.PD;

            % Precompute likelihood vectors for all measurements (shared across particles)
            ref_hmm = obj.particles{1}.hmms{1};  % any HMM — lookup table is shared
            L_vecs = zeros(size(obj.A_transition, 1), N_meas);
            for m = 1:N_meas
                L_vecs(:, m) = ref_hmm.likelihoodLookup(z(:, m));
            end

            % For each particle, compute per-target normalization constants
            % and sample joint association
            for i = 1:obj.N_p
                % target_weights(t, m) for m=1..N_meas, and target_weights(t, N_meas+1) = clutter
                target_weights = zeros(obj.N_t, N_meas + 1);

                for t = 1:obj.N_t
                    prob_t = obj.particles{i}.hmms{t}.ptarget_prob;  % [npx2 x 1]

                    for m = 1:N_meas
                        % Normalization constant: marginalizes over grid position
                        c_tm = full(prob_t)' * L_vecs(:, m);
                        target_weights(t, m) = obj.PD * c_tm;
                    end
                    target_weights(t, N_meas + 1) = p_clutter;  % missed detection weight
                end

                % Sample joint association using the same method as KF_RBPF_multi
                assoc_vec = HMM_RBPF_multi.sample_joint_association( ...
                    target_weights, obj.N_t, N_meas);

                obj.particles{i}.associations = assoc_vec;
                obj.particles{i}.association_history = ...
                    [obj.particles{i}.association_history, assoc_vec];
            end
        end

        % ------------------------------------------------------------------
        function generateAssociations_uniform(obj, z)
            % GENERATEASSOCIATIONS_UNIFORM  Independent uniform sampling.
            N_meas = size(z, 2);

            for i = 1:obj.N_p
                assoc_vec = zeros(obj.N_t, 1);
                for t = 1:obj.N_t
                    assoc_vec(t) = randi([0, N_meas]);
                end
                obj.particles{i}.associations = assoc_vec;
                obj.particles{i}.association_history = ...
                    [obj.particles{i}.association_history, assoc_vec];
            end
        end

        % ------------------------------------------------------------------
        function measurement_update(obj, z)
            % MEASUREMENT_UPDATE  Update each HMM with its sampled measurement.
            % Weight = product over targets of HMM normalization constants.

            if isempty(z)
                % Missed detection for all targets — weights stay at 1 (relative)
                for i = 1:obj.N_p
                    obj.particles{i}.weight = obj.particles{i}.weight;
                end
                return
            end

            N_meas = size(z, 2);
            ref_hmm = obj.particles{1}.hmms{1};
            L_vecs  = zeros(size(obj.A_transition, 1), N_meas);
            for m = 1:N_meas
                L_vecs(:, m) = ref_hmm.likelihoodLookup(z(:, m));
            end

            for i = 1:obj.N_p
                particle_weight = 1;

                for t = 1:obj.N_t
                    assoc_t = obj.particles{i}.associations(t);
                    prob_t  = obj.particles{i}.hmms{t}.ptarget_prob;

                    if assoc_t > 0 && assoc_t <= N_meas
                        % Associated to a real measurement
                        c_t = full(prob_t)' * L_vecs(:, assoc_t);
                        particle_weight = particle_weight * (obj.PD * c_t + eps);

                        % Update HMM (detection only — no magnitude)
                        obj.particles{i}.hmms{t}.measurement_update(z(:, assoc_t));

                    else
                        % Clutter / missed detection
                        particle_weight = particle_weight * (1 - obj.PD);
                        % HMM keeps its predicted distribution (no measurement update)
                    end
                end

                obj.particles{i}.weight = obj.particles{i}.weight * particle_weight;
            end
        end

        % ------------------------------------------------------------------
        function resample(obj)
            % RESAMPLE  Systematic resampling with deep HMM copies.
            w_vec = cellfun(@(p) p.weight, obj.particles);
            ESS_thresh = obj.ESS_threshold_percentage * obj.N_p;

            if obj.current_ESS < ESS_thresh
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
                    src = obj.particles{idx};

                    % Deep copy all HMM objects (critical — they are handle objects)
                    new_hmms = cell(1, obj.N_t);
                    for t = 1:obj.N_t
                        new_hmms{t} = HMM.copyHMM(src.hmms{t});
                    end

                    new_particles{i} = struct( ...
                        'hmms',               {new_hmms}, ...
                        'associations',       zeros(obj.N_t, 1), ...
                        'weight',             1 / obj.N_p, ...
                        'association_history',src.association_history);
                end
                obj.particles = new_particles;

                if obj.debug
                    fprintf('  HMM_RBPF_multi: resampled (ESS=%.1f)\n', obj.current_ESS);
                end
            end
        end

        % ------------------------------------------------------------------
        function [x_est_cell, P_est_cell] = getGaussianEstimate(obj)
            % GETGAUSSIANESTIMATE  Weighted mixture of per-particle HMM estimates.
            %
            % OUTPUTS:
            %   x_est_cell - {1 x N_t} of [2 x 1] position estimates
            %   P_est_cell - {1 x N_t} of [2 x 2] position covariances

            x_est_cell = cell(1, obj.N_t);
            P_est_cell = cell(1, obj.N_t);

            w_vec = cellfun(@(p) p.weight, obj.particles);

            for t = 1:obj.N_t
                x_mix = zeros(2, 1);
                P_mix = zeros(2, 2);

                % Weighted mean
                for i = 1:obj.N_p
                    [x_i, ~] = obj.particles{i}.hmms{t}.getGaussianEstimate();
                    x_mix = x_mix + w_vec(i) * x_i;
                end

                % Weighted covariance (mixture covariance formula)
                for i = 1:obj.N_p
                    [x_i, P_i] = obj.particles{i}.hmms{t}.getGaussianEstimate();
                    d = x_i - x_mix;
                    P_mix = P_mix + w_vec(i) * (P_i + d*d');
                end
                P_mix = 0.5*(P_mix + P_mix') + 1e-10*eye(2);

                x_est_cell{t} = x_mix;
                P_est_cell{t} = P_mix;
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

        function particles = init_particles(N_p, x0_cell, A_transition, ...
                                            pointlikelihood_image, uniform_init)
            % INIT_PARTICLES  Create particle array with N_t HMMs each.
            N_t = numel(x0_cell);
            particles = cell(1, N_p);

            for i = 1:N_p
                hmms = cell(1, N_t);
                for t = 1:N_t
                    x0_t = x0_cell{t};

                    if uniform_init
                        hmms{t} = HMM([], A_transition, pointlikelihood_image);
                    else
                        % Gaussian init around provided position
                        hmms{t} = HMM(x0_t, A_transition, pointlikelihood_image);
                    end
                end

                particles{i} = struct( ...
                    'hmms',               {hmms}, ...
                    'associations',       zeros(N_t, 1), ...
                    'weight',             1/N_p, ...
                    'association_history', []);
            end
        end

        % ------------------------------------------------------------------
        function assoc_vec = sample_joint_association(target_weights, N_t, N_meas)
            % SAMPLE_JOINT_ASSOCIATION  Sample exclusivity-constrained joint assoc.
            %
            % Identical logic to KF_RBPF_multi.generateAssociations_optimalimportancedist
            % but factored out as a static helper for reuse.
            %
            % INPUTS:
            %   target_weights - [N_t x (N_meas+1)] weight matrix
            %                    Column N_meas+1 is the clutter/miss weight
            %   N_t            - Number of targets
            %   N_meas         - Number of measurements

            if N_t == 1
                w = target_weights(1, :);
                w = w / (sum(w) + eps);
                cumW = cumsum(w);
                cumW(end) = 1.0;  % Clamp for floating-point safety
                idx  = find(cumW >= rand(), 1);
                if isempty(idx), idx = numel(cumW); end
                assoc_vec = zeros(1, 1);
                if idx <= N_meas, assoc_vec(1) = idx; end

            elseif N_t == 2
                assoc_matrix = zeros(N_meas+1, N_meas+1);
                for c1 = 0:N_meas
                    for c2 = 0:N_meas
                        if c1 > 0 && c2 > 0 && c1 == c2
                            assoc_matrix(c1+1, c2+1) = 0;  % exclusivity
                        else
                            assoc_matrix(c1+1, c2+1) = ...
                                target_weights(1, c1+1) * target_weights(2, c2+1);
                        end
                    end
                end
                flat = assoc_matrix(:);
                flat = flat / (sum(flat) + eps);
                cumW = cumsum(flat);
                cumW(end) = 1.0;  % Clamp for floating-point safety
                idx  = find(cumW >= rand(), 1);
                if isempty(idx), idx = numel(cumW); end
                [r, c] = ind2sub(size(assoc_matrix), idx);
                assoc_vec = [r-1; c-1];

            else
                % General N_t: enumerate all (M+1)^N_t hypotheses
                tensor_size = (N_meas+1) * ones(1, N_t);
                n_hyp = (N_meas+1)^N_t;
                hyp_w   = zeros(1, n_hyp);
                hyp_a   = zeros(n_hyp, N_t);

                for h = 1:n_hyp
                    sub = cell(1, N_t);
                    [sub{:}] = ind2sub(tensor_size, h);
                    a = cell2mat(sub) - 1;   % 0-indexed
                    hyp_a(h, :) = a;

                    meas_a = a(a > 0);
                    if length(meas_a) ~= length(unique(meas_a))
                        continue  % exclusivity violated
                    end
                    w = 1;
                    for t = 1:N_t
                        w = w * target_weights(t, a(t)+1);
                    end
                    hyp_w(h) = w;
                end

                hyp_w = hyp_w / (sum(hyp_w) + eps);
                cumW = cumsum(hyp_w);
                cumW(end) = 1.0;  % Clamp for floating-point safety
                idx = find(cumW >= rand(), 1);
                if isempty(idx), idx = n_hyp; end
                assoc_vec = hyp_a(idx, :)';
            end
        end

    end % static methods

end
