% HMM_RBPF.m
% Rao-Blackwellized Particle Filter with HMM Inner Filter for Data Association
%
% DESCRIPTION:
%   Implements RBPF for single target tracking with data association.
%   Each particle represents a discrete association hypothesis, with
%   continuous state tracked by an embedded grid-based HMM.
%
%   RBPF exploits conditional independence:
%     - Particles handle DISCRETE state: association hypothesis
%     - HMMs handle CONTINUOUS state: position probability over 128x128 grid
%
%   ALGORITHM (per timestep):
%     1. PREDICTION: Propagate each particle's HMM via transition matrix
%     2. ASSOCIATION: Sample/enumerate association hypotheses per particle
%     3. UPDATE: Update each particle's HMM with associated measurement
%     4. WEIGHTING: Compute weights from HMM posterior likelihood
%     5. RESAMPLING: Resample particles if ESS below threshold
%
% PARTICLE STRUCTURE:
%   particles{i} = struct with fields:
%     .association  - Integer: which measurement? (0 = clutter/no detection)
%     .hmm          - HMM object: grid-based filter for continuous state
%     .weight       - Scalar: particle weight
%
% PROPERTIES:
%   N_p              - Number of particles
%   N_x              - State dimension (always 2: [x, y])
%   N_z              - Measurement dimension (always 2: [x, y])
%   particles        - Cell array of particle structs {1 x N_p}
%
%   % HMM Model (for creating inner HMM filters)
%   A_transition          - State transition matrix [npx2 x npx2]
%   pointlikelihood_image - Detection likelihood lookup [npx2 x npx2]
%   pointlikelihood_mag   - Magnitude likelihood table [npx2 x 2] (optional)
%   magnitude_weight      - Weight on magnitude likelihood (default 1.0)
%
%   % Detection Model
%   PD               - Detection probability
%   PFA              - False alarm probability
%
%   % Control Flags
%   debug            - Enable debug output
%
% METHODS:
%   HMM_RBPF(x0, N_particles, A_transition, pointlikelihood_image, ...)
%   timestep(z, true_state, true_meas_flag, z_mag)
%   prediction()
%   generateAssociations(z)
%   measurement_update(z, z_mag)
%   resample()
%   getGaussianEstimate()
%
% EXAMPLE USAGE:
%   % Load precomputed HMM tables
%   load('precalc_imagegridHMMEmLike.mat', 'pointlikelihood_image');
%   load('precalc_imagegridHMMSTMn15.mat', 'A');
%
%   % Create RBPF
%   x0 = [0; 2];  % Initial position guess [x; y]
%   rbpf = HMM_RBPF(x0, 50, A, pointlikelihood_image, 'Debug', true);
%
%   % Tracking loop
%   for k = 1:num_steps
%       measurements = get_measurements(k);  % [2 x N_meas]
%       rbpf.timestep(measurements);
%       [x_est, P_est] = rbpf.getGaussianEstimate();
%   end
%
% SEE ALSO:
%   HMM, KF_RBPF, PDA_HMM, PDA_PF
%
% REFERENCES:
%   [1] Doucet et al. "On Sequential Monte Carlo Sampling Methods for
%       Bayesian Filtering" (1998)
%   [2] Sarkka & Svensson "Bayesian Filtering and Smoothing" (2023)
%       Chapter 11: Rao-Blackwellized Particle Filtering

classdef HMM_RBPF < DA_Filter
    % HMM_RBPF  Rao-Blackwellized Particle Filter with HMM inner filters.
    %
    % Inherits from DA_Filter.  See DA_Filter for the full interface contract,
    % including the storeHistory / history pattern for offline analysis.

    properties
        % -----------------------------------------------------------------
        % Filter dimensions
        % -----------------------------------------------------------------
        N_p     % Number of particles
        N_x     % State dimension (always 2 for HMM-based: [x, y])
        N_z     % Measurement dimension (always 2: [x, y])

        % -----------------------------------------------------------------
        % Particle filter state
        % -----------------------------------------------------------------
        % particles - Cell array of particle structs {1 x N_p}.
        %   Each struct contains:
        %     .association        - Integer measurement index (0 = clutter)
        %     .hmm                - HMM handle: inner grid-based filter
        %     .weight             - Scalar importance weight
        %     .association_history- [1 x k] past association indices
        %     .state_trajectory   - [2 x k] past MMSE position estimates
        particles

        % -----------------------------------------------------------------
        % HMM model matrices (shared / read-only across all inner HMMs)
        % -----------------------------------------------------------------
        A_transition            % Transition matrix              [npx2 x npx2]
        pointlikelihood_image   % Detection likelihood lookup    [npx2 x npx2]
        pointlikelihood_mag     % Magnitude likelihood table     [npx2 x 2] ([] = disabled)
        magnitude_weight        % Scalar weight on magnitude likelihood

        % -----------------------------------------------------------------
        % Detection model
        % -----------------------------------------------------------------
        PD  % Detection probability  (default: 0.95)
        PFA % False alarm probability (default: 0.05)

        % -----------------------------------------------------------------
        % Abstract properties required by DA_Filter
        % -----------------------------------------------------------------
        debug = false               % Enable verbose debug output
        DynamicPlot = false         % Enable real-time step-by-step visualization
        dynamic_figure_handle = []  % Figure handle used by DynamicPlot

        % -----------------------------------------------------------------
        % Association strategy
        % -----------------------------------------------------------------
        % association_strategy - How associations are sampled each timestep.
        %   'uniform'  : Draw uniformly from {0, 1, ..., N_meas}
        %   'optimal'  : Sample from optimal importance distribution
        %   'likelihood': Alias for 'optimal'
        association_strategy

        % -----------------------------------------------------------------
        % Resampling
        % -----------------------------------------------------------------
        ESS_threshold_percentage    % Fraction of N_p below which resampling triggers

    end

    methods

        function obj = HMM_RBPF(x0, N_particles, A_transition, pointlikelihood_image, varargin)
            % HMM_RBPF Constructor for HMM-based Rao-Blackwellized Particle Filter
            %
            % INPUTS:
            %   x0                    - Initial position [2 x 1] or [] for uniform
            %   N_particles           - Number of particles
            %   A_transition          - HMM transition matrix [npx2 x npx2]
            %   pointlikelihood_image - Detection likelihood lookup [npx2 x npx2]
            %
            % OPTIONAL NAME-VALUE PAIRS:
            %   'Debug'              - true/false (default false)
            %   'PD'                 - Detection probability (default 0.95)
            %   'PFA'                - False alarm probability (default 0.05)
            %   'ESSThreshold'       - Resampling threshold (default 0.5)
            %   'AssociationStrategy'- 'uniform', 'optimal', or 'likelihood'
            %   'PointlikelihoodMag' - [npx2 x 2] magnitude likelihood table
            %   'MagnitudeWeight'    - Scalar weight on magnitude likelihood
            %   'UniformInit'        - true/false uniform prior for all particles

            % Parse optional arguments
            p = inputParser;
            addParameter(p, 'Debug', false, @islogical);
            addParameter(p, 'PD', 0.95, @(x) x >= 0 && x <= 1);
            addParameter(p, 'PFA', 0.05, @(x) x >= 0 && x <= 1);
            addParameter(p, 'ESSThreshold', 0.5, @(x) x > 0 && x <= 1);
            addParameter(p, 'AssociationStrategy', 'optimal', @(x) ismember(x, {'uniform', 'optimal', 'likelihood'}));
            addParameter(p, 'UniformInit', false, @islogical);
            addParameter(p, 'PointlikelihoodMag', [], @(x) isempty(x) || isnumeric(x));
            addParameter(p, 'MagnitudeWeight', 1.0, @(x) isnumeric(x) && isscalar(x) && x > 0);
            parse(p, varargin{:});

            obj.debug                    = p.Results.Debug;
            obj.PD                       = p.Results.PD;
            obj.PFA                      = p.Results.PFA;
            obj.ESS_threshold_percentage = p.Results.ESSThreshold;
            obj.association_strategy     = p.Results.AssociationStrategy;

            % Store HMM model
            obj.A_transition          = A_transition;
            obj.pointlikelihood_image = pointlikelihood_image;
            obj.pointlikelihood_mag   = p.Results.PointlikelihoodMag;
            obj.magnitude_weight      = p.Results.MagnitudeWeight;

            % HMM is always 2D position
            obj.N_p = N_particles;
            obj.N_x = 2;
            obj.N_z = 2;

            % Initialize particles
            uniform_init = p.Results.UniformInit;
            obj.particles = HMM_RBPF.initialize_particles( ...
                N_particles, x0, A_transition, pointlikelihood_image, ...
                obj.pointlikelihood_mag, obj.magnitude_weight, ...
                uniform_init, obj.debug);

            % Initialize timestep counter.
            % obj.history is initialised to struct([]) by DA_Filter.
            obj.timestep_counter = 0;

            if obj.debug
                fprintf('HMM_RBPF initialized with %d particles.\n', N_particles);
                fprintf('  State dimension: %d (position only)\n', obj.N_x);
                fprintf('  Measurement dimension: %d\n', obj.N_z);
                fprintf('  Association strategy: %s\n', obj.association_strategy);
                fprintf('  PD = %.2f, PFA = %.2f\n', obj.PD, obj.PFA);
                if ~isempty(obj.pointlikelihood_mag)
                    fprintf('  Hybrid likelihood: ENABLED (weight=%.3f)\n', obj.magnitude_weight);
                end
            end
        end

        function timestep(obj, z, varargin)
            % TIMESTEP  Execute one HMM-RBPF predict-update cycle.
            %
            % SYNTAX:
            %   obj.timestep(z)
            %   obj.timestep(z, true_state)
            %   obj.timestep(z, true_state, z_mag)
            %
            % INPUTS:
            %   z          - Measurements [2 x N_measurements].
            %                Pass [] for a missed-detection step.
            %   true_state - (optional) Ground-truth state [N_x x 1].
            %                Used only for dynamic visualisation.
            %   z_mag      - (optional) Raw signal frame [128 x 128] for
            %                hybrid (detection × magnitude) likelihood.
            %
            % NOTE:
            %   History is NOT accumulated here.  Call storeHistory(z)
            %   after this method when running in test/analysis mode.

            z_mag = [];
            if nargin > 3, z_mag = varargin{2}; end

            % ----------------------------------------------------------
            % STEP 1: Prediction
            obj.prediction();

            % ----------------------------------------------------------
            % STEP 2: Association generation
            obj.generateAssociations(z);

            % ----------------------------------------------------------
            % STEP 3: Measurement update + weight computation
            obj.measurement_update(z, z_mag);

            % ----------------------------------------------------------
            % STEP 4: Capture ESS BEFORE resampling into obj.current_ESS
            weights = zeros(1, obj.N_p);
            for i = 1:obj.N_p
                weights(i) = obj.particles{i}.weight;
            end
            obj.current_ESS = 1 / sum(weights .^ 2);

            % ----------------------------------------------------------
            % STEP 5: Resampling
            obj.resample();

            % Advance counter (used as index into obj.history by storeHistory)
            obj.timestep_counter = obj.timestep_counter + 1;
        end

        function storeHistory(obj, measurements, varargin)
            % STOREHISTORY  Snapshot full HMM-RBPF state into obj.history.
            %
            % SYNTAX:
            %   obj.storeHistory(measurements)
            %   obj.storeHistory(measurements, true_state)
            %
            % INPUTS:
            %   measurements - Raw measurements from this timestep [2 x N_m].
            %   true_state   - (optional) Ground-truth state [N_x x 1].
            %                  Pass [] or omit when GT is unavailable.
            %
            % DESCRIPTION:
            %   Appends one struct to obj.history at index obj.timestep_counter.
            %
            %   Fields always written (DA_Filter contract):
            %     .x_est        [2 x 1]     - Gaussian mean estimate
            %     .P_est        [2 x 2]     - Gaussian covariance
            %     .measurements [2 x N_m]   - Raw measurements
            %     .true_state   [N_x x 1]   - GT state ([] if unavailable)
            %     .timestep_num scalar       - Current timestep index
            %     .ESS          scalar       - ESS before resampling
            %     .particle_weights [1 x N_p]- Normalised particle weights
            %     .particle_associations [1 x N_p] - Association indices
            %     .particle_entropies [1 x N_p]    - Per-particle HMM entropy
            %
            %   Additional fields when obj.store_full_history == true:
            %     .particle_states       [2 x N_p]   - MMSE position per particle
            %     .particle_assoc_hist   {1 x N_p}   - Full association histories
            %     .particle_trajectories {1 x N_p}   - MMSE position trajectories
            %     .particle_hmm_grids    {1 x N_p}   - Full ptarget_prob per particle
            %                                          (16384 x 1 per particle — large!)
            %
            % See also timestep, getGaussianEstimate, store_full_history

            % Parse optional ground-truth argument
            true_state = [];
            if nargin > 2 && ~isempty(varargin{1})
                true_state = varargin{1};
            end

            k = obj.timestep_counter;

            % ----------------------------------------------------------
            % Lightweight fields — always stored
            % ----------------------------------------------------------
            [x_est, P_est] = obj.getGaussianEstimate();

            particle_weights      = zeros(1, obj.N_p);
            particle_associations = zeros(1, obj.N_p);
            particle_entropies    = zeros(1, obj.N_p);
            for i = 1:obj.N_p
                particle_weights(i)      = obj.particles{i}.weight;
                particle_associations(i) = obj.particles{i}.association;
                particle_entropies(i)    = obj.particles{i}.hmm.getEntropy();
            end

            % DA_Filter contract fields
            obj.history(k).x_est                 = x_est;
            obj.history(k).P_est                 = P_est;
            obj.history(k).measurements          = measurements;
            obj.history(k).true_state            = true_state;
            obj.history(k).timestep_num          = k;
            obj.history(k).ESS                   = obj.current_ESS;
            obj.history(k).particle_weights      = particle_weights;
            obj.history(k).particle_associations = particle_associations;
            obj.history(k).particle_entropies    = particle_entropies;

            % ----------------------------------------------------------
            % Full-fidelity fields — only when store_full_history == true
            % ----------------------------------------------------------
            if obj.store_full_history
                particle_states    = zeros(obj.N_x, obj.N_p);
                particle_assoc_hist = cell(1, obj.N_p);
                particle_traj      = cell(1, obj.N_p);
                particle_grids     = cell(1, obj.N_p);

                for i = 1:obj.N_p
                    [x_i, ~]              = obj.particles{i}.hmm.getGaussianEstimate();
                    particle_states(:, i) = x_i;
                    particle_assoc_hist{i}= obj.particles{i}.association_history;
                    particle_traj{i}      = obj.particles{i}.state_trajectory;
                    % Full probability grid per particle (npx2 x 1 each)
                    particle_grids{i}     = full(obj.particles{i}.hmm.ptarget_prob);
                end

                obj.history(k).particle_states       = particle_states;
                obj.history(k).particle_assoc_hist   = particle_assoc_hist;
                obj.history(k).particle_trajectories = particle_traj;
                obj.history(k).particle_hmm_grids    = particle_grids;
            end
        end

        function prediction(obj)
            % PREDICTION Propagate all particle HMMs through transition model

            for i = 1:obj.N_p
                obj.particles{i}.hmm.prediction();
            end

            if obj.debug
                fprintf('Prediction step completed for all %d particles.\n', obj.N_p);
            end
        end

        function generateAssociations(obj, z)
            % GENERATEASSOCIATIONS Sample association hypotheses for particles

            switch obj.association_strategy
                case 'uniform'
                    obj.generateAssociations_uniform(z);
                case 'optimal'
                    obj.generateAssociations_optimal(z);
                case 'likelihood'
                    obj.generateAssociations_likelihood(z);
                otherwise
                    error('Unknown association strategy: %s', obj.association_strategy);
            end
        end

        function measurement_update(obj, z, z_mag)
            % MEASUREMENT_UPDATE Update particle HMMs and compute weights
            %
            % INPUTS:
            %   z     - Measurements [2 x N_measurements]
            %   z_mag - (optional) Signal frame for magnitude likelihood
            %
            % WEIGHT UPDATE:
            %   OIS strategy ('optimal' / 'likelihood'):
            %     w_k^i = w_{k-1}^i * Z_k^i
            %     where Z_k^i was stored in ois_weight_update during generateAssociations.
            %     Same pattern as KF_RBPF.
            %
            %   Uniform strategy:
            %     w_k^i = w_{k-1}^i * PD * marginal_likelihood  (detection)
            %     w_k^i = w_{k-1}^i * (1 - PD)                  (missed)

            if nargin < 3, z_mag = []; end

            use_ois = strcmp(obj.association_strategy, 'optimal') || ...
                      strcmp(obj.association_strategy, 'likelihood');

            for i = 1:obj.N_p
                assoc  = obj.particles{i}.association;
                w_prev = obj.particles{i}.weight;

                if assoc > 0
                    % Detection hypothesis: update HMM with associated measurement
                    z_i = z(:, assoc);
                    obj.particles{i}.hmm.measurement_update(z_i, z_mag);

                    if use_ois
                        % OIS: multiply prior weight by marginal likelihood sum Z_k^i
                        obj.particles{i}.weight = w_prev * obj.particles{i}.ois_weight_update;
                    else
                        % Uniform proposal: explicit marginal likelihood ratio
                        L     = obj.particles{i}.hmm.likelihood_prob;
                        prior = obj.particles{i}.hmm.prior_prob;
                        marginal_likelihood = max(full(sum(L .* prior)), 1e-300);
                        obj.particles{i}.weight = w_prev * obj.PD * marginal_likelihood;
                    end

                else
                    % Missed detection: set posterior = prior (no measurement update)
                    obj.particles{i}.hmm.posterior_prob  = obj.particles{i}.hmm.prior_prob;
                    obj.particles{i}.hmm.ptarget_prob    = obj.particles{i}.hmm.prior_prob;
                    obj.particles{i}.hmm.likelihood_prob = ones(obj.particles{i}.hmm.npx2, 1);

                    if use_ois
                        obj.particles{i}.weight = w_prev * obj.particles{i}.ois_weight_update;
                    else
                        obj.particles{i}.weight = w_prev * (1 - obj.PD);
                    end
                end

                % Store trajectory: append current MMSE estimate
                [x_est_i, ~] = obj.particles{i}.hmm.getGaussianEstimate();
                obj.particles{i}.state_trajectory = [obj.particles{i}.state_trajectory, x_est_i];
            end

            % Normalize weights
            total_weight = sum(cellfun(@(p) p.weight, obj.particles));
            if total_weight > 0
                for i = 1:obj.N_p
                    obj.particles{i}.weight = obj.particles{i}.weight / total_weight;
                end
            else
                warning('HMM_RBPF:ZeroWeights', 'All particle weights are zero. Setting uniform.');
                for i = 1:obj.N_p
                    obj.particles{i}.weight = 1 / obj.N_p;
                end
            end
        end

        function generateAssociations_uniform(obj, z)
            % GENERATEASSOCIATIONS_UNIFORM Uniform random association sampling
            N_measurements = size(z, 2);

            for i = 1:obj.N_p
                assoc = randi([0, N_measurements]);
                obj.particles{i}.association = assoc;
                obj.particles{i}.association_history = [obj.particles{i}.association_history, assoc];
            end
        end

        function generateAssociations_optimal(obj, z)
            % GENERATEASSOCIATIONS_OPTIMAL Optimal importance distribution sampling
            %
            % For the HMM, the marginal likelihood of measurement j given particle i:
            %   p(z_j | particle i) = sum_x [ L(z_j | x) * prior_i(x) ]
            %
            % Optimal importance distribution:
            %   q(c_k = j) proportional to PD * p(z_j | particle_i)  for j > 0
            %   q(c_k = 0) proportional to (1 - PD)                  for missed
            %
            % The unnormalized sum Z_k^i = sum_j association_weights(j) is the
            % marginal likelihood p(y_k | y_{1:k-1}, c_{1:k-1}^i) and is stored
            % in particle.ois_weight_update so measurement_update can apply:
            %   w_k^i = w_{k-1}^i * Z_k^i   (same as KF_RBPF OIS pattern)

            N_measurements = size(z, 2);
            p_missed = (1 - obj.PD);

            for i = 1:obj.N_p
                prior_i = obj.particles{i}.hmm.prior_prob;

                association_weights = zeros(1, N_measurements + 1);

                for j = 1:N_measurements
                    % Compute likelihood grid for measurement j
                    L_j = obj.particles{i}.hmm.likelihoodLookup(z(:, j));

                    % Marginal likelihood: sum( L(z_j|x) * prior(x) )
                    % Scale by npx2 to unnormalise: pointlikelihood_image rows
                    % sum to ~1/npx2, so the raw marginal is O(1/npx2) and
                    % cannot compete with p_missed = (1-PD) ≈ 0.05 without
                    % this correction.
                    marginal_j = full(sum(L_j .* prior_i)) * obj.particles{i}.hmm.npx2;
                    association_weights(j) = obj.PD * marginal_j;
                end

                % Missed detection
                association_weights(end) = p_missed;

                % Store Z_k^i BEFORE normalizing — this is the OIS incremental
                % weight factor p(y_k | y_{1:k-1}, c_{1:k-1}^i)
                Z_k_i = sum(association_weights);
                obj.particles{i}.ois_weight_update = max(Z_k_i, 1e-300);

                % Normalize to form proper sampling distribution
                if Z_k_i > 0
                    association_weights = association_weights / Z_k_i;
                else
                    % Fallback to uniform if all likelihoods are zero
                    association_weights = ones(1, N_measurements + 1) / (N_measurements + 1);
                end

                % Sample via inverse CDF
                cumulative = cumsum(association_weights);
                r = rand();
                sampled_idx = find(cumulative >= r, 1);

                if sampled_idx <= N_measurements
                    assoc = sampled_idx;
                else
                    assoc = 0;
                end

                obj.particles{i}.association = assoc;
                obj.particles{i}.association_history = [obj.particles{i}.association_history, assoc];
            end

        end

        function generateAssociations_likelihood(obj, z)
            % GENERATEASSOCIATIONS_LIKELIHOOD Likelihood-based sampling
            % (Delegates to optimal importance distribution)
            if obj.debug
                fprintf('Likelihood-based association: using optimal distribution.\n');
            end
            obj.generateAssociations_optimal(z);
        end

        function resample(obj)
            % RESAMPLE Systematic resampling of particles
            %
            % CRITICAL: Uses HMM.copyHMM() for deep copy of HMM handle objects.

            weights = cellfun(@(p) p.weight, obj.particles);
            ESS = 1 / sum(weights .^ 2);
            ESS_threshold = obj.ESS_threshold_percentage * obj.N_p;

            if ESS < ESS_threshold
                % Systematic resampling
                cumulative_weights = cumsum(weights);
                step = 1 / obj.N_p;
                start = rand * step;
                positions = start:step:1;

                new_particles = cell(1, obj.N_p);
                index = 1;

                for i = 1:obj.N_p
                    while positions(i) > cumulative_weights(index)
                        index = index + 1;
                    end

                    % Deep copy particle (CRITICAL: must use copyHMM for handle class)
                    new_hmm = HMM.copyHMM(obj.particles{index}.hmm);
                    new_particles{i} = struct( ...
                        'association', obj.particles{index}.association, ...
                        'hmm', new_hmm, ...
                        'weight', 1 / obj.N_p, ...
                        'ois_weight_update', obj.particles{index}.ois_weight_update, ...
                        'association_history', obj.particles{index}.association_history, ...
                        'state_trajectory', obj.particles{index}.state_trajectory);
                end

                obj.particles = new_particles;

                if obj.debug
                    fprintf('Resampling performed. ESS was: %.2f (threshold: %.2f)\n', ESS, ESS_threshold);
                end
            else
                if obj.debug
                    fprintf('No resampling needed. ESS: %.2f (threshold: %.2f)\n', ESS, ESS_threshold);
                end
            end
        end

        function [x_est, P_est] = getGaussianEstimate(obj)
            % GETGAUSSIANESTIMATE Weighted MMSE position estimate (mixture mean)
            %
            % OUTPUTS:
            %   x_est - Weighted mean position [2 x 1]
            %   P_est - Weighted covariance [2 x 2]
            %
            % Each particle's HMM gives a Gaussian estimate. The RBPF estimate
            % is the mixture: x_est = sum_i w_i * x_i
            %   P_est = sum_i w_i * (P_i + (x_i - x_est)*(x_i - x_est)')

            x_est = zeros(obj.N_x, 1);
            P_est = zeros(obj.N_x, obj.N_x);

            % First pass: weighted mean
            for i = 1:obj.N_p
                w_i = obj.particles{i}.weight;
                [x_i, ~] = obj.particles{i}.hmm.getGaussianEstimate();
                x_est = x_est + w_i * x_i;
            end

            % Second pass: weighted covariance (Gaussian mixture)
            for i = 1:obj.N_p
                w_i = obj.particles{i}.weight;
                [x_i, P_i] = obj.particles{i}.hmm.getGaussianEstimate();
                d = x_i - x_est;
                P_est = P_est + w_i * (P_i + d * d');
            end
        end

        function grid_2d = getMixtureGrid(obj)
            % GETMIXTUREGRID Weighted mixture of all particle HMM grids [128x128]
            %
            % Returns the particle-weighted sum of per-particle HMM posteriors,
            % reshaped to [grid_size x grid_size].  Use for per-step diagnostics.
            npx2    = obj.particles{1}.hmm.npx2;
            npx     = obj.particles{1}.hmm.grid_size;
            mixture = zeros(npx2, 1);
            for i = 1:obj.N_p
                mixture = mixture + obj.particles{i}.weight * full(obj.particles{i}.hmm.ptarget_prob);
            end
            grid_2d = reshape(mixture, [npx, npx]);
        end

        function [x_map, P_est] = getMAPEstimate(obj)
            % GETMAPESTIMATE Weighted MAP position estimate from mixture grid
            %
            % Builds the weighted mixture distribution over the spatial grid,
            % then returns the grid cell with maximum probability.
            %
            % OUTPUTS:
            %   x_map - MAP position [2 x 1]
            %   P_est - Covariance from MMSE estimate (for compatibility)

            % Build weighted mixture grid
            npx2   = obj.particles{1}.hmm.npx2;
            pxyvec = obj.particles{1}.hmm.pxyvec;
            mixture = zeros(npx2, 1);
            for i = 1:obj.N_p
                mixture = mixture + obj.particles{i}.weight * ...
                          full(obj.particles{i}.hmm.ptarget_prob);
            end

            % MAP: argmax of mixture grid
            [~, idx] = max(mixture);
            x_map = pxyvec(idx, :)';

            % Covariance from MMSE for compatibility
            [~, P_est] = obj.getGaussianEstimate();
        end

        function visualize(obj, varargin)
            % VISUALIZE  Render current HMM-RBPF state (post-hoc; see below).
            %
            % DESCRIPTION:
            %   Per-step real-time visualisation is not implemented for
            %   HMM_RBPF because the inner HMM grids are best inspected
            %   offline after a full run.  Use visualize_RBPFHMM_history()
            %   with the accumulated obj.history struct array instead:
            %
            %     visualize_RBPFHMM_history(filter, 'Animate', true, ...);
            %
            % See also storeHistory, visualize_RBPFHMM_history

            if obj.debug
                fprintf('[HMM_RBPF] visualize() called — use visualize_RBPFHMM_history() for post-hoc plots.\n');
            end
        end

        function setDetectionModel(obj, PD, PFA)
            % SETDETECTIONMODEL Update detection model parameters
            if PD <= 0 || PD > 1
                error('HMM_RBPF:InvalidPD', 'PD must be in (0, 1]');
            end
            if PFA < 0 || PFA >= 1
                error('HMM_RBPF:InvalidPFA', 'PFA must be in [0, 1)');
            end
            obj.PD = PD;
            obj.PFA = PFA;
            if obj.debug
                fprintf('[DETECTION MODEL] Updated: PD=%.3f, PFA=%.3f\n', obj.PD, obj.PFA);
            end
        end

    end

    methods (Static)

        function particles = initialize_particles(num_particles, x0, A_transition, ...
                pointlikelihood_image, pointlikelihood_mag, magnitude_weight, ...
                uniform_init, debug_flag)
            % INITIALIZE_PARTICLES Create particle array with HMM inner filters
            %
            % INPUTS:
            %   num_particles         - Number of particles
            %   x0                    - Initial position [2x1] or [] for uniform
            %   A_transition          - HMM transition matrix
            %   pointlikelihood_image - Detection likelihood lookup
            %   pointlikelihood_mag   - (optional) Magnitude likelihood table
            %   magnitude_weight      - (optional) Weight for magnitude likelihood
            %   uniform_init          - true = uniform prior for all particles
            %   debug_flag            - Enable debug output in inner HMMs

            if nargin < 5, pointlikelihood_mag = []; end
            if nargin < 6, magnitude_weight = 1.0; end
            if nargin < 7, uniform_init = false; end
            if nargin < 8, debug_flag = false; end

            particles = cell(1, num_particles);

            % Build common HMM constructor arguments
            hmm_opts = {'Debug', false, ...
                        'PointlikelihoodMag', pointlikelihood_mag, ...
                        'MagnitudeWeight', magnitude_weight};

            if uniform_init
                % All particles get uniform prior
                for i = 1:num_particles
                    inner_hmm = HMM([], A_transition, pointlikelihood_image, hmm_opts{:});

                    particles{i} = struct( ...
                        'association', 0, ...
                        'hmm', inner_hmm, ...
                        'weight', 1 / num_particles, ...
                        'ois_weight_update', 1.0, ...
                        'association_history', [], ...
                        'state_trajectory', []);
                end
            else
                % All particles get Gaussian prior centred on x0
                % (Diversity comes from association sampling, not initial state,
                %  because the HMM grid already represents the full distribution)
                for i = 1:num_particles
                    inner_hmm = HMM(x0, A_transition, pointlikelihood_image, hmm_opts{:});

                    % Get initial position estimate for trajectory seed
                    [x_init, ~] = inner_hmm.getGaussianEstimate();

                    particles{i} = struct( ...
                        'association', 0, ...
                        'hmm', inner_hmm, ...
                        'weight', 1 / num_particles, ...
                        'association_history', [], ...
                        'state_trajectory', x_init);
                end
            end

            if debug_flag
                if uniform_init
                    fprintf('Initialized %d particles with uniform prior.\n', num_particles);
                else
                    fprintf('Initialized %d particles with Gaussian prior.\n', num_particles);
                end
            end
        end

    end

end