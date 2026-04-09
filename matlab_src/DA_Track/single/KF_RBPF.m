% KF_RBPF.m
% Rao-Blackwellized Particle Filter with Kalman Filter for Data Association
%
% DESCRIPTION:
%   Implements RBPF for single target tracking with data association.
%   Each particle represents a discrete association hypothesis, with
%   continuous state tracked by an embedded Kalman Filter.
%
%   RBPF exploits conditional independence:
%     - Particles handle DISCRETE state: association hypothesis
%     - Kalman Filters handle CONTINUOUS state: position, velocity, etc.
%
%   ALGORITHM (per timestep):
%     1. PREDICTION: Propagate each particle's KF through dynamics
%     2. ASSOCIATION: Sample/enumerate association hypotheses per particle
%     3. UPDATE: Update each particle's KF with associated measurement
%     4. WEIGHTING: Compute weights from KF innovation likelihoods
%     5. RESAMPLING: Resample particles if ESS below threshold
%
% PARTICLE STRUCTURE:
%   particles{i} = struct with fields:
%     .association  - Integer: which measurement? (0 = clutter/no detection)
%     .kf           - KF object: Kalman filter for continuous state
%     .weight       - Scalar: particle weight
%
% PROPERTIES:
%   N_p              - Number of particles
%   N_x              - State dimension
%   N_z              - Measurement dimension
%   particles        - Cell array of particle structs {1 x N_p}
%   F, Q, H, R       - System model matrices (for creating KFs)
%
%   % Detection Model (inherited from DA_Filter pattern)
%   PD               - Detection probability
%   PFA              - False alarm probability
%
%   % Control Flags
%   debug            - Enable debug output
%   DynamicPlot      - Enable real-time visualization
%
% METHODS:
%   KF_RBPF(x0, N_particles, F, Q, H, R, ...)  - Constructor
%   timestep(z)                                  - Main algorithm loop
%   prediction()                                 - Propagate particle KFs
%   generateAssociations(z)                      - Create association hypotheses
%   measurement_update(z)                        - Update KFs and compute weights
%   resample()                                   - Particle resampling
%   getGaussianEstimate()                        - Extract mean state estimate
%
% EXAMPLE USAGE:
%   % Setup
%   F = ...; Q = ...; H = ...; R = ...;
%   x0 = [0; 0; 1; 1; 0; 0];  % Initial state guess
%   rbpf = KF_RBPF(x0, 1000, F, Q, H, R, 'Debug', true);
%
%   % Tracking loop
%   for k = 1:num_steps
%       measurements = get_measurements(k);  % [2 x N_meas]
%       rbpf.timestep(measurements);
%       [x_est, P_est] = rbpf.getGaussianEstimate();
%   end
%
% SEE ALSO:
%   KF, HMM_RBPF, PDA_PF, DA_Filter
%
% REFERENCES:
%   [1] Doucet et al. "On Sequential Monte Carlo Sampling Methods for
%       Bayesian Filtering" (1998)
%   [2] Särkkä & Svensson "Bayesian Filtering and Smoothing" (2023)
%       Chapter 11: Rao-Blackwellized Particle Filtering

classdef KF_RBPF < DA_Filter
    % KF_RBPF  Rao-Blackwellized Particle Filter with Kalman Filter inner filters.
    %
    % Inherits from DA_Filter.  See DA_Filter for the full interface contract,
    % including the storeHistory / history pattern for offline analysis.

    properties
        % -----------------------------------------------------------------
        % Filter dimensions
        % -----------------------------------------------------------------
        N_p                     % Number of particles
        N_x                     % State dimension
        N_z                     % Measurement dimension

        % -----------------------------------------------------------------
        % Particle filter state
        % -----------------------------------------------------------------
        % particles - Cell array of particle structs {1 x N_p}.
        %   Each struct contains:
        %     .association             - Integer measurement index (0 = clutter)
        %     .kf                      - KF handle: inner Kalman filter
        %     .weight                  - Scalar importance weight
        %     .ois_weight_update       - Marginal likelihood Z_k^i (OIS strategy)
        %     .association_history     - [1 x k] past association indices
        %     .state_trajectory        - [N_x x k] past KF state estimates
        %     .covariance_trajectory   - {1 x k} past KF covariances
        %     .innovation_trajectory   - [N_z x k] past innovations
        %     .innovation_cov_trajectory - {1 x k} past innovation covariances
        %     .measurement_trajectory  - [N_z x k] past associated measurements
        particles

        % -----------------------------------------------------------------
        % System model (shared across all inner KFs)
        % -----------------------------------------------------------------
        F   % State transition matrix            [N_x x N_x]
        Q   % Process noise covariance           [N_x x N_x]
        H   % Measurement matrix                 [N_z x N_x]
        R   % Measurement noise covariance       [N_z x N_z]

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
        %   'optimal'  : Sample from optimal importance distribution (Särkkä §11.3)
        %   'likelihood': Alias for 'optimal'
        association_strategy

        % -----------------------------------------------------------------
        % Resampling
        % -----------------------------------------------------------------
        ESS_threshold_percentage    % Fraction of N_p below which resampling triggers

        % -----------------------------------------------------------------
        % Visualization plot bounds (set via setPlotBounds for animations)
        % -----------------------------------------------------------------
        plot_bounds_x  = []  % [x_min, x_max] for position plot
        plot_bounds_y  = []  % [y_min, y_max] for position plot
        plot_bounds_vx = []  % [vx_min, vx_max] for velocity plot
        plot_bounds_vy = []  % [vy_min, vy_max] for velocity plot

        % -----------------------------------------------------------------
        % Visualize helper flags
        % -----------------------------------------------------------------
        true_meas_exists = true  % Whether a true-detection measurement exists
        gif_filename     = ''    % Path for GIF output ('' = disabled)
        gif_first_frame  = true  % Internal flag: first GIF frame not yet written
    end

    methods

        function obj = KF_RBPF(x0, N_particles, F, Q, H, R, varargin)
            % KF_RBPF Constructor for Rao-Blackwellized Particle Filter
            %
            % INPUTS:
            %   x0          - Initial state estimate [N_x x 1]
            %   N_particles - Number of particles
            %   F           - State transition matrix [N_x x N_x]
            %   Q           - Process noise covariance [N_x x N_x]
            %   H           - Measurement matrix [N_z x N_x]
            %   R           - Measurement noise covariance [N_z x N_z]
            %   varargin    - Name-value pairs: 'Debug', 'DynamicPlot', 'ESSThreshold'
            %
            % OUTPUTS:
            %   obj - Initialized KF_RBPF object

            % Parse optional arguments
            p = inputParser;
            addParameter(p, 'Debug', false, @islogical);
            addParameter(p, 'PD', 0.95, @(x) x >= 0 && x <= 1);
            addParameter(p, 'PFA', 0.05, @(x) x >= 0 && x <= 1);
            addParameter(p, 'ESSThreshold', 0.5, @(x) x > 0 && x <= 1);
            addParameter(p, 'AssociationStrategy', 'optimal', @(x) ismember(x, {'uniform', 'optimal', 'likelihood'}));
            addParameter(p, 'UniformInit', false, @islogical); % Add UniformInit parameter
            parse(p, varargin{:});

            obj.debug = p.Results.Debug;
            obj.PD = p.Results.PD;
            obj.PFA = p.Results.PFA;
            obj.ESS_threshold_percentage = p.Results.ESSThreshold;
            obj.association_strategy = p.Results.AssociationStrategy;
            uniform_init = p.Results.UniformInit;

            % Store dimensions
            obj.N_p = N_particles;
            obj.N_x = size(F, 1);
            obj.N_z = size(H, 1);

            % Store system matrices
            obj.F = F;
            obj.Q = Q;
            obj.H = H;
            obj.R = R;

            % Initialize particle array
            obj.particles = KF_RBPF.initialize_particles(N_particles, x0, F, Q, H, R, uniform_init);

            % Initialize timestep counter.
            % obj.history is initialised to struct([]) by DA_Filter.
            obj.timestep_counter = 0;

            if obj.debug
                fprintf('KF_RBPF initialized with %d particles.\n', N_particles);
                fprintf('  State dimension: %d\n', obj.N_x);
                fprintf('  Measurement dimension: %d\n', obj.N_z);
                fprintf('  Association strategy: %s\n', obj.association_strategy);
                fprintf('  PD = %.2f, PFA = %.2f\n', obj.PD, obj.PFA);
            end

        end

        function timestep(obj, z, varargin)
            % TIMESTEP  Execute one KF-RBPF predict-update cycle.
            %
            % SYNTAX:
            %   obj.timestep(z)
            %   obj.timestep(z, true_state)
            %
            % INPUTS:
            %   z          - Measurements [N_z x N_measurements].
            %                Pass [] for a missed-detection step.
            %   true_state - (optional) Ground-truth state [N_x x 1].
            %                Used only for dynamic visualisation; ignored
            %                in filter computations.
            %
            % ALGORITHM:
            %   1. Prediction  : Propagate each particle's embedded KF
            %   2. Association : Sample hypotheses per particle
            %   3. Update      : Update KFs and compute importance weights
            %   4. ESS         : Compute Effective Sample Size (stored in
            %                    obj.current_ESS BEFORE resampling)
            %   5. Resample    : Systematic resampling if ESS < threshold
            %
            % NOTE:
            %   History is NOT accumulated here.  Call storeHistory(z)
            %   after this method when running in test/analysis mode.

            true_state = [];
            if nargin > 2, true_state = varargin{1}; end

            % ----------------------------------------------------------
            % STEP 1: Prediction
            obj.prediction();

            % ----------------------------------------------------------
            % STEP 2: Association generation
            obj.generateAssociations(z);

            % ----------------------------------------------------------
            % STEP 3: Measurement update + weight computation
            obj.measurement_update(z);

            % ----------------------------------------------------------
            % STEP 4: Capture ESS BEFORE resampling.
            %   Stored in obj.current_ESS (inherited from DA_Filter) so
            %   that storeHistory() can read it without extra arguments.
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

            % ----------------------------------------------------------
            % Optional dynamic visualisation
            if obj.DynamicPlot
                obj.visualize(true_state, z);
            end
        end

        function storeHistory(obj, measurements, varargin)
            % STOREHISTORY  Snapshot full KF-RBPF state into obj.history.
            %
            % SYNTAX:
            %   obj.storeHistory(measurements)
            %   obj.storeHistory(measurements, true_state)
            %
            % INPUTS:
            %   measurements - Raw measurements from this timestep [N_z x N_m].
            %   true_state   - (optional) Ground-truth state [N_x x 1].
            %                  Pass [] or omit when GT is unavailable.
            %
            % DESCRIPTION:
            %   Appends one struct to obj.history at index obj.timestep_counter.
            %
            %   Fields always written (DA_Filter contract):
            %     .x_est        [N_x x 1]   - Gaussian mean estimate
            %     .P_est        [N_x x N_x] - Gaussian covariance
            %     .measurements [N_z x N_m] - Raw measurements
            %     .true_state   [N_x x 1]   - GT state ([] if unavailable)
            %     .timestep_num scalar       - Current timestep index
            %     .ESS          scalar       - ESS before resampling
            %     .particle_weights [1 x N_p]- Normalised particle weights
            %     .particle_associations [1 x N_p] - Association indices
            %
            %   Additional fields written when obj.store_full_history == true:
            %     .particle_states          [N_x x N_p] - KF state per particle
            %     .particle_association_histories {1 x N_p} - Full assoc. history
            %     .particle_trajectories    {1 x N_p}   - State trajectory per particle
            %     .particle_covariances     {1 x N_p}   - Cov trajectory per particle
            %     .particle_innovations     {1 x N_p}   - Innovation trajectory
            %     .particle_innovation_covs {1 x N_p}   - Innovation cov trajectory
            %     .particle_measurements    {1 x N_p}   - Associated meas trajectory
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

            particle_weights       = zeros(1, obj.N_p);
            particle_associations  = zeros(1, obj.N_p);
            for i = 1:obj.N_p
                particle_weights(i)      = obj.particles{i}.weight;
                particle_associations(i) = obj.particles{i}.association;
            end

            % DA_Filter contract fields
            obj.history(k).x_est         = x_est;
            obj.history(k).P_est         = P_est;
            obj.history(k).measurements  = measurements;
            obj.history(k).true_state    = true_state;
            obj.history(k).timestep_num  = k;
            % ESS captured before resampling in timestep() via obj.current_ESS
            obj.history(k).ESS                  = obj.current_ESS;
            obj.history(k).particle_weights      = particle_weights;
            obj.history(k).particle_associations = particle_associations;

            % ----------------------------------------------------------
            % Full-fidelity fields — only when store_full_history == true
            % ----------------------------------------------------------
            if obj.store_full_history
                particle_states       = zeros(obj.N_x, obj.N_p);
                particle_assoc_hist   = cell(1, obj.N_p);
                particle_trajectories = cell(1, obj.N_p);
                particle_covariances  = cell(1, obj.N_p);
                particle_innovations  = cell(1, obj.N_p);
                particle_innov_covs   = cell(1, obj.N_p);
                particle_measurements = cell(1, obj.N_p);

                for i = 1:obj.N_p
                    particle_states(:, i)       = obj.particles{i}.kf.x;
                    particle_assoc_hist{i}      = obj.particles{i}.association_history;
                    particle_trajectories{i}    = obj.particles{i}.state_trajectory;
                    particle_covariances{i}     = obj.particles{i}.covariance_trajectory;
                    particle_innovations{i}     = obj.particles{i}.innovation_trajectory;
                    particle_innov_covs{i}      = obj.particles{i}.innovation_cov_trajectory;
                    particle_measurements{i}    = obj.particles{i}.measurement_trajectory;
                end

                obj.history(k).particle_states                = particle_states;
                obj.history(k).particle_association_histories = particle_assoc_hist;
                obj.history(k).particle_trajectories          = particle_trajectories;
                obj.history(k).particle_covariances           = particle_covariances;
                obj.history(k).particle_innovations           = particle_innovations;
                obj.history(k).particle_innovation_covs       = particle_innov_covs;
                obj.history(k).particle_measurements          = particle_measurements;
            end
        end

        function prediction(obj)
            % PREDICTION Propagate particle Kalman filters through dynamics
            %
            % MODIFIES:
            %   Each particle's KF state and covariance
            %
            % ALGORITHM:
            %   For i = 1:N_p
            %       particles{i}.kf.predict()
            %   End

            % Loop over particles and call KF prediction
            for i = 1:obj.N_p
                obj.particles{i}.kf.prediction();
            end

            % Debug output if enabled
            if obj.debug
                disp('Prediction step completed for all particles.');
            end

        end

        function generateAssociations(obj, z)
            % GENERATEASSOCIATIONS Sample association hypotheses for particles
            %
            % INPUTS:
            %   z - Measurements [N_z x N_measurements]
            %
            % MODIFIES:
            %   Each particle's .association field
            %
            % ALGORITHM OPTIONS:
            %   Option 1 (Simple): Uniform random association
            %     - For each particle, randomly pick measurement index
            %     - association = 0 means clutter (no measurement)
            %     - association = j means use measurement z(:,j)
            %
            %   Option 2 (Optimal Importance Distribution): Sample from optimal
            %       distribution outlined in Sarrka
            %
            %   Option 3 (Likelihood-based): Sample proportional to KF likelihood
            %     - Compute likelihood for each measurement
            %     - Sample association with probability proportional to likelihood

            % Association generation switch
            switch obj.association_strategy
                case 'uniform'
                    obj.generateAssociations_uniform(z);
                case 'gating'
                    obj.generateAssociations_optimalimportancedist(z);
                case 'likelihood'
                    obj.generateAssociations_likelihood(z);
                otherwise
                    error('Unknown association strategy.');
            end

            % TODO: Choose association strategy
            % TODO: Implement association sampling per particle
            % TODO: Handle edge cases (no measurements, single measurement)

        end

        function measurement_update(obj, z)
            % MEASUREMENT_UPDATE Update particle KFs and compute weights
            %
            % INPUTS:
            %   z - Measurements [N_z x N_measurements]
            %
            % MODIFIES:
            %   Each particle's KF state, covariance, and weight
            %
            % WEIGHT UPDATE (sequential importance weights):
            %
            %   OIS strategy ('optimal' / 'gating'):
            %     The association c_k^i was drawn from the OPTIMAL importance
            %     distribution q(c_k^i) = p(c_k^i | y_k, y_{1:k-1}, c_{1:k-1}^i).
            %     The incremental weight factor is therefore the marginal likelihood:
            %
            %       w_k^i  ∝  w_{k-1}^i  *  p(y_k | y_{1:k-1}, c_{1:k-1}^i)
            %              =  w_{k-1}^i  *  Z_k^i
            %
            %     where Z_k^i = sum_j p(y_k | c_k^i=j) * p(c_k^i=j)
            %     was computed and stored in particle.ois_weight_update during
            %     generateAssociations_optimalimportancedist.
            %
            %   Uniform strategy:
            %     The association was drawn uniformly, so the IS weight is the
            %     raw target/proposal ratio (original behaviour):
            %       w_k^i  ∝  PD * N(z_assoc; H*x, S)   (detection)
            %              or  (1 - PD)                   (missed detection)

            use_ois_weights = strcmp(obj.association_strategy, 'optimal') || ...
                              strcmp(obj.association_strategy, 'gating');

            for i = 1:obj.N_p
                assoc = obj.particles{i}.association;
                w_prev = obj.particles{i}.weight; % weight carried from previous step

                if assoc > 0 % Detection hypothesis
                    % Measurement associated
                    z_i = z(:, assoc);
                    obj.particles{i}.kf.measurement_update(z_i);

                    % Store trajectory information
                    obj.particles{i}.state_trajectory = [obj.particles{i}.state_trajectory, obj.particles{i}.kf.x];
                    obj.particles{i}.covariance_trajectory{end + 1} = obj.particles{i}.kf.P;
                    obj.particles{i}.innovation_trajectory = [obj.particles{i}.innovation_trajectory, obj.particles{i}.kf.z];
                    obj.particles{i}.innovation_cov_trajectory{end + 1} = obj.particles{i}.kf.S;
                    obj.particles{i}.measurement_trajectory = [obj.particles{i}.measurement_trajectory, z_i];

                    if use_ois_weights
                        % OIS: multiply prior weight by the marginal likelihood
                        % Z_k^i = p(y_k | y_{1:k-1}, c_{1:k-1}^i) stored during association step
                        obj.particles{i}.weight = w_prev * obj.particles{i}.ois_weight_update;
                    else
                        % Uniform proposal: raw target/proposal likelihood ratio
                        innov = obj.particles{i}.kf.z;
                        S     = obj.particles{i}.kf.S;
                        likelihood = mvnpdf(innov', zeros(1, obj.N_z), S);
                        obj.particles{i}.weight = obj.PD * likelihood;
                    end

                else % Clutter/missed detection
                    % No measurement update, just prediction (store predicted state)
                    obj.particles{i}.state_trajectory = [obj.particles{i}.state_trajectory, obj.particles{i}.kf.x];
                    obj.particles{i}.covariance_trajectory{end + 1} = obj.particles{i}.kf.P;
                    obj.particles{i}.innovation_trajectory = [obj.particles{i}.innovation_trajectory, zeros(obj.N_z, 1)];
                    obj.particles{i}.innovation_cov_trajectory{end + 1} = [];
                    obj.particles{i}.measurement_trajectory = [obj.particles{i}.measurement_trajectory, NaN(obj.N_z, 1)];

                    if use_ois_weights
                        % OIS: same marginal likelihood factor regardless of sampled association
                        obj.particles{i}.weight = w_prev * obj.particles{i}.ois_weight_update;
                    else
                        % Uniform proposal: missed detection likelihood ratio
                        obj.particles{i}.weight = (1 - obj.PD);
                    end
                end

            end

            % Normalize weights
            total_weight = sum(cellfun(@(p) p.weight, obj.particles));

            for i = 1:obj.N_p
                obj.particles{i}.weight = obj.particles{i}.weight / total_weight;
            end

        end

        function generateAssociations_uniform(obj, z)
            % GENERATEASSOCIATIONS_UNIFORM Uniform random association sampling
            N_measurements = size(z, 2);

            for i = 1:obj.N_p
                % Randomly choose association index (0 = clutter)
                obj.particles{i}.association = randi([0, N_measurements]);
                obj.particles{i}.association_history = [obj.particles{i}.association_history, obj.particles{i}.association];
            end

        end

        function generateAssociations_optimalimportancedist(obj, z)
            % GENERATEASSOCIATIONS_OPTIMALIMPORTANCEDIST Optimal importance distribution sampling
            %
            % ALGORITHM (per Särkkä & Svensson, 2023, Section 11.3):
            %   Optimal importance distribution for data association:
            %   q(c_k^i | y_{1:k}, c_{1:k-1}^i) ∝ p(y_k | c_k^i, y_{1:k-1}, c_{1:k-1}^i) * p(c_k^i)
            %
            %   For each association hypothesis j:
            %     - If j = 0 (missed detection):
            %         p(y_k | c_k = 0) = (1 - P_D)
            %     - If j > 0 (measurement j is true detection):
            %         p(y_k | c_k = j) = P_D * N(y_k; H*x_k|k-1, S_k)
            %         where S_k = H*P_k|k-1*H' + R (innovation covariance)
            %
            %   The unnormalized sum:
            %     Z_k^i = sum_j p(y_k | c_k^i = j) * p(c_k^i = j)
            %   is the marginal likelihood p(y_k | y_{1:k-1}, c_{1:k-1}^i) and is
            %   stored in particle.ois_weight_update so that measurement_update can
            %   apply the correct sequential importance weight:
            %     w_k^i = w_{k-1}^i * Z_k^i
            %
            % NOTE: This is computed PER PARTICLE since each particle has different
            %       predicted state and thus different innovation likelihoods.

            N_measurements = size(z, 2);

            % Missed detection hypothesis probability (j=0)
            p_missed_detection = (1 - obj.PD);

            % Optimal importance distribution is PER PARTICLE
            for i = 1:obj.N_p
                % Compute UNnormalized association likelihoods for this particle
                % Index 1:N_measurements = measurement associations
                % Index N_measurements+1 = clutter/missed detection
                association_weights = zeros(1, N_measurements + 1);

                % Compute likelihood for each measurement association
                for j = 1:N_measurements
                    z_j = z(:, j);
                    % Compute innovation and covariance using particle's KF state
                    [innov, S] = obj.particles{i}.kf.getInnovation(z_j);

                    % Likelihood: p(y_k | c_k = j) = P_D * N(innovation; 0, S)
                    association_weights(j) = obj.PD * mvnpdf(innov', zeros(1, obj.N_z), S);
                end

                % Clutter/missed detection hypothesis (stored at end)
                association_weights(end) = p_missed_detection;

                % -------------------------------------------------------
                % Store unnormalized sum BEFORE normalizing.
                % This is Z_k^i = p(y_k | y_{1:k-1}, c_{1:k-1}^i), the
                % correct incremental weight factor for the OIS.
                % -------------------------------------------------------
                Z_k_i = sum(association_weights);
                obj.particles{i}.ois_weight_update = Z_k_i;

                % Normalize to form proper sampling distribution q(c_k^i | ...)
                association_weights = association_weights / Z_k_i;

                % Sample association using inverse CDF method
                cumulative_weights = cumsum(association_weights);
                r = rand();
                sampled_idx = find(cumulative_weights >= r, 1);

                % Convert sampled index to association value
                % Indices 1:N_measurements map to associations 1:N_measurements
                % Index N_measurements+1 maps to association 0 (clutter)
                if sampled_idx <= N_measurements
                    assoc = sampled_idx;
                else
                    assoc = 0; % Clutter/missed detection
                end

                obj.particles{i}.association = assoc;
                obj.particles{i}.association_history = [obj.particles{i}.association_history, assoc];
            end

        end

        function generateAssociations_likelihood(obj, z)
            % GENERATEASSOCIATIONS_LIKELIHOOD Likelihood-based association sampling
            % Falls back to uniform sampling (likelihood strategy not yet implemented).
            obj.generateAssociations_uniform(z);
        end

        function resample(obj)
            % RESAMPLE Systematic resampling of particles
            %
            % MODIFIES:
            %   obj.particles - Resampled particle array
            %
            % ALGORITHM:
            %   1. Check ESS = 1 / sum(weights^2)
            %   2. If ESS < threshold:
            %        - Systematic resampling to get new indices
            %        - Create new particle array
            %        - CRITICAL: Deep copy KF objects!
            %        - Reset weights to uniform
            %
            % CRITICAL BUG TO AVOID:
            %   DO NOT just copy particle structs directly!
            %   Must use KF.copyKF() to create independent KF objects.
            %   Otherwise resampled particles share same KF (disaster).

            % Compute Effective Sample Size (ESS)
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

                    % Deep copy particle (including trajectory information)
                    new_kf = KF.copyKF(obj.particles{index}.kf);
                    new_particles{i} = struct( ...
                        'association', obj.particles{index}.association, ...
                        'kf', new_kf, ...
                        'weight', 1 / obj.N_p, ...
                        'ois_weight_update', obj.particles{index}.ois_weight_update, ...
                        'association_history', obj.particles{index}.association_history, ...
                        'state_trajectory', obj.particles{index}.state_trajectory, ...
                        'covariance_trajectory', {obj.particles{index}.covariance_trajectory}, ...
                        'innovation_trajectory', obj.particles{index}.innovation_trajectory, ...
                        'innovation_cov_trajectory', {obj.particles{index}.innovation_cov_trajectory}, ...
                        'measurement_trajectory', obj.particles{index}.measurement_trajectory);
                end

                obj.particles = new_particles;

                if obj.debug
                    fprintf('Resampling performed. ESS: %.2f\n', ESS);
                end

            else

                if obj.debug
                    fprintf('No resampling needed. ESS: %.2f\n', ESS);
                end

            end

        end

        function [x_est, P_est] = getGaussianEstimate(obj)
            % GETGAUSSIANESTIMATE Extract weighted mean state estimate
            %
            % OUTPUTS:
            %   x_est - Weighted mean state [N_x x 1]
            %   P_est - Weighted covariance [N_x x N_x]
            %
            % ALGORITHM:
            %   x_est = sum_i w_i * x_i
            %   P_est = sum_i w_i * (P_i + (x_i - x_est)*(x_i - x_est)')

            x_est = zeros(obj.N_x, 1);
            P_est = zeros(obj.N_x, obj.N_x);

            for i = 1:obj.N_p
                w_i = obj.particles{i}.weight;
                x_i = obj.particles{i}.kf.x;
                P_i = obj.particles{i}.kf.P;

                x_est = x_est + w_i * x_i;
            end

            for i = 1:obj.N_p
                w_i = obj.particles{i}.weight;
                x_i = obj.particles{i}.kf.x;
                P_i = obj.particles{i}.kf.P;

                diff = x_i - x_est;
                P_est = P_est + w_i * (P_i + diff * diff');
            end

        end

        function visualize(obj, true_state, measurements)
            % VISUALIZE Plot particle distribution and estimates
            %
            % INPUTS:
            %   true_state   - (optional) Ground truth state [N_x x 1]
            %   measurements - (optional) Current measurements [N_z x N_meas]
            %
            % CREATES:
            %   1x4 layout:
            %   Subplot 1: Particles colored by weight (position space)
            %   Subplot 2: Particles colored by association (position space)
            %   Subplot 3: Velocity estimates (if N_x >= 4)
            %   Subplot 4: Association histogram

            % Handle optional arguments
            if nargin < 2 || isempty(true_state)
                true_state = [];
            end

            if nargin < 3 || isempty(measurements)
                measurements = [];
            end

            % Create or use existing figure
            if isempty(obj.dynamic_figure_handle) || ~isvalid(obj.dynamic_figure_handle)
                obj.dynamic_figure_handle = figure('Name', 'KF-RBPF Tracking', ...
                    'NumberTitle', 'off', 'Position', [50, 50, 1600, 400]);
            else
                figure(obj.dynamic_figure_handle);
                clf;
            end

            % Extract particle states and weights
            [x_est, P_est] = obj.getGaussianEstimate();
            particle_positions = zeros(obj.N_x, obj.N_p);
            particle_associations = zeros(1, obj.N_p);
            particle_weights = zeros(1, obj.N_p);

            for i = 1:obj.N_p
                particle_positions(:, i) = obj.particles{i}.kf.x;
                particle_associations(i) = obj.particles{i}.association;
                particle_weights(i) = obj.particles{i}.weight;
            end

            % Compute ESS
            ESS = 1 / sum(particle_weights .^ 2);

            % Define spatial bounds - use static bounds if set, otherwise compute dynamically
            if ~isempty(obj.plot_bounds_x) && ~isempty(obj.plot_bounds_y)
                % Use pre-set static bounds
                Xbounds = obj.plot_bounds_x;
                Ybounds = obj.plot_bounds_y;
            else
                % Compute dynamic bounds based on particles, measurements, and true state
                all_x = particle_positions(1, :);
                all_y = particle_positions(2, :);

                if ~isempty(measurements)
                    all_x = [all_x, measurements(1, :)];
                    all_y = [all_y, measurements(2, :)];
                end

                if ~isempty(true_state)
                    all_x = [all_x, true_state(1)];
                    all_y = [all_y, true_state(2)];
                end

                % Compute bounds with 20% margin
                x_range = max(all_x) - min(all_x);
                y_range = max(all_y) - min(all_y);
                margin = 0.2; % 20 % margin

                Xbounds = [min(all_x) - margin * x_range, max(all_x) + margin * x_range];
                Ybounds = [min(all_y) - margin * y_range, max(all_y) + margin * y_range];

                % Ensure minimum range for visualization
                if x_range < 0.5
                    x_center = mean(Xbounds);
                    Xbounds = [x_center - 0.5, x_center + 0.5];
                end

                if y_range < 0.5
                    y_center = mean(Ybounds);
                    Ybounds = [y_center - 0.5, y_center + 0.5];
                end

            end

            %% SUBPLOT 1: Position colored by weight
            subplot(1, 4, 1);
            cla; hold on;

            % Scatter particles
            scatter(particle_positions(1, :), particle_positions(2, :), 20, ...
                particle_weights, 'filled', 'MarkerFaceAlpha', 0.6);
            colormap('hot');
            cb = colorbar;
            cb.Label.String = 'Weight';

            % Plot mean estimate
            plot(x_est(1), x_est(2), 'go', 'MarkerSize', 12, 'LineWidth', 3);

            % Plot covariance ellipse
            if obj.N_x >= 2
                pos_cov = P_est(1:2, 1:2);
                ellipse_1sigma = obj.computeCovarianceEllipse(x_est(1:2), pos_cov, 1);
                plot(ellipse_1sigma(1, :), ellipse_1sigma(2, :), 'g-', 'LineWidth', 2);
            end

            % Plot true state
            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'md', 'MarkerSize', 10, ...
                    'LineWidth', 2, 'MarkerFaceColor', 'm');
            end

            % Plot measurements
            if ~isempty(measurements)
                plot(measurements(1, :), measurements(2, :), 'r+', ...
                    'MarkerSize', 10, 'LineWidth', 2);

                % Mark the closest measurement to true state with a star (correct association)
                % Only if a true measurement actually exists (not a missed detection)
                if ~isempty(true_state) && obj.true_meas_exists
                    % Compute distance from each measurement to true position
                    true_pos = true_state(1:2);
                    dists = vecnorm(measurements - true_pos, 2, 1);
                    [~, closest_idx] = min(dists);

                    % Plot star on top of closest measurement
                    plot(measurements(1, closest_idx), measurements(2, closest_idx), ...
                        'p', 'MarkerSize', 18, 'LineWidth', 2.5, ...
                        'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y');
                end

            end

            title(sprintf('Position (Weighted)\nESS: %.1f', ESS), 'Interpreter', 'latex');
            xlabel('X (m)'); ylabel('Y (m)');
            xlim(Xbounds); ylim(Ybounds);
            axis square; grid on;

            %% SUBPLOT 2: Position colored by association
            subplot(1, 4, 2);
            cla; hold on;

            % Get unique associations
            unique_assocs = unique(particle_associations);
            N_meas = max(unique_assocs);

            % Create colormap for associations
            if N_meas > 0
                assoc_colors = lines(N_meas + 1); % +1 for clutter
            else
                assoc_colors = [0.5 0.5 0.5]; % Gray for all clutter
            end

            % Plot particles by association
            for assoc = unique_assocs
                idx = particle_associations == assoc;

                if assoc == 0
                    % Clutter: gray
                    scatter(particle_positions(1, idx), particle_positions(2, idx), ...
                        20, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.4);
                else
                    % Measurement assoc: colored
                    scatter(particle_positions(1, idx), particle_positions(2, idx), ...
                        20, assoc_colors(assoc, :), 'filled', 'MarkerFaceAlpha', 0.7);
                end

            end

            % Plot mean estimate
            plot(x_est(1), x_est(2), 'ko', 'MarkerSize', 12, 'LineWidth', 3);

            % Plot covariance ellipse
            if obj.N_x >= 2
                pos_cov = P_est(1:2, 1:2);
                ellipse_1sigma = obj.computeCovarianceEllipse(x_est(1:2), pos_cov, 1);
                plot(ellipse_1sigma(1, :), ellipse_1sigma(2, :), 'k-', 'LineWidth', 2);
            end

            % Plot true state
            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'md', 'MarkerSize', 10, ...
                    'LineWidth', 2, 'MarkerFaceColor', 'm');
            end

            % Plot measurements with matching colors
            if ~isempty(measurements)

                for j = 1:size(measurements, 2)
                    plot(measurements(1, j), measurements(2, j), 'x', ...
                        'Color', assoc_colors(min(j, N_meas), :), ...
                        'MarkerSize', 12, 'LineWidth', 3);
                end

                % Mark the closest measurement to true state with a star (correct association)
                % Only if a true measurement actually exists (not a missed detection)
                if ~isempty(true_state) && obj.true_meas_exists
                    % Compute distance from each measurement to true position
                    true_pos = true_state(1:2);
                    dists = vecnorm(measurements - true_pos, 2, 1);
                    [~, closest_idx] = min(dists);

                    % Plot star on top of closest measurement
                    plot(measurements(1, closest_idx), measurements(2, closest_idx), ...
                        'p', 'MarkerSize', 18, 'LineWidth', 2.5, ...
                        'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'y');
                end

            end

            title('Position (Association)', 'Interpreter', 'latex');
            xlabel('X (m)'); ylabel('Y (m)');
            xlim(Xbounds); ylim(Ybounds);
            axis square; grid on;

            %% SUBPLOT 3: Velocity (if available) or Acceleration
            subplot(1, 4, 3);
            cla; hold on;

            if obj.N_x >= 4
                % Plot velocity
                scatter(particle_positions(3, :), particle_positions(4, :), 20, ...
                    particle_weights, 'filled', 'MarkerFaceAlpha', 0.6);
                colormap('hot');

                % Plot mean
                plot(x_est(3), x_est(4), 'go', 'MarkerSize', 12, 'LineWidth', 3);

                % Plot covariance ellipse
                vel_cov = P_est(3:4, 3:4);
                ellipse_1sigma = obj.computeCovarianceEllipse(x_est(3:4), vel_cov, 1);
                plot(ellipse_1sigma(1, :), ellipse_1sigma(2, :), 'g-', 'LineWidth', 2);

                % Plot true velocity
                if ~isempty(true_state) && length(true_state) >= 4
                    plot(true_state(3), true_state(4), 'md', 'MarkerSize', 10, ...
                        'LineWidth', 2, 'MarkerFaceColor', 'm');
                end

                % Compute velocity bounds - use static if set, otherwise dynamic
                if ~isempty(obj.plot_bounds_vx) && ~isempty(obj.plot_bounds_vy)
                    Vxbounds = obj.plot_bounds_vx;
                    Vybounds = obj.plot_bounds_vy;
                else
                    % Compute dynamic velocity bounds
                    all_vx = particle_positions(3, :);
                    all_vy = particle_positions(4, :);

                    if ~isempty(true_state) && length(true_state) >= 4
                        all_vx = [all_vx, true_state(3)];
                        all_vy = [all_vy, true_state(4)];
                    end

                    vx_range = max(all_vx) - min(all_vx);
                    vy_range = max(all_vy) - min(all_vy);
                    margin_v = 0.2;

                    Vxbounds = [min(all_vx) - margin_v * max(vx_range, 0.5), ...
                                    max(all_vx) + margin_v * max(vx_range, 0.5)];
                    Vybounds = [min(all_vy) - margin_v * max(vy_range, 0.5), ...
                                    max(all_vy) + margin_v * max(vy_range, 0.5)];
                end

                title('Velocity', 'Interpreter', 'latex');
                xlabel('V_x (m/s)'); ylabel('V_y (m/s)');
                xlim(Vxbounds); ylim(Vybounds);
                axis square; grid on;
            else
                % No velocity data
                text(0.5, 0.5, 'N/A (State dim < 4)', ...
                    'HorizontalAlignment', 'center', ...
                    'Units', 'normalized', 'FontSize', 14);
                title('Velocity', 'Interpreter', 'latex');
                axis off;
            end

            %% SUBPLOT 4: Association Histogram
            subplot(1, 4, 4);
            cla; hold on;

            % Count particles per association
            assoc_counts = histcounts(particle_associations, ...
                'BinEdges', -0.5:(max(particle_associations) + 0.5));

            % Create bar chart
            bar(0:max(particle_associations), assoc_counts, 'FaceColor', [0.3 0.5 0.8]);

            % Add percentage labels on bars
            for i = 0:max(particle_associations)

                if assoc_counts(i + 1) > 0
                    pct = 100 * assoc_counts(i + 1) / obj.N_p;
                    text(i, assoc_counts(i + 1), sprintf('%.1f%%', pct), ...
                        'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'bottom', 'FontSize', 9);
                end

            end

            title('Association Distribution', 'Interpreter', 'latex');
            xlabel('Association (0=Clutter)');
            ylabel('Particle Count');
            grid on;

            % Format x-axis
            if max(particle_associations) <= 10
                xticks(0:max(particle_associations));
            end

            %% Overall title
            sgtitle(sprintf('KF-RBPF Timestep %d | Strategy: %s | N_p=%d', ...
                obj.timestep_counter, obj.association_strategy, obj.N_p), ...
                'FontSize', 14, 'FontWeight', 'bold');

            drawnow;

            % Capture frame for GIF if enabled
            if ~isempty(obj.gif_filename)
                frame = getframe(gcf);
                im = frame2im(frame);
                [imind, cm] = rgb2ind(im, 256);

                if obj.gif_first_frame
                    imwrite(imind, cm, obj.gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
                    obj.gif_first_frame = false;
                else
                    imwrite(imind, cm, obj.gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
                end

            end

            % Add pause for animation visibility (0.2 seconds = 5 fps)
            if obj.DynamicPlot
                pause(0.2);
            end

        end

        % Getter and setter methods
    % Helper functions
    function setDetectionModel(obj, PD, PFA)
        % SETDETECTIONMODEL Set detection model parameters
        %
        % SYNTAX:
        %   obj.setDetectionModel(PD, PFA)
        %
        % INPUTS:
        %   PD  - Detection probability (0 < PD <= 1)
        %   PFA - False alarm probability (0 <= PFA < 1)
        %
        % DESCRIPTION:
        %   Updates the detection model parameters used in PDA data association.
        %   These parameters affect the weight hypothesis and clutter hypothesis
        %   calculations in the normalization constants computation.
        %
        % See also computeNormalizationConstants, measurement_update

        % Validate inputs
        if PD <= 0 || PD > 1
            error('PDA_PF:InvalidPD', 'Detection probability PD must be in range (0, 1]');
        end

        if PFA < 0 || PFA >= 1
            error('PDA_PF:InvalidPFA', 'False alarm probability PFA must be in range [0, 1)');
        end

        obj.PD = PD;
        obj.PFA = PFA;

        if obj.debug
            fprintf('[DETECTION MODEL] Updated: PD=%.3f, PFA=%.3f\n', obj.PD, obj.PFA);
        end

    end

    end

    

    methods (Static)
        % Helper functions can go here

        function particles = initialize_particles(num_particles, x0, F, Q, H, R, uniform_init)
            % INITIALIZE_PARTICLES Helper to create particle array with initial diversity
            %
            % INPUTS:
            %   num_particles - Number of particles
            %   x0            - Initial state estimate (mean)
            %   F, Q, H, R    - System model matrices
            %   uniform_init  - (optional) Boolean: if true, use uniform initialization
            % OUTPUTS:
            %   particles - Cell array of initialized particle structs
            %
            % ALGORITHM:
            %   If uniform_init = false (default):
            %       Particles are initialized with diverse initial states sampled
            %       from a distribution around x0 to enable localization from
            %       uncertain initial conditions.
            %   If uniform_init = true:
            %       Particles are uniformly distributed across state space:
            %       Position: x ∈ [-2, 2], y ∈ [0.5, 4]
            %       Velocity: vx, vy ∈ [-2, 2]
            %       Acceleration: ax, ay ∈ [-2, 2]

            if nargin < 7
                uniform_init = false;
            end

            particles = cell(1, num_particles);

            N_x = length(x0);

            if uniform_init
                % UNIFORM INITIALIZATION across state space
                % Define bounds for uniform initialization (matching PDA_PF)
                pos_bounds = [-2, 2; 0.5, 4]; % [x_min, x_max; y_min, y_max] - y >= 0.5
                vel_bounds = [-2, 2; -2, 2]; % [vx_min, vx_max; vy_min, vy_max]
                acc_bounds = [-2, 2; -2, 2]; % [ax_min, ax_max; ay_min, ay_max]

                % Initial KF covariance (conservative for uniform initialization)
                initial_kf_cov = 10 * eye(N_x);

                for i = 1:num_particles
                    % Initialize particle state uniformly
                    x_particle = zeros(N_x, 1);

                    % Position (x, y) - ensure y >= 0.5
                    x_particle(1) = pos_bounds(1, 1) + (pos_bounds(1, 2) - pos_bounds(1, 1)) * rand();
                    x_particle(2) = pos_bounds(2, 1) + (pos_bounds(2, 2) - pos_bounds(2, 1)) * rand();

                    % Velocity (vx, vy) if state dimension >= 4
                    if N_x >= 4
                        x_particle(3) = vel_bounds(1, 1) + (vel_bounds(1, 2) - vel_bounds(1, 1)) * rand();
                        x_particle(4) = vel_bounds(2, 1) + (vel_bounds(2, 2) - vel_bounds(2, 1)) * rand();
                    end

                    % Acceleration (ax, ay) if state dimension >= 6
                    if N_x >= 6
                        x_particle(5) = acc_bounds(1, 1) + (acc_bounds(1, 2) - acc_bounds(1, 1)) * rand();
                        x_particle(6) = acc_bounds(2, 1) + (acc_bounds(2, 2) - acc_bounds(2, 1)) * rand();
                    end

                    particles{i} = struct( ...
                        'association', 0, ... % Integer: which measurement index (0 = clutter)
                        'kf', KF(x_particle, initial_kf_cov, F, Q, H, R), ... % KF with uniform initial state
                        'weight', 1 / num_particles, ... % Uniform initial weights
                        'ois_weight_update', 1.0, ... % OIS marginal likelihood factor Z_k^i (default 1)
                        'association_history', [], ... % Initialize association history
                        'state_trajectory', x_particle, ... % Initialize state trajectory
                        'covariance_trajectory', [], ... % Initialize covariance trajectory (cell array)
                        'innovation_trajectory', [], ... % Initialize innovation trajectory
                        'innovation_cov_trajectory', [], ... % Innovation covariance trajectory
                        'measurement_trajectory', []); % Associated measurement trajectory
                end

            else
                % GAUSSIAN INITIALIZATION around x0
                % x0 is initialized at the first GT position with zero velocity/acceleration.
                % We want a good start: tight on position (we know it from GT), generous
                % on velocity and acceleration (initialised to 0, true values unknown).
                if N_x == 4
                    initial_uncertainty_std = [0.316; 0.316; 1.0; 1.0]; % [x, y, vx, vy]
                else
                    initial_uncertainty_std = [0.316; 0.316; 1.0; 1.0; 1.414; 1.414]; % [x,y,vx,vy,ax,ay]
                end

                % Initial covariance for particle state diversity
                initial_particle_cov = diag(initial_uncertainty_std .^ 2);

                % Initial KF covariance per particle — matches P0_init in run_experiment
                if N_x == 4
                    initial_kf_cov = diag([0.1, 0.1, 1.0, 1.0]);
                else
                    initial_kf_cov = diag([0.1, 0.1, 1.0, 1.0, 2.0, 2.0]);
                end

                for i = 1:num_particles
                    % Sample particle state from distribution around x0
                    x_particle = x0 + mvnrnd(zeros(N_x, 1), initial_particle_cov)';

                    particles{i} = struct( ...
                        'association', 0, ... % Integer: which measurement index (0 = clutter)
                        'kf', KF(x_particle, initial_kf_cov, F, Q, H, R), ... % KF with sampled initial state
                        'weight', 1 / num_particles, ... % Uniform initial weights
                        'ois_weight_update', 1.0, ... % OIS marginal likelihood factor Z_k^i (default 1)
                        'association_history', [], ... % Initialize association history
                        'state_trajectory', x_particle, ... % Initialize state trajectory
                        'covariance_trajectory', [], ... % Initialize covariance trajectory (cell array)
                        'innovation_trajectory', [], ... % Initialize innovation trajectory
                        'innovation_cov_trajectory', [], ... % Innovation covariance trajectory
                        'measurement_trajectory', []); % Associated measurement trajectory
                end

            end

        end

    end

    methods
        % Visualization helper methods

        function setPlotBounds(obj, all_measurements, all_true_states, margin)
            % SETPLOTBOUNDS Compute and set static plot bounds for animation
            %
            % INPUTS:
            %   all_measurements - Cell array of measurements {1 x N_timesteps}
            %                      Each cell contains [N_z x N_meas] measurements
            %   all_true_states  - (optional) True states [N_x x N_timesteps]
            %   margin           - (optional) Padding fraction (default: 0.2 = 20%)
            %
            % DESCRIPTION:
            %   Computes static axis bounds from all data to prevent axes from
            %   changing during animation. Should be called before running filter.
            %
            % EXAMPLE:
            %   rbpf.setPlotBounds(measurements, true_states, 0.2);

            if nargin < 3
                all_true_states = [];
            end

            if nargin < 4
                margin = 0.2; % 20 % margin
            end

            % Collect all position data
            all_x = [];
            all_y = [];
            all_vx = [];
            all_vy = [];

            % From measurements
            for k = 1:length(all_measurements)

                if ~isempty(all_measurements{k})
                    all_x = [all_x, all_measurements{k}(1, :)];
                    all_y = [all_y, all_measurements{k}(2, :)];
                end

            end

            % From true states
            if ~isempty(all_true_states)
                all_x = [all_x, all_true_states(1, :)];
                all_y = [all_y, all_true_states(2, :)];

                if size(all_true_states, 1) >= 4
                    all_vx = all_true_states(3, :);
                    all_vy = all_true_states(4, :);
                end

            end

            % Compute position bounds
            x_range = max(all_x) - min(all_x);
            y_range = max(all_y) - min(all_y);

            obj.plot_bounds_x = [min(all_x) - margin * x_range, max(all_x) + margin * x_range];
            obj.plot_bounds_y = [min(all_y) - margin * y_range, max(all_y) + margin * y_range];

            % Compute velocity bounds (if applicable)
            if ~isempty(all_vx) && ~isempty(all_vy)
                vx_range = max(all_vx) - min(all_vx);
                vy_range = max(all_vy) - min(all_vy);

                obj.plot_bounds_vx = [min(all_vx) - margin * max(vx_range, 0.5), ...
                                          max(all_vx) + margin * max(vx_range, 0.5)];
                obj.plot_bounds_vy = [min(all_vy) - margin * max(vy_range, 0.5), ...
                                          max(all_vy) + margin * max(vy_range, 0.5)];
            end

            if obj.debug
                fprintf('Static plot bounds set:\n');
                fprintf('  X: [%.2f, %.2f]\n', obj.plot_bounds_x(1), obj.plot_bounds_x(2));
                fprintf('  Y: [%.2f, %.2f]\n', obj.plot_bounds_y(1), obj.plot_bounds_y(2));

                if ~isempty(obj.plot_bounds_vx)
                    fprintf('  Vx: [%.2f, %.2f]\n', obj.plot_bounds_vx(1), obj.plot_bounds_vx(2));
                    fprintf('  Vy: [%.2f, %.2f]\n', obj.plot_bounds_vy(1), obj.plot_bounds_vy(2));
                end

            end

        end

        function ellipse_points = computeCovarianceEllipse(obj, mean_pos, cov_matrix, n_sigma)
            % COMPUTECOVARIANCEELLIPSE Compute points for covariance ellipse
            %
            % INPUTS:
            %   mean_pos    - Mean position [2x1]
            %   cov_matrix  - Covariance matrix [2x2]
            %   n_sigma     - Number of standard deviations (1 for 1σ, 3 for 3σ)
            %
            % OUTPUTS:
            %   ellipse_points - Points defining the ellipse [2xN]
            %
            % DESCRIPTION:
            %   Computes ellipse points using eigenvalue decomposition of the
            %   covariance matrix to handle arbitrary orientation and scaling.

            % Number of points for smooth ellipse
            n_points = 100;
            theta = linspace(0, 2 * pi, n_points);

            % Unit circle points
            unit_circle = [cos(theta); sin(theta)];

            % Eigenvalue decomposition of covariance matrix
            [eigvec, eigval] = eig(cov_matrix);

            % Ensure positive eigenvalues (numerical stability)
            eigval = max(eigval, eps);

            % Scale by n_sigma and square root of eigenvalues
            scaling = n_sigma * sqrt(eigval);

            % Transform unit circle to ellipse
            ellipse_centered = eigvec * scaling * unit_circle;

            % Translate to mean position
            ellipse_points = ellipse_centered + mean_pos;
        end

    end

end
