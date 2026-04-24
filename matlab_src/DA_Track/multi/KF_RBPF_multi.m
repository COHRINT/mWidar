% KF_RBPF_multi.m
% Multi-Target Rao-Blackwellized Particle Filter with Kalman Filters
%
% DESCRIPTION:
%   Implements RBPF for MULTI-TARGET tracking with joint data association.
%   Each particle represents a joint discrete association hypothesis for ALL targets,
%   with continuous states tracked by embedded Kalman Filters (one per target).
%
%   RBPF exploits conditional independence:
%     - Particles handle DISCRETE state: joint association hypothesis tensor
%     - Kalman Filters handle CONTINUOUS state: position, velocity, etc. per target
%
%   ALGORITHM (per timestep):
%     1. PREDICTION: Propagate each particle's KFs (all N_t targets) through dynamics
%     2. ASSOCIATION: Sample joint association hypotheses from optimal importance distribution
%     3. UPDATE: Update each target's KF with associated measurement (or no measurement)
%     4. WEIGHTING: Compute weights from joint KF innovation likelihoods
%     5. RESAMPLING: Resample particles if ESS below threshold
%
% PARTICLE STRUCTURE:
%   particles{i} = struct with fields:
%     .associations  - Vector [N_t x 1]: which measurement for each target? (0 = clutter/missed detection)
%     .kfs           - Cell array {1 x N_t}: Kalman filters for each target's continuous state
%     .weight        - Scalar: particle weight
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

classdef KF_RBPF_multi < handle % TODO: Inherit from DA_Filter if available

    properties
        % Filter Dimensions
        N_p % Number of particles
        N_x % State dimension (per target)
        N_z % Measurement dimension
        N_t % Number of targets (assumed known and fixed)

        % Particle Filter State
        particles % Cell array of particle structs {1 x N_p}
        % Each struct: {associations (N_t x 1), kfs (cell {1 x N_t}), weight}

        % System Model (for creating KF objects)
        F % State transition matrix [N_x x N_x]
        Q % Process noise covariance [N_x x N_x]
        H % Measurement matrix [N_z x N_x]
        R % Measurement noise covariance [N_z x N_z]

        % Detection Model Parameters
        PD % Detection probability (default: 0.95)
        PFA % False alarm probability (default: 0.05)

        % Control Flags
        debug % Enable debug output (default: false)

        % Association Strategy
        association_strategy % Strategy for generating associations: 'uniform', 'gating', 'likelihood'

        % Resampling Parameters
        ESS_threshold_percentage % ESS threshold for resampling (default: 0.5)

        % Timestep counter
        timestep_counter

        % History storage for post-processing visualization
        history % Struct array storing timestep data

        % Visualization support (DynamicPlot)
        DynamicPlot              % Enable real-time visualization (default: false)
        dynamic_figure_handle    % Figure handle for real-time plot
        plot_bounds_x            % [xmin, xmax] static position bounds ([] = auto)
        plot_bounds_y            % [ymin, ymax] static position bounds ([] = auto)
        gif_filename             % Path for GIF export ([] = disabled)
        gif_first_frame          % Internal flag for GIF writing
    end

    methods

        function obj = KF_RBPF_multi(x0_cell, N_particles, F, Q, H, R, varargin)
            % KF_RBPF_MULTI Constructor for Multi-Target Rao-Blackwellized Particle Filter
            %
            % INPUTS:
            %   x0_cell     - Cell array {1 x N_targets} of initial state estimates [N_x x 1] per target
            %   N_particles - Number of particles
            %   F           - State transition matrix [N_x x N_x]
            %   Q           - Process noise covariance [N_x x N_x]
            %   H           - Measurement matrix [N_z x N_x]
            %   R           - Measurement noise covariance [N_z x N_z]
            %   varargin    - Name-value pairs: 'Debug', 'PD', 'PFA', 'ESSThreshold', etc.
            %
            % OUTPUTS:
            %   obj - Initialized KF_RBPF_multi object

            % Parse optional arguments
            p = inputParser;
            addParameter(p, 'Debug', false, @islogical);
            addParameter(p, 'DynamicPlot', false, @islogical);
            addParameter(p, 'PD', 0.95, @(x) x >= 0 && x <= 1);
            addParameter(p, 'PFA', 0.05, @(x) x >= 0 && x <= 1);
            addParameter(p, 'ESSThreshold', 0.5, @(x) x > 0 && x <= 1);
            addParameter(p, 'AssociationStrategy', 'optimal', @(x) ismember(x, {'uniform', 'optimal', 'likelihood'}));
            addParameter(p, 'UniformInit', false, @islogical);
            % InitSigmaPos / InitSigmaVel: std-dev for Gaussian particle init.
            % Default (5 m, 2 m/s) is appropriate for unknown initial state.
            % Set smaller (e.g., 0.3 / 0.2) when initialising from known GT.
            addParameter(p, 'InitSigmaPos', 5.0, @(x) x >= 0);
            addParameter(p, 'InitSigmaVel', 2.0, @(x) x >= 0);
            parse(p, varargin{:});

            obj.debug = p.Results.Debug;
            obj.DynamicPlot = p.Results.DynamicPlot;
            obj.PD = p.Results.PD;
            obj.PFA = p.Results.PFA;
            obj.ESS_threshold_percentage = p.Results.ESSThreshold;
            obj.association_strategy = p.Results.AssociationStrategy;
            obj.N_t = length(x0_cell); % Number of targets from initial states
            uniform_init = p.Results.UniformInit;
            init_sigma_pos = p.Results.InitSigmaPos;
            init_sigma_vel = p.Results.InitSigmaVel;

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
            obj.particles = KF_RBPF_multi.initialize_particles(N_particles, x0_cell, F, Q, H, R, uniform_init, init_sigma_pos, init_sigma_vel);

            % Initialize visualization properties
            obj.dynamic_figure_handle = [];
            obj.plot_bounds_x = [];
            obj.plot_bounds_y = [];
            obj.gif_filename  = [];
            obj.gif_first_frame = true;

            % Initialize timestep counter and history storage
            obj.timestep_counter = 0;
            obj.history = struct('measurements', {}, 'true_state', {}, 'true_meas_flag', {}, ...
                'particle_states', {}, 'particle_associations', {}, ...
                'particle_weights', {}, 'particle_association_histories', {}, ...
                'particle_trajectories', {}, ... % Full state trajectories for each particle
                'particle_covariances', {}, ... % Covariance trajectories for each particle
                'particle_innovations', {}, ... % Innovation trajectories for each particle
                'particle_innovation_covs', {}, ... % Innovation covariance trajectories
                'particle_measurements', {}, ... % Associated measurements for each particle
                'estimate', {}, 'covariance', {}, 'ESS', {});

            if obj.debug
                fprintf('KF_RBPF_multi initialized with %d particles tracking %d targets.\n', N_particles, obj.N_t);
                fprintf('  State dimension (per target): %d\n', obj.N_x);
                fprintf('  Measurement dimension: %d\n', obj.N_z);
                fprintf('  Association strategy: %s\n', obj.association_strategy);
                fprintf('  PD = %.2f, PFA = %.2f\n', obj.PD, obj.PFA);
            end

        end

        function timestep(obj, z, true_state, true_meas_flag)
            % TIMESTEP Execute one RBPF timestep
            %
            % INPUTS:
            %   z               - Measurements [N_z x N_measurements]
            %   true_state      - (optional) Ground truth state for visualization [N_x x 1]
            %   true_meas_flag  - (optional) Boolean indicating if a true measurement exists
            %
            % ALGORITHM:
            %   1. Prediction: Propagate each particle's KF
            %   2. Association: Generate hypotheses for each particle
            %   3. Update: Update KFs with associated measurements
            %   4. Resample: If ESS < threshold
            %   5. Visualize: Update plots if enabled

            % Handle optional arguments
            if nargin < 3
                true_state = [];
            end

            if nargin < 4
                true_meas_flag = true; % Default: assume true measurement exists
            end

            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % STEP 1: Prediction
            obj.prediction();

            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % STEP 2: Association Generation
            obj.generateAssociations(z);

            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % STEP 3: Measurement Update
            obj.measurement_update(z);

            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % STEP 3.5: Compute ESS BEFORE resampling
            weights = zeros(1, obj.N_p);

            for i = 1:obj.N_p
                weights(i) = obj.particles{i}.weight;
            end

            ESS_before_resample = 1 / sum(weights .^ 2);

            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % STEP 4: Resampling
            obj.resample();

            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % STEP 5: Store history for post-processing visualization
            obj.timestep_counter = obj.timestep_counter + 1;
            obj.storeHistory(z, true_state, true_meas_flag, ESS_before_resample);

        end

        function storeHistory(obj, measurements, true_state, true_meas_flag, ESS_before_resample)
            % STOREHISTORY Save current timestep data for later visualization (MULTI-TARGET)
            %
            % INPUTS:
            %   measurements        - Measurements [N_z x N_measurements]
            %   true_state          - Ground truth states (cell {1 x N_t} or matrix) (optional)
            %   true_meas_flag      - Boolean indicating if true measurement exists
            %   ESS_before_resample - ESS computed before resampling

            k = obj.timestep_counter;

            % Extract particle states, associations, weights for all targets
            % For multi-target: store states as cell array of matrices
            particle_states_cell = cell(1, obj.N_t);

            for t = 1:obj.N_t
                particle_states_cell{t} = zeros(obj.N_x, obj.N_p);
            end

            particle_associations = zeros(obj.N_t, obj.N_p); % [N_t x N_p] matrix
            particle_weights = zeros(1, obj.N_p);
            particle_association_histories = cell(1, obj.N_p);

            for i = 1:obj.N_p
                % Extract states for each target
                for t = 1:obj.N_t
                    particle_states_cell{t}(:, i) = obj.particles{i}.kfs{t}.x;
                end

                particle_associations(:, i) = obj.particles{i}.associations;
                particle_weights(i) = obj.particles{i}.weight;
                particle_association_histories{i} = obj.particles{i}.association_history;
            end

            % Get current estimates (cell array per target)
            [x_est_cell, P_est_cell] = obj.getGaussianEstimate();

            % Store in history
            obj.history(k).measurements = measurements;
            obj.history(k).true_state = true_state;
            obj.history(k).true_meas_flag = true_meas_flag;
            obj.history(k).particle_states = particle_states_cell; % Cell {1 x N_t} of [N_x x N_p]
            obj.history(k).particle_associations = particle_associations; % [N_t x N_p]
            obj.history(k).particle_weights = particle_weights;
            obj.history(k).particle_association_histories = particle_association_histories;
            obj.history(k).estimate = x_est_cell; % Cell {1 x N_t}
            obj.history(k).covariance = P_est_cell; % Cell {1 x N_t}
            obj.history(k).ESS = ESS_before_resample;

        end

        function prediction(obj)
            % PREDICTION Propagate particle Kalman filters through dynamics
            %
            % MODIFIES:
            %   Each particle's KF state and covariance (for ALL targets)
            %
            % ALGORITHM:
            %   For i = 1:N_p
            %       For t = 1:N_t
            %           particles{i}.kfs{t}.predict()
            %       End
            %   End

            % Loop over particles
            for i = 1:obj.N_p
                % Loop over targets within each particle
                for t = 1:obj.N_t
                    obj.particles{i}.kfs{t}.prediction();
                end

            end

            % Debug output if enabled
            if obj.debug
                disp('Prediction step completed for all particles and targets.');
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
                case 'optimal'
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
            % MEASUREMENT_UPDATE Update particle KFs and compute weights (MULTI-TARGET)
            %
            % INPUTS:
            %   z - Measurements [N_z x N_measurements]
            %
            % MODIFIES:
            %   Each particle's KF states, covariances, and weight
            %
            % ALGORITHM:
            %   For i = 1:N_p
            %       particle_weight = 1
            %       For t = 1:N_t (each target)
            %           c_t = particles{i}.associations(t)
            %           If c_t > 0 (detection hypothesis for target t)
            %               z_t = z(:, c_t)
            %               particles{i}.kfs{t}.update(z_t)
            %               particle_weight *= P_D * likelihood_from_KF_innovation
            %           Else (clutter/missed detection for target t)
            %               particle_weight *= (1 - P_D) * clutter_density
            %           End
            %       End
            %       particles{i}.weight = particle_weight
            %   End
            %   Normalize weights

            for i = 1:obj.N_p
                % Update each target's KF using its sampled association
                for t = 1:obj.N_t
                    assoc_t = obj.particles{i}.associations(t);
                    if assoc_t > 0
                        obj.particles{i}.kfs{t}.measurement_update(z(:, assoc_t));
                    end
                    % assoc_t == 0: missed detection — KF keeps predicted state
                end

                % Per Sarkka RBPF: w_k^i = w_{k-1}^i * Z^i
                % Z^i is the normalizing constant of the optimal importance distribution
                % p(z_k | x_{k|k-1}^i), computed in generateAssociations_optimalimportancedist.
                obj.particles{i}.weight = obj.particles{i}.weight * obj.particles{i}.Z_i;
            end

            % Normalize weights across all particles
            total_weight = sum(cellfun(@(p) p.weight, obj.particles));

            if total_weight < eps || ~isfinite(total_weight)
                % Weight collapse — reset to uniform so tracking can recover
                if obj.debug
                    warning('KF_RBPF_multi:WeightCollapse', ...
                        'All particle weights collapsed at step %d — resetting to uniform.', ...
                        obj.timestep_counter + 1);
                end
                for i = 1:obj.N_p
                    obj.particles{i}.weight = 1 / obj.N_p;
                end
            else
                for i = 1:obj.N_p
                    obj.particles{i}.weight = obj.particles{i}.weight / total_weight;
                end
            end

        end

        function generateAssociations_uniform(obj, z)
            % GENERATEASSOCIATIONS_UNIFORM Uniform random association sampling
            % Each target independently draws a random association in [0, N_meas].
            N_measurements = size(z, 2);

            for i = 1:obj.N_p
                % Sample an independent random association for each target
                assoc_vec = zeros(obj.N_t, 1);
                for t = 1:obj.N_t
                    assoc_vec(t) = randi([0, N_measurements]);
                end
                obj.particles{i}.associations = assoc_vec;
                obj.particles{i}.association_history = ...
                    [obj.particles{i}.association_history, assoc_vec];
            end

        end

        function generateAssociations_optimalimportancedist(obj, z)
            % GENERATEASSOCIATIONS_OPTIMALIMPORTANCEDIST Multi-target optimal importance distribution
            %
            % ALGORITHM (Multi-Target Joint Data Association):
            %   The optimal importance distribution for multi-target tracking is:
            %
            %   q(c_{1:N_t}^i | y_{1:k}, c_{1:k-1}^i) ∝ ∏_{t=1}^{N_t} p(y_k | c_t^i, y_{1:k-1}, c_{1:k-1}^i) * p(c_t^i)
            %
            %   subject to the EXCLUSIVITY CONSTRAINT: c_i ≠ c_j for all i ≠ j (i,j > 0)
            %
            %   For 2 targets (generalizes to N_t targets):
            %   - Build association tensor A[i,j] where:
            %       * i = association for target 1 (0 = clutter, 1:M = measurements)
            %       * j = association for target 2 (0 = clutter, 1:M = measurements)
            %
            %   - Compute joint weights:
            %       * If i = j AND i > 0: A[i,j] = 0 (EXCLUSIVITY: same measurement forbidden)
            %       * If i = 0 OR j = 0: A[i,j] = p(c_1=i) * p(c_2=j) (at least one clutter)
            %       * If i ≠ j AND i,j > 0: A[i,j] = p(c_1=i) * p(c_2=j) (different measurements OK)
            %
            %   where p(c_t = m) = P_D * N(y_m - H*x_t; 0, S_t) for m > 0
            %         p(c_t = 0) = (1 - P_D) for clutter
            %
            % NOTE: This is computed PER PARTICLE since each has different predicted states.

            N_measurements = size(z, 2);

            % Clutter model parameters
            clutter_density = 1; % Uniform clutter (can be made configurable)
            p_clutter = (1 - obj.PD); % Weight for clutter/missed detection

            % For each particle, sample joint association from exclusivity-constrained distribution
            for i = 1:obj.N_p
                % Build multi-target association tensor
                % For N_t targets and M measurements, need (M+1)^N_t tensor
                % Dimension: each target can associate with 0 (clutter) or 1:M (measurements)

                Z_i = 0; % normalizing constant of optimal proposal (filled by each branch)

                if obj.N_t == 1
                    % Single target - use original algorithm
                    association_weights = zeros(1, N_measurements + 1);

                    for j = 1:N_measurements
                        z_j = z(:, j);
                        [innov, S] = obj.particles{i}.kfs{1}.getInnovation(z_j);
                        association_weights(j) = obj.PD * mvnpdf(innov', zeros(1, obj.N_z), S);
                    end

                    association_weights(end) = p_clutter;

                    % Normalizing constant for RBPF weight (Sarkka: w^i *= Z^i)
                    Z_i = sum(association_weights);
                    if Z_i < eps, Z_i = eps; end
                    association_weights = association_weights / Z_i;
                    cumulative_weights = cumsum(association_weights);
                    cumulative_weights(end) = 1.0;  % Clamp for floating-point safety
                    r = rand();
                    sampled_idx = find(cumulative_weights >= r, 1);
                    if isempty(sampled_idx), sampled_idx = numel(cumulative_weights); end
                    obj.particles{i}.associations(1) = (sampled_idx <= N_measurements) * sampled_idx;

                elseif obj.N_t == 2
                    % Two targets - use 2D association matrix with exclusivity
                    % Rows: target 1 associations (0:M)
                    % Cols: target 2 associations (0:M)
                    assoc_matrix = zeros(N_measurements + 1, N_measurements + 1);

                    % Compute marginal weights for each target-measurement pair
                    target1_weights = zeros(1, N_measurements + 1);
                    target2_weights = zeros(1, N_measurements + 1);

                    % Target 1 weights: index 1 = missed detection, index m+1 = measurement m
                    target1_weights(1) = p_clutter;
                    for m = 1:N_measurements
                        [innov, S] = obj.particles{i}.kfs{1}.getInnovation(z(:, m));
                        target1_weights(m+1) = obj.PD * mvnpdf(innov', zeros(1, obj.N_z), S);
                    end

                    % Target 2 weights: index 1 = missed detection, index m+1 = measurement m
                    target2_weights(1) = p_clutter;
                    for m = 1:N_measurements
                        [innov, S] = obj.particles{i}.kfs{2}.getInnovation(z(:, m));
                        target2_weights(m+1) = obj.PD * mvnpdf(innov', zeros(1, obj.N_z), S);
                    end

                    % Build joint distribution with exclusivity constraint
                    for c1 = 0:N_measurements

                        for c2 = 0:N_measurements
                            idx1 = c1 + 1; % MATLAB 1-indexing
                            idx2 = c2 + 1;

                            if (c1 > 0) && (c2 > 0) && (c1 == c2)
                                % EXCLUSIVITY: Same measurement for both targets is forbidden
                                assoc_matrix(idx1, idx2) = 0;
                            else
                                % Independent associations (product of marginals)
                                assoc_matrix(idx1, idx2) = target1_weights(idx1) * target2_weights(idx2);
                            end

                        end

                    end

                    % Flatten matrix to vector for sampling
                    assoc_vec = assoc_matrix(:);
                    Z_i = sum(assoc_vec);  % normalizing constant for RBPF weight
                    if Z_i < eps, Z_i = eps; end
                    assoc_vec = assoc_vec / Z_i;

                    % Sample from flattened distribution
                    cumulative_weights = cumsum(assoc_vec);
                    cumulative_weights(end) = 1.0;  % Clamp for floating-point safety
                    r = rand();
                    sampled_idx = find(cumulative_weights >= r, 1);
                    if isempty(sampled_idx), sampled_idx = numel(cumulative_weights); end

                    % Convert linear index back to (c1, c2)
                    [row, col] = ind2sub(size(assoc_matrix), sampled_idx);
                    obj.particles{i}.associations(1) = row - 1; % Convert back to 0-indexed
                    obj.particles{i}.associations(2) = col - 1;

                else
                    % General N_t targets - use recursive tensor construction
                    % Build N_t-dimensional tensor with exclusivity constraints
                    tensor_size = (N_measurements + 1) * ones(1, obj.N_t);
                    assoc_tensor = zeros(tensor_size);

                    % Compute marginal weights for each target
                    target_weights = zeros(obj.N_t, N_measurements + 1);

                    for t = 1:obj.N_t
                        % index 1 = missed detection, index m+1 = measurement m
                        target_weights(t, 1) = p_clutter;
                        for m = 1:N_measurements
                            [innov, S] = obj.particles{i}.kfs{t}.getInnovation(z(:, m));
                            target_weights(t, m+1) = obj.PD * mvnpdf(innov', zeros(1, obj.N_z), S);
                        end
                    end

                    % Enumerate all possible joint associations
                    % For N_t targets, we have (M+1)^N_t hypotheses
                    num_hypotheses = (N_measurements + 1) ^ obj.N_t;
                    hypothesis_weights = zeros(1, num_hypotheses);
                    hypothesis_assocs = zeros(num_hypotheses, obj.N_t);

                    for h = 1:num_hypotheses
                        % Convert linear index to multi-dimensional association
                        sub_indices = cell(1, obj.N_t);
                        [sub_indices{:}] = ind2sub(tensor_size, h);
                        assoc_hypothesis = cell2mat(sub_indices) - 1; % 0-indexed associations
                        hypothesis_assocs(h, :) = assoc_hypothesis;

                        % Check exclusivity constraint
                        meas_assocs = assoc_hypothesis(assoc_hypothesis > 0);

                        if length(meas_assocs) ~= length(unique(meas_assocs))
                            % Duplicate measurement assignments - forbidden
                            hypothesis_weights(h) = 0;
                        else
                            % Valid hypothesis - compute product of marginals
                            weight = 1;

                            for t = 1:obj.N_t
                                weight = weight * target_weights(t, assoc_hypothesis(t) + 1);
                            end

                            hypothesis_weights(h) = weight;
                        end

                    end

                    % Normalize and sample (Z_i = normalizing constant for RBPF weight)
                    Z_i = sum(hypothesis_weights);
                    if Z_i < eps, Z_i = eps; end
                    hypothesis_weights = hypothesis_weights / Z_i;
                    cumulative_weights = cumsum(hypothesis_weights);
                    cumulative_weights(end) = 1.0;  % Clamp for floating-point safety
                    r = rand();
                    sampled_h = find(cumulative_weights >= r, 1);
                    if isempty(sampled_h), sampled_h = num_hypotheses; end

                    % Extract sampled association
                    obj.particles{i}.associations = hypothesis_assocs(sampled_h, :)';
                end

                % Store normalizing constant for use in measurement_update
                obj.particles{i}.Z_i = Z_i;

                % Store association history
                obj.particles{i}.association_history = [obj.particles{i}.association_history, obj.particles{i}.associations];
            end

        end

        function generateAssociations_likelihood(obj, z)
            % GENERATEASSOCIATIONS_LIKELIHOOD Likelihood-based association sampling

            % PASS FOR THE MOMENT
            fprintf('Likelihood-based association generation not yet implemented.\n');
            fprintf('Using uniform random associations instead.\n');
            obj.generateAssociations_uniform(z);
        end

        function resample(obj)
            % RESAMPLE Systematic resampling of particles (MULTI-TARGET)
            %
            % MODIFIES:
            %   obj.particles - Resampled particle array
            %
            % ALGORITHM:
            %   1. Check ESS = 1 / sum(weights^2)
            %   2. If ESS < threshold:
            %        - Systematic resampling to get new indices
            %        - Create new particle array
            %        - CRITICAL: Deep copy ALL KF objects for ALL targets!
            %        - Reset weights to uniform
            %
            % CRITICAL BUG TO AVOID:
            %   DO NOT just copy particle structs directly!
            %   Must use KF.copyKF() to create independent KF objects for EACH target.
            %   Otherwise resampled particles share same KF (disaster).

            % Compute Effective Sample Size (ESS)
            weights = cellfun(@(p) p.weight, obj.particles);
            ESS = 1 / sum(weights .^ 2);
            ESS_threshold = obj.ESS_threshold_percentage * obj.N_p;

            if ESS < ESS_threshold
                % Systematic resampling
                cumulative_weights = cumsum(weights);
                cumulative_weights(end) = 1.0;  % Force exact 1 for floating-point safety
                step = 1 / obj.N_p;
                start = rand * step;
                positions = start + (0:obj.N_p-1) * step;  % Always exactly N_p elements

                new_particles = cell(1, obj.N_p);
                index = 1;

                for i = 1:obj.N_p

                    while positions(i) > cumulative_weights(index) && index < obj.N_p
                        index = index + 1;
                    end

                    % Deep copy ALL KFs for this particle (one per target)
                    new_kfs = cell(1, obj.N_t);

                    for t = 1:obj.N_t
                        new_kfs{t} = KF.copyKF(obj.particles{index}.kfs{t});
                    end

                    % Create new particle with deep-copied KFs
                    new_particles{i} = struct( ...
                        'associations', obj.particles{index}.associations, ...
                        'kfs', {new_kfs}, ...
                        'weight', 1 / obj.N_p, ...
                        'association_history', obj.particles{index}.association_history);
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

        function [x_est_cell, P_est_cell] = getGaussianEstimate(obj)
            % GETGAUSSIANESTIMATE Extract weighted mean state estimates (MULTI-TARGET)
            %
            % OUTPUTS:
            %   x_est_cell - Cell array {1 x N_t} of weighted mean states [N_x x 1] per target
            %   P_est_cell - Cell array {1 x N_t} of weighted covariances [N_x x N_x] per target
            %
            % ALGORITHM:
            %   For each target t:
            %     x_est{t} = sum_i w_i * x_i^t
            %     P_est{t} = sum_i w_i * (P_i^t + (x_i^t - x_est{t})*(x_i^t - x_est{t})')

            x_est_cell = cell(1, obj.N_t);
            P_est_cell = cell(1, obj.N_t);

            % Compute estimates for each target independently
            for t = 1:obj.N_t
                x_est = zeros(obj.N_x, 1);
                P_est = zeros(obj.N_x, obj.N_x);

                % Compute weighted mean
                for i = 1:obj.N_p
                    w_i = obj.particles{i}.weight;
                    x_i = obj.particles{i}.kfs{t}.x;
                    x_est = x_est + w_i * x_i;
                end

                % Compute weighted covariance
                for i = 1:obj.N_p
                    w_i = obj.particles{i}.weight;
                    x_i = obj.particles{i}.kfs{t}.x;
                    P_i = obj.particles{i}.kfs{t}.P;

                    diff = x_i - x_est;
                    P_est = P_est + w_i * (P_i + diff * diff');
                end

                x_est_cell{t} = x_est;
                P_est_cell{t} = P_est;
            end

        end

        function visualize(obj, true_states, measurements)
            % VISUALIZE Plot multi-target particle distribution and estimates
            %
            % INPUTS:
            %   true_states  - (optional) Ground truth cell {1xN_t} or [N_x x N_t]
            %   measurements - (optional) Current measurements [N_z x N_meas]
            %
            % CREATES:
            %   1x4 layout:
            %   Subplot 1: Particles colored by weight (position space)
            %   Subplot 2: Particles colored by association (position space)
            %   Subplot 3: Velocity estimates (if N_x >= 4)
            %   Subplot 4: Association histogram

            % Handle optional arguments
            if nargin < 2, true_states = []; end
            if nargin < 3, measurements = []; end

            % Normalize true_states to cell array {1 x N_t}
            if ~isempty(true_states) && ~iscell(true_states)
                % Matrix [N_x x N_t] passed — convert to cell
                gt_cell = cell(1, obj.N_t);
                for t = 1:obj.N_t
                    gt_cell{t} = true_states(:, t);
                end
                true_states = gt_cell;
            end

            % Create or reuse figure
            if isempty(obj.dynamic_figure_handle) || ~isvalid(obj.dynamic_figure_handle)
                obj.dynamic_figure_handle = figure('Name', 'KF-RBPF-Multi Tracking', ...
                    'NumberTitle', 'off', 'Position', [50, 50, 400*obj.N_t, 800]);
            else
                figure(obj.dynamic_figure_handle);
                clf;
            end

            % Get estimates and particle weights
            [x_est_cell, P_est_cell] = obj.getGaussianEstimate();
            weights = cellfun(@(p) p.weight, obj.particles);
            ESS = 1 / sum(weights .^ 2);
            colors = lines(obj.N_t);

            % Compute global spatial bounds from all particles + measurements
            all_x = [];  all_y = [];
            for i = 1:obj.N_p
                for t = 1:obj.N_t
                    pos = obj.particles{i}.kfs{t}.x(1:2);
                    all_x(end+1) = pos(1); %#ok<AGROW>
                    all_y(end+1) = pos(2); %#ok<AGROW>
                end
            end
            if ~isempty(measurements)
                all_x = [all_x, measurements(1,:)];
                all_y = [all_y, measurements(2,:)];
            end
            if ~isempty(true_states)
                for t = 1:obj.N_t
                    all_x(end+1) = true_states{t}(1);
                    all_y(end+1) = true_states{t}(2);
                end
            end

            if ~isempty(obj.plot_bounds_x)
                Xb = obj.plot_bounds_x; Yb = obj.plot_bounds_y;
            else
                xr = max(all_x) - min(all_x); yr = max(all_y) - min(all_y);
                mg = 0.15;
                Xb = [min(all_x)-mg*max(xr,0.5), max(all_x)+mg*max(xr,0.5)];
                Yb = [min(all_y)-mg*max(yr,0.5), max(all_y)+mg*max(yr,0.5)];
            end

            %% ROW 1: one subplot per target — position clouds
            for t = 1:obj.N_t
                subplot(2, obj.N_t, t);
                cla; hold on;

                % Particle positions for this target, colored by weight
                pos_t = zeros(2, obj.N_p);
                assoc_t = zeros(1, obj.N_p);
                for i = 1:obj.N_p
                    pos_t(:,i) = obj.particles{i}.kfs{t}.x(1:2);
                    assoc_t(i) = obj.particles{i}.associations(t);
                end

                scatter(pos_t(1,:), pos_t(2,:), 15, weights, 'filled', ...
                    'MarkerFaceAlpha', 0.5);
                colormap('hot');

                % Estimate + 1-sigma ellipse
                xe = x_est_cell{t};
                Pe = P_est_cell{t};
                plot(xe(1), xe(2), 'o', 'Color', colors(t,:), ...
                    'MarkerSize', 10, 'LineWidth', 2.5, 'MarkerFaceColor', colors(t,:));
                th = linspace(0, 2*pi, 100);
                try
                    L = chol(Pe(1:2,1:2))';
                    ell = xe(1:2) + L * [cos(th); sin(th)];
                    plot(ell(1,:), ell(2,:), '-', 'Color', colors(t,:), 'LineWidth', 1.5);
                catch, end

                % Ground truth
                if ~isempty(true_states)
                    gt = true_states{t};
                    plot(gt(1), gt(2), 'd', 'Color', colors(t,:), ...
                        'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'w');
                end

                % Measurements
                if ~isempty(measurements)
                    plot(measurements(1,:), measurements(2,:), 'r+', ...
                        'MarkerSize', 8, 'LineWidth', 1.5);
                end

                xlim(Xb); ylim(Yb);
                xlabel('X (m)'); ylabel('Y (m)');
                title(sprintf('Target %d  |  ESS=%.0f', t, ESS));
                grid on; axis square;
            end

            %% ROW 2: one subplot per target — MAP association bar
            for t = 1:obj.N_t
                subplot(2, obj.N_t, obj.N_t + t);
                cla; hold on;

                assoc_t = zeros(1, obj.N_p);
                for i = 1:obj.N_p
                    assoc_t(i) = obj.particles{i}.associations(t);
                end

                max_a = max(assoc_t);
                if max_a < 1, max_a = 1; end
                counts = histcounts(assoc_t, 'BinEdges', -0.5:(max_a+0.5));
                bar(0:max_a, counts, 'FaceColor', colors(t,:));

                xlabel('Association (0=miss)'); ylabel('Count');
                title(sprintf('T%d Assoc. Dist.', t));
                if max_a <= 10, xticks(0:max_a); end
                grid on;
            end

            sgtitle(sprintf('KF-RBPF-Multi | k=%d | Strategy=%s | N_p=%d | N_t=%d', ...
                obj.timestep_counter, obj.association_strategy, obj.N_p, obj.N_t), ...
                'FontSize', 12, 'FontWeight', 'bold');
            drawnow;

            % GIF export
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

            if obj.DynamicPlot, pause(0.1); end

        end

    end

    methods (Static)
        % Helper functions can go here

        function particles = initialize_particles(num_particles, x0_cell, F, Q, H, R, uniform_init, sigma_pos, sigma_vel)
            % INITIALIZE_PARTICLES Helper to create multi-target particle array
            %
            % INPUTS:
            %   num_particles - Number of particles
            %   x0_cell       - Cell array {1 x N_targets} of initial state estimates [N_x x 1]
            %   F, Q, H, R    - System model matrices
            %   uniform_init  - (optional) Boolean: if true, use uniform initialization
            %   sigma_pos     - (optional) Gaussian init position std-dev (default 5.0 m)
            %   sigma_vel     - (optional) Gaussian init velocity std-dev (default 2.0 m/s)
            % OUTPUTS:
            %   particles - Cell array of initialized particle structs

            if nargin < 7, uniform_init = false; end
            if nargin < 8, sigma_pos = 5.0; end
            if nargin < 9, sigma_vel = 2.0; end

            particles = cell(1, num_particles);
            num_targets = length(x0_cell);
            N_x = length(x0_cell{1}); % State dimension (assumed same for all targets)

            if uniform_init
                % UNIFORM INITIALIZATION across state space
                pos_bounds = [-2, 2; 0.5, 4]; % [x_min, x_max; y_min, y_max]
                vel_bounds = [-2, 2; -2, 2]; % [vx_min, vx_max; vy_min, vy_max]
                acc_bounds = [-2, 2; -2, 2]; % [ax_min, ax_max; ay_min, ay_max]

                initial_kf_cov = 10 * eye(N_x); % Conservative covariance

                for i = 1:num_particles
                    kfs = cell(1, num_targets);

                    for t = 1:num_targets
                        % Initialize each target state uniformly
                        x_particle = zeros(N_x, 1);

                        % Position (x, y)
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

                        kfs{t} = KF(x_particle, initial_kf_cov, F, Q, H, R);
                    end

                    particles{i} = struct( ...
                        'associations', zeros(num_targets, 1), ... % [N_t x 1] associations (0 = clutter)
                        'kfs', {kfs}, ... % Cell {1 x N_t} of KF objects
                        'weight', 1 / num_particles, ... % Uniform initial weight
                        'association_history', [], ... % History: [N_t x K] matrix over time
                        'Z_i', 1); % RBPF normalizing constant (set by generateAssociations)
                end

            else
                % GAUSSIAN INITIALIZATION around provided initial states
                initial_uncertainty_std = ones(N_x, 1);
                initial_uncertainty_std(1:min(2, N_x))     = sigma_pos;  % x, y
                if N_x >= 4
                    initial_uncertainty_std(3:4) = sigma_vel;             % vx, vy
                end
                % Acceleration states (if 6-DOF): default small uncertainty
                if N_x >= 6
                    initial_uncertainty_std(5:6) = 1.0;
                end

                initial_particle_cov = diag(initial_uncertainty_std .^ 2);
                initial_kf_cov = diag(initial_uncertainty_std .^ 2);  % KF init matches particle spread

                for i = 1:num_particles
                    kfs = cell(1, num_targets);

                    for t = 1:num_targets
                        % Sample particle state from distribution around x0{t}
                        x_particle = x0_cell{t} + mvnrnd(zeros(N_x, 1), initial_particle_cov)';
                        kfs{t} = KF(x_particle, initial_kf_cov, F, Q, H, R);
                    end

                    particles{i} = struct( ...
                        'associations', zeros(num_targets, 1), ... % [N_t x 1] associations
                        'kfs', {kfs}, ... % Cell {1 x N_t} of KF objects
                        'weight', 1 / num_particles, ... % Uniform initial weight
                        'association_history', [], ... % History storage
                        'Z_i', 1); % RBPF normalizing constant (set by generateAssociations)
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
