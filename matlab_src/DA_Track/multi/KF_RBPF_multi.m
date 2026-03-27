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
            addParameter(p, 'PD', 0.95, @(x) x >= 0 && x <= 1);
            addParameter(p, 'PFA', 0.05, @(x) x >= 0 && x <= 1);
            addParameter(p, 'ESSThreshold', 0.5, @(x) x > 0 && x <= 1);
            addParameter(p, 'AssociationStrategy', 'optimal', @(x) ismember(x, {'uniform', 'optimal', 'likelihood'}));
            addParameter(p, 'UniformInit', false, @islogical);
            parse(p, varargin{:});

            obj.debug = p.Results.Debug;
            obj.PD = p.Results.PD;
            obj.PFA = p.Results.PFA;
            obj.ESS_threshold_percentage = p.Results.ESSThreshold;
            obj.association_strategy = p.Results.AssociationStrategy;
            obj.N_t = length(x0_cell); % Number of targets from initial states
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
            obj.particles = KF_RBPF_multi.initialize_particles(N_particles, x0_cell, F, Q, H, R, uniform_init);

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

            clutter_density = 1; % Placeholder for clutter model

            for i = 1:obj.N_p
                % Initialize particle weight (will be product over all targets)
                particle_weight = 1;

                % Process each target
                for t = 1:obj.N_t
                    assoc_t = obj.particles{i}.associations(t);

                    if assoc_t > 0 % Detection hypothesis for target t
                        % Measurement associated with target t
                        z_t = z(:, assoc_t);
                        obj.particles{i}.kfs{t}.measurement_update(z_t);

                        % Compute likelihood contribution from this target
                        innov = obj.particles{i}.kfs{t}.z; % Innovation
                        S = obj.particles{i}.kfs{t}.S; % Innovation covariance
                        likelihood = mvnpdf(innov', zeros(1, obj.N_z), S);
                        particle_weight = particle_weight * (obj.PD * likelihood);

                    else % Clutter/missed detection for target t
                        % No measurement update for this target - keep predicted state
                        % Weight contribution from missed detection
                        particle_weight = particle_weight * ((1 - obj.PD) * clutter_density);
                    end

                end

                % Assign computed weight to particle
                obj.particles{i}.weight = particle_weight;
            end

            % Normalize weights across all particles
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

                if obj.N_t == 1
                    % Single target - use original algorithm
                    association_weights = zeros(1, N_measurements + 1);

                    for j = 1:N_measurements
                        z_j = z(:, j);
                        [innov, S] = obj.particles{i}.kfs{1}.getInnovation(z_j);
                        association_weights(j) = obj.PD * mvnpdf(innov', zeros(1, obj.N_z), S);
                    end

                    association_weights(end) = p_clutter;

                    % Normalize and sample
                    association_weights = association_weights / sum(association_weights);
                    cumulative_weights = cumsum(association_weights);
                    r = rand();
                    sampled_idx = find(cumulative_weights >= r, 1);
                    obj.particles{i}.associations(1) = (sampled_idx <= N_measurements) * sampled_idx;

                elseif obj.N_t == 2
                    % Two targets - use 2D association matrix with exclusivity
                    % Rows: target 1 associations (0:M)
                    % Cols: target 2 associations (0:M)
                    assoc_matrix = zeros(N_measurements + 1, N_measurements + 1);

                    % Compute marginal weights for each target-measurement pair
                    target1_weights = zeros(1, N_measurements + 1);
                    target2_weights = zeros(1, N_measurements + 1);

                    % Target 1 weights
                    for m = 1:N_measurements
                        [innov, S] = obj.particles{i}.kfs{1}.getInnovation(z(:, m));
                        target1_weights(m) = obj.PD * mvnpdf(innov', zeros(1, obj.N_z), S);
                    end

                    target1_weights(end) = p_clutter;

                    % Target 2 weights
                    for m = 1:N_measurements
                        [innov, S] = obj.particles{i}.kfs{2}.getInnovation(z(:, m));
                        target2_weights(m) = obj.PD * mvnpdf(innov', zeros(1, obj.N_z), S);
                    end

                    target2_weights(end) = p_clutter;

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
                    assoc_vec = assoc_vec / sum(assoc_vec); % Normalize

                    % Sample from flattened distribution
                    cumulative_weights = cumsum(assoc_vec);
                    r = rand();
                    sampled_idx = find(cumulative_weights >= r, 1);

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

                        for m = 1:N_measurements
                            [innov, S] = obj.particles{i}.kfs{t}.getInnovation(z(:, m));
                            target_weights(t, m) = obj.PD * mvnpdf(innov', zeros(1, obj.N_z), S);
                        end

                        target_weights(t, end) = p_clutter;
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

                    % Normalize and sample
                    hypothesis_weights = hypothesis_weights / sum(hypothesis_weights);
                    cumulative_weights = cumsum(hypothesis_weights);
                    r = rand();
                    sampled_h = find(cumulative_weights >= r, 1);

                    % Extract sampled association
                    obj.particles{i}.associations = hypothesis_assocs(sampled_h, :)';
                end

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
                step = 1 / obj.N_p;
                start = rand * step;
                positions = start:step:1;

                new_particles = cell(1, obj.N_p);
                index = 1;

                for i = 1:obj.N_p

                    while positions(i) > cumulative_weights(index)
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

    end

    methods (Static)
        % Helper functions can go here

        function particles = initialize_particles(num_particles, x0_cell, F, Q, H, R, uniform_init)
            % INITIALIZE_PARTICLES Helper to create multi-target particle array
            %
            % INPUTS:
            %   num_particles - Number of particles
            %   x0_cell       - Cell array {1 x N_targets} of initial state estimates [N_x x 1]
            %   F, Q, H, R    - System model matrices
            %   uniform_init  - (optional) Boolean: if true, use uniform initialization
            % OUTPUTS:
            %   particles - Cell array of initialized particle structs
            %
            % ALGORITHM:
            %   Each particle maintains N_targets Kalman filters (one per target).
            %   If uniform_init = false (default):
            %       Each target's KF is initialized with diverse states sampled
            %       from distribution around corresponding x0.
            %   If uniform_init = true:
            %       Each target's KF is uniformly distributed across state space.

            if nargin < 7
                uniform_init = false;
            end

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
                        'association_history', []); % History: [N_t x K] matrix over time
                end

            else
                % GAUSSIAN INITIALIZATION around provided initial states
                % Define initial uncertainty for particle diversity
                if N_x == 4
                    initial_uncertainty_std = [5.0; 5.0; 2.0; 2.0]; % [x, y, vx, vy]
                else
                    initial_uncertainty_std = ones(N_x, 1);
                    initial_uncertainty_std(1:min(2, N_x)) = 5.0; % Position

                    if N_x >= 4
                        initial_uncertainty_std(3:4) = 2.0; % Velocity
                    end

                end

                initial_particle_cov = diag(initial_uncertainty_std .^ 2);
                initial_kf_cov = 10 * eye(N_x); % Conservative KF covariance

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
                        'association_history', []); % History storage
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
