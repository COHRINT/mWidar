classdef GNN_PF < handle
    % GNN_PF Global Nearest Neighbor Particle Filter
    %
    % DESCRIPTION:
    %   Implements a hybrid particle filter combining GNN data association
    %   with particle filtering for single target tracking. Supports both
    %   standard Gaussian likelihood and precomputed spatial likelihood
    %   lookup tables for radar/sensor fusion applications.
    %
    % PROPERTIES:
    %   N_p, N_x, N_z        - Filter dimensions (particles, states, measurements)
    %   particles, weights   - Particle filter state [N_x x N_p], [N_p x 1]
    %   F, Q, H             - System model matrices
    %   pointlikelihood_image - Precomputed likelihood lookup table
    %   debug, validate     - Control flags for debugging and validation
    %
    % METHODS:
    %   GNN_PF              - Constructor
    %   timestep            - Process single time step with GNN algorithm
    %   prediction          - Particle prediction step
    %   resample            - Bootstrap particle resampling
    %
    % EXAMPLE:
    %   pf = GNN_PF(x0, 1000, F, Q, H);
    %   [x_est, P_est] = pf.timestep(x_prior, P_prior, measurements);
    %
    % See also particleFilter, trackingPF

    properties
        % Filter Parameters
        N_p % Number of particles
        N_x % State dimension
        N_z % Measurement dimension

        % Filter State
        particles % Particle states [N_x x N_p]
        weights % Particle weights [N_p x 1]

        % System Model
        F % State transition matrix [N_x x N_x]
        Q % Process noise covariance [N_x x N_x]
        H % Measurement matrix [N_z x N_x]

        % Likelihood Model
        pointlikelihood_image % Precomputed likelihood lookup table [128^2 x 128^2]

        % Control Flags
        debug = false % Enable debug output and validation
        validate = false % Enable input/output validation checks
    end

    methods

        function obj = GNN_PF(x0, N_particles, F, Q, H, pointlikelihood_image)
            % GNN_PF Constructor for Global Nearest Neighbor Particle Filter
            %
            % SYNTAX:
            %   obj = GNN_PF(x0, N_particles, F, Q, H)
            %   obj = GNN_PF(x0, N_particles, F, Q, H, pointlikelihood_image)
            %
            % INPUTS:
            %   x0                  - Initial state estimate [N_x x 1]
            %   N_particles         - Number of particles (integer > 0)
            %   F                   - State transition matrix [N_x x N_x]
            %   Q                   - Process noise covariance [N_x x N_x]
            %   H                   - Measurement matrix [N_z x N_x]
            %   pointlikelihood_image - (optional) Precomputed likelihood lookup table [128^2 x 128^2]
            %
            % OUTPUTS:
            %   obj - Initialized GNN_PF object
            %
            % DESCRIPTION:
            %   Creates and initializes a GNN particle filter with uniform particle
            %   distribution around the initial state. Uses provided likelihood
            %   table for hybrid filtering applications.
            %
            % NOTE:
            %   Load likelihood table in calling script using:
            %   load('supplemental/precalc_imagegridHMMEmLike.mat', 'pointlikelihood_image');
            %
            % See also timestep, prediction, resample

            % Validate inputs
            if nargin < 5
                error('GNN_PF:InvalidInput', 'Minimum 5 inputs required: {x0, N_particles, F, Q, H}');
            end

            if nargin < 6 || isempty(pointlikelihood_image)
                % Initialize with empty matrix - will be set later if needed
                pointlikelihood_image = [];
            end

            % Initialize basic properties
            obj.N_p = N_particles;
            obj.N_x = size(x0, 1);
            obj.N_z = size(H, 1);

            % Initialize particles with small spread around initial guess
            init_spread = [0.1, 0.1, 0.25, 0.25, 0.5, 0.5]; % Position, velocity, acceleration spreads
            obj.particles = repmat(x0(:), 1, N_particles);

            % Add Gaussian noise to each state component
            for i = 1:obj.N_x
                obj.particles(i, :) = obj.particles(i, :) + init_spread(i) * randn(1, N_particles);
            end

            obj.weights = ones(N_particles, 1) / N_particles;

            % Store system matrices
            obj.F = F;
            obj.Q = Q;
            obj.H = H;

            % Store likelihood lookup table
            obj.pointlikelihood_image = pointlikelihood_image;

            % Validate dimensions if likelihood image is provided
            if ~isempty(pointlikelihood_image)
                expected_dim = 128 ^ 2;
                [rows, cols] = size(obj.pointlikelihood_image);

                if obj.debug
                    fprintf('Debug: Loaded likelihood table - Rows: %d, Cols: %d\n', rows, cols);
                    fprintf('Debug: Expected dimensions: %d x %d\n', expected_dim, expected_dim);
                end

                if rows ~= expected_dim || cols ~= expected_dim
                    warning('GNN_PF:DimensionMismatch', ...
                        'Likelihood model dimensions (%dx%d) do not match expected grid size (%dx%d)', ...
                        rows, cols, expected_dim, expected_dim);
                end

            end

        end

        % timestep: Iterate GNN_PF through a single timestep
        %{
            args:
                obj -> class object
                ? -> prior distribution
                z -> measurement
            outputs:
                ? -> posterior distribution
        %}
        function timestep(obj, z)
            %{
             GNN_PF Timestep function -- Implements the GNN-PF algorithm for a single timestep
             0. Initialization -- Resample the particles (only if weights are not uniform)
             1. Prediction Step -- Particles are propogated through the dynamics model (weights unchanged)
             2. Validation Step -- Validate the particles against the measurements (gating)
                -- This step is optional, but can be used to reduce the number of measurement updates
             3. Data Association Step -- Compute the probability of each measurement being a true detection
                -- This step is optional given optimizations present for GNN, and is built into step 4
                -- HOWEVER, this MUST be done for GNN_PF as the maximum beta weight will be used to update the particles
              4. State Estimation Step -- Update the particle weights based on the measurements and associations
                 -- This step is required for both PDA and GNN, and is the final step in the algorithm
                 -- For PDA, all measurements will be passed through the state estimation step and mixed accordingly
                 -- For GNN and one-measurement-at-a-time, only one measurement will be passed through the state estimation step
            %}

            % Step 0: Initialization
            % Resample the particles if necessary (only if weights are not uniform)
            % This will only NOT be called if
            %    1) This is the first timestep (weights are uniform by default)
            %    2) The weights are already uniform (no measurement updates yet)
            if any(obj.weights ~= 1 / length(obj.weights))
                obj.resample();
            end

            % Step 1: Prediction Step
            % Particles are propagated through the dynamics model (weights unchanged)
            if obj.debug
                obj.printState('Before Prediction');
            end

            obj.prediction();

            if obj.debug
                obj.printState('After Prediction');
            end

            % Step 2: Validation step - Gate measurements using particle filter estimate
            % COMMENT OUT THIS ENTIRE BLOCK TO DISABLE GATING
            if ~isempty(z)
                % Get current particle filter estimate for gating
                [x_est, P_est] = obj.getGaussianEstimate();
                z_hat = x_est(1:2); % Predicted measurement (only position)
                S = P_est(1:2, 1:2); % Use position covariance only
                
                % Compute 2-sigma bounds for gating
                sigma_bounds = 2 * sqrt(diag(S));
                lower_bound = z_hat - sigma_bounds;
                upper_bound = z_hat + sigma_bounds;
                
                % Apply gating: Keep measurements within 2-sigma bounds
                in_bounds = all(z >= lower_bound & z <= upper_bound, 1);
                z_valid = z(:, in_bounds);
                has_valid_meas = ~isempty(z_valid);
                
                % if obj.debug
                    fprintf('Debug: Gating - Input: %d measurements, Output: %d valid measurements\n', ...
                        size(z, 2), size(z_valid, 2));
                % end
                
                % Use gated measurements for state estimation
                if has_valid_meas
                    z_to_process = z_valid;
                else
                    z_to_process = []; % No valid measurements - missed detection
                end
            else
                z_to_process = z; % No measurements to gate
            end
            % END GATING BLOCK - COMMENT OUT TO HERE TO DISABLE GATING
            
            % UNCOMMENT THIS LINE TO DISABLE GATING (and comment out the block above)
            % z_to_process = z; % Use all measurements without gating

            % Step 3: Data Association Step -- Simplified, no beta computation needed

            % Step 4: State Estimation Step
            obj.state_Estimation(z_to_process);

            if obj.debug
                obj.printState('After Measurement Update');
            end

        end

        function prediction(obj)
            % PREDICTION Propagate particles through system dynamics
            %
            % SYNTAX:
            %   obj.prediction()
            %
            % DESCRIPTION:
            %   Applies state transition model to all particles with additive
            %   process noise. Uses vectorized operations for computational
            %   efficiency. Particle weights remain unchanged.
            %
            % MODIFIES:
            %   obj.particles - Updated particle states [N_x x N_p]
            %
            % ALGORITHM:
            %   particles = F * particles + process_noise
            %   where process_noise ~ N(0, Q) for each particle
            %
            % See also timestep, resample

            if obj.debug
                fprintf('Debug: F matrix [%dx%d], particles [%dx%d]\n', ...
                    size(obj.F, 1), size(obj.F, 2), size(obj.particles, 1), size(obj.particles, 2));
            end

            % Propagate particles through state transition model
            obj.particles = obj.F * obj.particles;

            % Add process noise to each particle
            process_noise = mvnrnd(zeros(1, obj.N_x), obj.Q, obj.N_p)';
            obj.particles = obj.particles + process_noise;

        end

        % Validation (Gating) Step: Trims down detections to a set of detections that pass through the validation ellipse
        %{
            % TODO: Decide if we want gating in these classes
            args:
                S -> innovation covariance
                z_hat -> predicted measurment
                z -> set of all detections
            outputs:
                valid_z -> set of detections that pass through validation ellipse
                meas -> boolean, set to false if no measurments pass through the ellipse (missed detection)

                %}

            function [valid_z, meas] = Validation(obj, S, z_hat, z)
                meas = true; % default
                gamma = chi2inv(0.05/2, 2); % Threshold
                valid_z = [];

                for j = 1:size(z, 2)
                    detection = z(:, j);
                    Nu = (detection - z_hat)' / S * (detection - z_hat); % Validation ellipse (NIS statistic)

                    if Nu < gamma % Validation gate
                        valid_z = [valid_z detection]; % Append new measurment onto validated list
                    end

                end

                if isempty(valid_z)
                    fprintf('No Valid Measurments \n')
                    meas = false; % missed detection
                end

            end
        

        % GNN: Standard Global Nearest Neighbor algorithm
        %{
            args:
                ? -> prior distribution
                O -> set of valid measurments

            outputs:
                beta -> probability of data association (size len(O)+1 x 1)
                    beta(0) = P(all clutter | z_hat)
                    beta(i) = P(i NOT clutter | z_hat)

        %}
        function [beta] = Data_Association(obj, z_hat, O, S)

            % % Tuning parameters
            % lambda = 2.5;
            % PD = 0.95;
            % PG = 0.95;

            % % Pre allocate space
            % likelihood = zeros(1, size(valid_z, 2));
            % beta = zeros(1, size(valid_z, 2));

            % % Compute likelihood of each validated measurment
            % for j = 1:size(valid_z, 2)
            %     likelihood(j) = (mvnpdf(valid_z(:, j), z_hat, S) * PD) / lambda;
            % end

            % sum_likelihood = sum(likelihood, 2); % sum

            % for j = 1:size(valid_z, 2)
            %     beta(j) = likelihood(j) / (1 - PD * PG + sum_likelihood); % Compute beta values
            % end

            % beta0 = (1 - PD * PG) / (1 - PD * PG + sum_likelihood); % beta0 -> probability of no detections being true

        end

        % state_Estimation: Hybrid_PF weight update and state estimation
        %{
            args:
                beta -> probability of detection i being the true target
                weights -> weights of particles
                O -> set of validated measurments

            outputs:
                posterior -> posterior weights

        %}
        function state_Estimation(obj, z)
            % STATE_ESTIMATION Update particle weights based on measurement likelihood (GNN)
            %
            % SYNTAX:
            %   obj.state_Estimation(z)
            %
            % INPUTS:
            %   z    - Current measurements [N_z x N_measurements] (after gating)
            %
            % DESCRIPTION:
            %   Implements simplified GNN weight update: w_final = sum(w_measurement_i)
            %   Sums likelihood contributions from all measurements.
            %   Handles missed detection case when z is empty.
            %
            % MODIFIES:
            %   obj.weights - Updated and normalized particle weights [N_p x 1]
            %
            % See also timestep, prediction, likelihoodLookup

            % Handle missed detection case (no valid measurements after gating)
            if isempty(z)
                if obj.debug
                    fprintf('Debug: Missed detection - no valid measurements after gating\n');
                end
                % Keep weights unchanged for missed detection
                return;
            end

            % Check if likelihood image is available
            if isempty(obj.pointlikelihood_image)
                error('GNN_PF:NoLikelihoodData', ...
                'Likelihood lookup table not provided. Load and pass to constructor.');
            end

            % Debug: Print measurement info
            if obj.debug
                fprintf('Debug: Processing %d valid measurements after gating\n', size(z, 2));

                for i = 1:size(z, 2)
                    fprintf('Debug: Valid measurement %d: [%.4f, %.4f]\n', i, z(1, i), z(2, i));
                end

            end

            % Initialize total likelihood accumulator
            likelihood_total = zeros(obj.N_p, 1);

            % Add contribution from clutter (uniform over all particles)
            PD = .95; % Probability of detection
            PFA = 0.05; % Probability of false alarm
            clutter_likelihood = (1 - PD) * PFA / (obj.N_p);

            % Likelihood total starts with clutter contribution
            likelihood_total = clutter_likelihood * ones(obj.N_p, 1);

            % Add contribution from each measurement
            for i = 1:size(z, 2)
                % Get likelihood for this measurement
                likelihood_i = obj.likelihoodLookup(z(:, i));

                % Add contribution (no beta weighting in simplified version)
                likelihood_total = likelihood_total + likelihood_i;

                if obj.debug
                    fprintf('Debug: Measurement %d likelihood range: [%.6f, %.6f]\n', ...
                        i, min(likelihood_i), max(likelihood_i));
                end

            end

            % Apply Gaussian weighting (using centroid of measurements for now)
            if size(z, 2) > 0
                z_centroid = mean(z, 2); % Simple centroid approach
                sf = 0.15;
                dx = obj.particles(1, :) - z_centroid(1); % [1 x N_p]
                dy = obj.particles(2, :) - z_centroid(2); % [1 x N_p]
                dist_sq = dx .^ 2 + dy .^ 2; % [1 x N_p]
                gauss_weights = exp(-dist_sq / (2 * sf ^ 2)); % [1 x N_p]

                % Apply Gaussian weighting
                likelihood_total = likelihood_total .* gauss_weights';
            end

            % Final weight update and normalization
            new_weights = likelihood_total + eps;
            obj.weights = new_weights / sum(new_weights);

            % Debug: Print final weight stats
            if obj.debug
                fprintf('Debug: Final weights - min: %.6f, max: %.6f, sum: %.6f\n', ...
                    min(obj.weights), max(obj.weights), sum(obj.weights));
            end

        end

        function weight_updates = computeIndividualWeightUpdates(obj, z)
            % COMPUTEINDIVIDUALWEIGHTUPDATES Compute separate weight updates for each measurement (for GNN)
            %
            % SYNTAX:
            %   weight_updates = obj.computeIndividualWeightUpdates(z)
            %
            % INPUTS:
            %   z - Current measurements [N_z x N_measurements]
            %
            % OUTPUTS:
            %   weight_updates - Cell array of weight updates for each measurement
            %                   weight_updates{i} contains [N_p x 1] weights for measurement i
            %                   weight_updates{end} contains clutter weights (uniform)
            %
            % DESCRIPTION:
            %   Computes individual normalized weight updates for each measurement separately.
            %   Used by GNN algorithm to compare and select the best measurement association.
            %   Each weight update includes both likelihood lookup and Gaussian weighting.
            %
            % See also state_Estimation, likelihoodLookup

            % Check if likelihood image is available
            if isempty(obj.pointlikelihood_image)
                error('GNN_PF:NoLikelihoodData', ...
                'Likelihood lookup table not provided. Load and pass to constructor.');
            end

            N_measurements = size(z, 2);
            weight_updates = cell(N_measurements + 1, 1); % +1 for clutter

            % Compute weight update for each measurement individually
            for i = 1:N_measurements
                % Get likelihood for this measurement
                likelihood_i = obj.likelihoodLookup(z(:, i));

                % Apply Gaussian weighting for this specific measurement
                sf = 0.15;
                dx = obj.particles(1, :) - z(1, i); % [1 x N_p]
                dy = obj.particles(2, :) - z(2, i); % [1 x N_p]
                dist_sq = dx .^ 2 + dy .^ 2; % [1 x N_p]
                gauss_weights = exp(-dist_sq / (2 * sf ^ 2)); % [1 x N_p]

                % Combine likelihoods and Gaussian weights
                combined_weights = likelihood_i .* gauss_weights' + eps;

                % Normalize this individual weight update
                weight_updates{i} = combined_weights / sum(combined_weights);

                if obj.debug
                    fprintf('Debug: Measurement %d individual weights - min: %.6f, max: %.6f\n', ...
                        i, min(weight_updates{i}), max(weight_updates{i}));
                end

            end

            % Add clutter weight update (uniform distribution)
            clutter_weights = ones(obj.N_p, 1) / obj.N_p;
            weight_updates{end} = clutter_weights;

            if obj.debug
                fprintf('Debug: Computed %d individual weight updates (%d measurements + clutter)\n', ...
                    length(weight_updates), N_measurements);
            end

        end

        function likelihood_raw = likelihoodLookup(obj, z)
            % LIKELIHOODLOOKUP Get likelihood values for particles given measurement
            %
            % SYNTAX:
            %   likelihood_raw = obj.likelihoodLookup(z)
            %
            % INPUTS:
            %   z - Current measurement [N_z x 1]
            %
            % OUTPUTS:
            %   likelihood_raw - Likelihood values for all particles [N_p x 1]
            %
            % DESCRIPTION:
            %   Performs vectorized lookup of likelihood values from precomputed
            %   lookup table for all particles given a measurement. Handles boundary
            %   constraints and ensures proper indexing.

            % Define spatial grid parameters (should match precomputed lookup table)
            npx = 128;
            xgrid = linspace(-2, 2, npx);
            ygrid = linspace(0, 4, npx);

            % Find measurement grid point (computed once, used for all particles)
            [~, meas_x_idx] = min(abs(xgrid - z(1)));
            [~, meas_y_idx] = min(abs(ygrid - z(2)));
            meas_linear_idx = sub2ind([npx, npx], meas_y_idx, meas_x_idx);

            % Debug: Print measurement indices
            if obj.debug
                fprintf('Debug: Measurement indices - x_idx: %d, y_idx: %d, linear_idx: %d\n', ...
                    meas_x_idx, meas_y_idx, meas_linear_idx);
            end

            % Vectorized grid point finding for all particles
            [~, px_indices] = min(abs(obj.particles(1, :)' - xgrid), [], 2); % [N_p x 1]
            [~, py_indices] = min(abs(obj.particles(2, :)' - ygrid), [], 2); % [N_p x 1]

            % Enforce boundary constraints
            px_indices = max(1, min(npx, px_indices));
            py_indices = max(1, min(npx, py_indices));

            % Convert to linear indices for all particles
            particle_linear_indices = sub2ind([npx, npx], py_indices, px_indices); % [N_p x 1]

            % Bounds checking with informative error messages
            if meas_linear_idx > size(obj.pointlikelihood_image, 1) || meas_linear_idx < 1
                error('GNN_PF:MeasurementOutOfBounds', ...
                    'Measurement linear index %d out of bounds [1, %d]. Measurement may be outside spatial grid.', ...
                    meas_linear_idx, size(obj.pointlikelihood_image, 1));
            end

            if any(particle_linear_indices > size(obj.pointlikelihood_image, 2)) || any(particle_linear_indices < 1)
                error('GNN_PF:ParticlesOutOfBounds', ...
                    'Particle linear indices out of bounds [1, %d]. Some particles may be outside spatial grid.', ...
                    size(obj.pointlikelihood_image, 2));
            end

            % Vectorized likelihood lookup from precomputed model
            likelihood_raw = obj.pointlikelihood_image(meas_linear_idx, particle_linear_indices);

            % Ensure likelihood_raw is a column vector
            if size(likelihood_raw, 1) == 1 && size(likelihood_raw, 2) == obj.N_p
                likelihood_raw = likelihood_raw';
            end

        end

        function [state_est, state_est_covariance] = getGaussianEstimate(obj)
            % GETGAUSSIANESTIMATE Compute Gaussian state estimate from particles
            %
            % SYNTAX:
            %   [state_est, state_est_covariance] = obj.getGaussianEstimate()
            %
            % OUTPUTS:
            %   state_est - Estimated state mean [N_x x 1]
            %   state_est_covariance - Estimated state covariance [N_x x N_x]
            %
            % DESCRIPTION:
            %   Computes the weighted mean and covariance of the particle states
            %   using the current particle weights. Provides a Gaussian approximation
            %   of the particle filter state.

            % Compute weighted mean
            state_est = obj.particles * obj.weights;

            % Compute weighted covariance using more robust formula
            deviations = obj.particles - state_est; % [N_x x N_p]

            % Vectorized weighted covariance calculation
            weighted_deviations = deviations .* sqrt(obj.weights'); % [N_x x N_p]
            state_est_covariance = (weighted_deviations * weighted_deviations') / sum(obj.weights);

            % Ensure covariance is real and symmetric (numerical stability)
            state_est_covariance = real(state_est_covariance);
            state_est_covariance = 0.5 * (state_est_covariance + state_est_covariance');

            % Add small regularization to ensure positive definiteness
            state_est_covariance = state_est_covariance +1e-8 * eye(size(state_est_covariance, 1));
        end

        function printState(obj, label)
            % PRINTSTATE Print current state estimate in readable format
            %
            % SYNTAX:
            %   obj.printState()
            %   obj.printState(label)
            %
            % INPUTS:
            %   label - (optional) String label to identify this printout
            %
            % DESCRIPTION:
            %   Prints the current particle filter state estimate in a clean,
            %   readable format including position, velocity, acceleration,
            %   and weight statistics.

            if nargin < 2
                label = 'Current State';
            end

            % Compute mean state
            mean_state = mean(obj.particles, 2);

            % Compute effective sample size
            eff_sample_size = 1 / sum(obj.weights .^ 2);

            % Print header
            fprintf('\n--- %s ---\n', label);
            fprintf('Position:     [%.4f, %.4f] m\n', mean_state(1), mean_state(2));
            fprintf('Velocity:     [%.4f, %.4f] m/s\n', mean_state(3), mean_state(4));
            fprintf('Acceleration: [%.4f, %.4f] m/s²\n', mean_state(5), mean_state(6));
            fprintf('Weights:      min=%.6f, max=%.6f, sum=%.6f\n', ...
                min(obj.weights), max(obj.weights), sum(obj.weights));
            fprintf('Particles:    %d total\n', obj.N_p);
            fprintf('Eff. Sample:  %.1f (%.1f%%)\n', eff_sample_size, 100 * eff_sample_size / obj.N_p);
            fprintf('-------------------\n');
        end

        % Helper functions
        function loadLikelihoodData(obj, likelihood_file_path)
            % LOADLIKELIHOODDATA Load precomputed likelihood lookup table for hybrid PF
            %
            % SYNTAX:
            %   obj.loadLikelihoodData(likelihood_file_path)
            %
            % INPUTS:
            %   likelihood_file_path - string, path to .mat file containing 'pointlikelihood_image'
            %
            % DESCRIPTION:
            %   Loads and validates the precomputed likelihood lookup table used by the
            %   hybrid particle filter for measurement updates. Expected table size is
            %   128^2 x 128^2 corresponding to spatial grid discretization.
            %
            % See also GNN_PF, prediction, resample

            try
                % Check if file exists
                if ~exist(likelihood_file_path, 'file')
                    error('GNN_PF:FileNotFound', ...
                        'Likelihood file not found: %s', likelihood_file_path);
                end

                % Load the likelihood data
                likelihood_data = load(likelihood_file_path, 'pointlikelihood_image');

                % Validate that the expected variable exists
                if ~isfield(likelihood_data, 'pointlikelihood_image')
                    error('GNN_PF:InvalidData', ...
                        'Variable "pointlikelihood_image" not found in file: %s', likelihood_file_path);
                end

                % Store as class property (immutable lookup table)
                obj.pointlikelihood_image = likelihood_data.pointlikelihood_image;

                % Validate dimensions (expecting 128^2 x 128^2 based on your original code)
                expected_dim = 128 ^ 2;
                [rows, cols] = size(obj.pointlikelihood_image);

                % Debug: Print detailed dimension info
                if obj.debug
                    fprintf('Debug: Loaded likelihood table - Rows: %d, Cols: %d\n', rows, cols);
                    fprintf('Debug: Expected dimensions: %d x %d\n', expected_dim, expected_dim);
                end

                if rows ~= expected_dim || cols ~= expected_dim
                    warning('GNN_PF:DimensionMismatch', ...
                        'Likelihood model dimensions (%dx%d) do not match expected grid size (%dx%d)', ...
                        rows, cols, expected_dim, expected_dim);
                end

                if obj.debug
                    fprintf('Successfully loaded likelihood lookup table from: %s\n', likelihood_file_path);
                    fprintf('Likelihood table dimensions: %dx%d\n', rows, cols);
                end

            catch ME
                error('GNN_PF:LoadError', ...
                    'Failed to load likelihood data: %s', ME.message);
            end

        end

        function resample(obj)
            % RESAMPLE Bootstrap resampling for particle filter
            %
            % SYNTAX:
            %   obj.resample()
            %
            % DESCRIPTION:
            %   Performs bootstrap resampling to reduce particle degeneracy. Uses
            %   systematic resampling with linear search. Particles are resampled
            %   according to their weights and weights are reset to uniform.
            %   Handles edge cases with invalid or zero weights.
            %
            % MODIFIES:
            %   obj.particles - resampled particle states [N_x x N_p]
            %   obj.weights   - reset to uniform [N_p x 1]
            %
            % See also GNN_PF, prediction, timestep

            % First, check if weights are valid
            if any(isnan(obj.weights)) || any(isinf(obj.weights))
                warning('Invalid weights detected before processing');
                % Replace invalid weights with small values
                obj.weights(isnan(obj.weights) | isinf(obj.weights)) = eps;
            end

            % Add small epsilon to zero weights to prevent issues
            epsilon = eps; % Using MATLAB's eps
            obj.weights = obj.weights + epsilon;
            obj.weights = obj.weights / sum(obj.weights); % Renormalize after adding epsilon

            % Compute cumulative sum of weights
            sampcdf = cumsum(obj.weights);

            % Draw uniform random numbers
            urands = rand(1, obj.N_p);

            % Use simple linear search for resampling indices
            indsampsout = zeros(1, obj.N_p);

            for i = 1:obj.N_p
                indsampsout(i) = find(sampcdf >= urands(i), 1, 'first');

                if isempty(indsampsout(i))
                    indsampsout(i) = obj.N_p;
                end

            end

            % Resample particles directly
            obj.particles = obj.particles(:, indsampsout);

            % Set uniform weights
            obj.weights = (1 / obj.N_p) * ones(size(obj.weights));
        end

        % Setter methods for properties - REMOVED because they cause issues with direct assignment
        % MATLAB allows direct property assignment for handle classes
        % Use: obj.debug = true; instead of custom setters

    end

end
