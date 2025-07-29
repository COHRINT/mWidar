classdef GNN_PF < DA_Filter
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
    %   pf = GNN_PF(x0, 1000, F, Q, H, pointlikelihood_image, 'Debug', true, 'DynamicPlot', true);
    %   pf.timestep(measurements);
    %   [x_est, P_est] = pf.getGaussianEstimate();
    %
    % See also DA_Filter, GNN_HMM, particleFilter, trackingPF

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

        % Control Flags (inherited from DA_Filter)
        debug = false % Enable debug output and validation
        validate = false % Enable input/output validation checks
        DynamicPlot = false % Enable real-time visualization during timesteps

        % Validation Parameters
        validation_sigma_bounds = 2 % Number of sigma bounds for measurement gating (default: 2)

        % Dynamic Plotting (inherited from DA_Filter)
        dynamic_figure_handle % Figure handle for dynamic plotting
    end

    methods

        function obj = GNN_PF(x0, N_particles, F, Q, H, pointlikelihood_image, varargin)
            % GNN_PF Constructor for Global Nearest Neighbor Particle Filter
            %
            % SYNTAX:
            %   obj = GNN_PF(x0, N_particles, F, Q, H)
            %   obj = GNN_PF(x0, N_particles, F, Q, H, pointlikelihood_image)
            %   obj = GNN_PF(..., 'Debug', true, 'DynamicPlot', true, 'ValidationSigma', 3)
            %
            % INPUTS:
            %   x0                  - Initial state estimate [N_x x 1]
            %   N_particles         - Number of particles (integer > 0)
            %   F                   - State transition matrix [N_x x N_x]
            %   Q                   - Process noise covariance [N_x x N_x]
            %   H                   - Measurement matrix [N_z x N_x]
            %   pointlikelihood_image - (optional) Precomputed likelihood lookup table [128^2 x 128^2]
            %   varargin            - Name-value pairs: 'Debug', true/false, 'DynamicPlot', true/false, 'ValidationSigma', numeric
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

            % Parse options using parent class utility
            options = DA_Filter.parseFilterOptions(varargin{:});
            obj.debug = options.Debug;
            obj.DynamicPlot = options.DynamicPlot;
            obj.validation_sigma_bounds = options.ValidationSigma;

            % Initialize basic properties
            obj.N_p = N_particles;
            obj.N_x = size(x0, 1);
            obj.N_z = size(H, 1);

            % Initialize particles with small spread around initial guess
            % init_spread = [0.1, 0.1, 0.25, 0.25, 0.5, 0.5]; % Position, velocity, acceleration spreads
            obj.particles = repmat(x0(:), 1, N_particles);

            % Add Gaussian noise to each state component
            % for i = 1:obj.N_x
            %     obj.particles(i, :) = obj.particles(i, :);% + init_spread(i) * randn(1, N_particles);
            % end

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
                    fprintf('\n=== GNN_PF INITIALIZATION ===\n');
                    fprintf('Particles: %d, States: %d, Measurements: %d\n', N_particles, obj.N_x, obj.N_z);
                    fprintf('Likelihood table: %dx%d (expected: %dx%d)\n', rows, cols, expected_dim, expected_dim);
                    fprintf('============================\n\n');
                end

                if rows ~= expected_dim || cols ~= expected_dim
                    warning('GNN_PF:DimensionMismatch', ...
                        'Likelihood model dimensions (%dx%d) do not match expected grid size (%dx%d)', ...
                        rows, cols, expected_dim, expected_dim);
                end

            end

            % Initialize dynamic plotting if enabled with shorter height for 1x3 square axes
            obj.initializeDynamicPlot('GNN-PF Dynamic Tracking', [100, 100, 1400, 600]);

        end

        %% ========== TIMESTEP ==========
        function timestep(obj, z, varargin)
            % TIMESTEP Implements the GNN-PF algorithm for a single timestep
            %
            % SYNTAX:
            %   obj.timestep(z)
            %   obj.timestep(z, true_state)
            %
            % INPUTS:
            %   z          - Measurements [2 x N_measurements]
            %   true_state - (optional) True state for visualization
            %
            % DESCRIPTION:
            %   Executes one timestep of the GNN-PF algorithm including:
            %   0. Initialization - Resample particles if weights are non-uniform
            %   1. Prediction Step - Propagate particles through dynamics model
            %   2. Validation Step - Gate measurements using particle estimate
            %   3. Data Association Step - Select best measurement using GNN
            %   4. Measurement Update - Update particle weights using selected measurement
            %   5. Visualization - Update dynamic plot if enabled
            %
            % ALGORITHM NOTES:
            %   - Resampling only occurs if weights are non-uniform
            %   - Gating uses configurable sigma bounds based on current state estimate
            %   - GNN selects single best measurement based on likelihood comparison
            %   - Supports optional real-time visualization
            %
            % See also prediction, measurement_update, resample

            if obj.debug
                fprintf('\n=== GNN-PF TIMESTEP START ===\n');
                fprintf('Input: %d measurements\n', size(z, 2));

                if ~isempty(z)

                    for i = 1:size(z, 2)
                        fprintf('  Meas %d: [%.3f, %.3f]\n', i, z(1, i), z(2, i));
                    end

                end

                fprintf('------------------------------\n');
            end

            % Step 0: Initialization
            % Resample the particles if necessary (only if weights are not uniform)
            % This will only NOT be called if
            %    1) This is the first timestep (weights are uniform by default)
            %    2) The weights are already uniform (no measurement updates yet)
            if any(obj.weights ~= 1 / length(obj.weights))
                % print effective sample size
                ESS = 1 / sum(obj.weights .^ 2);

                if obj.debug
                    fprintf('[RESAMPLING] ESS: %.2f, resampling particles...\n', ESS);
                end

                obj.resample();
            end

            % Step 1: Prediction Step
            % Particles are propagated through the dynamics model (weights unchanged)
            if obj.debug
                fprintf('[PREDICTION] Applying dynamics model...\n');
            end

            obj.prediction();

            if obj.debug
                fprintf('[PREDICTION] Complete.\n');
            end

            % Step 2: Validation step - Gate measurements using particle filter estimate
            % COMMENT OUT THIS BLOCK TO DISABLE GATING
            [z_to_process, has_valid_meas] = obj.Validation(z);

            % END GATING BLOCK - COMMENT OUT TO HERE TO DISABLE GATING

            % UNCOMMENT THIS LINE TO DISABLE GATING (and comment out the block above)
            % z_to_process = z; % Use all measurements without gating

            % Step 3: Data Association Step - GNN selects best measurement
            % TODO: Implement GNN data association to select single best measurement
            % TODO: Compare likelihoods/costs for each measurement and choose optimal one
            % TODO: Handle missed detection case when no measurements pass gating
            selected_measurement = obj.Data_Association(z_to_process);

            % Step 4: Measurement Update Step
            obj.measurement_update(selected_measurement);

            if obj.debug
                [x_est, P_est] = obj.getGaussianEstimate();
                fprintf('\nOutput: State estimate [%.4f, %.4f] m\n', x_est(1), x_est(2));
                fprintf('        Covariance trace: %.6f\n', trace(P_est));
                fprintf('=== GNN-PF TIMESTEP END ===\n\n');
            end

            % Update dynamic plot if enabled
            if obj.DynamicPlot

                if nargin > 2
                    true_state = varargin{1};
                    obj.updateDynamicPlot(z_to_process, true_state);
                else
                    obj.updateDynamicPlot(z_to_process);
                end

            end

        end

        %% ========== PREDICTION STEP ==========
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
                fprintf('[PREDICTION] F matrix [%dx%d], particles [%dx%d]\n', ...
                    size(obj.F, 1), size(obj.F, 2), size(obj.particles, 1), size(obj.particles, 2));
            end

            % Propagate particles through state transition model
            obj.particles = obj.F * obj.particles;

            % Add process noise to each particle
            process_noise = mvnrnd(zeros(1, obj.N_x), obj.Q, obj.N_p)';
            obj.particles = obj.particles + process_noise;

        end

        %% ========== MEASUREMENT VALIDATION ==========
        function [z_valid, has_valid_meas] = Validation(obj, z)
            % VALIDATION Gate measurements using particle filter state estimate
            %
            % SYNTAX:
            %   [z_valid, has_valid_meas] = obj.Validation(z)
            %
            % INPUTS:
            %   z - Raw measurements [N_z x N_meas]
            %
            % OUTPUTS:
            %   z_valid        - Validated measurements [N_z x N_valid]
            %   has_valid_meas - Boolean flag indicating if any measurements passed validation
            %
            % DESCRIPTION:
            %   Applies configurable sigma gating to filter measurements that fall within
            %   bounds around the current particle filter state estimate. Uses position-only
            %   gating based on current state covariance and obj.validation_sigma_bounds.
            %
            % ALGORITHM:
            %   1. Get current particle filter estimate (mean and covariance)
            %   2. Compute sigma bounds based on obj.validation_sigma_bounds and position uncertainty
            %   3. Accept measurements within bounds: lower_bound <= z <= upper_bound
            %
            % See also timestep, measurement_update

            % Handle empty measurement case
            if isempty(z)
                z_valid = [];
                has_valid_meas = false;

                if obj.debug
                    fprintf('[GATING] No measurements to validate\n');
                end

                return;
            end

            % Get current particle filter estimate for gating
            [x_est, P_est] = obj.getGaussianEstimate();
            z_hat = x_est(1:2); % Predicted measurement (only position)
            S = P_est(1:2, 1:2); % Use position covariance only

            % Compute sigma bounds for gating using configurable parameter
            sigma_bounds = obj.validation_sigma_bounds * sqrt(diag(S));
            lower_bound = z_hat - sigma_bounds;
            upper_bound = z_hat + sigma_bounds;

            % Apply gating: Keep measurements within sigma bounds
            in_bounds = all(z >= lower_bound & z <= upper_bound, 1);
            z_valid = z(:, in_bounds);
            has_valid_meas = ~isempty(z_valid);

            if obj.debug
                fprintf('[GATING] Input: %d measurements, Output: %d valid measurements\n', ...
                    size(z, 2), size(z_valid, 2));

                if obj.debug && size(z, 2) > 0
                    fprintf('[GATING] Using %.1f-sigma bounds: x=[%.3f, %.3f], y=[%.3f, %.3f]\n', ...
                        obj.validation_sigma_bounds, lower_bound(1), upper_bound(1), lower_bound(2), upper_bound(2));
                end

            end

        end

        %% ========== GLOBAL NEAREST NEIGHBOR DATA ASSOCIATION ==========
        function selected_measurement = Data_Association(obj, z_valid)
            % DATA_ASSOCIATION Global Nearest Neighbor data association for validated measurements
            %
            % SYNTAX:
            %   selected_measurement = obj.Data_Association(z_valid)
            %
            % INPUTS:
            %   z_valid - Validated measurements [N_z x N_valid]
            %
            % OUTPUTS:
            %   selected_measurement - Single selected measurement [N_z x 1] or empty if none selected
            %
            % DESCRIPTION:
            %   Implements GNN data association algorithm to select the single best
            %   measurement from the validated set. The selection is based on maximizing
            %   the likelihood or minimizing a cost function.
            %
            % TODO: Implement proper GNN cost function (e.g., Mahalanobis distance)
            % TODO: Compare different cost metrics: likelihood, innovation covariance, etc.
            % TODO: Handle edge cases: no measurements, single measurement, multiple measurements
            % TODO: Consider track quality/confidence in association decision
            %
            % ALGORITHM:
            %   1. If no measurements: return empty (missed detection)
            %   2. If single measurement: return that measurement
            %   3. If multiple measurements: compute cost/likelihood for each and select best
            %
            % See also timestep, measurement_update, Validation

            % Handle empty measurement case (missed detection)
            if isempty(z_valid)
                selected_measurement = [];

                if obj.debug
                    fprintf('[GNN DATA ASSOCIATION] No validated measurements - missed detection\n');
                end

                return;
            end

            % Handle single measurement case
            if size(z_valid, 2) == 1
                selected_measurement = z_valid;

                if obj.debug
                    fprintf('[GNN DATA ASSOCIATION] Single measurement - auto-selected [%.3f, %.3f]\n', ...
                        selected_measurement(1), selected_measurement(2));
                end

                return;
            end

            % Multiple measurements case - need to select best one
            N_measurements = size(z_valid, 2);

            if obj.debug
                fprintf('[GNN DATA ASSOCIATION] Selecting best from %d measurements\n', N_measurements);
            end

            % TODO: Implement proper GNN cost function
            % For now, use a simple approach based on likelihood comparison
            
            % Check if likelihood image is available for proper cost computation
            if isempty(obj.pointlikelihood_image)
                % Fallback: Select measurement closest to predicted position
                [x_est, ~] = obj.getGaussianEstimate();
                predicted_pos = x_est(1:2); % Position only
                
                distances = zeros(1, N_measurements);
                for i = 1:N_measurements
                    distances(i) = norm(z_valid(:, i) - predicted_pos);
                end
                
                [~, best_idx] = min(distances);
                selected_measurement = z_valid(:, best_idx);
                
                if obj.debug
                    fprintf('[GNN] Selected measurement %d (closest to prediction): [%.3f, %.3f]\n', ...
                        best_idx, selected_measurement(1), selected_measurement(2));
                end
                
                return;
            end

            % TODO: Implement proper likelihood-based or cost-based selection
            % Option 1: Maximum likelihood approach
            % - Compute weight update for each measurement individually
            % - Select measurement that maximizes total likelihood
            
            % Option 2: Minimum cost approach (Mahalanobis distance)
            % - Compute innovation covariance S for current estimate
            % - Calculate Mahalanobis distance: (z - z_hat)' * S^(-1) * (z - z_hat)
            % - Select measurement with minimum distance
            
            % Option 3: Information-theoretic approach
            % - Compute expected information gain for each measurement
            % - Select measurement that maximizes information gain
            
            % For now, use maximum likelihood approach using existing infrastructure
            weight_updates = obj.computeIndividualWeightUpdates(z_valid);
            
            % Compute total likelihood for each measurement (sum of all particle weights)
            total_likelihoods = zeros(1, N_measurements);
            for i = 1:N_measurements
                total_likelihoods(i) = sum(weight_updates{i});
            end
            
            % Select measurement with maximum total likelihood
            [max_likelihood, best_idx] = max(total_likelihoods);
            selected_measurement = z_valid(:, best_idx);

            if obj.debug
                fprintf('[GNN] Measurement likelihoods: [');
                for i = 1:N_measurements
                    fprintf('%.4f ', total_likelihoods(i));
                end
                fprintf(']\n');
                fprintf('[GNN] Selected measurement %d (max likelihood %.4f): [%.3f, %.3f]\n', ...
                    best_idx, max_likelihood, selected_measurement(1), selected_measurement(2));
            end

        end

        %% ========== MEASUREMENT UPDATE ==========
        function measurement_update(obj, z)
            % MEASUREMENT_UPDATE Update particle weights based on selected measurement (GNN)
            %
            % SYNTAX:
            %   obj.measurement_update(z)
            %
            % INPUTS:
            %   z    - Selected measurement [N_z x 1] (from GNN data association)
            %
            % DESCRIPTION:
            %   Implements GNN measurement update using single selected measurement.
            %   Unlike PDA which combines all measurements, GNN uses only the best
            %   measurement as determined by the data association step.
            %   Handles missed detection case when z is empty.
            %
            % MODIFIES:
            %   obj.weights - Updated and normalized particle weights [N_p x 1]
            %
            % TODO: Optimize likelihood computation for single measurement case
            % TODO: Consider different weight update strategies for GNN vs PDA
            % TODO: Implement proper missed detection handling with detection probability
            %
            % See also timestep, prediction, likelihoodLookup, Data_Association

            % Handle missed detection case (no measurement selected by GNN)
            if isempty(z)

                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] No measurement selected - missed detection case\n');
                end

                % TODO: Apply proper missed detection probability model
                % For now, keep weights unchanged for missed detection
                return;
            end

            % Check if likelihood image is available
            if isempty(obj.pointlikelihood_image)
                error('GNN_PF:NoLikelihoodData', ...
                'Likelihood lookup table not provided. Load and pass to constructor.');
            end

            % Debug: Print measurement info
            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Processing selected measurement: [%.3f, %.3f]\n', z(1), z(2));
            end

            % Single measurement case - simpler than PDA
            if obj.debug
                fprintf('  Selected meas: [%.3f, %.3f] -> ', z(1), z(2));
            end

            % Get likelihood for the selected measurement
            likelihood_meas = obj.likelihoodLookup(z);

            % Apply Gaussian weighting for the selected measurement
            sf = 0.15;
            dx = obj.particles(1, :) - z(1); % [1 x N_p]
            dy = obj.particles(2, :) - z(2); % [1 x N_p]
            dist_sq = dx .^ 2 + dy .^ 2; % [1 x N_p]
            gauss_weights = exp(-dist_sq / (2 * sf ^ 2)); % [1 x N_p]

            % Apply Gaussian weighting
            likelihood_total = likelihood_meas .* gauss_weights';

            if obj.debug
                fprintf('[GAUSSIAN MASK] Applied around selected measurement [%.3f, %.3f]\n', z(1), z(2));
            end

            % Final weight update and normalization
            new_weights = likelihood_total + eps;
            obj.weights = new_weights / sum(new_weights);

            % Debug: Print final weight stats
            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Complete. Weights: min=%.6f, max=%.6f, sum=%.6f\n', ...
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
            % See also measurement_update, likelihoodLookup, Data_Association

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
                    fprintf('[GNN] Measurement %d individual weights: min=%.6f, max=%.6f\n', ...
                        i, min(weight_updates{i}), max(weight_updates{i}));
                end

            end

            % Add clutter weight update (uniform distribution)
            clutter_weights = ones(obj.N_p, 1) / obj.N_p;
            weight_updates{end} = clutter_weights;

            if obj.debug
                fprintf('[GNN] Computed %d individual weight updates (%d measurements + clutter)\n', ...
                    length(weight_updates), N_measurements);
            end

        end

        %% ========== LIKELIHOOD COMPUTATION ==========
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
                fprintf('    Grid indices: x=%d, y=%d, linear=%d\n', ...
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

        %% ========== STATE ESTIMATION ==========
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

            if obj.debug
                fprintf('\n=== LOADING LIKELIHOOD DATA ===\n');
                fprintf('Loading from: %s\n', likelihood_file_path);
            end

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

                if obj.debug
                    fprintf('[VALIDATION] Loaded table dimensions: %dx%d\n', rows, cols);
                    fprintf('[VALIDATION] Expected dimensions: %dx%d\n', expected_dim, expected_dim);
                end

                if rows ~= expected_dim || cols ~= expected_dim
                    warning('GNN_PF:DimensionMismatch', ...
                        'Likelihood model dimensions (%dx%d) do not match expected grid size (%dx%d)', ...
                        rows, cols, expected_dim, expected_dim);
                end

                if obj.debug
                    fprintf('[SUCCESS] Likelihood lookup table loaded successfully\n');
                    fprintf('================================\n\n');
                end

            catch ME

                if obj.debug
                    fprintf('[ERROR] Failed to load likelihood data: %s\n', ME.message);
                    fprintf('===============================\n\n');
                end

                error('GNN_PF:LoadError', ...
                    'Failed to load likelihood data: %s', ME.message);
            end

        end

        %% ========== PARTICLE RESAMPLING ==========
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

            if obj.debug
                fprintf('\n=== RESAMPLING ===\n');
                fprintf('[WEIGHTS] Pre-resample: min=%.6f, max=%.6f, sum=%.6f\n', ...
                    min(obj.weights), max(obj.weights), sum(obj.weights));
            end

            % First, check if weights are valid
            if any(isnan(obj.weights)) || any(isinf(obj.weights))

                if obj.debug
                    fprintf('[WARNING] Invalid weights detected, replacing with epsilon\n');
                end

                warning('Invalid weights detected before processing');
                % Replace invalid weights with small values
                obj.weights(isnan(obj.weights) | isinf(obj.weights)) = eps;
            end

            % Add small epsilon to zero weights to prevent issues
            epsilon = eps; % Using MATLAB's eps
            obj.weights = obj.weights + epsilon;
            obj.weights = obj.weights / sum(obj.weights); % Renormalize after adding epsilon

            if obj.debug
                fprintf('[NORMALIZATION] Post-normalization: sum=%.6f\n', sum(obj.weights));
            end

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

            if obj.debug
                fprintf('[PARTICLES] Resampled particles, weights now uniform\n');
                fprintf('[RESULT] New weights: min=%.6f, max=%.6f, sum=%.6f\n', ...
                    min(obj.weights), max(obj.weights), sum(obj.weights));
                fprintf('==================\n\n');
            end

        end

        %% ========== VISUALIZATION ==========
        function visualize(obj, figure_handle, title_str, measurements, true_state)
            % VISUALIZE Plot current particle distribution and state estimates
            %
            % SYNTAX:
            %   obj.visualize()
            %   obj.visualize(figure_handle)
            %   obj.visualize(figure_handle, title_str)
            %   obj.visualize(figure_handle, title_str, measurements)
            %   obj.visualize(figure_handle, title_str, measurements, true_state)
            %
            % INPUTS:
            %   figure_handle - (optional) Figure handle to plot in
            %   title_str     - (optional) Title string for plot
            %   measurements  - (optional) Current measurements [N_z x N_meas]
            %   true_state    - (optional) True state for comparison [N_x x 1]
            %
            % DESCRIPTION:
            %   Creates visualization of particle filter state with subplots for:
            %   1. Position particles with zoom inset
            %   2. Velocity particles
            %   3. Acceleration particles
            %   Based on visualization style from test_hybrid_PF.m

            if nargin < 2 || isempty(figure_handle)
                figure;
                % Set default figure size and position optimized for 1x3 layout
                % Only set position for new figures, preserve user resizing
                current_pos = get(gcf, 'Position');

                if current_pos(3) == 560 && current_pos(4) == 420 % Default MATLAB size
                    set(gcf, 'Position', [100, 100, 1200, 400]);
                end

            else
                figure(figure_handle);
            end

            if nargin < 3 || isempty(title_str)
                title_str = 'GNN-PF Particle Distribution';
            end

            if nargin < 4
                measurements = [];
            end

            if nargin < 5
                true_state = [];
            end

            % Get current state estimate
            [mean_state, state_cov] = obj.getGaussianEstimate();

            % Compute effective sample size for main title
            eff_sample_size = 1 / sum(obj.weights .^ 2);

            % Define spatial bounds (matching test_hybrid_PF)
            Xbounds = [-2, 2];
            Ybounds = [0, 4];

            % Subplot 1: Position with full scene view and inset zoom
            ax1 = subplot(1, 3, 1);
            cla;

            % Scatter plot of particle positions colored by weights
            h_scatter = scatter(obj.particles(1, :), obj.particles(2, :), 20, obj.weights, ...
                'filled', 'MarkerFaceAlpha', 0.6);
            hold on

            % Plot particle filter estimate
            plot(mean_state(1), mean_state(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3, ...
                'DisplayName', 'PF Estimate');

            % Plot covariance ellipses (1σ and 3σ)
            pos_cov = state_cov(1:2, 1:2); % Position covariance only

            % 1σ covariance ellipse (solid white)
            ellipse_1sigma = obj.computeCovarianceEllipse(mean_state(1:2), pos_cov, 1);
            plot(ellipse_1sigma(1, :), ellipse_1sigma(2, :), 'w-', 'LineWidth', 2, ...
                'DisplayName', '1\sigma Covariance');

            % 3σ covariance ellipse (dotted white)
            ellipse_validation = obj.computeCovarianceEllipse(mean_state(1:2), pos_cov, obj.validation_sigma_bounds);
            plot(ellipse_validation(1, :), ellipse_validation(2, :), 'w:', 'LineWidth', 2, ...
                'DisplayName', 'Validation');

            % Plot true state if provided
            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'd', 'Color', 'm', ...
                    'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm', ...
                    'DisplayName', 'True Position');
            end

            % Plot measurements if provided
            if ~isempty(measurements)
                % Plot all measurements with a single legend entry in bright orange
                plot(measurements(1, :), measurements(2, :), '+', 'Color', [1 0.5 0], ...
                    'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Measurements');

                % Draw rectangle indicating inset zoom region around first measurement
                if size(measurements, 2) > 0
                    current_meas = measurements(:, 1);
                    zoom_rect = rectangle('Position', [current_meas(1) - 0.3, current_meas(2) - 0.3, 0.6, 0.6], ...
                        'EdgeColor', 'k', 'LineWidth', 1, 'LineStyle', '--');
                end

            end

            title('Position', 'Interpreter', 'latex');
            xlabel('X (m)'), ylabel('Y (m)');
            xlim(Xbounds), ylim(Ybounds);
            axis square;
            legend('Location', 'northwest');

            % Create inset zoom window if measurements are provided
            if ~isempty(measurements) && size(measurements, 2) > 0
                current_meas = measurements(:, 1);

                % Define inset location in data coordinates (positioned in upper right)
                inset_x_range = [0.5, 1.8];
                inset_y_range = [2.7, 3.9];

                % Convert data coordinates to normalized figure coordinates
                ax1_pos = get(ax1, 'Position');
                xlims = get(ax1, 'XLim');
                ylims = get(ax1, 'YLim');

                % Normalize inset position within the main subplot
                inset_left_norm = (inset_x_range(1) - xlims(1)) / (xlims(2) - xlims(1));
                inset_bottom_norm = (inset_y_range(1) - ylims(1)) / (ylims(2) - ylims(1));
                inset_width_norm = (inset_x_range(2) - inset_x_range(1)) / (xlims(2) - xlims(1));
                inset_height_norm = (inset_y_range(2) - inset_y_range(1)) / (ylims(2) - ylims(1));

                % Convert to figure coordinates
                inset_pos = [ax1_pos(1) + inset_left_norm * ax1_pos(3), ...
                                 ax1_pos(2) + inset_bottom_norm * ax1_pos(4), ...
                                 inset_width_norm * ax1_pos(3), ...
                                 inset_height_norm * ax1_pos(4)];

                inset_ax = axes('Position', inset_pos);
                scatter(obj.particles(1, :), obj.particles(2, :), 15, obj.weights, ...
                    'filled', 'MarkerFaceAlpha', 0.7);
                hold on
                plot(mean_state(1), mean_state(2), 'ro', 'MarkerSize', 8, 'LineWidth', 2);

                % Add covariance ellipses to inset
                ellipse_1sigma = obj.computeCovarianceEllipse(mean_state(1:2), pos_cov, 1);
                plot(ellipse_1sigma(1, :), ellipse_1sigma(2, :), 'w-', 'LineWidth', 1.5);

                ellipse_3sigma = obj.computeCovarianceEllipse(mean_state(1:2), pos_cov, 3);
                plot(ellipse_3sigma(1, :), ellipse_3sigma(2, :), 'w:', 'LineWidth', 1.5);

                if ~isempty(true_state)
                    plot(true_state(1), true_state(2), 'd', 'Color', 'm', ...
                        'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', 'm');
                end

                plot(current_meas(1), current_meas(2), '+', 'Color', [1 0.5 0], ...
                    'MarkerSize', 8, 'LineWidth', 2);

                xlim([current_meas(1) - 0.3, current_meas(1) + 0.3]);
                ylim([current_meas(2) - 0.3, current_meas(2) + 0.3]);
                axis square;
                set(inset_ax, 'FontSize', 8, 'Box', 'on', 'XTick', [], 'YTick', []);

                % Store inset handle for later use
                inset_handle = inset_ax;

                % Return focus to main subplot
                axes(ax1);
            end

            % Subplot 2: Velocity estimates (if state dimension >= 4)
            if obj.N_x >= 4
                subplot(1, 3, 2);
                cla;

                scatter(obj.particles(3, :), obj.particles(4, :), 20, obj.weights, ...
                    'filled', 'MarkerFaceAlpha', 0.6);
                hold on
                plot(mean_state(3), mean_state(4), 'ro', 'MarkerSize', 10, 'LineWidth', 3, ...
                    'DisplayName', 'PF Estimate');

                % Plot velocity covariance ellipses
                vel_cov = state_cov(3:4, 3:4); % Velocity covariance
                ellipse_1sigma_vel = obj.computeCovarianceEllipse(mean_state(3:4), vel_cov, 1);
                plot(ellipse_1sigma_vel(1, :), ellipse_1sigma_vel(2, :), 'w-', 'LineWidth', 2, ...
                    'DisplayName', '1\sigma Covariance');

                ellipse_3sigma_vel = obj.computeCovarianceEllipse(mean_state(3:4), vel_cov, 3);
                plot(ellipse_3sigma_vel(1, :), ellipse_3sigma_vel(2, :), 'w:', 'LineWidth', 2, ...
                    'DisplayName', '3\sigma Covariance');

                if ~isempty(true_state) && length(true_state) >= 4
                    plot(true_state(3), true_state(4), 'd', 'Color', 'm', ...
                        'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm', ...
                        'DisplayName', 'True Velocity');
                end

                title('Velocity', 'Interpreter', 'latex');
                xlabel('V_x (m/s)'), ylabel('V_y (m/s)');
                axis square;
                legend('Location', 'northwest');
            end

            % Subplot 3: Acceleration estimates (if state dimension >= 6)
            if obj.N_x >= 6
                subplot(1, 3, 3);
                cla;

                scatter(obj.particles(5, :), obj.particles(6, :), 20, obj.weights, ...
                    'filled', 'MarkerFaceAlpha', 0.6);
                hold on
                plot(mean_state(5), mean_state(6), 'ro', 'MarkerSize', 10, 'LineWidth', 3, ...
                    'DisplayName', 'PF Estimate');

                % Plot acceleration covariance ellipses
                acc_cov = state_cov(5:6, 5:6); % Acceleration covariance
                ellipse_1sigma_acc = obj.computeCovarianceEllipse(mean_state(5:6), acc_cov, 1);
                plot(ellipse_1sigma_acc(1, :), ellipse_1sigma_acc(2, :), 'w-', 'LineWidth', 2, ...
                    'DisplayName', '1\sigma Covariance');

                ellipse_3sigma_acc = obj.computeCovarianceEllipse(mean_state(5:6), acc_cov, 3);
                plot(ellipse_3sigma_acc(1, :), ellipse_3sigma_acc(2, :), 'w:', 'LineWidth', 2, ...
                    'DisplayName', '3\sigma Covariance');

                if ~isempty(true_state) && length(true_state) >= 6
                    plot(true_state(5), true_state(6), 'd', 'Color', 'm', ...
                        'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm', ...
                        'DisplayName', 'True Acceleration');
                end

                title('Acceleration', 'Interpreter', 'latex');
                xlabel('A_x (m/s^2)'), ylabel('A_y (m/s^2)');
                axis square;
                legend('Location', 'northwest');
            end

            % Add main title with filter name, timestep, and effective sample size
            filter_name = strrep(class(obj), '_', '\_'); % Fix LaTeX typesetting

            if contains(title_str, 'Step')
                % Extract step number from title_str
                step_match = regexp(title_str, 'Step (\d+)', 'tokens');

                if ~isempty(step_match)
                    step_num = step_match{1}{1};
                    main_title = sprintf('%s - Timestep %s (ESS: %.1f)', filter_name, step_num, eff_sample_size);
                else
                    main_title = sprintf('%s (ESS: %.1f)', filter_name, eff_sample_size);
                end

            else
                main_title = sprintf('%s (ESS: %.1f)', filter_name, eff_sample_size);
            end

            sgtitle(main_title, 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');

            % Add colorbar spanning reduced height on the right
            if obj.N_x >= 4 % Only add colorbar if we have multiple subplots
                cb = colorbar('Position', [0.92, 0.25, 0.02, 0.5]);
                cb.Label.String = 'Particle Weight';
                cb.Label.Interpreter = 'latex';
            end

            % Bring inset axes to front if it exists
            if exist('inset_handle', 'var') && isvalid(inset_handle)
                uistack(inset_handle, 'top');
            end

        end

        function ellipse_points = computeCovarianceEllipse(obj, mean_pos, cov_matrix, n_sigma)
            % COMPUTECOVARIANCEELLIPSE Compute points for covariance ellipse
            %
            % SYNTAX:
            %   ellipse_points = obj.computeCovarianceEllipse(mean_pos, cov_matrix, n_sigma)
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
