classdef PDA_HMM < DA_Filter
    % PDA_HMM Probabilistic Data Association Hidden Markov Model
    %
    % DESCRIPTION:
    %   Implements a hybrid tracker combining PDA data association with HMM
    %   grid-based state estimation for single target tracking. Uses discrete
    %   probability distributions over spatial grids instead of particles.
    %   Supports precomputed spatial likelihood lookup tables for radar/sensor
    %   fusion applications.
    %
    % RECENT UPDATES (v2.0):
    %   - Added validation region gating with configurable sigma bounds
    %   - Added getter/setter methods for detection model parameters
    %   - Added computeNormalizationConstants method for GNN compatibility
    %   - Added helper methods for likelihood computation
    %   - Enhanced debugging and state reporting capabilities
    %   - Improved compatibility with particle filter interfaces
    %
    % PROPERTIES:
    %   grid_size, npx2          - Grid dimensions (128x128, flattened to 16384)
    %   ptarget_prob            - Current probability distribution [npx2 x 1]
    %   A_transition            - HMM state transition matrix [npx2 x npx2]
    %   pointlikelihood_image   - Precomputed likelihood lookup table [npx2 x npx2]
    %   xgrid, ygrid            - Spatial grid coordinates
    %   debug, validate         - Control flags for debugging and validation
    %
    % METHODS:
    %   PDA_HMM                 - Constructor
    %   timestep                - Process single time step with PDA-HMM algorithm
    %   prediction              - HMM prediction step using transition matrix
    %   measurement_update      - PDA measurement update with grid likelihood
    %   getGaussianEstimate     - Extract mean and covariance from grid distribution
    %
    % EXAMPLE:
    %   hmm = PDA_HMM(x0, A_transition, pointlikelihood_image, 'Debug', true, 'DynamicPlot', true);
    %   hmm.timestep(z);
    %   [x_est, P_est] = hmm.getGaussianEstimate();
    %
    % See also DA_Filter, PDA_PF, test_HMM

    properties
        % Grid Parameters
        grid_size = 128 % Grid size (128x128)
        npx2 % Total grid points (128^2 = 16384)

        % Scene Parameters
        Xbounds = [-2, 2] % X bounds of scene in meters
        Ybounds = [0, 4] % Y bounds of scene in meters

        % Grid State
        ptarget_prob % Current probability distribution [npx2 x 1]

        % Local copies for visualization
        prior_prob % Prior probability distribution [npx2 x 1]
        likelihood_prob % Combined likelihood distribution [npx2 x 1]
        posterior_prob % Posterior probability distribution [npx2 x 1]

        % HMM Model
        A_transition % State transition matrix [npx2 x npx2]
        pointlikelihood_image % Precomputed likelihood lookup table [npx2 x npx2]

        % Spatial Grid
        xgrid % X coordinate grid [1 x 128]
        ygrid % Y coordinate grid [1 x 128]
        pxgrid % X meshgrid [128 x 128]
        pygrid % Y meshgrid [128 x 128]
        pxyvec % Vectorized coordinates [npx2 x 2]
        dx, dy % Grid resolution

        % PDA Parameters
        PD = 0.95 % Probability of detection
        PFA = 0.05 % Probability of false alarm

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

        function obj = PDA_HMM(x0, A_transition, pointlikelihood_image, varargin)
            % PDA_HMM Constructor for Probabilistic Data Association HMM
            %
            % SYNTAX:
            %   obj = PDA_HMM(x0, A_transition, pointlikelihood_image)
            %   obj = PDA_HMM(..., 'Debug', true, 'DynamicPlot', true, 'ValidationSigma', 3)
            %
            % INPUTS:
            %   x0                    - Initial state estimate [2 x 1] (position only)
            %   A_transition          - HMM transition matrix [npx2 x npx2]
            %   pointlikelihood_image - Precomputed likelihood lookup table [npx2 x npx2]
            %   varargin              - Name-value pairs:
            %                          'Debug', true/false
            %                          'DynamicPlot', true/false
            %                          'ValidationSigma', numeric
            %
            % OUTPUTS:
            %   obj - Initialized PDA_HMM object
            %
            % DESCRIPTION:
            %   Creates and initializes a PDA-HMM filter with Gaussian probability
            %   distribution around the initial state position on the spatial grid.
            %
            % See also timestep, prediction, measurement_update

            % Validate inputs
            if nargin < 3
                error('PDA_HMM:InvalidInput', 'Requires 3 inputs: {x0, A_transition, pointlikelihood_image}');
            end

            % Parse options using parent class utility
            options = DA_Filter.parseFilterOptions(varargin{:});
            obj.debug = options.Debug;
            obj.DynamicPlot = options.DynamicPlot;
            obj.validation_sigma_bounds = options.ValidationSigma;

            % Initialize grid parameters
            obj.npx2 = obj.grid_size ^ 2;

            % Create spatial grid
            obj.xgrid = linspace(obj.Xbounds(1), obj.Xbounds(2), obj.grid_size);
            obj.ygrid = linspace(obj.Ybounds(1), obj.Ybounds(2), obj.grid_size);
            [obj.pxgrid, obj.pygrid] = meshgrid(obj.xgrid, obj.ygrid);
            obj.pxyvec = [obj.pxgrid(:), obj.pygrid(:)];
            obj.dx = obj.xgrid(2) - obj.xgrid(1);
            obj.dy = obj.ygrid(2) - obj.ygrid(1);

            % Store HMM matrices
            obj.A_transition = A_transition;
            obj.pointlikelihood_image = pointlikelihood_image;

            % Validate matrix dimensions
            if size(A_transition, 1) ~= obj.npx2 || size(A_transition, 2) ~= obj.npx2
                error('PDA_HMM:InvalidDimensions', ...
                    'Transition matrix dimensions (%dx%d) do not match grid size (%d)', ...
                    size(A_transition, 1), size(A_transition, 2), obj.npx2);
            end

            if size(pointlikelihood_image, 1) ~= obj.npx2 || size(pointlikelihood_image, 2) ~= obj.npx2
                error('PDA_HMM:InvalidDimensions', ...
                    'Likelihood model dimensions (%dx%d) do not match grid size (%d)', ...
                    size(pointlikelihood_image, 1), size(pointlikelihood_image, 2), obj.npx2);
            end

            % Initialize probability distribution around initial position
            obj.initializeProbabilityDistribution(x0);

            if obj.debug
                fprintf('\n=== PDA_HMM INITIALIZATION ===\n');
                fprintf('Grid: %dx%d (%.4fm resolution)\n', ...
                    obj.grid_size, obj.grid_size, obj.dx);
                fprintf('Scene bounds: X[%.1f, %.1f], Y[%.1f, %.1f] m\n', ...
                    obj.Xbounds(1), obj.Xbounds(2), obj.Ybounds(1), obj.Ybounds(2));
                fprintf('Validation: %.1f-sigma bounds\n', obj.validation_sigma_bounds);
                fprintf('Detection: PD=%.3f, PFA=%.3f\n', obj.PD, obj.PFA);
                fprintf('==============================\n\n');
            end

            % Initialize dynamic plotting if enabled
            obj.initializeDynamicPlot('PDA-HMM Dynamic Tracking', [100, 100, 800, 600]);
        end

        function initializeProbabilityDistribution(obj, x0)
            % INITIALIZEPROBABILITYDISTRIBUTION Initialize grid probability around x0
            %
            % SYNTAX:
            %   obj.initializeProbabilityDistribution(x0)
            %
            % INPUTS:
            %   x0 - Initial position estimate [2 x 1]
            %
            % DESCRIPTION:
            %   Creates Gaussian probability distribution centered at x0 with
            %   appropriate initial uncertainty.

            % Initialize with Gaussian distribution around initial position
            obj.ptarget_prob = sparse(obj.npx2, 1);
            sigma_init = 0.3; % Initial uncertainty in meters

            for i = 1:obj.npx2
                x_pos = obj.pxyvec(i, 1);
                y_pos = obj.pxyvec(i, 2);
                dist_sq = (x_pos - x0(1)) ^ 2 + (y_pos - x0(2)) ^ 2;
                obj.ptarget_prob(i) = exp(-dist_sq / (2 * sigma_init ^ 2));
            end

            % Normalize probability distribution
            obj.ptarget_prob = obj.ptarget_prob / sum(obj.ptarget_prob);

            if obj.debug
                fprintf('-> Initialized probability distribution around [%.3f, %.3f]\n', x0(1), x0(2));
            end

        end

        %% ========== TIMESTEP ==========
        function timestep(obj, z, varargin)
            % TIMESTEP Process single time step with PDA-HMM algorithm
            %
            % SYNTAX:
            %   obj.timestep(z)
            %   obj.timestep(z, true_state)
            %
            % INPUTS:
            %   z          - Current measurements [2 x N_measurements]
            %   true_state - (optional) True state for visualization
            %
            % DESCRIPTION:
            %   Implements full PDA-HMM algorithm:
            %   1. Prediction step using HMM transition matrix
            %   2. Measurement validation using configurable sigma gating
            %   3. Measurement update with PDA likelihood combination
            %   Use getGaussianEstimate() to extract state estimates after timestep.
            %
            % See also prediction, measurement_update, getGaussianEstimate

            if obj.debug
                fprintf('\n=== PDA-HMM TIMESTEP START ===\n');
                fprintf('Input: %d measurements\n', size(z, 2));

                if ~isempty(z)

                    for i = 1:size(z, 2)
                        fprintf('  Meas %d: [%.3f, %.3f]\n', i, z(1, i), z(2, i));
                    end

                end

                fprintf('------------------------------\n');
            end

            % Step 1: Prediction step
            obj.prediction();

            % Step 2: Measurement validation (gating)
            z_original = z; % Store original measurements for debugging
            [z_valid, has_valid_meas] = obj.Validation(z);
            z_to_process = z_valid; % Use validated measurements

            % Step 3: Measurement update with PDA
            obj.measurement_update(z_to_process);

            if obj.debug
                [x_est, P_est] = obj.getGaussianEstimate();
                fprintf('\nOutput: State estimate [%.4f, %.4f] m\n', x_est(1), x_est(2));
                fprintf('        Covariance trace: %.6f\n', trace(P_est));
                fprintf('=== PDA-HMM TIMESTEP END ===\n\n');
            end

            % Update dynamic plot if enabled
            if obj.DynamicPlot

                if nargin > 2
                    true_state = varargin{1};
                    obj.updateDynamicPlot(z_to_process, true_state, z_original);
                else
                    obj.updateDynamicPlot(z_to_process, [], z_original);
                end

            end

        end

        %% ========== PREDICTION STEP ==========
        function prediction(obj)
            % PREDICTION HMM prediction step using transition matrix
            %
            % SYNTAX:
            %   obj.prediction()
            %
            % DESCRIPTION:
            %   Applies HMM state transition model to propagate probability
            %   distribution forward in time using pre-computed transition matrix.
            %
            % MODIFIES:
            %   obj.ptarget_prob - Updated predicted probability distribution
            %   obj.prior_prob   - Stores copy of predicted distribution
            %
            % ALGORITHM:
            %   P(x_k | z_1:k-1) = A_transition * P(x_k-1 | z_1:k-1)
            %
            % See also timestep, measurement_update

            if obj.debug
                fprintf('[PREDICTION] Applying HMM transition matrix...\n');
                fprintf('[PREDICTION] Prior sum before transition: %.6f\n', full(sum(obj.ptarget_prob)));
            end

            % Apply state transition model
            obj.ptarget_prob = obj.A_transition * obj.ptarget_prob;

            % Ensure probability distribution remains normalized
            obj.ptarget_prob = obj.ptarget_prob / sum(obj.ptarget_prob);

            % Store copy of prior for visualization
            obj.prior_prob = obj.ptarget_prob;

            if obj.debug
                fprintf('[PREDICTION] Prior sum after transition: %.6f\n', full(sum(obj.ptarget_prob)));
                fprintf('[PREDICTION] Prior max value: %.6f\n', full(max(obj.ptarget_prob)));
                fprintf('[PREDICTION] Prior spread (std): %.6f\n', full(std(obj.ptarget_prob)));
                fprintf('[PREDICTION] Complete.\n');
            end

        end

        %% ========== MEASUREMENT VALIDATION ==========
        function [z_valid, has_valid_meas] = Validation(obj, z)
            % VALIDATION Gate measurements using HMM grid state estimate
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
            %   ellipsoidal bounds around the current HMM grid state estimate. Uses Gaussian
            %   approximation extracted from the grid distribution and Mahalanobis distance
            %   thresholding with obj.validation_sigma_bounds.
            %
            % ALGORITHM:
            %   1. Extract Gaussian estimate (mean, covariance) from grid distribution
            %   2. Scale covariance by validation_sigma_bounds²
            %   3. Apply ellipsoidal gating using Mahalanobis distance threshold
            %   4. Accept measurements within chi-square threshold for 2 DOF
            %
            % See also timestep, measurement_update, getGaussianEstimate

            % Handle empty measurement case
            if isempty(z)
                z_valid = [];
                has_valid_meas = false;

                if obj.debug
                    fprintf('[GATING] No measurements to validate\n');
                end

                return;
            end

            % Get current state estimate and covariance from grid distribution
            [x_est, P_est] = obj.getGaussianEstimate();
            
            % Scale covariance for gating (similar to PDA_PF)
            gate_covariance = obj.validation_sigma_bounds^2 * P_est;
            
            % Apply ellipsoidal gating for each measurement
            z_valid = [];
            valid_count = 0;
            
            for i = 1:size(z, 2)
                meas_i = z(:, i);
                
                % Compute Mahalanobis distance
                innovation = meas_i - x_est;
                mahal_dist_sq = innovation' / gate_covariance * innovation;
                
                % Chi-square threshold for 2D (95% confidence for 2-sigma)
                chi2_threshold = obj.validation_sigma_bounds^2 * 2; % 2 DOF
                
                if mahal_dist_sq <= chi2_threshold
                    % Measurement passes gate
                    z_valid = [z_valid, meas_i];
                    valid_count = valid_count + 1;
                    
                    if obj.debug
                        fprintf('[GATING] Meas %d: [%.3f, %.3f] PASSED (dist²=%.3f <= %.3f)\n', ...
                            i, meas_i(1), meas_i(2), mahal_dist_sq, chi2_threshold);
                    end
                else
                    if obj.debug
                        fprintf('[GATING] Meas %d: [%.3f, %.3f] REJECTED (dist²=%.3f > %.3f)\n', ...
                            i, meas_i(1), meas_i(2), mahal_dist_sq, chi2_threshold);
                    end
                end
            end
            
            has_valid_meas = ~isempty(z_valid);

            if obj.debug
                fprintf('[GATING] Input: %d measurements, Output: %d valid measurements\n', ...
                    size(z, 2), valid_count);
                fprintf('[GATING] Using %.1f-sigma ellipsoidal gate (chi²=%.1f)\n', ...
                    obj.validation_sigma_bounds, chi2_threshold);
            end

        end

        %% ========== MEASUREMENT UPDATE (PDA) ==========
        function measurement_update(obj, z)
            % MEASUREMENT_UPDATE PDA measurement update with grid likelihood
            %
            % SYNTAX:
            %   obj.measurement_update(z)
            %
            % INPUTS:
            %   z - Current measurements [2 x N_measurements]
            %
            % DESCRIPTION:
            %   Implements PDA measurement update for grid-based HMM:
            %   1. Compute likelihood grid for each measurement
            %   2. Add all likelihood grids together (PDA combination)
            %   3. Add false positive (clutter) contribution as uniform scalar
            %   4. Apply combined likelihood to posterior
            %
            % MODIFIES:
            %   obj.ptarget_prob - Updated posterior probability distribution
            %
            % TODO: Implement proper PDA beta coefficients for measurement weights
            % TODO: Add validation gating to filter measurements
            % TODO: Optimize likelihood grid computation for multiple measurements
            %
            % See also timestep, prediction, likelihoodLookup

            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Starting PDA likelihood computation...\n');
            end

            % Compute normalization constants for GNN compatibility (optional)
            % normalization_constants = obj.computeNormalizationConstants(z);

            % Handle missed detection case
            if isempty(z)

                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] No measurements - missed detection case\n');
                end

                % TODO: Apply missed detection probability (1-PD) to posterior
                % For now, keep prediction unchanged

                % Store uniform likelihood for visualization (no information)
                obj.likelihood_prob = ones(obj.npx2, 1) / obj.npx2;
                obj.posterior_prob = obj.ptarget_prob; % Posterior equals prior for missed detection

                return;
            end

            N_measurements = size(z, 2);

            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Processing %d measurements:\n', N_measurements);
            end

            % Initialize combined likelihood grid
            clutter_contribution = (1 - obj.PD) * obj.PFA / obj.npx2;
            likelihood_total = clutter_contribution * ones(obj.npx2, 1);

            if obj.debug
                fprintf('[CLUTTER] Added clutter contribution: %.8f per grid cell\n', clutter_contribution);
            end

            % TEMPORARY FIX: CATEGORICAL SCALING
            % Get normalization constant for each measurement hypothesis
            normalization_constants = obj.computeNormalizationConstants(z);
            normalization_constants = normalization_constants / sum(normalization_constants); % Normalize

            likelihood_total = likelihood_total * normalization_constants(end); % Scale by clutter constant

            % Add likelihood contribution from each measurement
            for i = 1:N_measurements

                if obj.debug
                    fprintf('  Meas %d: [%.3f, %.3f] -> ', i, z(1, i), z(2, i));
                end

                % Get likelihood grid for this measurement using helper function
                likelihood_meas = obj.computeLikelihoodForMeasurement(z(:, i));

                % TODO: Apply proper PDA beta coefficient here instead of simple addition
                % For now, just add the likelihood grids together
                likelihood_total = likelihood_total + likelihood_meas * normalization_constants(i);
            end

            % Compute posterior: P(x_k | z_1:k) ∝ P(z_k | x_k) * P(x_k | z_1:k-1)
            obj.ptarget_prob = obj.ptarget_prob .* likelihood_total + eps;

            % Store copies for visualization
            obj.likelihood_prob = likelihood_total;

            % Normalize posterior distribution
            obj.ptarget_prob = obj.ptarget_prob / sum(obj.ptarget_prob);
            obj.posterior_prob = obj.ptarget_prob; % Store copy of posterior

            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Complete. Posterior: sum=%.6f, max=%.6f\n', ...
                    full(sum(obj.ptarget_prob)), full(max(obj.ptarget_prob)));
            end

        end

        %% ========== NORMALIZATION CONSTANTS (FOR GNN) ==========
        function normalization_constants = computeNormalizationConstants(obj, z)
            % COMPUTENORMALIZATIONCONSTANTS Compute normalization constants for each measurement hypothesis (for GNN)
            %
            % SYNTAX:
            %   normalization_constants = obj.computeNormalizationConstants(z)
            %
            % INPUTS:
            %   z - Current measurements [2 x N_measurements]
            %
            % OUTPUTS:
            %   normalization_constants - Vector of normalization constants [N_measurements + 1 x 1]
            %                           normalization_constants(i) = sum over grid of likelihood_i * prior
            %                           normalization_constants(end) = clutter hypothesis constant
            %
            % DESCRIPTION:
            %   Computes normalization constants for GNN-style data association by calculating
            %   the integral (sum) of likelihood * prior over the entire grid for each measurement
            %   hypothesis separately. This is the HMM equivalent of marginalizing over particles.
            %
            % ALGORITHM:
            %   For each measurement i:
            %     likelihood_grid_i = likelihoodLookup(measurement_i)
            %     normalization_constants(i) = sum(likelihood_grid_i .* prior_prob)
            %
            % See also measurement_update, likelihoodLookup

            N_measurements = size(z, 2);
            normalization_constants = zeros(N_measurements + 1, 1); % +1 for clutter

            % Handle empty measurement case -- just return clutter
            if N_measurements == 0
                % Only clutter hypothesis
                clutter_constant = (1 - obj.PD) * obj.PFA / obj.npx2;
                normalization_constants(1) = clutter_constant;

                if obj.debug
                    fprintf('[GNN] No measurements - only clutter hypothesis: %.6f\n', clutter_constant);
                end

                return;
            end

            % Compute normalization constant for each measurement individually
            for i = 1:N_measurements
                % Get likelihood grid for this measurement using helper function
                likelihood_grid = obj.computeLikelihoodForMeasurement(z(:, i));

                % Marginalize over the entire grid: sum(likelihood * prior)
                normalization_constants(i) = sum((obj.npx2 * obj.PD) * likelihood_grid .* obj.prior_prob);

                if obj.debug
                    fprintf('[GNN] Measurement %d normalization constant: %.6f\n', ...
                        i, normalization_constants(i));
                end

            end

            % Add clutter hypothesis normalization constant
            % Clutter hypothesis: (1-PD) * PFA / grid_area
            clutter_constant = (1 - obj.PD) * obj.PFA / obj.npx2;
            normalization_constants(end) = clutter_constant;

            if obj.debug
                fprintf('[GNN] Clutter hypothesis constant: %.6f\n', clutter_constant);
                fprintf('[GNN] Computed %d normalization constants (%d measurements + clutter)\n', ...
                    length(normalization_constants), N_measurements);
            end

        end

        %% ========== LIKELIHOOD COMPUTATION ==========
        function likelihood_grid = computeLikelihoodForMeasurement(obj, measurement)
            % COMPUTELIKELIHOODFORMEASUREMENT Compute grid likelihood for a single measurement
            %
            % SYNTAX:
            %   likelihood_grid = obj.computeLikelihoodForMeasurement(measurement)
            %
            % INPUTS:
            %   measurement - Single measurement [2 x 1]
            %
            % OUTPUTS:
            %   likelihood_grid - Likelihood values for all grid points [npx2 x 1]
            %
            % DESCRIPTION:
            %   Abstracts the common pattern of computing grid likelihood for a measurement
            %   using likelihood lookup and Gaussian masking. Used by both measurement_update
            %   and computeNormalizationConstants to avoid code duplication.
            %
            % See also measurement_update, computeNormalizationConstants, likelihoodLookup

            % Get likelihood for this measurement
            likelihood_grid = obj.likelihoodLookup(measurement);

            % Apply Gaussian mask (optional - can be disabled if needed)
            sf = 0.15; % scaling factor for Gaussian mask
            gaussmask = mvnpdf(obj.pxyvec, measurement', sf * eye(2));
            gaussmask(gaussmask < 0.1 * max(gaussmask)) = 0; % threshold small values

            % Combine likelihood and Gaussian mask and scale appropriately
            likelihood_grid = (obj.PD * obj.npx2) * likelihood_grid .* gaussmask;

            if obj.debug
                fprintf('    Likelihood: max=%.6f, sum=%.6f\n', ...
                    full(max(likelihood_grid)), full(sum(likelihood_grid)));
            end

        end

        function likelihood_grid = likelihoodLookup(obj, measurement)
            % LIKELIHOODLOOKUP Get likelihood grid for given measurement
            %
            % SYNTAX:
            %   likelihood_grid = obj.likelihoodLookup(measurement)
            %
            % INPUTS:
            %   measurement - Current measurement [2 x 1]
            %
            % OUTPUTS:
            %   likelihood_grid - Likelihood values for all grid points [npx2 x 1]
            %
            % DESCRIPTION:
            %   Retrieves likelihood values from precomputed lookup table for all
            %   grid points given a specific measurement location.
            %
            % See also measurement_update

            % Find closest grid point to measurement
            [~, meas_x_idx] = min(abs(obj.xgrid - measurement(1)));
            [~, meas_y_idx] = min(abs(obj.ygrid - measurement(2)));
            meas_linear_idx = sub2ind([obj.grid_size, obj.grid_size], meas_y_idx, meas_x_idx);

            if obj.debug
                fprintf('    Grid indices: x=%d, y=%d, linear=%d\n', ...
                    meas_x_idx, meas_y_idx, meas_linear_idx);
            end

            % Bounds checking
            if meas_linear_idx > size(obj.pointlikelihood_image, 1) || meas_linear_idx < 1
                error('PDA_HMM:IndexError', ...
                    'Measurement linear index %d out of bounds [1, %d]', ...
                    meas_linear_idx, size(obj.pointlikelihood_image, 1));
            end

            % Extract likelihood column from lookup table
            likelihood_grid = (obj.PD * obj.npx2) * obj.pointlikelihood_image(meas_linear_idx, :)' + eps;

            % Ensure output is column vector
            if size(likelihood_grid, 2) > 1
                likelihood_grid = likelihood_grid';
            end

            if obj.debug
                fprintf('    Likelihood: max=%.6f, sum=%.6f\n', ...
                    full(max(likelihood_grid)), full(sum(likelihood_grid)));
            end

        end

        %% ========== COMPATIBILITY METHODS ==========
        function ess = getEffectiveSampleSize(obj)
            % GETEFFECTIVESAMPLESIZE Get effective sample size equivalent for HMM
            %
            % SYNTAX:
            %   ess = obj.getEffectiveSampleSize()
            %
            % OUTPUTS:
            %   ess - Effective sample size measure based on probability distribution entropy
            %
            % DESCRIPTION:
            %   Computes an HMM equivalent of particle filter effective sample size
            %   using the entropy of the probability distribution. Lower entropy
            %   corresponds to higher effective sample size (more concentrated distribution).
            %
            % NOTE:
            %   This is primarily for compatibility with particle filter interfaces.
            %   The mapping is: ESS = exp(-entropy), normalized to [0, npx2] range.

            entropy = obj.getEntropy();

            % Convert entropy to ESS-like measure
            % High entropy -> low ESS, Low entropy -> high ESS
            max_entropy = log(obj.npx2); % Maximum possible entropy (uniform distribution)
            normalized_entropy = entropy / max_entropy; % [0, 1]
            ess = obj.npx2 * (1 - normalized_entropy); % [0, npx2]

            if obj.debug
                fprintf('[ESS] Entropy: %.4f, Normalized: %.4f, ESS: %.1f\n', ...
                    entropy, normalized_entropy, ess);
            end

        end

        function printState(obj, label)
            % PRINTSTATE Print current HMM state information for debugging
            %
            % SYNTAX:
            %   obj.printState(label)
            %
            % INPUTS:
            %   label - String label for the output
            %
            % DESCRIPTION:
            %   Prints detailed state information including position estimate,
            %   uncertainty measures, and probability distribution statistics.

            if nargin < 2
                label = 'HMM State';
            end

            % Get current estimates
            [x_est, P_est] = obj.getGaussianEstimate();
            [x_map, map_prob] = obj.getMAPEstimate();
            entropy = obj.getEntropy();
            ess = obj.getEffectiveSampleSize();

            % Print header
            fprintf('\n--- %s ---\n', label);
            fprintf('MMSE Estimate: [%.4f, %.4f] m\n', x_est(1), x_est(2));
            fprintf('MAP Estimate:  [%.4f, %.4f] m (prob=%.6f)\n', x_map(1), x_map(2), full(map_prob));
            fprintf('Covariance:    det=%.8f, trace=%.6f\n', det(P_est), trace(P_est));
            fprintf('Uncertainty:   entropy=%.4f, ESS=%.1f\n', entropy, ess);
            fprintf('Distribution:  min=%.8f, max=%.8f, sum=%.8f\n', ...
                full(min(obj.ptarget_prob)), full(max(obj.ptarget_prob)), full(sum(obj.ptarget_prob)));
            fprintf('Grid:          %dx%d (%d points)\n', obj.grid_size, obj.grid_size, obj.npx2);
            fprintf('-------------------\n');
        end

        %% ========== STATE ESTIMATION ==========
        function [x_est, P_est] = getGaussianEstimate(obj)
            % GETGAUSSIANESTIMATE Extract Gaussian estimate from grid distribution
            %
            % SYNTAX:
            %   [x_est, P_est] = obj.getGaussianEstimate()
            %
            % OUTPUTS:
            %   x_est - State estimate (position mean) [2 x 1]
            %   P_est - State covariance estimate [2 x 2]
            %
            % DESCRIPTION:
            %   Computes MMSE estimate (mean) and covariance from the discrete
            %   probability distribution over the spatial grid.
            %
            % See also timestep

            % Compute MMSE estimate (mean of distribution)
            x_est = sum(obj.pxyvec .* repmat(obj.ptarget_prob, [1, 2]), 1)';

            % Compute covariance matrix
            % E[xx^T] - μμ^T where μ is the mean
            second_moment = reshape(sum([obj.pxyvec(:, 1) .^ 2, ...
                                             obj.pxyvec(:, 1) .* obj.pxyvec(:, 2), ...
                                             obj.pxyvec(:, 2) .* obj.pxyvec(:, 1), ...
                                             obj.pxyvec(:, 2) .^ 2] .* ...
                repmat(obj.ptarget_prob, [1, 4]), 1), [2 2]);

            P_est = second_moment - x_est * x_est';

            % Ensure covariance is symmetric and positive semi-definite
            P_est = 0.5 * (P_est + P_est');
            P_est = P_est +1e-8 * eye(2); % Small regularization

            if obj.debug
                % Convert from sparse to dense matrices for computation
                x_est = full(x_est);
                P_est = full(P_est);

                fprintf('[GAUSSIAN EST] Mean=[%.4f, %.4f], det(P)=%.8f\n', ...
                    x_est(1), x_est(2), det(P_est));
            end

        end

        %% ========== MAP ESTIMATION ==========
        function [x_map, map_prob] = getMAPEstimate(obj)
            % GETMAPESTIMATE Get Maximum A Posteriori estimate from grid
            %
            % SYNTAX:
            %   [x_map, map_prob] = obj.getMAPEstimate()
            %
            % OUTPUTS:
            %   x_map    - MAP position estimate [2 x 1]
            %   map_prob - Probability value at MAP estimate
            %
            % DESCRIPTION:
            %   Finds the grid point with highest probability density.

            [map_prob, map_idx] = max(obj.ptarget_prob);
            x_map = obj.pxyvec(map_idx, :)';

            if obj.debug
                fprintf('[MAP EST] Position=[%.4f, %.4f], prob=%.6f\n', ...
                    x_map(1), x_map(2), full(map_prob));
            end

        end

        %% ========== UNCERTAINTY ANALYSIS ==========
        function entropy = getEntropy(obj)
            % GETENTROPY Compute entropy of current probability distribution
            %
            % SYNTAX:
            %   entropy = obj.getEntropy()
            %
            % OUTPUTS:
            %   entropy - Shannon entropy of current distribution
            %
            % DESCRIPTION:
            %   Computes H = -sum(p * log(p)) as measure of uncertainty.

            % Avoid log(0) by adding small epsilon
            p_safe = obj.ptarget_prob + eps;
            entropy = -sum(obj.ptarget_prob .* log(p_safe));

            if obj.debug
                fprintf('[ENTROPY] Current entropy: %.4f\n', full(entropy));
            end

        end

        %% ========== VISUALIZATION ==========
        function visualize(obj, figure_handle, title_str, z_valid, true_state, z_all)
            % VISUALIZE Plot prior, likelihood, and posterior distributions in 1x3 layout
            %
            % SYNTAX:
            %   obj.visualize()
            %   obj.visualize(figure_handle)
            %   obj.visualize(figure_handle, title_str)
            %   obj.visualize(figure_handle, title_str, z_valid, true_state)
            %   obj.visualize(figure_handle, title_str, z_valid, true_state, z_all)
            %
            % INPUTS:
            %   figure_handle - (optional) Figure handle to plot in
            %   title_str     - (optional) Title string for main plot
            %   z_valid       - (optional) Validated measurements [2 x N_valid] for overlay
            %   true_state    - (optional) True state [N x 1] for overlay
            %   z_all         - (optional) All measurements [2 x N_all] for visualization
            %
            % DESCRIPTION:
            %   Creates 1x3 subplot showing prior, likelihood, and posterior
            %   distributions with proper axis labels, colorbars, and overlays.
            %   Colors validated measurements differently from rejected ones.

            if nargin < 2 || isempty(figure_handle)
                figure;
            else
                figure(figure_handle);
            end

            if nargin < 3
                title_str = 'PDA\_HMM Step';
            end

            if nargin < 4
                z_valid = [];
            end

            if nargin < 5
                true_state = [];
            end

            if nargin < 6
                z_all = [];
            end

            % Subplot 1: Prior P(x_k | z_1:k-1)
            subplot(1, 3, 1);

            if ~isempty(obj.prior_prob)
                prior_grid = reshape(obj.prior_prob, [obj.grid_size, obj.grid_size]);
                imagesc(obj.xgrid, obj.ygrid, prior_grid);
                set(gca, 'YDir', 'normal');
                % Individual color scale for prior
                clim([min(obj.prior_prob), max(obj.prior_prob)]);
                colorbar;
            else
                % Plot current distribution if prior not available
                prob_grid = reshape(obj.ptarget_prob, [obj.grid_size, obj.grid_size]);
                imagesc(obj.xgrid, obj.ygrid, prob_grid);
                set(gca, 'YDir', 'normal');
                clim([min(obj.ptarget_prob), max(obj.ptarget_prob)]);
                colorbar;
            end

            xlabel('X (m)');
            ylabel('Y (m)');
            title('Prior $P(x_k | z_{1:k-1})$');
            xlim(obj.Xbounds);
            ylim(obj.Ybounds);
            axis equal tight;

            % Add true state overlay if provided
            hold on;

            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'd', 'Color', 'm', ...
                    'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm');
            end

            hold off;

            % Subplot 2: Likelihood P(z_k | x_k)
            subplot(1, 3, 2);

            if ~isempty(obj.likelihood_prob)
                likelihood_grid = reshape(obj.likelihood_prob, [obj.grid_size, obj.grid_size]);
                imagesc(obj.xgrid, obj.ygrid, likelihood_grid);
                set(gca, 'YDir', 'normal');
                % Individual color scale for likelihood
                clim([min(obj.likelihood_prob), max(obj.likelihood_prob)]);
                colorbar;
            else
                % Show empty plot if likelihood not available
                imagesc(obj.xgrid, obj.ygrid, zeros(obj.grid_size));
                set(gca, 'YDir', 'normal');
                clim([0, 1]);
                colorbar;
            end

            xlabel('X (m)');
            ylabel('Y (m)');
            title('Likelihood $P(z_k | x_k)$');
            xlim(obj.Xbounds);
            ylim(obj.Ybounds);
            axis equal tight;

            % Add measurements overlay if provided
            hold on;

            % Plot all measurements (if available) in gray/rejected color
            if ~isempty(z_all)
                plot(z_all(1, :), z_all(2, :), 'x', 'Color', [0.7 0.7 0.7], ...
                    'MarkerSize', 4, 'LineWidth', 1);
                
                % Plot validated measurements (if available) in bright color on top
                if ~isempty(z_valid)
                    plot(z_valid(1, :), z_valid(2, :), '+', 'Color', [1 0.5 0], ...
                        'MarkerSize', 3, 'LineWidth', 1);
                end
            else
                % Backward compatibility: if only z_valid provided, treat as all measurements
                if ~isempty(z_valid)
                    plot(z_valid(1, :), z_valid(2, :), '+', 'Color', [1 0.5 0], ...
                        'MarkerSize', 3, 'LineWidth', 1);
                end
            end

            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'd', 'Color', 'm', ...
                    'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm');
            end

            hold off;

            % Subplot 3: Posterior P(x_k | z_1:k)
            subplot(1, 3, 3);

            if ~isempty(obj.posterior_prob)
                posterior_grid = reshape(obj.posterior_prob, [obj.grid_size, obj.grid_size]);
                imagesc(obj.xgrid, obj.ygrid, posterior_grid);
                set(gca, 'YDir', 'normal');
                % Individual color scale for posterior
                clim([min(obj.posterior_prob), max(obj.posterior_prob)]);
                colorbar;
            else
                % Plot current distribution if posterior not available
                prob_grid = reshape(obj.ptarget_prob, [obj.grid_size, obj.grid_size]);
                imagesc(obj.xgrid, obj.ygrid, prob_grid);
                set(gca, 'YDir', 'normal');
                clim([min(obj.ptarget_prob), max(obj.ptarget_prob)]);
                colorbar;
            end

            xlabel('X (m)');
            ylabel('Y (m)');
            title('Posterior $P(x_k | z_{1:k})$');
            xlim(obj.Xbounds);
            ylim(obj.Ybounds);
            axis equal tight;

            % Add state estimates and overlays
            hold on;
            [x_mmse, ~] = obj.getGaussianEstimate();
            [x_map, ~] = obj.getMAPEstimate();

            plot(x_mmse(1), x_mmse(2), 'ro', 'MarkerSize', 8, 'LineWidth', 2, ...
                'DisplayName', 'MMSE');
            plot(x_map(1), x_map(2), 'bs', 'MarkerSize', 8, 'LineWidth', 2, ...
                'DisplayName', 'MAP');

            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'd', 'Color', 'm', ...
                    'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm', ...
                    'DisplayName', 'True Position');
            end

            legend('Location', 'best');
            hold off;

            % Add main title with proper LaTeX formatting
            % Fix LaTeX typesetting for filter name
            if contains(title_str, 'PDA-HMM') || contains(title_str, 'PDA_HMM')
                title_str = strrep(title_str, 'PDA-HMM', 'PDA\_HMM');
                title_str = strrep(title_str, 'PDA_HMM', 'PDA\_HMM');
            end

            sgtitle(title_str, 'FontSize', 14, 'FontWeight', 'bold');

            % Preserve existing figure position if figure exists, otherwise set default
            if exist('figure_handle', 'var') && ~isempty(figure_handle) && isvalid(figure_handle)
                % Keep existing position - don't override user resizing
            else
                % Set default position only for new figures
                set(gcf, 'Position', [100, 100, 1200, 400]);
            end

        end

        function updateDynamicPlot(obj, measurements, true_state, all_measurements)
            % UPDATEDYNAMICPLOT Update dynamic plot during timestep execution (PDA_HMM override)
            %
            % SYNTAX:
            %   obj.updateDynamicPlot(measurements)
            %   obj.updateDynamicPlot(measurements, true_state)
            %   obj.updateDynamicPlot(measurements, true_state, all_measurements)
            %
            % INPUTS:
            %   measurements     - Used measurements [N_z x N_measurements] (after gating)
            %   true_state       - (optional) True state for comparison
            %   all_measurements - (optional) All measurements [N_z x N_measurements] (before gating)
            %
            % DESCRIPTION:
            %   Updates the dynamic visualization if enabled. Overrides parent method
            %   to support displaying both all measurements and used measurements with
            %   color coding for validation status.
            
            if ~obj.DynamicPlot || isempty(obj.dynamic_figure_handle) || ...
               ~isvalid(obj.dynamic_figure_handle)
                return;
            end
            
            % Handle optional arguments
            if nargin < 3 || isempty(true_state)
                true_state = [];
            end
            
            if nargin < 4
                all_measurements = [];
            end
            
            % Increment timestep counter
            obj.timestep_counter = obj.timestep_counter + 1;
            
            % Create title with timestep information
            title_str = sprintf('%s Real-time Tracking (Step %d)', ...
                class(obj), obj.timestep_counter);
            
            % Call PDA_HMM-specific visualization with both measurement sets
            obj.visualize(obj.dynamic_figure_handle, title_str, measurements, true_state, all_measurements);
            
            drawnow; % Force immediate update
            
            % Capture frame for animation after plot is updated
            obj.captureFrame();
            
            pause(0.01); % Small pause for smooth animation
        end

        %% ========== GETTER/SETTER METHODS ==========
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
            %   These parameters affect the likelihood combination and clutter hypothesis
            %   calculations in the measurement update and normalization constants.
            %
            % See also computeNormalizationConstants, measurement_update

            % Validate inputs
            if PD <= 0 || PD > 1
                error('PDA_HMM:InvalidPD', 'Detection probability PD must be in range (0, 1]');
            end

            if PFA < 0 || PFA >= 1
                error('PDA_HMM:InvalidPFA', 'False alarm probability PFA must be in range [0, 1)');
            end

            obj.PD = PD;
            obj.PFA = PFA;

            if obj.debug
                fprintf('[DETECTION MODEL] Updated: PD=%.3f, PFA=%.3f\n', obj.PD, obj.PFA);
            end

        end

        function loadLikelihoodData(obj, likelihood_file_path)
            % LOADLIKELIHOODDATA Load precomputed likelihood lookup table for hybrid HMM
            %
            % SYNTAX:
            %   obj.loadLikelihoodData(likelihood_file_path)
            %
            % INPUTS:
            %   likelihood_file_path - string, path to .mat file containing 'pointlikelihood_image'
            %
            % DESCRIPTION:
            %   Loads and validates the precomputed likelihood lookup table used by the
            %   hybrid HMM filter for measurement updates. Expected table size is
            %   128^2 x 128^2 corresponding to spatial grid discretization.
            %
            % See also PDA_HMM, measurement_update, likelihoodLookup

            if obj.debug
                fprintf('\n=== LOADING LIKELIHOOD DATA ===\n');
                fprintf('Loading from: %s\n', likelihood_file_path);
            end

            try
                % Check if file exists
                if ~exist(likelihood_file_path, 'file')
                    error('PDA_HMM:FileNotFound', ...
                        'Likelihood file not found: %s', likelihood_file_path);
                end

                % Load the likelihood data
                likelihood_data = load(likelihood_file_path, 'pointlikelihood_image');

                % Validate that the expected variable exists
                if ~isfield(likelihood_data, 'pointlikelihood_image')
                    error('PDA_HMM:InvalidData', ...
                        'Variable "pointlikelihood_image" not found in file: %s', likelihood_file_path);
                end

                % Store as class property (immutable lookup table)
                obj.pointlikelihood_image = likelihood_data.pointlikelihood_image;

                % Validate dimensions (expecting 128^2 x 128^2 based on grid)
                expected_dim = obj.npx2;
                [rows, cols] = size(obj.pointlikelihood_image);

                if obj.debug
                    fprintf('[VALIDATION] Loaded table dimensions: %dx%d\n', rows, cols);
                    fprintf('[VALIDATION] Expected dimensions: %dx%d\n', expected_dim, expected_dim);
                end

                if rows ~= expected_dim || cols ~= expected_dim
                    warning('PDA_HMM:DimensionMismatch', ...
                        'Likelihood model dimensions (%dx%d) do not match expected grid size (%dx%d)', ...
                        rows, cols, expected_dim, expected_dim);
                end

                if obj.debug
                    fprintf('[SUCCESS] Likelihood lookup table loaded successfully\n');
                    fprintf('===============================\n\n');
                end

            catch ME
                error('PDA_HMM:LoadError', ...
                    'Failed to load likelihood data: %s', ME.message);
            end

        end

        %% ========== VALIDATION AND DEBUGGING ==========
        function validateState(obj)
            % VALIDATESTATE Check internal state consistency
            %
            % DESCRIPTION:
            %   Validates that probability distribution sums to 1 and other
            %   consistency checks for debugging purposes.

            prob_sum = sum(obj.ptarget_prob);

            if abs(full(prob_sum) - 1.0) > 1e-6
                warning('PDA_HMM:Normalization', ...
                    'Probability distribution not normalized: sum = %.8f', full(prob_sum));
            end

            if any(obj.ptarget_prob < 0)
                warning('PDA_HMM:NegativeProbability', ...
                'Negative probabilities detected');
            end

            if obj.debug
                fprintf('[VALIDATION] State: sum=%.8f, min=%.8f, max=%.8f\n', ...
                    full(prob_sum), full(min(obj.ptarget_prob)), full(max(obj.ptarget_prob)));
            end

        end

    end

end
