classdef GNN_HMM < DA_Filter
    % GNN_HMM Global Nearest Neighbor Hidden Markov Model
    %
    % DESCRIPTION:
    %   Implements a hybrid tracker combining GNN data association with HMM
    %   grid-based state estimation for single target tracking. Uses discrete
    %   probability distributions over spatial grids instead of particles.
    %   Supports precomputed spatial likelihood lookup tables for radar/sensor
    %   fusion applications.
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
    %   GNN_HMM                 - Constructor
    %   timestep                - Process single time step with GNN-HMM algorithm
    %   prediction              - HMM prediction step using transition matrix
    %   measurement_update      - GNN measurement update with grid likelihood
    %   getGaussianEstimate     - Extract mean and covariance from grid distribution
    %
    % EXAMPLE:
    %   hmm = GNN_HMM(x0, A_transition, pointlikelihood_image, 'Debug', true, 'DynamicPlot', true);
    %   hmm.timestep(measurements);
    %   [x_est, P_est] = hmm.getGaussianEstimate();
    %
    % See also DA_Filter, GNN_PF, test_HMM

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

        % GNN Parameters
        PD = 0.95 % Probability of detection
        PFA = 0.05 % Probability of false alarm
        lambda_clutter = 2.5 % Clutter density parameter

        % Control Flags (inherited from DA_Filter)
        debug = false % Enable debug output and validation
        validate = false % Enable input/output validation checks
        DynamicPlot = false % Enable real-time visualization during timesteps

        % Dynamic Plotting (inherited from DA_Filter)
        dynamic_figure_handle % Figure handle for dynamic plotting
    end

    methods

        function obj = GNN_HMM(x0, A_transition, pointlikelihood_image, varargin)
            % GNN_HMM Constructor for Global Nearest Neighbor HMM
            %
            % SYNTAX:
            %   obj = GNN_HMM(x0, A_transition, pointlikelihood_image)
            %   obj = GNN_HMM(..., 'Debug', true, 'DynamicPlot', true)
            %
            % INPUTS:
            %   x0                    - Initial state estimate [2 x 1] (position only)
            %   A_transition          - HMM transition matrix [npx2 x npx2]
            %   pointlikelihood_image - Precomputed likelihood lookup table [npx2 x npx2]
            %   varargin              - Name-value pairs: 'Debug', true/false, 'DynamicPlot', true/false
            %
            % OUTPUTS:
            %   obj - Initialized GNN_HMM object
            %
            % DESCRIPTION:
            %   Creates and initializes a GNN-HMM filter with Gaussian probability
            %   distribution around the initial state position on the spatial grid.
            %
            % See also timestep, prediction, measurement_update

            % Validate inputs
            if nargin < 3
                error('GNN_HMM:InvalidInput', 'Requires 3 inputs: {x0, A_transition, pointlikelihood_image}');
            end

            % Parse options using parent class utility
            options = DA_Filter.parseFilterOptions(varargin{:});
            obj.debug = options.Debug;
            obj.DynamicPlot = options.DynamicPlot;

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
                error('GNN_HMM:InvalidDimensions', ...
                    'Transition matrix dimensions (%dx%d) do not match grid size (%d)', ...
                    size(A_transition, 1), size(A_transition, 2), obj.npx2);
            end

            if size(pointlikelihood_image, 1) ~= obj.npx2 || size(pointlikelihood_image, 2) ~= obj.npx2
                error('GNN_HMM:InvalidDimensions', ...
                    'Likelihood model dimensions (%dx%d) do not match grid size (%d)', ...
                    size(pointlikelihood_image, 1), size(pointlikelihood_image, 2), obj.npx2);
            end

            % Initialize probability distribution around initial position
            obj.initializeProbabilityDistribution(x0);

            if obj.debug
                fprintf('\n=== GNN_HMM INITIALIZATION ===\n');
                fprintf('Grid: %dx%d (%.4fm resolution)\n', ...
                    obj.grid_size, obj.grid_size, obj.dx);
                fprintf('Scene bounds: X[%.1f, %.1f], Y[%.1f, %.1f] m\n', ...
                    obj.Xbounds(1), obj.Xbounds(2), obj.Ybounds(1), obj.Ybounds(2));
                fprintf('==============================\n\n');
            end

            % Initialize dynamic plotting if enabled
            obj.initializeDynamicPlot('GNN-HMM Dynamic Tracking', [100, 100, 800, 600]);
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

        function timestep(obj, measurements, varargin)
            % TIMESTEP Process single time step with GNN-HMM algorithm
            %
            % SYNTAX:
            %   obj.timestep(measurements)
            %   obj.timestep(measurements, true_state)
            %
            % INPUTS:
            %   measurements - Current measurements [2 x N_measurements]
            %   true_state   - (optional) True state for visualization
            %
            % DESCRIPTION:
            %   Implements full GNN-HMM algorithm:
            %   1. Prediction step using HMM transition matrix
            %   2. Data association step using GNN to select best measurement
            %   3. Measurement update with selected measurement likelihood
            %   Use getGaussianEstimate() to extract state estimates after timestep.
            %
            % See also prediction, measurement_update, getGaussianEstimate

            if obj.debug
                fprintf('\n=== GNN-HMM TIMESTEP START ===\n');
                fprintf('Input: %d measurements\n', size(measurements, 2));

                if ~isempty(measurements)

                    for i = 1:size(measurements, 2)
                        fprintf('  Meas %d: [%.3f, %.3f]\n', i, measurements(1, i), measurements(2, i));
                    end

                end

                fprintf('------------------------------\n');
            end

            % Step 1: Prediction step
            obj.prediction();

            % Step 2: Data Association step - GNN selects best measurement
            % TODO: Implement GNN data association to select single best measurement
            % TODO: Compare likelihoods/costs for each measurement and choose optimal one
            % TODO: Handle missed detection case when no measurements available
            % TODO: Consider validation gating before data association
            selected_measurement = obj.Data_Association(measurements);

            % Step 3: Measurement update with selected measurement
            obj.measurement_update(selected_measurement);

            if obj.debug
                [x_est, P_est] = obj.getGaussianEstimate();
                fprintf('\nOutput: State estimate [%.4f, %.4f] m\n', x_est(1), x_est(2));
                fprintf('        Covariance trace: %.6f\n', trace(P_est));
                fprintf('=== GNN-HMM TIMESTEP END ===\n\n');
            end

            % Update dynamic plot if enabled
            if obj.DynamicPlot

                if nargin > 2
                    true_state = varargin{1};
                    obj.updateDynamicPlot(measurements, true_state);
                else
                    obj.updateDynamicPlot(measurements);
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

        %% ========== GLOBAL NEAREST NEIGHBOR DATA ASSOCIATION ==========
        function selected_measurement = Data_Association(obj, measurements)
            % DATA_ASSOCIATION Global Nearest Neighbor data association for measurements
            %
            % SYNTAX:
            %   selected_measurement = obj.Data_Association(measurements)
            %
            % INPUTS:
            %   measurements - Current measurements [2 x N_measurements]
            %
            % OUTPUTS:
            %   selected_measurement - Single selected measurement [2 x 1] or empty if none selected
            %
            % DESCRIPTION:
            %   Implements GNN data association algorithm to select the single best
            %   measurement from the available set. The selection is based on maximizing
            %   the likelihood or minimizing a cost function for grid-based HMM.
            %
            % TODO: Implement proper GNN cost function for grid-based HMM
            % TODO: Compare different cost metrics: integrated likelihood, maximum likelihood, etc.
            % TODO: Handle edge cases: no measurements, single measurement, multiple measurements
            % TODO: Consider validation gating before data association
            % TODO: Implement Mahalanobis distance or other distance metrics for grid-based systems
            %
            % ALGORITHM:
            %   1. If no measurements: return empty (missed detection)
            %   2. If single measurement: return that measurement
            %   3. If multiple measurements: compute cost/likelihood for each and select best
            %
            % See also timestep, measurement_update, likelihoodLookup

            % Handle empty measurement case (missed detection)
            if isempty(measurements)
                selected_measurement = [];

                if obj.debug
                    fprintf('[GNN DATA ASSOCIATION] No measurements - missed detection\n');
                end

                return;
            end

            % Handle single measurement case
            if size(measurements, 2) == 1
                selected_measurement = measurements;

                if obj.debug
                    fprintf('[GNN DATA ASSOCIATION] Single measurement - auto-selected [%.3f, %.3f]\n', ...
                        selected_measurement(1), selected_measurement(2));
                end

                return;
            end

            % Multiple measurements case - need to select best one
            N_measurements = size(measurements, 2);

            if obj.debug
                fprintf('[GNN DATA ASSOCIATION] Selecting best from %d measurements\n', N_measurements);
            end

            % TODO: Implement proper GNN cost function for grid-based HMM
            % For now, use a simple approach based on likelihood comparison
            
            % TODO: Option 1: Maximum integrated likelihood approach
            % - Compute likelihood grid for each measurement
            % - Integrate likelihood * prior for each measurement: sum(likelihood_grid .* prior_prob)
            % - Select measurement with maximum integrated likelihood
            
            % TODO: Option 2: Maximum likelihood at MAP estimate approach
            % - Get current MAP estimate from grid
            % - Evaluate likelihood at MAP position for each measurement
            % - Select measurement with maximum likelihood at MAP
            
            % TODO: Option 3: Minimum "distance" approach (grid-based)
            % - Define distance metric suitable for grid-based systems
            % - Could use KL divergence between predicted and measurement-updated distributions
            % - Select measurement with minimum distance/cost
            
            % For now, use maximum integrated likelihood approach
            integrated_likelihoods = zeros(1, N_measurements);
            
            for i = 1:N_measurements
                % Get likelihood grid for this measurement
                likelihood_grid = obj.likelihoodLookup(measurements(:, i));
                
                % Integrate likelihood with current prior
                integrated_likelihoods(i) = sum(likelihood_grid .* obj.ptarget_prob);
                
                if obj.debug
                    fprintf('[GNN] Measurement %d integrated likelihood: %.6f\n', i, integrated_likelihoods(i));
                end
            end
            
            % Select measurement with maximum integrated likelihood
            [max_likelihood, best_idx] = max(integrated_likelihoods);
            selected_measurement = measurements(:, best_idx);

            if obj.debug
                fprintf('[GNN] Integrated likelihoods: [');
                for i = 1:N_measurements
                    fprintf('%.6f ', integrated_likelihoods(i));
                end
                fprintf(']\n');
                fprintf('[GNN] Selected measurement %d (max integrated likelihood %.6f): [%.3f, %.3f]\n', ...
                    best_idx, max_likelihood, selected_measurement(1), selected_measurement(2));
            end

        end

        %% ========== MEASUREMENT UPDATE ==========
        function measurement_update(obj, measurement)
            % MEASUREMENT_UPDATE GNN measurement update with selected measurement
            %
            % SYNTAX:
            %   obj.measurement_update(measurement)
            %
            % INPUTS:
            %   measurement - Selected measurement [2 x 1] (from GNN data association)
            %
            % DESCRIPTION:
            %   Implements GNN measurement update for grid-based HMM using single
            %   selected measurement. Unlike PDA which combines all measurements,
            %   GNN uses only the best measurement as determined by data association.
            %   Handles missed detection case when measurement is empty.
            %
            % MODIFIES:
            %   obj.ptarget_prob - Updated posterior probability distribution
            %
            % TODO: Optimize likelihood computation for single measurement case
            % TODO: Consider different update strategies for GNN vs PDA
            % TODO: Implement proper missed detection handling with detection probability
            % TODO: Remove or modify Gaussian mask for single measurement case
            %
            % See also timestep, prediction, likelihoodLookup, Data_Association

            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Starting GNN likelihood computation...\n');
            end

            % Handle missed detection case (no measurement selected by GNN)
            if isempty(measurement)

                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] No measurement selected - missed detection case\n');
                end

                % TODO: Apply missed detection probability (1-PD) to posterior
                % For now, keep prediction unchanged

                % Store uniform likelihood for visualization (no information)
                obj.likelihood_prob = ones(obj.npx2, 1) / obj.npx2;
                obj.posterior_prob = obj.ptarget_prob; % Posterior equals prior for missed detection

                return;
            end

            % Single measurement case - simpler than PDA
            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Processing selected measurement: [%.3f, %.3f]\n', ...
                    measurement(1), measurement(2));
            end

            % Get likelihood grid for the selected measurement
            likelihood_meas = obj.likelihoodLookup(measurement);

            % TODO: Apply Gaussian mask - consider if this is appropriate for GNN
            % Unlike PDA which needs to handle multiple measurements, GNN has already
            % selected the best measurement, so Gaussian mask might be redundant
            sf = 0.15; % scaling factor for Gaussian mask
            gaussmask = mvnpdf(obj.pxyvec, measurement', sf * eye(2));
            gaussmask(gaussmask < 0.1 * max(gaussmask)) = 0; % threshold small values

            % Apply Gaussian mask to likelihood
            likelihood_total = likelihood_meas .* gaussmask;

            if obj.debug
                fprintf('[GAUSSIAN MASK] Applied around selected measurement [%.3f, %.3f]\n', ...
                    measurement(1), measurement(2));
                fprintf('[LIKELIHOOD] Max likelihood: %.6f, sum: %.6f\n', ...
                    full(max(likelihood_total)), full(sum(likelihood_total)));
            end

            % Compute posterior: P(x_k | z_1:k) ∝ P(z_k | x_k) * P(x_k | z_1:k-1)
            obj.ptarget_prob = obj.ptarget_prob .* likelihood_total;

            % Store copies for visualization
            obj.likelihood_prob = likelihood_total / sum(likelihood_total); % Normalize for visualization

            % Normalize posterior distribution
            obj.ptarget_prob = obj.ptarget_prob / sum(obj.ptarget_prob);
            obj.posterior_prob = obj.ptarget_prob; % Store copy of posterior

            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Complete. Posterior: sum=%.6f, max=%.6f\n', ...
                    full(sum(obj.ptarget_prob)), full(max(obj.ptarget_prob)));
            end

        end

        %% ========== LIKELIHOOD COMPUTATION ==========
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
            % See also measurement_update, Data_Association

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
                error('GNN_HMM:IndexError', ...
                    'Measurement linear index %d out of bounds [1, %d]', ...
                    meas_linear_idx, size(obj.pointlikelihood_image, 1));
            end

            % Extract likelihood column from lookup table
            likelihood_grid = obj.pointlikelihood_image(meas_linear_idx, :)';

            % Ensure output is column vector
            if size(likelihood_grid, 2) > 1
                likelihood_grid = likelihood_grid';
            end

            if obj.debug
                fprintf('    Likelihood: max=%.6f, sum=%.6f\n', ...
                    full(max(likelihood_grid)), full(sum(likelihood_grid)));
            end

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

        %% ========== VALIDATION AND DEBUGGING ==========
        function validateState(obj)
            % VALIDATESTATE Check internal state consistency
            %
            % DESCRIPTION:
            %   Validates that probability distribution sums to 1 and other
            %   consistency checks for debugging purposes.

            prob_sum = sum(obj.ptarget_prob);

            if abs(full(prob_sum) - 1.0) > 1e-6
                warning('GNN_HMM:Normalization', ...
                    'Probability distribution not normalized: sum = %.8f', full(prob_sum));
            end

            if any(obj.ptarget_prob < 0)
                warning('GNN_HMM:NegativeProbability', ...
                'Negative probabilities detected');
            end

            if obj.debug
                fprintf('[VALIDATION] State: sum=%.8f, min=%.8f, max=%.8f\n', ...
                    full(prob_sum), full(min(obj.ptarget_prob)), full(max(obj.ptarget_prob)));
            end

        end

        %% ========== VISUALIZATION ==========
        function visualize(obj, figure_handle, title_str, measurements, true_state)
            % VISUALIZE Plot prior, likelihood, and posterior distributions in 1x3 layout
            %
            % SYNTAX:
            %   obj.visualize()
            %   obj.visualize(figure_handle)
            %   obj.visualize(figure_handle, title_str)
            %   obj.visualize(figure_handle, title_str, measurements, true_state)
            %
            % INPUTS:
            %   figure_handle - (optional) Figure handle to plot in
            %   title_str     - (optional) Title string for main plot
            %   measurements  - (optional) Current measurements [2 x N] for overlay
            %   true_state    - (optional) True state [N x 1] for overlay
            %
            % DESCRIPTION:
            %   Creates 1x3 subplot showing prior, likelihood, and posterior
            %   distributions with proper axis labels, colorbars, and overlays.

            if nargin < 2 || isempty(figure_handle)
                figure;
            else
                figure(figure_handle);
            end

            if nargin < 3
                title_str = 'GNN\_HMM Step';
            end

            if nargin < 4
                measurements = [];
            end

            if nargin < 5
                true_state = [];
            end

            % Set consistent color limits for all subplots
            if ~isempty(obj.prior_prob) && ~isempty(obj.likelihood_prob) && ~isempty(obj.posterior_prob)
                max_prob = max([max(obj.prior_prob), max(obj.likelihood_prob), max(obj.posterior_prob)]);
                color_lims = [0, max_prob];
            else
                color_lims = [0, 1];
            end

            % Subplot 1: Prior P(x_k | z_1:k-1)
            subplot(1, 3, 1);

            if ~isempty(obj.prior_prob)
                prior_grid = reshape(obj.prior_prob, [obj.grid_size, obj.grid_size]);
                imagesc(obj.xgrid, obj.ygrid, prior_grid);
                set(gca, 'YDir', 'normal');
                clim(color_lims);
                colorbar;
            else
                % Plot current distribution if prior not available
                prob_grid = reshape(obj.ptarget_prob, [obj.grid_size, obj.grid_size]);
                imagesc(obj.xgrid, obj.ygrid, prob_grid);
                set(gca, 'YDir', 'normal');
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
                clim(color_lims);
                colorbar;
            else
                % Show empty plot if likelihood not available
                imagesc(obj.xgrid, obj.ygrid, zeros(obj.grid_size));
                set(gca, 'YDir', 'normal');
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

            if ~isempty(measurements)
                plot(measurements(1, :), measurements(2, :), '+', 'Color', [1 0.5 0], ...
                    'MarkerSize', 10, 'LineWidth', 3);
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
                clim(color_lims);
                colorbar;
            else
                % Plot current distribution if posterior not available
                prob_grid = reshape(obj.ptarget_prob, [obj.grid_size, obj.grid_size]);
                imagesc(obj.xgrid, obj.ygrid, prob_grid);
                set(gca, 'YDir', 'normal');
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

            if ~isempty(measurements)
                plot(measurements(1, :), measurements(2, :), '+', 'Color', [1 0.5 0], ...
                    'MarkerSize', 10, 'LineWidth', 3, 'DisplayName', 'Measurements');
            end

            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'd', 'Color', 'm', ...
                    'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm', ...
                    'DisplayName', 'True Position');
            end

            legend('Location', 'best');
            hold off;

            % Add main title with proper LaTeX formatting
            % Fix LaTeX typesetting for filter name
            if contains(title_str, 'GNN-HMM') || contains(title_str, 'GNN_HMM')
                title_str = strrep(title_str, 'GNN-HMM', 'GNN\_HMM');
                title_str = strrep(title_str, 'GNN_HMM', 'GNN\_HMM');
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

    end

end
