classdef MC_PF < DA_Filter
    % MC_PF Sequential Importance Resampling Data Association Particle Filter
    %
    % DESCRIPTION:
    %   Implements a hybrid SIR particle filter with Sequential Importance Resampling
    %   for single target data association and tracking.
    %   Supports both standard Gaussian likelihood and precomputed spatial
    %   likelihood lookup tables for radar/sensor fusion applications.
    %
    %   SIR ALGORITHM:
    %   1. Prediction: Propagate particles through dynamics
    %   2. Update: Multiply previous weights by new likelihood and normalize
    %   3. Resample: If ESS < configurable threshold, resample and reset to uniform weights
    %
    % PROPERTIES:
    %   N_p, N_x, N_z        - Filter dimensions (particles, states, measurements)
    %   particles, weights   - Particle filter state [N_x x N_p], [N_p x 1]
    %   F, Q, H             - System model matrices
    %   pointlikelihood_image - Precomputed likelihood lookup table
    %   pointlikelihood_mag   - Precomputed likelihood magnitude lookup table (hybrid likelihood)
    %   PD, PFA               - Detection probability and false alarm probability
    %   debug, validate       - Control flags for debugging and validation
    %
    % METHODS:
    %   MC_PF                - Constructor
    %   timestep              - Process single time step with SIR MC algorithm
    %   prediction          - Particle prediction step
    %   resample            - Bootstrap particle resampling
    %
    % EXAMPLE:
    %   pf = MC_PF(x0, 1000, F, Q, H, pointlikelihood_image, 'Debug', true, 'DynamicPlot', true);
    %   pf.timestep(measurements);
    %   [x_est, P_est] = pf.getGaussianEstimate();
    %
    % See also DA_Filter, PDA_HMM, particleFilter, trackingPF

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
        pointlikelihood_mag % Precomputed likelihood magnitude lookup table (hybrid likelihood) [128^2 x 2]

        % Control Flags (inherited from DA_Filter)
        debug = false % Enable debug output and validation
        validate = false % Enable input/output validation checks
        DynamicPlot = false % Enable real-time visualization during timesteps

        % Validation Parameters
        validation_sigma_bounds = 2 % Number of sigma bounds for measurement gating (default: 2)

        % Detection Model Parameters
        PD = 0.95 % Detection probability (probability of detecting true target)
        PFA = 0.05 % False alarm probability (probability of false measurement)

        % SIR Resampling Parameters
        ESS_threshold_percentage = 0.95 % ESS threshold as percentage of N_p for resampling (default: 95 % - behaves like bootstrap)
        hybrid_resample_fraction = 0.8 % Fraction of particles to resample (remaining are uniform) for robustness (default: 0.8 = 80% resample, 20% uniform)

        % Dynamic Plotting (inherited from DA_Filter)
        dynamic_figure_handle % Figure handle for dynamic plotting

        % Updated Likelihood
        composite_likelihood

        % Visualization Storage (for comprehensive likelihood breakdown)
        current_signal = [] % Current mWidar signal [128 x 128]
        detection_likelihood_field = [] % Detection likelihood field [128^2 x 1]
        magnitude_likelihood_field = [] % Magnitude likelihood field [128^2 x 1]
        combined_likelihood_field = [] % Combined likelihood field [128^2 x 1]
        % current_measurement = [] % Current measurement position [2 x 1] % LEGACY: don't delete yet incase it does something useful
        particle_detection_likelihoods = [] % Individual particle detection likelihoods [N_p x 1]
        particle_magnitude_likelihoods = [] % Individual particle magnitude likelihoods [N_p x 1]

        % Update GIF filename for dynamic plotting (inherited from DA_Filter)
        gif_filename = '' % Filename for saving dynamic plot as GIF (empty = no GIF)
        gif_frame_counter = 0; % Frame counter for GIF creation
    end

    methods

        function obj = MC_PF(x0, N_particles, F, Q, H, pointlikelihood_image, pointlikelihood_mag, varargin) % MC_PF Constructor for SIR Monte Carlo Data Association Particle Filter
            %
            % SYNTAX:
            %   obj = MC_PF(x0, N_particles, F, Q, H)
            %   obj = MC_PF(x0, N_particles, F, Q, H, pointlikelihood_image)
            %   obj = MC_PF(..., 'Debug', true, 'DynamicPlot', true, 'ValidationSigma', 3, 'ESSThreshold', 0.15, 'UniformInit', true)
            %
            % INPUTS:
            %   x0                  - Initial state estimate [N_x x 1]
            %   N_particles         - Number of particles (integer > 0)
            %   F                   - State transition matrix [N_x x N_x]
            %   Q                   - Process noise covariance [N_x x N_x]
            %   H                   - Measurement matrix [N_z x N_x]
            %   pointlikelihood_image - (optional) Precomputed likelihood lookup table [128^2 x 128^2]
            %   pointlikelihood_mag   - (optional) Precomputed likelihood magnitude lookup table (hybrid likelihood) [128^2 x 2]
            %   varargin            - Name-value pairs:
            %                        'Debug', true/false
            %                        'DynamicPlot', true/false
            %                        'ValidationSigma', numeric
            %                        'ESSThreshold', numeric (0-1, default: 0.20)
            %                        'UniformInit', true/false (default: false)
            %                        'HybridResampleFraction', numeric (0-1, default: 0.8)
            %
            % OUTPUTS:
            %   obj - Initialized SIR MC_PF object
            %
            % DESCRIPTION:
            %   Creates and initializes a SIR MC-DA particle filter with particle
            %   distribution around the initial state or uniformly over entire space.
            %   Uses provided likelihood table for hybrid filtering applications.
            %   Implements Sequential Importance Resampling with configurable ESS threshold.
            %
            % NOTE:
            %   Load likelihood table in calling script using:
            %   load('supplemental/precalc_imagegridHMMEmLike.mat', 'pointlikelihood_image');
            %
            %   When 'UniformInit' is true, particles are initialized uniformly over:
            %   Position: x ∈ [-2, 2], y ∈ [0, 4]
            %   Velocity: vx, vy ∈ [-2, 2]
            %   Acceleration: ax, ay ∈ [-2, 2]
            %
            % See also timestep, prediction, resample

            % Validate inputs
            if nargin < 5
                error('MC_PF:InvalidInput', 'Minimum 5 inputs required: {x0, N_particles, F, Q, H}');
            end

            if nargin < 6 || isempty(pointlikelihood_image)
                % Initialize with empty matrix - will be set later if needed
                pointlikelihood_image = [];
            end

            if nargin < 7 || isempty(pointlikelihood_mag)
                % Initialize with empty matrix - will be set later if needed
                pointlikelihood_mag = [];
                obj.composite_likelihood = false;
            else
                obj.composite_likelihood = true;
            end

            % Parse UniformInit flag first (before calling parent parseFilterOptions)
            uniform_init = false; % Default to false for backwards compatibility
            hybrid_resample_frac = 0.8; % Default to 0.8 for hybrid resampling
            filtered_varargin = {};

            % Filter out UniformInit and HybridResampleFraction parameters and collect remaining arguments
            i = 1;

            while i <= length(varargin)

                if i < length(varargin) && strcmpi(varargin{i}, 'UniformInit')
                    uniform_init = varargin{i + 1};
                    i = i + 2; % Skip both parameter name and value
                elseif i < length(varargin) && strcmpi(varargin{i}, 'HybridResampleFraction')
                    hybrid_resample_frac = varargin{i + 1};
                    i = i + 2; % Skip both parameter name and value
                else
                    filtered_varargin{end + 1} = varargin{i};
                    i = i + 1;
                end

            end

            % Parse remaining options using parent class utility
            options = DA_Filter.parseFilterOptions(filtered_varargin{:});
            obj.debug = options.Debug;
            obj.DynamicPlot = options.DynamicPlot;
            obj.validation_sigma_bounds = options.ValidationSigma;

            % Set hybrid resample fraction
            obj.hybrid_resample_fraction = hybrid_resample_frac;

            % Initialize basic properties -- N_x is still state dimension, not augmented
            obj.N_p = N_particles;
            obj.N_x = size(x0, 1);
            obj.N_z = size(H, 1);

            % Initialize particles - either uniform over space or around initial guess
            if uniform_init
                % Uniform initialization over entire space
                if obj.debug
                    fprintf('[UNIFORM INIT] Initializing particles uniformly over entire space\n');
                end

                % Define bounds for uniform initialization
                pos_bounds = [-2, 2; 0.5, 4]; % [x_min, x_max; y_min, y_max] - y >= 0.5
                vel_bounds = [-2, 2; -2, 2]; % [vx_min, vx_max; vy_min, vy_max]
                acc_bounds = [-2, 2; -2, 2]; % [ax_min, ax_max; ay_min, ay_max]

                % Initialize particles uniformly -- NOTE N_x+1 BECAUSE WE'RE TRACKING ASSOCIATIONS
                obj.particles = zeros(obj.N_x + 1, N_particles);

                % Position (x, y) - ensure y >= 0.5
                obj.particles(1, :) = pos_bounds(1, 1) + (pos_bounds(1, 2) - pos_bounds(1, 1)) * rand(1, N_particles);
                obj.particles(2, :) = pos_bounds(2, 1) + (pos_bounds(2, 2) - pos_bounds(2, 1)) * rand(1, N_particles);

                % Velocity (vx, vy) if state dimension >= 4
                if obj.N_x >= 4
                    obj.particles(3, :) = vel_bounds(1, 1) + (vel_bounds(1, 2) - vel_bounds(1, 1)) * rand(1, N_particles);
                    obj.particles(4, :) = vel_bounds(2, 1) + (vel_bounds(2, 2) - vel_bounds(2, 1)) * rand(1, N_particles);
                end

                % Acceleration (ax, ay) if state dimension >= 6
                if obj.N_x >= 6
                    obj.particles(5, :) = acc_bounds(1, 1) + (acc_bounds(1, 2) - acc_bounds(1, 1)) * rand(1, N_particles);
                    obj.particles(6, :) = acc_bounds(2, 1) + (acc_bounds(2, 2) - acc_bounds(2, 1)) * rand(1, N_particles);
                end

                if obj.debug
                    fprintf('[UNIFORM INIT] Particles initialized over bounds:\n');
                    fprintf('  Position: x ∈ [%.1f, %.1f], y ∈ [%.1f, %.1f]\n', pos_bounds(1, :), pos_bounds(2, :));

                    if obj.N_x >= 4
                        fprintf('  Velocity: vx ∈ [%.1f, %.1f], vy ∈ [%.1f, %.1f]\n', vel_bounds(1, :), vel_bounds(2, :));
                    end

                    if obj.N_x >= 6
                        fprintf('  Acceleration: ax ∈ [%.1f, %.1f], ay ∈ [%.1f, %.1f]\n', acc_bounds(1, :), acc_bounds(2, :));
                    end

                end

            else
                % Standard initialization with small spread around initial guess
                if obj.debug
                    fprintf('[STANDARD INIT] Initializing particles around initial state\n');
                end

                init_spread = [0.1, 0.1, 0.25, 0.25, 0.5, 0.5]; % Position, velocity, acceleration spreads
                obj.particles = repmat(x0(:), 1, N_particles);

                % Add Gaussian noise to each state component to provide diversity
                for i = 1:obj.N_x
                    obj.particles(i, :) = obj.particles(i, :) + init_spread(i) * randn(1, N_particles);
                end

                % Ensure all particles have y >= 0.5
                obj.particles(2, obj.particles(2, :) < 0.5) = 0.5;

                % OPTIONAL: Initialize velocity and acceleration to zero mean if ground truth is suspect
                % Uncomment these lines if ground truth initial velocity/acceleration seem wrong:
                % obj.particles(3, :) = 0 + init_spread(3) * randn(1, N_particles); % vx ~ N(0, 0.25²)
                % obj.particles(4, :) = 0 + init_spread(4) * randn(1, N_particles); % vy ~ N(0, 0.25²)
                % obj.particles(5, :) = 0 + init_spread(5) * randn(1, N_particles); % ax ~ N(0, 0.5²)
                % obj.particles(6, :) = 0 + init_spread(6) * randn(1, N_particles); % ay ~ N(0, 0.5²)
            end

            obj.weights = ones(N_particles, 1) / N_particles;

            % Store system matrices
            obj.F = F;
            obj.Q = Q;
            obj.H = H;

            % Store likelihood lookup table
            obj.pointlikelihood_image = pointlikelihood_image;
            obj.pointlikelihood_mag = pointlikelihood_mag;

            % Validate dimensions if likelihood image is provided
            if ~isempty(pointlikelihood_image)
                expected_dim = 128 ^ 2;
                [rows, cols] = size(obj.pointlikelihood_image);

                if obj.debug
                    fprintf('\n=== MC_PF INITIALIZATION ===\n');
                    fprintf('Particles: %d, Augmented States: %d, Measurements: %d\n', N_particles, obj.N_x, obj.N_z);
                    fprintf('ESS Threshold: %.1f%% (%d particles)\n', 100 * obj.ESS_threshold_percentage, round(obj.ESS_threshold_percentage * obj.N_p));
                    fprintf('Hybrid Resample Fraction: %.1f%% (%d resampled, %d uniform)\n', ...
                        100 * obj.hybrid_resample_fraction, ...
                        round(obj.hybrid_resample_fraction * obj.N_p), ...
                        obj.N_p - round(obj.hybrid_resample_fraction * obj.N_p));
                    fprintf('Detection likelihood table: %dx%d (expected: %dx%d)\n', rows, cols, expected_dim, expected_dim);

                    % Validate magnitude likelihood dimensions if provided
                    if ~isempty(pointlikelihood_mag)
                        [mag_rows, mag_cols] = size(obj.pointlikelihood_mag);
                        fprintf('Magnitude likelihood table: %dx%d (expected: %dx2)\n', mag_rows, mag_cols, expected_dim);

                        if mag_rows ~= expected_dim || mag_cols ~= 2
                            warning('PDA_PF:MagDimensionMismatch', ...
                                'Magnitude likelihood dimensions (%dx%d) do not match expected size (%dx2)', ...
                                mag_rows, mag_cols, expected_dim);
                        end

                        fprintf('Composite likelihood enabled: Yes\n');
                    else
                        fprintf('Composite likelihood enabled: No\n');
                    end

                    fprintf('============================\n\n');
                end

                if rows ~= expected_dim || cols ~= expected_dim
                    warning('PDA_PF:DimensionMismatch', ...
                        'Likelihood model dimensions (%dx%d) do not match expected grid size (%dx%d)', ...
                        rows, cols, expected_dim, expected_dim);
                end

            end

            % Initialize dynamic plotting if enabled with larger size for comprehensive 2x5 likelihood visualization
            obj.initializeDynamicPlot('MC-PF Dynamic Tracking', [100, 100, 1800, 800]);

            % Initialize GIF filename to empty (no GIF by default)
            obj.gif_filename = '';
            obj.gif_frame_counter = 0;

        end

        %% ========== TIMESTEP ==========
        function timestep(obj, z, varargin)
            % TIMESTEP Implements the SIR MC-PF algorithm for a single timestep
            %
            % SYNTAX:
            %   obj.timestep(z)
            %   obj.timestep(z, true_state)
            %
            % INPUTS:
            %   z          - Measurements: either [2 x N_measurements] matrix or struct with .det and .mag fields
            %   true_state - (optional) True state for visualization
            %
            % DESCRIPTION:
            %   Executes one timestep of the SIR MC-PF algorithm including:
            %   1. Prediction Step - Propagate particles through dynamics model
            %   2. Validation Step - Gate measurements using particle estimate
            %   3. Data Association - Sample measurement association for each particle
            %   4. Measurement Update - Update particle weights using sampled associations
            %   5. Resampling - Resample if ESS falls below configurable threshold
            %   6. Visualization - Update dynamic plot if enabled
            %
            % TODO: Implement MC-DA augmented state [x_state, x_association]
            % TODO: Sample association hypothesis for each particle from discrete distribution
            % TODO: Update weights based on individual particle's sampled association
            %
            % MEASUREMENT FORMAT:
            %   - Legacy: z as [2 x N_measurements] matrix
            %   - New: z as struct with z.det [2 x N_measurements] and z.mag [128 x 128]
            %
            % ALGORITHM NOTES:
            %   - Uses Sequential Importance Resampling (SIR) paradigm
            %   - Resampling occurs at END of timestep if ESS < threshold
            %   - Gating uses configurable sigma bounds based on current state estimate
            %   - MC-DA: Each particle samples its own measurement association
            %   - Supports optional real-time visualization
            %
            % See also prediction, measurement_update, resample

            % Handle both legacy measurement format and new struct format
            if isstruct(z)
                % New format: measurement struct with .det and .mag fields
                z_det = z.det;
                z_mag = z.mag;
                measurement_struct_format = true;

                % Store signal data for comprehensive visualization
                if obj.composite_likelihood && ~isempty(z_mag)
                    obj.current_signal = z_mag;

                    if obj.debug
                        fprintf('Stored signal data for comprehensive visualization\n');
                    end

                end

            else
                % Legacy format: direct measurement matrix
                z_det = z;
                z_mag = [];
                measurement_struct_format = false;
            end

            if obj.debug
                fprintf('\n=== SIR PDA-PF TIMESTEP START ===\n');

                if measurement_struct_format
                    fprintf('Input: measurement struct format with %d detections and %dx%d signal\n', ...
                        size(z_det, 2), size(z_mag, 1), size(z_mag, 2));
                else
                    fprintf('Input: legacy format with %d measurements\n', size(z_det, 2));
                end

                if ~isempty(z_det)

                    for i = 1:size(z_det, 2)
                        fprintf('  Meas %d: [%.3f, %.3f]\n', i, z_det(1, i), z_det(2, i));
                    end

                end

                fprintf('------------------------------\n');
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
            % Store original measurements for visualization (use detections)
            z_original = z_det;

            % Apply gating to get valid measurements
            [z_gated, has_valid_meas] = obj.Validation(z_det);
            
            % Find indices of gated measurements in original measurement set
            % These are the valid association indices for MC-DA sampling
            valid_indices = [];
            if has_valid_meas
                for i = 1:size(z_gated, 2)
                    idx = find(all(bsxfun(@eq, z_det, z_gated(:, i)), 1), 1);
                    if ~isempty(idx)
                        valid_indices = [valid_indices, idx];
                    end
                end
            end
            
            % Add clutter hypothesis index (N_measurements + 1)
            clutter_index = size(z_det, 2) + 1;
            valid_indices = [valid_indices, clutter_index];
            
            if obj.debug
                if has_valid_meas
                    fprintf('[GATING] %d/%d measurements passed gating\n', length(valid_indices)-1, size(z_det, 2));
                    fprintf('[GATING] Valid association indices: ');
                    fprintf('%d ', valid_indices);
                    fprintf('\n');
                else
                    fprintf('[GATING] No measurements passed gating - only clutter hypothesis available\n');
                    fprintf('[GATING] Valid association indices: %d (clutter only)\n', clutter_index);
                end
            end

            % Step 3: Data Association Step -- Sample associations for each particle
            % MC-DA: Each particle samples an association hypothesis from valid indices only
            % - Association indices: valid_indices (gated measurements + clutter)
            % - Particles can only associate with measurements that passed gating or clutter
            % THIS IS UNIFORM RANDOM SAMPLING PLACEHOLDER - REPLACE WITH PROPER MC-DA SAMPLING
            if isempty(valid_indices)
                % No valid measurements - all particles associate with clutter
                obj.particles(end, :) = clutter_index;
            else
                % Sample uniformly from valid indices (gated measurements + clutter)
                sample_idx = randi(length(valid_indices), 1, obj.N_p);
                obj.particles(end, :) = valid_indices(sample_idx);
            end

            % Step 4: Measurement Update Step (SIR: multiply by previous weights)
            % Pass ALL measurements (not just gated ones) so particles can compute likelihoods
            % for their sampled associations, but only gated associations are in particles(end,:)
            obj.measurement_update(z_det, z_mag);

            % validate measurements by setting weights of out-of-bounds particles to zero
            % Check bounds and enforce x,y in bounds 
            obj.weights(obj.particles(1, :) < -2) = eps; % Enforce x >= -2
            obj.weights(obj.particles(2, :) < 0.5) = eps; % Enforce y >= 0.5
            obj.weights(obj.particles(1, :) > 2) = eps; % Enforce x <= 2
            obj.weights(obj.particles(2, :) > 4) = eps; % Enforce y <= 4



            % Step 5: SIR Resampling - Check ESS and resample if needed
            ESS = 1 / sum(obj.weights .^ 2);
            ESS_threshold = obj.ESS_threshold_percentage * obj.N_p;

            if ESS < ESS_threshold

                if obj.debug
                    fprintf('\n');
                    fprintf('╔════════════════════════════════════════╗\n');
                    fprintf('║          SIR RESAMPLING TRIGGER        ║\n');
                    fprintf('║                                        ║\n');
                    fprintf('║  ESS: %6.1f < Threshold: %6.1f      ║\n', ESS, ESS_threshold);
                    fprintf('║  ESS Percentage: %5.1f%% < %.0f%%        ║\n', 100 * ESS / obj.N_p, 100 * obj.ESS_threshold_percentage);
                    fprintf('║                                        ║\n');
                    fprintf('║        RESAMPLING PARTICLES...         ║\n');
                    fprintf('╚════════════════════════════════════════╝\n');
                end

                obj.resample();

                if obj.debug
                    fprintf('╔════════════════════════════════════════╗\n');
                    fprintf('║         RESAMPLING COMPLETE            ║\n');
                    fprintf('║     Weights reset to uniform           ║\n');
                    fprintf('╚════════════════════════════════════════╝\n');
                    fprintf('\n');
                end

            else

                if obj.debug
                    fprintf('[SIR] ESS: %.1f (%.1f%%) >= Threshold: %.1f (%.0f%%) - No resampling needed\n', ...
                        ESS, 100 * ESS / obj.N_p, ESS_threshold, 100 * obj.ESS_threshold_percentage);
                end

            end

            if obj.debug
                [x_est, P_est] = obj.getGaussianEstimate();
                fprintf('\nOutput: State estimate [%.4f, %.4f] m\n', x_est(1), x_est(2));
                fprintf('        Covariance trace: %.6f\n', trace(P_est));
                fprintf('=== SIR MC-PF TIMESTEP END ===\n\n');
            end

            % Update dynamic plot if enabled
            if obj.DynamicPlot

                if nargin > 2
                    true_state = varargin{1};
                    obj.updateDynamicPlot(z_det, true_state, z_original);
                else
                    obj.updateDynamicPlot(z_det, [], z_original);
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

            % Only propagate the first N_x rows (kinematic state), leave association untouched
            obj.particles(1:obj.N_x, :) = obj.F * obj.particles(1:obj.N_x, :);

            % Add process noise to kinematic state only
            process_noise = mvnrnd(zeros(1, obj.N_x), obj.Q, obj.N_p)';
            obj.particles(1:obj.N_x, :) = obj.particles(1:obj.N_x, :) + process_noise;

            % Ensure all particles maintain y >= 0.5 after prediction (kinematic state only)
            obj.particles(2, obj.particles(2, :) < 0.5) = 0.5;

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
            %   Applies gating based on the probability of detection -- if the probability
            % of a detection is less than that of a false alarm (as defined by hyperparameters
            % PD and PFA), the measurements is rejected. Otherwise, it is accepted and
            % passed to the measurement update step.
            %
            % ALGORITHM:
            %   1. Calculate normalization Constants of measurements (p(z))
            %   2. Compare each constant to the false alarm rate (last hypothesis)
            %   3. Accept measurements with p(z) > PFA
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

            % NEW GATING SCHEMA
            normalization_constants = obj.computeNormalizationConstants(z);

            % Create a binary mask for validation measurements -- threshold is the false
            %   alarm rate (last hypothesis)
            threshold = normalization_constants(end) * obj.N_p;
            z_valid = z(:, normalization_constants(1:end - 1) > threshold);
            has_valid_meas = ~isempty(z_valid);

            if obj.debug
                fprintf('[GATING] Input: %d measurements, Output: %d valid measurements\n', ...
                    size(z, 2), size(z_valid, 2));

            end

        end

        %% ========== MEASUREMENT UPDATE (MC-DA) ==========
        function measurement_update(obj, z, z_mag)
            % MEASUREMENT_UPDATE Update particle weights based on measurement likelihood (SIR MC-DA)
            %
            % SYNTAX:
            %   obj.measurement_update(z)
            %   obj.measurement_update(z, z_mag)
            %
            % INPUTS:
            %   z    - Current measurements [N_z x N_measurements] (after gating)
            %   z_mag - (optional) mWidar signal [128 x 128] for composite likelihood
            %
            % DESCRIPTION:
            %   TODO: FIXME - Convert from PDA to MC-DA weight update
            %   MC-DA: w_new = w_old * likelihood(z_sampled | x_particle)
            %   Each particle uses ONLY its sampled association hypothesis, not a mixture.
            %   For particle i with sampled association a_i:
            %     - If a_i <= N_measurements: weight by likelihood(z_a_i | x_i)
            %     - If a_i = N_measurements + 1: weight by clutter hypothesis
            %
            %   CURRENT: Implements SIR PDA weight update: w_new = w_old * likelihood_total
            %   Sums likelihood contributions from all measurements plus clutter hypothesis.
            %   Handles missed detection case when z is empty.
            %   Uses SIR paradigm: multiply previous weights by new likelihood and normalize.
            %   If z_mag is provided and magnitude likelihood is available, uses composite likelihood.
            %
            % MODIFIES:
            %   obj.weights - Updated and normalized particle weights [N_p x 1]
            %
            % See also timestep, computeWeightsForMeasurement, computeNormalizationConstants

            % Handle optional z_mag parameter for backwards compatibility
            if nargin < 3
                z_mag = [];
            end

            % Store previous weights for SIR update
            previous_weights = obj.weights;

            % Handle missed detection case (no valid measurements after gating)
            if isempty(z)

                if obj.debug
                    fprintf('[MEASUREMENT UPDATE] No measurements - missed detection case\n');
                end

                % Keep weights unchanged for missed detection
                return;
            end

            % Check if likelihood image is available
            if isempty(obj.pointlikelihood_image)
                error('PDA_PF:NoLikelihoodData', ...
                'Likelihood lookup table not provided. Load and pass to constructor.');
            end

            % Debug: Print measurement info
            if obj.debug
                fprintf('[MEASUREMENT UPDATE] Processing %d measurements:\n', size(z, 2));
            end

            % Step 1: Compute magnitude likelihood for all particles (if available)
            magnitude_likelihood = ones(obj.N_p, 1);

            if obj.composite_likelihood && ~isempty(obj.pointlikelihood_mag) && ~isempty(z_mag)
                % Define spatial grid parameters
                npx = 128;
                xgrid = linspace(-2, 2, npx);
                ygrid = linspace(0, 4, npx);

                % Vectorized grid point finding for all particles
                [~, px_indices] = min(abs(obj.particles(1, :)' - xgrid), [], 2); % [N_p x 1]
                [~, py_indices] = min(abs(obj.particles(2, :)' - ygrid), [], 2); % [N_p x 1]

                % Enforce boundary constraints
                px_indices = max(1, min(npx, px_indices));
                py_indices = max(1, min(npx, py_indices));

                % Convert to linear indices for all particles
                particle_linear_indices = sub2ind([npx, npx], py_indices, px_indices); % [N_p x 1]

                % Magnitude likelihood lookup and calculation (computed ONCE per timestep)
                likelihood_mag_values = obj.pointlikelihood_mag(particle_linear_indices, :); % [N_p x 2]
                % Flip z_mag for correct orientation
                % z_mag = flipud(z_mag);
                magnitude_likelihood = normpdf(z_mag(particle_linear_indices), likelihood_mag_values(:, 1), likelihood_mag_values(:, 2));
                magnitude_likelihood = magnitude_likelihood(:) / sum(magnitude_likelihood); % Ensure column vector

                if obj.debug
                    fprintf('[MAGNITUDE LIKELIHOOD] Computed once for timestep: min/max/mean = %.2e / %.2e / %.2e\n', ...
                        min(magnitude_likelihood), max(magnitude_likelihood), mean(magnitude_likelihood));

                    fprintf("DEBUGGING -- THIS IS THE RANGE OF THE MAG LIKELIHOODS\n")

                end

                % Store for visualization
                obj.particle_magnitude_likelihoods = magnitude_likelihood;
                obj.current_signal = z_mag;
            else
                % No composite likelihood available - use constant magnitude likelihood (no effect)

                obj.particle_magnitude_likelihoods = magnitude_likelihood; % No magnitude data

                if obj.debug
                    fprintf('[MAGNITUDE LIKELIHOOD] Using constant (no composite likelihood available)\n');
                end

            end

            % Step 2: Compute detection likelihood for each measurement and update weights
            detection_likelihood = zeros(obj.N_p, 1); % Column vector for all particles

            % Initialize total likelihood with clutter hypothesis and normalization constants
            clutter_likelihood = (1 - obj.PD) * obj.PFA / obj.N_p;
            normalization_constants = obj.computeNormalizationConstants(z);
            normalization_constants = normalization_constants / sum(normalization_constants); % Normalize

            % Clutter hypothesis index
            clutter_idx = size(z, 2) + 1;

            % Iterate through all possible measurement associations (including clutter)
            for measurement_idx = 1:clutter_idx
                % Find particles associated with this measurement
                particle_mask = (obj.particles(end, :) == measurement_idx); % Logical mask [1 x N_p]

                if ~any(particle_mask)
                    continue; % Skip if no particles associated with this measurement
                end

                if measurement_idx == clutter_idx
                    % Clutter hypothesis - assign clutter likelihood
                    detection_likelihood(particle_mask) = clutter_likelihood * normalization_constants(end);
                    continue;
                end

                % True detection - compute weights for ALL particles, then select associated ones
                all_weights = obj.computeWeightsForMeasurement(z(:, measurement_idx)); % [N_p x 1]
                detection_likelihood(particle_mask) = all_weights(particle_mask);% normalization_constants(measurement_idx);

            end

            obj.particle_detection_likelihoods = detection_likelihood;

            new_weights = previous_weights .* detection_likelihood .* magnitude_likelihood + eps;
            obj.weights = new_weights / sum(new_weights);

            % Debug: Print final weight stats
            if obj.debug
                fprintf('[SIR MEASUREMENT UPDATE] Complete. Weights: min=%.6f, max=%.6f, sum=%.6f\n', ...
                    min(obj.weights), max(obj.weights), sum(obj.weights));
            end

        end

        function normalization_constants = computeNormalizationConstants(obj, z, z_mag)
            % COMPUTENORMALIZATIONCONSTANTS Compute normalization constants for each
            % measurement hypothesis (including clutter)
            %
            % SYNTAX:
            %   normalization_constants = obj.computeNormalizationConstants(z)
            %   normalization_constants = obj.computeNormalizationConstants(z, z_mag)
            %
            % INPUTS:
            %   z - Current measurements [N_z x N_measurements]
            %   z_mag - (optional) mWidar signal [128 x 128] for composite likelihood
            %
            % OUTPUTS:
            %   normalization_constants - Vector of normalization constants [N_measurements + 1 x 1]
            %                           normalization_constants(i) = sum_p w_p^(i) for measurement i
            %                           normalization_constants(end) = clutter hypothesis constant
            %
            % DESCRIPTION:
            %   TODO: FIXME - MC-DA: These normalization constants are used for sampling associations
            %   In MC-DA, normalization constants define the discrete distribution over associations:
            %   P(association = i | x_particle) = normalization_constants(i) / sum(normalization_constants)
            %   Each particle samples from this distribution to get its association hypothesis.
            %
            %   CURRENT: Computes normalization constants for GNN-style data association by calculating
            %   the sum of particle weights for each measurement hypothesis separately.
            %   Uses proper detection model and marginalizes out particle dimension.
            %   This is also used for gating decision in the Validation step, as well
            %   as for normalizing weights in the measurement update step.
            %
            % See also computeWeightsForMeasurement, measurement_update

            % Handle optional z_mag parameter for backwards compatibility
            if nargin < 3
                z_mag = [];
            end

            % Check if likelihood image is available
            if isempty(obj.pointlikelihood_image)
                error('PDA_PF:NoLikelihoodData', ...
                'Likelihood lookup table not provided. Load and pass to constructor.');
            end

            N_measurements = size(z, 2);
            normalization_constants = zeros(N_measurements + 1, 1); % +1 for clutter

            % Compute normalization constant for each measurement individually
            for i = 1:N_measurements
                % Get weights for this measurement using helper function

                measurement_weights = obj.computeWeightsForMeasurement(z(:, i));

                fprintf("Size of measuremnt_weights");
                disp(size(measurement_weights))

                % Marginalize out particles (sum across all particles)
                normalization_constants(i) = sum(measurement_weights);

                if obj.debug
                    fprintf('[PDA] Measurement %d normalization constant: %.6f\n', ...
                        i, normalization_constants(i));
                end

            end

            % Add clutter hypothesis normalization constant
            % Clutter hypothesis: (1-PD) * PFA / N_p
            % FIXME: Read in to this and why 
            clutter_constant = (1 - obj.PD) * obj.PFA / obj.N_p;
            normalization_constants(end) = clutter_constant;

            if obj.debug
                fprintf('[GNN] Clutter hypothesis constant: %.6f\n', clutter_constant);
                fprintf('[GNN] Computed %d normalization constants (%d measurements + clutter)\n', ...
                    length(normalization_constants), N_measurements);
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
                fprintf('    Likelihood Computation: Grid indices: x=%d, y=%d, linear=%d\n', ...
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
                error('PDA_PF:MeasurementOutOfBounds', ...
                    'Measurement linear index %d out of bounds [1, %d]. Measurement may be outside spatial grid.', ...
                    meas_linear_idx, size(obj.pointlikelihood_image, 1));
            end

            if any(particle_linear_indices > size(obj.pointlikelihood_image, 2)) || any(particle_linear_indices < 1)
                error('PDA_PF:ParticlesOutOfBounds', ...
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

        function weights = computeWeightsForMeasurement(obj, z)
            % COMPUTEWEIGHTSFORMEASUREMENT Compute particle weights for a single measurement (DETECTION ONLY)
            %
            % SYNTAX:
            %   weights = obj.computeWeightsForMeasurement(z)
            %
            % INPUTS:
            %   z - Single measurement [N_z x 1]
            %
            % OUTPUTS:
            %   weights - Particle weights for this measurement [N_p x 1] (DETECTION ONLY)
            %
            % DESCRIPTION:
            %   Computes detection-only particle weights for a measurement.
            %   Magnitude likelihood is handled separately in the measurement update function.
            %
            % See also measurement_update, computeIndividualWeightUpdates, likelihoodLookup

            % Always use detection-only likelihood (magnitude handled separately)
            if obj.debug
                fprintf('    Computing detection-only weights\n');
            end

            % LEGACY: don't delete yet incase it does something useful
            % Store current measurement
            % obj.current_measurement = z;

            % Get detection-only likelihood for this measurement
            likelihood = obj.likelihoodLookup(z);

            % Store individual particle likelihoods for visualization
            obj.particle_detection_likelihoods = likelihood;

            % Apply Gaussian weighting
            sf = 0.15;
            dx = obj.particles(1, :) - z(1); % [1 x N_p]
            dy = obj.particles(2, :) - z(2); % [1 x N_p]
            dist_sq = dx .^ 2 + dy .^ 2; % [1 x N_p]
            gauss_weights = exp(-dist_sq / (2 * sf ^ 2)); % [1 x N_p]

            % Combine likelihood and Gaussian weights with detection probability scaling
            weights = (obj.PD / obj.N_p) * (likelihood .* gauss_weights') + eps;

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
            %   to produce a Gaussian approximation of the posterior distribution.

            % Compute weighted mean (only kinematic state, not association index)
            state_est = obj.particles(1:obj.N_x, :) * obj.weights;

            % Compute weighted covariance using more robust formula
            deviations = obj.particles(1:obj.N_x, :) - state_est; % [N_x x N_p]

            % Vectorized weighted covariance calculation
            weighted_deviations = deviations .* sqrt(obj.weights'); % [N_x x N_p]
            state_est_covariance = (weighted_deviations * weighted_deviations') / sum(obj.weights);

            % Ensure covariance is real and symmetric (numerical stability)
            state_est_covariance = real(state_est_covariance);
            state_est_covariance = 0.5 * (state_est_covariance + state_est_covariance');

            % Add small regularization to ensure positive definiteness
            state_est_covariance = state_est_covariance +1e-8 * eye(size(state_est_covariance, 1));
        end

        function [association_dist] = getAssociationDistribution(obj)
            % GETASSOCIATIONDISTRIBUTION Compute empirical distribution of particle associations
            %
            % SYNTAX:
            %   association_dist = obj.getAssociationDistribution()
            %
            % OUTPUTS:
            %   association_dist - Vector of association probabilities [N_measurements + 1 x 1]
            %                     association_dist(i) = P(association = i) for measurement i
            %                     association_dist(end) = P(clutter/missed detection)
            %
            % DESCRIPTION:
            %   Computes the empirical distribution of measurement associations
            %   across all particles based on their weights. This provides insight
            %   into which measurements are most likely associated with the target.
            %   Clutter is treated as just another hypothesis (index N_measurements + 1).

            % Get max association index from particles
            max_assoc_idx = max(obj.particles(end, :));
            
            % Initialize distribution: one bin for each association hypothesis
            % Indices 1 to N_measurements are measurements, last index is clutter
            association_dist = zeros(max_assoc_idx, 1);

            % Sum weights for each association hypothesis (measurements AND clutter)
            for i = 1:max_assoc_idx
                association_dist(i) = sum(obj.weights(obj.particles(end, :) == i));
            end

            % Normalize to form a valid probability distribution
            association_dist = association_dist / sum(association_dist);
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
            %   Updates the detection model parameters used in MC-DA data association.
            %   These parameters affect the association probabilities and clutter hypothesis
            %   calculations in the normalization constants computation.
            %
            % See also computeNormalizationConstants, measurement_update

            % Validate inputs
            if PD <= 0 || PD > 1
                error('MC_PF:InvalidPD', 'Detection probability PD must be in range (0, 1]');
            end

            if PFA < 0 || PFA >= 1
                error('MC_PF:InvalidPFA', 'False alarm probability PFA must be in range [0, 1)');
            end

            obj.PD = PD;
            obj.PFA = PFA;

            if obj.debug
                fprintf('[DETECTION MODEL] Updated: PD=%.3f, PFA=%.3f\n', obj.PD, obj.PFA);
            end

        end

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
            % See also PDA_PF, prediction, resample

            if obj.debug
                fprintf('\n=== LOADING LIKELIHOOD DATA ===\n');
                fprintf('Loading from: %s\n', likelihood_file_path);
            end

            try
                % Check if file exists
                if ~exist(likelihood_file_path, 'file')
                    error('PDA_PF:FileNotFound', ...
                        'Likelihood file not found: %s', likelihood_file_path);
                end

                % Load the likelihood data
                likelihood_data = load(likelihood_file_path, 'pointlikelihood_image');

                % Validate that the expected variable exists
                if ~isfield(likelihood_data, 'pointlikelihood_image')
                    error('PDA_PF:InvalidData', ...
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
                    warning('PDA_PF:DimensionMismatch', ...
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

                error('PDA_PF:LoadError', ...
                    'Failed to load likelihood data: %s', ME.message);
            end

        end

        %% ========== PARTICLE RESAMPLING ==========
        function resample(obj)
            % TODO: FIXME - MC-DA: Resample both kinematic state AND association indices
            % When resampling particles, must preserve association hypothesis for each particle
            % Augmented state [x_state, x_association] should be resampled together
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
            % See also PDA_PF, prediction, timestep

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

            

            % --- Hybrid resampling: configurable fraction resampled, remainder uniform ---
            % ROBUSTNESS FEATURE: When shit hits the fan (particle deprivation, weight collapse),
            % this prevents complete loss of diversity by injecting uniform particles.
            % Benefits:
            %   - Prevents filter divergence when all particles collapse to single mode
            %   - Maintains exploration capability even with frequent resampling
            %   - Recovers from tracking loss by reintroducing spatial diversity
            %   - Default 80/20 split balances exploitation vs exploration
            % 
            % Configurable via constructor: MC_PF(..., 'HybridResampleFraction', 0.9)
            % Set to 1.0 for pure bootstrap resampling (no robustness injection)
            % Set to 0.5 for aggressive exploration (50% uniform particles)

            % Use hybrid resampling fraction as configured
            N_resample = round(obj.hybrid_resample_fraction * obj.N_p);
            N_uniform = obj.N_p - N_resample;

            if obj.debug
                fprintf('[HYBRID RESAMPLE] Resampling %d particles (%.1f%%), injecting %d uniform (%.1f%%)\n', ...
                    N_resample, 100 * obj.hybrid_resample_fraction, ...
                    N_uniform, 100 * (1 - obj.hybrid_resample_fraction));
            end

            % 1. Resample configured fraction of particles from weighted distribution
            sampcdf = cumsum(obj.weights);
            urands = rand(1, N_resample);
            indsampsout = zeros(1, N_resample);

            for i = 1:N_resample
                indsampsout(i) = find(sampcdf >= urands(i), 1, 'first');
                if isempty(indsampsout(i))
                    indsampsout(i) = obj.N_p;
                end
            end

            % Resampled particles
            particles_resampled = obj.particles(:, indsampsout);

            % 2. Uniformly initialize remaining particles over the search area (robustness injection)
            % Define bounds for uniform initialization
            pos_bounds = [-2, 2; 0.5, 4]; % [x_min, x_max; y_min, y_max]
            vel_bounds = [-2, 2; -2, 2];
            acc_bounds = [-2, 2; -2, 2];

            N_x = obj.N_x;
            particles_uniform = zeros(N_x + 1, N_uniform);

            % Position (x, y)
            particles_uniform(1, :) = pos_bounds(1, 1) + (pos_bounds(1, 2) - pos_bounds(1, 1)) * rand(1, N_uniform);
            particles_uniform(2, :) = pos_bounds(2, 1) + (pos_bounds(2, 2) - pos_bounds(2, 1)) * rand(1, N_uniform);

            % Velocity (vx, vy) if state dimension >= 4
            if N_x >= 4
                particles_uniform(3, :) = vel_bounds(1, 1) + (vel_bounds(1, 2) - vel_bounds(1, 1)) * rand(1, N_uniform);
                particles_uniform(4, :) = vel_bounds(2, 1) + (vel_bounds(2, 2) - vel_bounds(2, 1)) * rand(1, N_uniform);
            end

            % Acceleration (ax, ay) if state dimension >= 6
            if N_x >= 6
                particles_uniform(5, :) = acc_bounds(1, 1) + (acc_bounds(1, 2) - acc_bounds(1, 1)) * rand(1, N_uniform);
                particles_uniform(6, :) = acc_bounds(2, 1) + (acc_bounds(2, 2) - acc_bounds(2, 1)) * rand(1, N_uniform);
            end

            % Association index: randomly assign (for uniform particles)
            % Use 1 (first measurement) or N_measurements+1 (clutter), or random if you prefer
            particles_uniform(end, :) = 1;

            % Combine both sets
            obj.particles = [particles_resampled, particles_uniform];

            % Ensure all particles have y >= 0.5 after resampling
            obj.particles(2, obj.particles(2, :) < 0.5) = 0.5;

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
        function visualize(obj, figure_handle, title_str, measurements, true_state, all_measurements)
            % VISUALIZE Plot current particle distribution and state estimates
            %
            % SYNTAX:
            %   obj.visualize()
            %   obj.visualize(figure_handle)
            %   obj.visualize(figure_handle, title_str)
            %   obj.visualize(figure_handle, title_str, measurements)
            %   obj.visualize(figure_handle, title_str, measurements, true_state)
            %   obj.visualize(figure_handle, title_str, measurements, true_state, all_measurements)
            %
            % INPUTS:
            %   figure_handle    - (optional) Figure handle to plot in
            %   title_str        - (optional) Title string for plot
            %   measurements     - (optional) Used measurements [N_z x N_meas] (after gating)
            %   true_state       - (optional) True state for comparison [N_x x 1]
            %   all_measurements - (optional) All measurements [N_z x N_meas] (before gating)
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
                title_str = 'PDA-PF Particle Distribution';
            end

            if nargin < 4
                measurements = [];
            end

            if nargin < 5
                true_state = [];
            end

            if nargin < 6
                all_measurements = [];
            end

            % Get current state estimate
            [mean_state, state_cov] = obj.getGaussianEstimate();

            % Sort particles by weight (ascending) so highest weights plot on top
            [weights_sorted, sort_idx] = sort(obj.weights, 'ascend');
            particles_sorted = obj.particles(:, sort_idx);

            % Compute effective sample size for main title
            eff_sample_size = 1 / sum(obj.weights .^ 2);

            % Determine colorbar bounds based on weight distribution
            uniform_threshold = 1e-6; % Threshold for considering weights uniform
            weight_range = max(obj.weights) - min(obj.weights);
            uniform_weight = 1 / obj.N_p;

            if weight_range < uniform_threshold
                % Weights are uniform - set bounds to [0, 2/N_p] for better visibility
                colorbar_min = 0;
                colorbar_max = 2 * uniform_weight;
            else
                % Weights are not uniform - set bounds to [0, max(weight)]
                colorbar_min = 0;
                colorbar_max = max(obj.weights);
            end

            % Define spatial bounds (matching test_hybrid_PF)
            Xbounds = [-2, 2];
            Ybounds = [0, 4];

            % Subplot 1: Position with full scene view
            ax1 = subplot(1, 3, 1);
            set(ax1, 'Position', [0.08, 0.15, 0.25, 0.7]); % Wider subplots
            cla;

            % Scatter plot of particle positions colored by weights (sorted for proper layering)
            h_scatter = scatter(particles_sorted(1, :), particles_sorted(2, :), 20, weights_sorted, ...
                'filled', 'MarkerFaceAlpha', 0.6);
            caxis([colorbar_min, colorbar_max]); % Set consistent color limits
            cb1 = colorbar;
            set(cb1, 'TickLabelFormat', '%.4f');
            hold on

            % Plot particle filter estimate
            plot(mean_state(1), mean_state(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);

            % Plot covariance ellipses (1σ and validation bounds)
            pos_cov = state_cov(1:2, 1:2); % Position covariance only

            % 1σ covariance ellipse (solid black)
            ellipse_1sigma = obj.computeCovarianceEllipse(mean_state(1:2), pos_cov, 1);
            plot(ellipse_1sigma(1, :), ellipse_1sigma(2, :), 'k-', 'LineWidth', 2);

            % Validation sigma covariance ellipse (dotted black)
            ellipse_validation = obj.computeCovarianceEllipse(mean_state(1:2), pos_cov, obj.validation_sigma_bounds);
            plot(ellipse_validation(1, :), ellipse_validation(2, :), 'k:', 'LineWidth', 2);

            % Plot true state if provided
            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'd', 'Color', 'm', ...
                    'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm');
            end

            % Plot all measurements first (orange +) if provided
            if ~isempty(all_measurements)
                plot(all_measurements(1, :), all_measurements(2, :), '+', 'Color', [1 0.5 0], ...
                    'MarkerSize', 8, 'LineWidth', 2);
            end

            % Plot used measurements (red +) if provided - these will appear on top
            if ~isempty(measurements)
                plot(measurements(1, :), measurements(2, :), '+', 'Color', 'r', ...
                    'MarkerSize', 8, 'LineWidth', 2);
            end

            title('Position', 'Interpreter', 'latex');
            xlabel('X (m)'), ylabel('Y (m)');
            xlim(Xbounds), ylim(Ybounds);
            axis square;

            % Subplot 2: Velocity estimates (if state dimension >= 4)
            if obj.N_x >= 4
                ax2 = subplot(1, 3, 2);
                set(ax2, 'Position', [0.36, 0.15, 0.25, 0.7]); % Wider subplots
                cla;

                scatter(particles_sorted(3, :), particles_sorted(4, :), 20, weights_sorted, ...
                    'filled', 'MarkerFaceAlpha', 0.6);
                caxis([colorbar_min, colorbar_max]); % Set consistent color limits
                cb2 = colorbar;
                set(cb2, 'TickLabelFormat', '%.4f');
                hold on
                plot(mean_state(3), mean_state(4), 'ro', 'MarkerSize', 10, 'LineWidth', 3);

                % Plot velocity covariance ellipses
                vel_cov = state_cov(3:4, 3:4); % Velocity covariance
                ellipse_1sigma_vel = obj.computeCovarianceEllipse(mean_state(3:4), vel_cov, 1);
                plot(ellipse_1sigma_vel(1, :), ellipse_1sigma_vel(2, :), 'k-', 'LineWidth', 2);

                ellipse_validation_vel = obj.computeCovarianceEllipse(mean_state(3:4), vel_cov, obj.validation_sigma_bounds);
                plot(ellipse_validation_vel(1, :), ellipse_validation_vel(2, :), 'k:', 'LineWidth', 2);

                if ~isempty(true_state) && length(true_state) >= 4
                    plot(true_state(3), true_state(4), 'd', 'Color', 'm', ...
                        'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm');
                end

                title('Velocity', 'Interpreter', 'latex');
                xlabel('V_x (m/s)'), ylabel('V_y (m/s)');
                axis square;
            end

            % Subplot 3: Acceleration estimates (if state dimension >= 6)
            if obj.N_x >= 6
                ax3 = subplot(1, 3, 3);
                set(ax3, 'Position', [0.64, 0.15, 0.25, 0.7]); % Wider subplots
                cla;

                scatter(particles_sorted(5, :), particles_sorted(6, :), 20, weights_sorted, ...
                    'filled', 'MarkerFaceAlpha', 0.6);
                caxis([colorbar_min, colorbar_max]); % Set consistent color limits
                cb3 = colorbar;
                set(cb3, 'TickLabelFormat', '%.4f');
                hold on
                plot(mean_state(5), mean_state(6), 'ro', 'MarkerSize', 10, 'LineWidth', 3);

                % Plot acceleration covariance ellipses
                acc_cov = state_cov(5:6, 5:6); % Acceleration covariance
                ellipse_1sigma_acc = obj.computeCovarianceEllipse(mean_state(5:6), acc_cov, 1);
                plot(ellipse_1sigma_acc(1, :), ellipse_1sigma_acc(2, :), 'k-', 'LineWidth', 2);

                ellipse_validation_acc = obj.computeCovarianceEllipse(mean_state(5:6), acc_cov, obj.validation_sigma_bounds);
                plot(ellipse_validation_acc(1, :), ellipse_validation_acc(2, :), 'k:', 'LineWidth', 2);

                if ~isempty(true_state) && length(true_state) >= 6
                    plot(true_state(5), true_state(6), 'd', 'Color', 'm', ...
                        'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm');
                end

                title('Acceleration', 'Interpreter', 'latex');
                xlabel('A_x (m/s^2)'), ylabel('A_y (m/s^2)');
                axis square;
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

            % sgtitle(main_title, 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');

            % Add colorbar spanning height on the right
            if obj.N_x >= 4 % Only add colorbar if we have multiple subplots
                cb = colorbar('Position', [0.92, 0.15, 0.02, 0.7]);
                cb.Label.String = 'Particle Weight';
                cb.Label.Interpreter = 'latex';
                set(cb, 'TickLabelFormat', '%.4f');
                % Set colorbar limits based on weight distribution
                caxis([colorbar_min, colorbar_max]);
            end

        end

        function updateDynamicPlot(obj, measurements, true_state, all_measurements)
            % UPDATEDYNAMICPLOT Update dynamic plot during timestep execution (PDA_PF override)
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
            %   Updates the dynamic visualization with comprehensive likelihood breakdown.
            %   Shows 2x5 panel layout with detection, magnitude, combined likelihood fields
            %   plus analysis panels similar to test_hybrid_PF visualization.

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

            % Check if we have particle likelihood data
            % Always show comprehensive visualization if particle likelihood data is available
            if isempty(obj.particle_detection_likelihoods)
                % Fallback to simple visualization if no particle likelihood data available
                title_str = sprintf('%s Real-time Tracking (Step %d)', ...
                    class(obj), obj.timestep_counter);
                obj.visualize(obj.dynamic_figure_handle, title_str, measurements, true_state, all_measurements);
                drawnow;
                obj.captureFrame();
                pause(0.01);
                return;
            end

            % Determine if we have magnitude/signal data (composite likelihood mode)
            has_magnitude_data = ~isempty(obj.particle_magnitude_likelihoods);

            % Set figure and clear for comprehensive visualization
            figure(obj.dynamic_figure_handle);
            clf;

            % Create 2x5 tiled layout for comprehensive likelihood visualization
            t = tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');

            % Set overall title with mode indication
            if has_magnitude_data
                mode_str = 'Hybrid Likelihood (Particle-Based)';
            else
                mode_str = 'Detection-Only Likelihood (Particle-Based)';
            end

            % title(t, sprintf('%s Real-time Tracking - %s (Step %d)', class(obj), mode_str, obj.timestep_counter), ...
            %       'Interpreter', 'latex', 'FontSize', 14);

            % Get current measurement (first measurement if multiple)
            if ~isempty(measurements)
                current_meas = measurements(:, 1);
            else
                current_meas = [0; 2]; % Default center position
            end

            % Get current particle distribution
            particles = obj.particles;
            weights = obj.weights;
            mean_state = particles * weights;

            % Sort particles by weight (ascending) so highest weights plot on top
            [weights_sorted, sort_idx] = sort(weights, 'ascend');
            particles_sorted = particles(:, sort_idx);

            % Convert sparse likelihood data to full if needed
            det_likes = obj.particle_detection_likelihoods;

            if issparse(det_likes)
                det_likes = full(det_likes);
            end

            % Ensure det_likes is a column vector
            if size(det_likes, 1) == 1
                det_likes = det_likes';
            end

            % Handle magnitude likelihood data if available
            if has_magnitude_data
                mag_likes = obj.particle_magnitude_likelihoods;

                if issparse(mag_likes)
                    mag_likes = full(mag_likes);
                end

                % Ensure mag_likes is a column vector
                if size(mag_likes, 1) == 1
                    mag_likes = mag_likes';
                end

                combined_likes = det_likes .* mag_likes;
            else
                mag_likes = [];
                combined_likes = det_likes; % Use detection only for "combined"
            end

            % Ensure combined_likes is a column vector for scatter
            if size(combined_likes, 1) == 1
                combined_likes = combined_likes';
            end

            % Sort likelihood arrays by same indices as particles (for proper layering)
            det_likes_sorted = det_likes(sort_idx);
            if has_magnitude_data
                mag_likes_sorted = mag_likes(sort_idx);
                combined_likes_sorted = combined_likes(sort_idx);
            else
                mag_likes_sorted = [];
                combined_likes_sorted = det_likes_sorted;
            end

            mean_state = particles * weights;
            
            % Determine colorbar limits for weight-based plots
            % If weights are uniform (just resampled), use [0, 1] for better visibility
            weight_range = max(weights) - min(weights);
            uniform_threshold = 1e-6; % Threshold for considering weights uniform
            
            if weight_range < uniform_threshold
                % Weights are uniform - use fixed [0, 1] scale
                weight_caxis_limits = [0, 1];
            else
                % Weights are not uniform - use [0, max(weight)]
                weight_caxis_limits = [0, max(weights_sorted)];
            end

            % Row 1: Particle Likelihood Visualizations
            % Tile 1: Particles colored by Detection Likelihood
            nexttile(1);
            scatter(particles_sorted(1, :), particles_sorted(2, :), 40, det_likes_sorted, 'filled', 'MarkerFaceAlpha', 0.7);
            hold on;
            plot(current_meas(1), current_meas(2), '+', 'Color', [0.2 0.2 0.2], 'MarkerSize', 12, 'LineWidth', 3);
            plot(mean_state(1), mean_state(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'mo', 'MarkerSize', 8, 'LineWidth', 2);
            end

            title('Detection Likelihood', 'Interpreter', 'latex');
            xlabel('$X$ (m)', 'Interpreter', 'latex');
            ylabel('$Y$ (m)', 'Interpreter', 'latex');
            xlim([-2, 2]); ylim([0, 4]);
            axis square;
            cb = colorbar;
            cb.Ruler.TickLabelFormat = '%.2f';
            caxis([0, max(det_likes_sorted)]);
            colormap(gca, 'parula');

            % Tile 2: Particles colored by Magnitude Likelihood
            nexttile(2);

            if has_magnitude_data
                scatter(particles_sorted(1, :), particles_sorted(2, :), 40, mag_likes_sorted, 'filled', 'MarkerFaceAlpha', 0.7);
                hold on;
                plot(current_meas(1), current_meas(2), '+', 'Color', [0.2 0.2 0.2], 'MarkerSize', 12, 'LineWidth', 3);
                plot(mean_state(1), mean_state(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

                if ~isempty(true_state)
                    plot(true_state(1), true_state(2), 'mo', 'MarkerSize', 8, 'LineWidth', 2);
                end

                title('Magnitude Likelihood', 'Interpreter', 'latex');
                xlabel('$X$ (m)', 'Interpreter', 'latex');
                ylabel('$Y$ (m)', 'Interpreter', 'latex');
                xlim([-2, 2]); ylim([0, 4]);
                axis square;
                cb = colorbar;
                cb.Ruler.TickLabelFormat = '%.2f';
                caxis([0, max(mag_likes_sorted)]);
                colormap(gca, 'parula');
            else
                % Show detection particles with note
                scatter(particles_sorted(1, :), particles_sorted(2, :), 40, det_likes_sorted, 'filled', 'MarkerFaceAlpha', 0.3);
                hold on;
                plot(current_meas(1), current_meas(2), '+', 'Color', [0.2 0.2 0.2], 'MarkerSize', 12, 'LineWidth', 3);
                text(0.5, 0.5, {'Magnitude Likelihood', 'Not Available', '(Detection-Only Mode)'}, ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'Units', 'normalized', 'FontSize', 12, 'Color', [0.5 0.5 0.5]);
                title('Magnitude Likelihood - N/A', 'Interpreter', 'latex');
                xlabel('$X$ (m)', 'Interpreter', 'latex');
                ylabel('$Y$ (m)', 'Interpreter', 'latex');
                xlim([-2, 2]); ylim([0, 4]);
                axis square;
            end

            % Tile 3: Particles colored by Combined Likelihood
            nexttile(3);
            scatter(particles_sorted(1, :), particles_sorted(2, :), 40, combined_likes_sorted, 'filled', 'MarkerFaceAlpha', 0.7);
            hold on;
            plot(current_meas(1), current_meas(2), '+', 'Color', [0.2 0.2 0.2], 'MarkerSize', 12, 'LineWidth', 3);
            plot(mean_state(1), mean_state(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'mo', 'MarkerSize', 8, 'LineWidth', 2);
            end

            if has_magnitude_data
                title('Combined Likelihood', 'Interpreter', 'latex');
                colormap(gca, 'parula');
            else
                title('Detection-Only Likelihood', 'Interpreter', 'latex');
                colormap(gca, 'hot');
            end

            xlabel('$X$ (m)', 'Interpreter', 'latex');
            ylabel('$Y$ (m)', 'Interpreter', 'latex');
            xlim([-2, 2]); ylim([0, 4]);
            axis square;
            cb = colorbar;
            cb.Ruler.TickLabelFormat = '%.2f';
            caxis([0, max(combined_likes_sorted)]);

            % Tile 4: mWidar Signal with Detections
            nexttile(4);

            if has_magnitude_data && ~isempty(obj.current_signal)
                % Display mWidar signal as background
                npx = 128;
                xvec = linspace(-2, 2, npx);
                yvec = linspace(0, 4, npx);
                signal_normalized = obj.current_signal / max(obj.current_signal(:));
                imagesc(xvec, yvec, reshape(signal_normalized, [npx, npx]));
                set(gca, 'YDir', 'normal');
                hold on;

                % Overlay all measurements (detections) prominently
                if ~isempty(all_measurements)
                    plot(all_measurements(1, :), all_measurements(2, :), 'wo', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'yellow');
                end

                if ~isempty(measurements)
                    plot(measurements(1, :), measurements(2, :), 'ro', 'MarkerSize', 12, 'LineWidth', 3, 'MarkerFaceColor', 'red');
                end

                % Add current measurement with distinct marker
                % plot(current_meas(1), current_meas(2), 'r*', 'MarkerSize', 15, 'LineWidth', 3);

                % Find and mark the most probable association with a cyan star
                association_dist = obj.getAssociationDistribution();
                if ~isempty(all_measurements) && length(association_dist) >= 2
                    % Find the measurement with highest association probability (exclude clutter)
                    [max_prob, max_idx] = max(association_dist(1:end-1)); % Exclude clutter hypothesis
                    
                    if max_idx <= size(all_measurements, 2)
                        % Mark the most probable association with a distinctive cyan star
                        plot(all_measurements(1, max_idx), all_measurements(2, max_idx), '*', ...
                            'Color', [0 1 1], 'MarkerSize', 30, 'LineWidth', 4);
                        
                        % Add text label
                        text(all_measurements(1, max_idx), all_measurements(2, max_idx) + 0.2, ...
                            sprintf('MAP: %.1f%%', 100*max_prob), ...
                            'HorizontalAlignment', 'center', 'FontSize', 9, ...
                            'FontWeight', 'bold', 'Color', [0 1 1], 'BackgroundColor', [0 0 0 0.5]);
                    end
                end

                % Show state estimate
                plot(mean_state(1), mean_state(2), 'bs', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'blue');

                if ~isempty(true_state)
                    plot(true_state(1), true_state(2), 'mo', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'magenta');
                end

                title('mWidar Signal + Detections', 'Interpreter', 'latex');
                xlabel('$X$ (m)', 'Interpreter', 'latex');
                ylabel('$Y$ (m)', 'Interpreter', 'latex');
                xlim([-2, 2]); ylim([0, 4]);
                axis square;
                % caxis([0, .01]); % Fixed colorbar range for normalized signal
                colorbar;
                colormap(gca, 'parula');
                
            else
                % Show detections without signal background (Detection-only mode)
                % Create a simple background grid for context
                npx = 128;
                xvec = linspace(-2, 2, npx);
                yvec = linspace(0, 4, npx);
                [X, Y] = meshgrid(xvec, yvec);
                background = 0.1 * ones(size(X)); % Light background
                imagesc(xvec, yvec, background);
                set(gca, 'YDir', 'normal');
                hold on;

                % Overlay all measurements (detections) prominently
                if ~isempty(all_measurements)
                    plot(all_measurements(1, :), all_measurements(2, :), 'ko', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'yellow');
                end

                if ~isempty(measurements)
                    plot(measurements(1, :), measurements(2, :), 'ro', 'MarkerSize', 12, 'LineWidth', 3, 'MarkerFaceColor', 'red');
                end

                % Add current measurement with distinct marker
                plot(current_meas(1), current_meas(2), 'r*', 'MarkerSize', 15, 'LineWidth', 3);

                % Find and mark the most probable association with a cyan star
                association_dist = obj.getAssociationDistribution();
                if ~isempty(measurements) && length(association_dist) >= 1
                    % Find the measurement with highest association probability (exclude clutter)
                    [max_prob, max_idx] = max(association_dist(1:end-1)); % Exclude clutter hypothesis
                    
                    if max_idx <= size(measurements, 2)
                        % Mark the most probable association with a distinctive cyan star
                        plot(measurements(1, max_idx), measurements(2, max_idx), '*', ...
                            'Color', [0 1 1], 'MarkerSize', 30, 'LineWidth', 4);
                        
                        % Add text label
                        text(measurements(1, max_idx), measurements(2, max_idx) + 0.2, ...
                            sprintf('MAP: %.1f%%', 100*max_prob), ...
                            'HorizontalAlignment', 'center', 'FontSize', 9, ...
                            'FontWeight', 'bold', 'Color', [0 1 1], 'BackgroundColor', [0 0 0 0.5]);
                    end
                end

                % Show state estimate
                plot(mean_state(1), mean_state(2), 'bs', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'blue');

                if ~isempty(true_state)
                    plot(true_state(1), true_state(2), 'mo', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'magenta');
                end

                text(0.5, 0.05, 'Signal Not Available (Detection-Only Mode)', ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized', ...
                    'FontSize', 10, 'Color', [0.5 0.5 0.5], 'BackgroundColor', 'white');
                title('Detections Only', 'Interpreter', 'latex');
                xlabel('$X$ (m)', 'Interpreter', 'latex');
                ylabel('$Y$ (m)', 'Interpreter', 'latex');
                xlim([-2, 2]); ylim([0, 4]);
                axis square;
                caxis([0, .01]); % Fixed colorbar range
                colorbar;
            end

            % Tile 5: Particles colored by Final Weights
            nexttile(5);
            scatter(particles_sorted(1, :), particles_sorted(2, :), 40, weights_sorted, 'filled', 'MarkerFaceAlpha', 0.7);
            hold on;
            plot(mean_state(1), mean_state(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);

            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'o', 'Color', [0.2 0.2 0.2], ...
                    'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', [0.7 0.7 0.7]);
            end

            plot(current_meas(1), current_meas(2), '+', 'Color', [0.2 0.2 0.2], ...
                'MarkerSize', 12, 'LineWidth', 3);
            title('Particle Weights', 'Interpreter', 'latex');
            xlabel('$X$ (m)', 'Interpreter', 'latex');
            ylabel('$Y$ (m)', 'Interpreter', 'latex');
            xlim([-2, 2]); ylim([0, 4]);
            axis square;
            cb = colorbar;
            cb.Ruler.TickLabelFormat = '%.4f';
            caxis(weight_caxis_limits);
            colormap(gca, 'parula');

            % Row 2: Analysis and Comparison
            if ~isempty(obj.particle_detection_likelihoods)
                % Tile 6: Particle State Distribution Violin Plot
                nexttile(6);
                
                % Create violin plot data for state dimensions
                % We'll show distributions for: X, Y, Vx, Vy (if available)
                state_labels = {};
                state_data = {};
                state_means = [];
                state_truth = [];
                
                % Position X
                state_labels{end+1} = 'X';
                state_data{end+1} = particles(1, :)';
                state_means(end+1) = mean_state(1);
                if ~isempty(true_state)
                    state_truth(end+1) = true_state(1);
                else
                    state_truth(end+1) = NaN;
                end
                
                % Position Y
                state_labels{end+1} = 'Y';
                state_data{end+1} = particles(2, :)';
                state_means(end+1) = mean_state(2);
                if ~isempty(true_state) && length(true_state) >= 2
                    state_truth(end+1) = true_state(2);
                else
                    state_truth(end+1) = NaN;
                end
                
                % Velocity X (if available)
                if size(particles, 1) >= 4
                    state_labels{end+1} = 'Vx';
                    state_data{end+1} = particles(3, :)';
                    state_means(end+1) = mean_state(3);
                    if ~isempty(true_state) && length(true_state) >= 4
                        state_truth(end+1) = true_state(3);
                    else
                        state_truth(end+1) = NaN;
                    end
                    
                    % Velocity Y
                    state_labels{end+1} = 'Vy';
                    state_data{end+1} = particles(4, :)';
                    state_means(end+1) = mean_state(4);
                    if ~isempty(true_state) && length(true_state) >= 4
                        state_truth(end+1) = true_state(4);
                    else
                        state_truth(end+1) = NaN;
                    end
                end
                
                % Use MATLAB's built-in violin plot if available, otherwise create custom
                try
                    % Try using violinplot from File Exchange (if user has it)
                    violinplot(state_data, state_labels);
                catch
                    % Fallback: Create simple box-and-whisker style plot with distribution overlay
                    hold on;
                    n_states = length(state_data);
                    colors = lines(n_states);
                    
                    for i = 1:n_states
                        data = state_data{i};
                        
                        % Create kernel density estimate for violin shape
                        [f, xi] = ksdensity(data, 'NumPoints', 50);
                        f = f / max(f) * 0.35; % Normalize width
                        
                        % Plot left and right sides of violin
                        fill([i - f, i + fliplr(f)], [xi, fliplr(xi)], colors(i, :), ...
                            'FaceAlpha', 0.6, 'EdgeColor', colors(i, :), 'LineWidth', 1.5);
                        
                        % Plot median line
                        plot([i-0.35, i+0.35], [median(data), median(data)], 'k-', 'LineWidth', 2);
                        
                        % Plot mean as circle
                        plot(i, state_means(i), 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
                        
                        % Plot true state as diamond (if available)
                        if ~isnan(state_truth(i))
                            plot(i, state_truth(i), 'md', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm');
                        end
                    end
                    
                    % Format plot
                    xlim([0.5, n_states + 0.5]);
                    xticks(1:n_states);
                    xticklabels(state_labels);
                    grid on;
                end
                
                xlabel('State Variable', 'Interpreter', 'latex');
                ylabel('Value', 'Interpreter', 'latex');
                title('Particle State Distributions', 'Interpreter', 'latex');
                legend({'Distribution', 'Median', 'Mean', 'Truth'}, 'Location', 'best', 'FontSize', 8);
                axis square;
            else
                % Tile 6: No likelihood data available
                nexttile(6);
                text(0.5, 0.5, {'Particle likelihood', 'data not available'}, ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'Units', 'normalized', 'FontSize', 12, 'Color', [0.5 0.5 0.5]);
                title('Likelihood Components');
            end

            if has_magnitude_data && ~isempty(obj.particle_detection_likelihoods) && ~isempty(obj.particle_magnitude_likelihoods)
                % Tile 7: Likelihood Component Histograms (Hybrid Mode)
                nexttile(7);
                det_likes = obj.particle_detection_likelihoods;
                mag_likes = obj.particle_magnitude_likelihoods;

                % Convert to full matrices if sparse
                if issparse(det_likes)
                    det_likes = full(det_likes);
                end

                if issparse(mag_likes)
                    mag_likes = full(mag_likes);
                end

                det_norm = det_likes / max(det_likes);
                mag_norm = mag_likes / max(mag_likes);
                combined_norm = (det_likes .* mag_likes) / max(det_likes .* mag_likes);

                edges = linspace(0, 1, 20);
                histogram(det_norm, edges, 'FaceAlpha', 0.6, 'FaceColor', 'r', 'EdgeColor', 'none');
                hold on;
                histogram(mag_norm, edges, 'FaceAlpha', 0.6, 'FaceColor', 'b', 'EdgeColor', 'none');
                histogram(combined_norm, edges, 'FaceAlpha', 0.6, 'FaceColor', 'g', 'EdgeColor', 'none');

                xlabel('Normalized Likelihood', 'Interpreter', 'latex');
                ylabel('Number of Particles', 'Interpreter', 'latex');
                title(['Likelihood Distributions (Step ', num2str(obj.timestep_counter), ')'], 'Interpreter', 'latex');
                legend('Detection', 'Magnitude', 'Combined', 'Location', 'northeast');
                grid on;
                axis square;
            elseif ~isempty(obj.particle_detection_likelihoods)
                % Tile 7: Detection Likelihood Histogram Only (Detection-Only Mode)
                nexttile(7);
                det_likes = obj.particle_detection_likelihoods;

                if issparse(det_likes)
                    det_likes = full(det_likes);
                end

                det_norm = det_likes / max(det_likes);

                edges = linspace(0, 1, 20);
                histogram(det_norm, edges, 'FaceAlpha', 0.6, 'FaceColor', 'r', 'EdgeColor', 'none');

                xlabel('Normalized Detection Likelihood', 'Interpreter', 'latex');
                ylabel('Number of Particles', 'Interpreter', 'latex');
                title(['Detection Likelihood Distribution (Step ', num2str(obj.timestep_counter), ')'], 'Interpreter', 'latex');
                legend('Detection Only', 'Location', 'northeast');
                grid on;
                axis square;
            else
                % Tile 7: No likelihood data available
                nexttile(7);
                text(0.5, 0.5, {'Particle likelihood', 'data not available'}, ...
                    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
                    'Units', 'normalized', 'FontSize', 12, 'Color', [0.5 0.5 0.5]);
                title('Likelihood Distributions');
            end

            % Tile 8: Velocity estimates
            nexttile(8);

            if size(particles, 1) >= 4 % Check if we have velocity states
                scatter(particles_sorted(3, :), particles_sorted(4, :), 20, weights_sorted, 'filled', 'MarkerFaceAlpha', 0.6);
                hold on;
                plot(mean_state(3), mean_state(4), 'ro', 'MarkerSize', 10, 'LineWidth', 3);

                if ~isempty(true_state) && length(true_state) >= 4
                    plot(true_state(3), true_state(4), 'o', 'Color', [0.7 0.7 0.7], ...
                        'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', [0.7 0.7 0.7]);
                end

                title(['Velocity (Step ', num2str(obj.timestep_counter), ')'], 'Interpreter', 'latex');
                xlabel('$V_x$ (m/s)', 'Interpreter', 'latex');
                ylabel('$V_y$ (m/s)', 'Interpreter', 'latex');
                axis square;
                cb = colorbar;
                cb.Ruler.TickLabelFormat = '%.4f';
                caxis(weight_caxis_limits);
            else
                text(0.5, 0.5, 'Velocity states\nnot available', 'HorizontalAlignment', 'center');
                title('Velocity');
            end

            % Tile 9: Acceleration estimates
            nexttile(9);

            if size(particles, 1) >= 6 % Check if we have acceleration states
                scatter(particles_sorted(5, :), particles_sorted(6, :), 20, weights_sorted, 'filled', 'MarkerFaceAlpha', 0.6);
                hold on;
                plot(mean_state(5), mean_state(6), 'ro', 'MarkerSize', 10, 'LineWidth', 3);

                if ~isempty(true_state) && length(true_state) >= 6
                    plot(true_state(5), true_state(6), 'o', 'Color', [0.7 0.7 0.7], ...
                        'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', [0.7 0.7 0.7]);
                end

                title(['Acceleration (Step ', num2str(obj.timestep_counter), ')'], 'Interpreter', 'latex');
                xlabel('$A_x$ (m/s$^2$)', 'Interpreter', 'latex');
                ylabel('$A_y$ (m/s$^2$)', 'Interpreter', 'latex');
                axis square;
                cb = colorbar;
                cb.Ruler.TickLabelFormat = '%.4f';
                caxis(weight_caxis_limits);
            else
                text(0.5, 0.5, 'Acceleration states\nnot available', 'HorizontalAlignment', 'center');
                title('Acceleration');
            end

            % Tile 10: Association Distribution Histogram
            nexttile(10);
            cla;
            
            % Get association distribution from particles
            association_dist = obj.getAssociationDistribution();
            N_associations = length(association_dist);
            
            % Create bar plot of association distribution
            bar_handle = bar(1:N_associations, association_dist, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
            hold on;
            
            % Find the measurement closest to ground truth and mark it with a star
            if ~isempty(true_state) && ~isempty(measurements)
                % Calculate distances from each measurement to ground truth
                N_meas = size(measurements, 2);
                distances = zeros(1, N_meas);
                for meas_idx = 1:N_meas
                    dx = measurements(1, meas_idx) - true_state(1);
                    dy = measurements(2, meas_idx) - true_state(2);
                    distances(meas_idx) = sqrt(dx^2 + dy^2);
                end
                
                % Find index of closest measurement
                [~, closest_idx] = min(distances);
                
                % Place a star at the top of the correct association bin
                star_y = association_dist(closest_idx) * 1.15; % Place 15% above bar
                plot(closest_idx, star_y, '*', 'Color', [1 0.843 0], 'MarkerSize', 25, 'LineWidth', 3);
                
                % Add text label
                text(closest_idx, star_y * 1.1, 'True Assoc.', ...
                    'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.8 0.5 0]);
            end
            
            % Formatting
            xlabel('Association Hypothesis', 'Interpreter', 'latex', 'FontSize', 11);
            ylabel('Probability', 'Interpreter', 'latex', 'FontSize', 11);
            title(['Association Distribution (Step ', num2str(obj.timestep_counter), ')'], 'Interpreter', 'latex', 'FontSize', 12);
            
            % Create x-axis labels
            x_labels = cell(1, N_associations);
            for i = 1:N_associations-1
                x_labels{i} = sprintf('M%d', i); % M1, M2, M3, etc.
            end
            x_labels{end} = 'Clutter'; % Last bin is clutter hypothesis
            
            set(gca, 'XTick', 1:N_associations, 'XTickLabel', x_labels);
            xtickangle(45);
            ylim([0, max(association_dist) * 1.3]); % Give room for star
            grid on;
            axis square;
            
            % Add a legend if star is present
            if ~isempty(true_state) && ~isempty(measurements)
                legend('Association Prob.', 'Ground Truth', 'Location', 'northeast', 'FontSize', 9);
            end

            drawnow; % Force immediate update

            % Capture frame for animation after plot is updated
            obj.captureFrame();

            pause(0.01); % Small pause for smooth animation

            % --- GIF writing ---
            if ~isempty(obj.gif_filename)
                frame = getframe(gcf);
                im = frame2im(frame);
                [A, map] = rgb2ind(im, 256);

                if obj.gif_frame_counter == 1
                    imwrite(A, map, obj.gif_filename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
                else
                    imwrite(A, map, obj.gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
                end

                obj.gif_frame_counter = obj.gif_frame_counter + 1;
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
