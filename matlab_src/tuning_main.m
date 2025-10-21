function tuning_main()
    % TUNING_MAIN - Run mWidar tracking analysis for tuning experiments
    %
    % This function is designed for extensive tuning testing with:
    % 1. Environment variables set at top (no command line arguments)
    % 2. Comprehensive performance metric storage for all filter types
    % 3. Animation generation and saving during loop execution
    % 4. Trajectory and deviation plots generated after every timestep
    %
    % ENVIRONMENT VARIABLES (modify these at the top):
    %   DATASET      - Tuning dataset: "T1_no_clutter", "T2_low_clutter", "T3_medium_clutter", "T4_simulated_clutter"
    %   FILTER_TYPE  - Filter type: "KF", "HybridPF", "HMM"
    %   DA_METHOD    - Data association: "PDA", "GNN"
    %   PLOT_MODE    - Plot mode: "interactive", "animation", "none"
    %   DEBUG        - Enable debug output: true/false
    %   DYNAMIC_PLOT - Enable dynamic plotting during filtering: true/false

    % Turn off ALL warnings 
    warning('off', 'all');

    %% ========== ENVIRONMENT VARIABLES (MODIFY THESE) ==========

    % === TUNING CONFIGURATION ===
    DATASET = "T4_simulated_clutter"; % Tuning dataset: "T1_no_clutter", "T2_low_clutter", "T3_medium_clutter", "T4_simulated_clutter"
    FILTER_TYPE = "HybridPF"; % Filter type: "KF", "HybridPF", "HMM"
    DA_METHOD = "PDA"; % Data association: "PDA", "GNN"
    PLOT_MODE = "animation"; % Plot mode: "interactive", "animation", "none"
    DEBUG = true; % Debug output
    DYNAMIC_PLOT = true; % Dynamic plotting during filtering
    INITIALIZE_TRUE = true; % Enable true/uninformative initialization for all filters (maximum uncertainty)

    % === SAVE DIRECTORIES ===
    TUNING_PLOTS_SAVE_DIR = fullfile("..", "figures", "DA_Track", "tuning_plots");

    %% ========== PARAMETER VALIDATION ==========

    % Validate dataset - now using tuning datasets
    valid_datasets = ["T1_no_clutter", "T2_low_clutter", "T3_medium_clutter", "T4_simulated_clutter"];

    if ~ismember(DATASET, valid_datasets)
        error('Invalid tuning dataset. Options: %s', strjoin(valid_datasets, ', '));
    end

    % Validate filter type
    valid_filters = ["KF", "HybridPF", "HMM"];

    if ~ismember(FILTER_TYPE, valid_filters)
        error('Invalid filter type. Options: %s', strjoin(valid_filters, ', '));
    end

    % Validate DA method
    valid_da_methods = ["PDA", "GNN"];

    if ~ismember(DA_METHOD, valid_da_methods)
        error('Invalid DA method. Options: %s', strjoin(valid_da_methods, ', '));
    end

    % Validate plot mode
    valid_plot_modes = ["interactive", "animation", "none"];

    if ~ismember(PLOT_MODE, valid_plot_modes)
        error('Invalid plot mode. Options: %s', strjoin(valid_plot_modes, ', '));
    end

    %% --- Display Configuration ---
    fprintf('=== TUNING CONFIGURATION ===\n');
    fprintf('Dataset: %s\n', DATASET);
    fprintf('Filter type: %s\n', FILTER_TYPE);
    fprintf('DA method: %s\n', DA_METHOD);
    fprintf('Plot mode: %s\n', PLOT_MODE);
    fprintf('Debug enabled: %s\n', string(DEBUG));
    fprintf('Dynamic plot enabled: %s\n', string(DYNAMIC_PLOT));
    fprintf('True initialization: %s\n', string(INITIALIZE_TRUE));
    fprintf('Tuning plots save dir: %s\n', TUNING_PLOTS_SAVE_DIR);
    fprintf('=============================\n');

    %% ========== WORKSPACE SETUP ==========
    %% --- Environment Configuration ---
    clc; close all
    % Add paths for MATLAB functions
    addpath(fullfile('DA_Track'))
    addpath(fullfile('supplemental'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObj'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObjTune'))

    % Create save directory for tuning plots
    if ~exist(TUNING_PLOTS_SAVE_DIR, 'dir')
        mkdir(TUNING_PLOTS_SAVE_DIR);
        fprintf('Created tuning plots save directory: %s\n', TUNING_PLOTS_SAVE_DIR);
    end

    % Create save directory for animations
    animation_save_dir = fullfile("..", "figures", "DA_Track");

    if ~exist(animation_save_dir, 'dir')
        mkdir(animation_save_dir);
        fprintf('Created animation save directory: %s\n', animation_save_dir);
    end

    %% --- Load Supplemental Data ---
    % Load tuning dataset
    load(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObjTune', DATASET + '.mat'), 'Data');

    % Set plotting flags based on PLOT_MODE parameter
    INTERACTIVE = (PLOT_MODE == "interactive");
    ANIMATION = (PLOT_MODE == "animation");

    %% --- Initialize Variables ---
    % Load Data
    GT = Data.GT;
    GT_meas = GT(1:2, :);
    z = Data.y;
    signal = Data.signal;

    % Create measurements structure for HybridPF with both detections and signals
    measurements = struct();
    measurements.z = z;        % Cell array of detection measurements [2 x N_meas] for each timestep
    measurements.signal = signal;  % Cell array of mWidar signals [128 x 128] for each timestep

    % Store original measurements for visualization (before any filtering)
    Data.y_original = Data.y; % Keep original measurements for plotting

    n_k = size(GT, 2);
    performance = cell(1, n_k);

    % Initialize validation sigma parameter (will be set based on filter type)
    validation_sigma = 2; % Default value

    %% ========== FILTER PARAMETERS SETUP ==========
    %% --- Basic Parameters ---
    dt = 0.1; % sec

    %% --- Define Kalman Filter Matrices ---
    % Define KF Matrices state vector - {x, y, vx, vy, ax, ay}
    % Correct continuous-time dynamics matrix for constant acceleration model
    A = [0 0 1 0 0 0; % dx/dt = vx
         0 0 0 1 0 0; % dy/dt = vy
         0 0 0 0 1 0; % dvx/dt = ax
         0 0 0 0 0 1; % dvy/dt = ay
         0 0 0 0 0 0; % dax/dt = 0 (constant acceleration)
         0 0 0 0 0 0]; % day/dt = 0 (constant acceleration)

    F_KF = expm(A * dt);

    %% --- Define Particle Filter Matrices ---
    % Use the direct discrete-time formulation (matches test_hybrid_PF)
    F_PF = [1, 0, dt, 0, dt ^ 2/2, 0;
            0, 1, 0, dt, 0, dt ^ 2/2;
            0, 0, 1, 0, dt, 0;
            0, 0, 0, 1, 0, dt;
            0, 0, 0, 0, 1, 0;
            0, 0, 0, 0, 0, 1];

    %% --- Define Noise Matrices ---
    Q = eye(6) * 1e-1;
    Q_KF = diag([1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2]);
    Q_PF = diag([1e-3, 1e-3, 1e-2, 1e-2, 1e-1, 1e-1]);

    R = 0.1 * eye(2);

    %% --- Define Observation Matrix ---
    H = [1 0 0 0 0 0;
         0 1 0 0 0 0];

    %% --- Define Initial Covariance ---
    P0 = diag([0.1 0.1 0.25 0.25 0.5 0.5]);

    %% ========== FILTER INITIALIZATION ==========

    switch FILTER_TYPE

        case "KF"
            %% --- Kalman Filter Setup ---
            fprintf("Using Kalman Filter ")

            % Handle true initialization for KF
            if INITIALIZE_TRUE
                % True initialization: zero mean with very large covariance
                initial_state = zeros(size(GT(:, 1))); % Zero mean for all states -- but fixate on position
                initial_state(1:2) = GT(1:2, 1);
                initial_covariance = diag([1, 1, 10, 10, 5, 5]); % Very large covariance

                if DEBUG
                    fprintf("with TRUE INITIALIZATION (zero mean, large covariance)\n");
                    fprintf("Initial covariance diagonal: [%.1f, %.1f, %.1f, %.1f, %.1f, %.1f]\n", ...
                        diag(initial_covariance));
                end

            else
                % Standard initialization using ground truth
                initial_state = GT(:, 1);
                initial_covariance = P0;

                if DEBUG
                    fprintf("with STANDARD INITIALIZATION (GT-based)\n");
                end

            end

            if DA_METHOD == "PDA"
                current_class = PDA_KF(initial_state, initial_covariance, F_KF, Q_KF, R, H, 'Debug', DEBUG, "DynamicPlot", DYNAMIC_PLOT);

            elseif DA_METHOD == "GNN"
                current_class = GNN_KF(initial_state, initial_covariance, F_KF, Q_KF, R, H, 'Debug', DEBUG, "DynamicPlot", DYNAMIC_PLOT);
            else
                error('Unknown data association method: %s', DA_METHOD);
            end

            fprintf("with %s data association\n", DA_METHOD);

            %% --- Initialize Comprehensive Performance Storage for KF ---
            [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate(); % Current estimate

            % Store comprehensive KF metrics
            performance{1}.x_predicted = current_class.x_predicted; % Predicted state
            performance{1}.P_predicted = current_class.P_predicted; % Predicted covariance
            performance{1}.z_predicted = current_class.z_predicted; % Predicted measurement
            performance{1}.S_innovation = current_class.S_innovation; % Innovation covariance
            performance{1}.innovation = []; % Innovation (empty for first step)
            performance{1}.K_gain = []; % Kalman gain (empty for first step)

            % Store initial measurements (empty for first timestep)
            performance{1}.measurements_original = [];
            performance{1}.measurements_used = [];

        case 'HybridPF'
            %% --- Hybrid Particle Filter Setup ---
            fprintf("Using Hybrid Particle Filter ")
            load(fullfile('supplemental', 'precalc_imagegridHMMEmLikeMag.mat'), 'pointlikelihood_image');
            pointlikelihood_mag = pointlikelihood_image; 
            clear pointlikelihood_image;
            load(fullfile('supplemental', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');

            if DA_METHOD == "PDA"
                fprintf("with PDA data association\n");

                validation_sigma = 5; % Set validation sigma for PDA
                current_class = PDA_PF(GT(:, 1), 10000, F_PF, Q_PF, H, pointlikelihood_image, pointlikelihood_mag, "Debug", DEBUG, "DynamicPlot", DYNAMIC_PLOT, "ValidationSigma", validation_sigma, "UniformInit", INITIALIZE_TRUE);
                % current_class.setDetectionModel(0.99, 0.80); % Set default detection model parameters (PD, PFA)
                current_class.setDetectionModel(0.99, 0.2); % Set default detection model parameters
                current_class.ESS_threshold_percentage = .10;
                
                % Enable composite likelihood mode for comprehensive visualization
                current_class.composite_likelihood = true;
                fprintf('-> Enabled composite likelihood mode for comprehensive visualization\n');
                current_class.gif_filename = 'coolgif.gif'; % No GIF by default

            elseif DA_METHOD == "GNN"
                fprintf("with GNN data association\n");
                Q_PF = Q;
                validation_sigma = 5; % Set validation sigma for GNN
                current_class = GNN_PF(GT(:, 1), 10000, F_PF, Q_PF, H, pointlikelihood_image, "Debug", DEBUG, "DynamicPlot", DYNAMIC_PLOT, "ValidationSigma", validation_sigma, "UniformInit", INITIALIZE_TRUE);

            else
                error('Unknown data association method: %s', DA_METHOD);
            end

            %% --- Initialize Comprehensive Performance Storage for PF ---
            performance{1}.particles = current_class.particles; % Particle states
            performance{1}.weights = current_class.weights; % Particle weights
            performance{1}.N_eff = []; % Effective sample size (empty for first step)
            performance{1}.resampled = false; % Resampling flag
            [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate(); % Gaussian approximation

            % Store initial measurements (empty for first timestep)
            performance{1}.measurements_original = [];
            performance{1}.measurements_used = [];

        case 'HMM'
            %% --- Hidden Markov Model Setup ---
            fprintf("Using HMM Filter ")
            load(fullfile('supplemental', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');
            load(fullfile('supplemental', 'precalc_imagegridHMMSTMn15.mat'), 'A');
            A_slow = A; clear A; % Clear A to avoid confusion with the next load
            load(fullfile('supplemental', 'precalc_imagegridHMMSTMn30.mat'), 'A');
            A_fast = A; clear A; % Clear A to avoid confusion with the next load

            % Handle true initialization for HMM
            if INITIALIZE_TRUE
                % Use center of field as dummy initial position (will be overridden)
                initial_position = [0; 2]; % Center of [-2,2] x [0,4] space

                if DEBUG
                    fprintf("with TRUE INITIALIZATION (uniform over field)\n");
                end

            else
                % Standard initialization using ground truth
                initial_position = GT(1:2, 1);

                if DEBUG
                    fprintf("with STANDARD INITIALIZATION (GT-based)\n");
                end

            end

            if DA_METHOD == "PDA"
                fprintf("with PDA data association\n");
                validation_sigma = 5; % Set validation sigma for PDA
                current_class = PDA_HMM(initial_position, A_slow, pointlikelihood_image, "Debug", DEBUG, "DynamicPlot", DYNAMIC_PLOT, "ValidationSigma", validation_sigma);

            elseif DA_METHOD == "GNN"
                fprintf("with GNN data association\n");
                validation_sigma = 5; % Set validation sigma for GNN
                current_class = GNN_HMM(initial_position, A_slow, pointlikelihood_image, "Debug", DEBUG, "DynamicPlot", DYNAMIC_PLOT, "ValidationSigma", validation_sigma);

            else
                error('Unknown data association method: %s', DA_METHOD);
            end

            % Override with uniform distribution if true initialization is requested
            if INITIALIZE_TRUE
                current_class.ptarget_prob = ones(current_class.npx2, 1) / current_class.npx2;

                if DEBUG
                    fprintf("-> Overrode with uniform distribution over entire field\n");
                end

            end

            %% --- Initialize Comprehensive Performance Storage for HMM ---
            [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate(); % Gaussian approximation
            performance{1}.prior_prob = current_class.ptarget_prob; % Prior distribution
            performance{1}.likelihood_prob = []; % Likelihood (empty for first step)
            performance{1}.posterior_prob = current_class.ptarget_prob; % Posterior distribution
            performance{1}.entropy = -sum(current_class.ptarget_prob .* log(current_class.ptarget_prob +1e-10)); % Information entropy

            % Store initial measurements (empty for first timestep)
            performance{1}.measurements_original = [];
            performance{1}.measurements_used = [];

        otherwise
            error('Unknown filter type: %s', FILTER_TYPE);

    end

    %% ========== STATE ESTIMATION LOOP ==========
    for i = 2:n_k
        fprintf('Processing time step %d/%d\n', i, n_k);

        %% --- Filter-Specific Timestep Processing ---
        switch FILTER_TYPE
            case "KF"
                %% --- Kalman Filter Timestep ---
                current_meas = z{i}; % Use original measurements

                % Store original measurements for visualization
                performance{i}.measurements_original = current_meas;
                performance{i}.measurements_used = current_meas; % KF uses all measurements

                % Perform Timestep Update using new standardized API
                current_class.timestep(current_meas, GT(:, i)); % Pass ground truth for visualization

                %% --- Update Comprehensive Performance Metrics for KF ---
                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();

                % Store comprehensive KF metrics
                performance{i}.x_predicted = current_class.x_predicted; % Predicted state
                performance{i}.P_predicted = current_class.P_predicted; % Predicted covariance
                performance{i}.z_predicted = current_class.z_predicted; % Predicted measurement
                performance{i}.S_innovation = current_class.S_innovation; % Innovation covariance

                % Calculate innovation and Kalman gain if measurements exist
                if ~isempty(current_meas)
                    performance{i}.innovation = current_meas(:, 1) - current_class.z_predicted; % Innovation for first measurement
                    performance{i}.K_gain = current_class.P_predicted * current_class.H' / current_class.S_innovation; % Kalman gain
                else
                    performance{i}.innovation = [];
                    performance{i}.K_gain = [];
                end

            case 'HybridPF'
                %% --- Hybrid Particle Filter Timestep ---
                current_meas_raw = measurements.z{i}; % Get raw detection measurements
                current_signal = measurements.signal{i}; % Get pre-calculated mWidar signal

                % Store original measurements for visualization
                performance{i}.measurements_original = current_meas_raw;

                % Get the actual used measurements by calling the validation method
                [measurements_used, ~] = current_class.Validation(current_meas_raw);
                performance{i}.measurements_used = measurements_used;

                % Create measurement struct with both detection and magnitude signal
                measurement_struct = struct();
                measurement_struct.det = current_meas_raw; % Detection measurements [2 x N_meas]
                measurement_struct.mag = current_signal;   % Pre-calculated mWidar signal [128 x 128]

                % Debug: Print measurement filtering info
                if DEBUG && ~isempty(current_meas_raw)
                    fprintf('  Step %d: %d original measurements, %d used measurements, signal size %dx%d\n', ...
                        i, size(current_meas_raw, 2), size(measurements_used, 2), size(current_signal, 1), size(current_signal, 2));
                end

                % Store pre-timestep particle info for comparison
                particles_before = current_class.particles;
                weights_before = current_class.weights;

                % Perform Timestep Update with measurement struct (includes dynamic plotting if enabled)
                if i < 1
                    current_class.timestep(measurement_struct.det, GT(:, i)); % Pass ground truth for visualization
                else
                    current_class.timestep(measurement_struct, GT(:, i)); % Pass ground truth for visualization
                end

                %% --- Update Comprehensive Performance Metrics for PF ---
                performance{i}.particles = current_class.particles; % Updated particle states
                performance{i}.weights = current_class.weights; % Updated particle weights

                % Calculate effective sample size
                performance{i}.N_eff = 1 / sum(current_class.weights .^ 2); % Effective sample size

                % Detect resampling (check if particles changed significantly)
                performance{i}.resampled = ~isequal(particles_before, current_class.particles);

                % Update Gaussian estimate
                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();

            case 'HMM'
                %% --- Hidden Markov Model Timestep ---
                current_meas = z{i}; % Use original measurements

                % Store original measurements for visualization
                performance{i}.measurements_original = current_meas;

                % Get the actual used measurements by calling the validation method
                [measurements_used, ~] = current_class.Validation(current_meas);
                performance{i}.measurements_used = measurements_used;

                % Debug: Print measurement filtering info
                if DEBUG && ~isempty(current_meas)
                    fprintf('  Step %d: %d original measurements, %d used measurements\n', ...
                        i, size(current_meas, 2), size(measurements_used, 2));
                end

                % Perform Timestep Update (includes dynamic plotting if enabled)
                current_class.timestep(current_meas, GT(1:2, i)); % Pass ground truth for visualization

                %% --- Update Comprehensive Performance Metrics for HMM ---
                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();
                performance{i}.prior_prob = current_class.prior_prob; % Prior distribution
                performance{i}.likelihood_prob = current_class.likelihood_prob; % Likelihood distribution
                performance{i}.posterior_prob = current_class.posterior_prob; % Posterior distribution

                % Calculate information entropy
                posterior_safe = current_class.posterior_prob +1e-10; % Add small epsilon to avoid log(0)
                performance{i}.entropy = -sum(posterior_safe .* log(posterior_safe)); % Information entropy

            otherwise
                error('Unknown filter type in timestep loop: %s', FILTER_TYPE);
        end

        %% --- Built-in DA_Track Animation ---
        % The DA_Track classes already handle their own dynamic plotting during timestep()
        % No additional animation code needed - the dynamic plotting is built-in!

    end % End of for loop

    %% ========== FINAL TUNING PLOT GENERATION ==========
    % Generate tuning plots at the end of simulation
    fprintf('Generating final tuning plots...\n');

    try
        mWidar_TuningPlot(performance, Data, n_k, FILTER_TYPE, DATASET, DA_METHOD, TUNING_PLOTS_SAVE_DIR);
        fprintf('Tuning plots generated successfully.\n');
    catch ME
        warning('Failed to generate final tuning plot: %s', ME.message);
    end

    %% ========== FINAL SUMMARY ==========
    fprintf('\n=== TUNING EXPERIMENT COMPLETED ===\n');
    fprintf('Filter: %s-%s\n', FILTER_TYPE, DA_METHOD);
    fprintf('Dataset: %s\n', DATASET);
    fprintf('Timesteps processed: %d\n', n_k);
    fprintf('Performance metrics stored for all timesteps: Yes\n');
    fprintf('Final tuning plots saved to: %s\n', TUNING_PLOTS_SAVE_DIR);

    %% ========== SAVE CAPTURED FRAMES ANIMATION ==========
    if DYNAMIC_PLOT && current_class.hasFrames()
        fprintf('\n=== SAVING CAPTURED FRAMES ANIMATION ===\n');

        % Create animation directory if it doesn't exist
        animation_dir = fullfile("..", "figures", "DA_Track", "captured_animations");

        if ~exist(animation_dir, 'dir')
            mkdir(animation_dir);
            fprintf('Created captured animation directory: %s\n', animation_dir);
        end

        % Generate filename for captured frames animation (GIF only)
        gif_filename = sprintf('%s_%s_%s_captured.gif', FILTER_TYPE, DATASET, DA_METHOD);
        gif_path = fullfile(animation_dir, gif_filename);

        % Save animation from captured frames as GIF
        current_class.saveAnimation(char(gif_path), 'FrameRate', 5);

        % Clear frames to free memory
        current_class.clearFrames();

        fprintf('Captured frames GIF saved to: %s\n', gif_path);
        fprintf('===========================================\n');
    else

        if ~DYNAMIC_PLOT
            fprintf('\nNo animation to save - DYNAMIC_PLOT was disabled.\n');
        else
            fprintf('\nNo frames captured - check if plotting occurred correctly.\n');
            fprintf('Captured frame count: %d\n', current_class.getFrameCount());
        end

    end

    %% ========== FINAL ANIMATION GENERATION ==========
    if ANIMATION
        fprintf('Generating animation...\n');
        saveString = FILTER_TYPE + "_" + DATASET + "_" + DA_METHOD + ".gif";
        animation_path = fullfile("..", "figures", "DA_Track", "tuning_animations", saveString);

        % Animation plotting function - use actual simulation time vector
        time_vector = 0:dt:(n_k - 1) * dt;
        mWidar_FilterPlot_Distribution(performance, Data, time_vector, FILTER_TYPE, validation_sigma, animation_path);
        fprintf('Animation saved to: %s\n', animation_path);
    end

    fprintf('===================================\n');

    %% --- Future Enhancements ---
    % NEES(current_class,initial_state,A,10,M,G)

end
