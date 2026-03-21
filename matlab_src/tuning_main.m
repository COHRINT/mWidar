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
    FILTER_TYPE = "HMM_RBPF"; % Filter type: "KF", "HybridPF", "HMM", "RBPF", "HMM_RBPF", "MC_PF"
    DA_METHOD = "PDA"; % Data association: "PDA", "GNN" (not used for RBPF / MC_PF - has built - in uniform association)
    PLOT_MODE = "none"; % Plot mode: "interactive", "animation", "none"
    DEBUG = true; % Debug output
    DYNAMIC_PLOT = false; % Dynamic plotting during filtering (not used for RBPF/MC_PF - post-processing only)
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
    valid_filters = ["KF", "HybridPF", "HMM", "RBPF", "HMM_RBPF", "MC_PF"];

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
    measurements.z = z; % Cell array of detection measurements [2 x N_meas] for each timestep
    measurements.signal = signal; % Cell array of mWidar signals [128 x 128] for each timestep

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
    Q_RBPF = diag([1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3]); % More certain acceleration for PF to improve performance in tuning

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

        case 'RBPF'
            %% --- Rao-Blackwellized Particle Filter Setup ---
            fprintf("Using KF-RBPF ");

            % Handle true initialization for RBPF
            if INITIALIZE_TRUE
                % True initialization: zero mean with position from GT
                initial_state = zeros(size(GT(:, 1)));
                initial_state(1:2) = GT(1:2, 1); % Use true initial position

                if DEBUG
                    fprintf("with TRUE INITIALIZATION (uncertain velocity)\n");
                    fprintf("Initial state: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", initial_state);
                end

            else
                % Standard initialization using ground truth
                initial_state = GT(:, 1);

                if DEBUG
                    fprintf("with STANDARD INITIALIZATION (GT-based)\n");
                end

            end

            % Create RBPF with uniform association strategy
            current_class = KF_RBPF(initial_state, 100, F_KF, Q_KF, H, R, ...
                'Debug', DEBUG, 'AssociationStrategy', 'uniform', 'UniformInit', INITIALIZE_TRUE);

            % current_class.setDetectionModel(0.99, 0.25); % PD=0.99, PFA=0.25
            % current_class.ESS_threshold_percentage = 0.2; % Resample at 20 % ESS

            if INITIALIZE_TRUE
                fprintf("(uniform association, 50 particles, UNIFORM INITIALIZATION)\n");
            else
                fprintf("(uniform association, 50 particles)\n");
            end

            %% --- Initialize Comprehensive Performance Storage for RBPF ---
            [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate();

            % Store RBPF-specific metrics
            performance{1}.particles = current_class.particles; % Particle array
            performance{1}.N_p = current_class.N_p; % Number of particles
            performance{1}.ESS = []; % Effective sample size (empty for first step)

            % Store initial measurements (empty for first timestep)
            performance{1}.measurements_original = [];
            performance{1}.measurements_used = [];

        case 'HMM_RBPF'
            %% --- HMM Rao-Blackwellized Particle Filter Setup ---
            fprintf("Using HMM-RBPF ");

            % Load HMM lookup tables
            load(fullfile('supplemental', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');
            load(fullfile('supplemental', 'precalc_imagegridHMMSTMn15.mat'), 'A');
            A_hmm = A; clear A;

            % Handle initialization
            if INITIALIZE_TRUE
                initial_position = [];  % HMM constructor treats [] as uniform prior

                if DEBUG
                    fprintf("with UNIFORM INITIALIZATION (maximum uncertainty)\n");
                end
            else
                initial_position = GT(1:2, 1);

                if DEBUG
                    fprintf("with STANDARD INITIALIZATION (GT-based position)\n");
                end
            end

            % Create HMM-RBPF
            current_class = HMM_RBPF(initial_position, 1000, A_hmm, pointlikelihood_image, ...
                'Debug', DEBUG, 'AssociationStrategy', 'optimal', ...
                'UniformInit', INITIALIZE_TRUE, 'PD', 0.95, 'PFA', 0.05);

            fprintf("(1000 particles, optimal association, uniform=%s)\n", string(INITIALIZE_TRUE));
            %% --- Initialize Performance Storage for HMM-RBPF ---
            [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate();
            performance{1}.particles = current_class.particles;
            performance{1}.N_p = current_class.N_p;
            performance{1}.ESS = [];
            performance{1}.measurements_original = [];
            performance{1}.measurements_used = [];

        case 'MC_PF'
            %% --- PDA_PF (Hybrid PDA Particle Filter) Setup ---
            fprintf("Using PDA_PF (Hybrid PDA Particle Filter) ");

            % Load likelihood tables
            load(fullfile('supplemental', 'precalc_imagegridHMMEmLikeMag.mat'), 'pointlikelihood_image');
            pointlikelihood_mag = pointlikelihood_image;
            clear pointlikelihood_image;
            load(fullfile('supplemental', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');

            % Handle true/uninformative initialization
            if INITIALIZE_TRUE
                % Maximum uncertainty - uniform over entire space
                initial_state = [0; 2; 0; 0; 0; 0]; % Dummy state, will be overwritten by uniform init

                if DEBUG
                    fprintf("with UNIFORM INITIALIZATION (maximum uncertainty)\n");
                end

            else
                % Standard initialization from ground truth
                initial_state = GT(:, 1);

                if DEBUG
                    fprintf("with STANDARD INITIALIZATION (GT-based)\n");
                end

            end

            % Create PDA_PF (instead of MC_PF which was performing poorly)
            current_class = PDA_PF(initial_state, 1000, F_PF, Q_PF, H, ...
                pointlikelihood_image, pointlikelihood_mag, ...
                'Debug', DEBUG, 'DynamicPlot', false, ...
                'UniformInit', INITIALIZE_TRUE); % 'ValidationSigma', 5,

            % Set detection model parameters (matching main.m)
            current_class.setDetectionModel(0.99, 0.25); % PD=0.99, PFA=0.25
            current_class.ESS_threshold_percentage = 0.2; % Resample at 20 % ESS
            current_class.hybrid_resample_fraction = 0.99; % 99 % resampled, 1 % uniform

            % Enable composite likelihood for visualization
            current_class.composite_likelihood = true;

            fprintf("(1000 particles, PDA algorithm, ESS threshold=20%%, hybrid_resample=99%%)\n");

            %% --- Initialize Comprehensive Performance Storage for PDA_PF ---
            [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate();

            % Store PF-specific metrics
            performance{1}.particles = current_class.particles; % Particle array
            performance{1}.N_p = current_class.N_p; % Number of particles
            performance{1}.ESS = []; % Effective sample size (empty for first step)

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
                measurement_struct.mag = current_signal; % Pre-calculated mWidar signal [128 x 128]

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

            case 'RBPF'
                %% --- Rao-Blackwellized Particle Filter Timestep ---
                current_meas = z{i}; % Use original measurements

                % Store original measurements for visualization
                performance{i}.measurements_original = current_meas;
                performance{i}.measurements_used = current_meas; % RBPF uses all measurements

                % Debug: Print measurement info
                if DEBUG && ~isempty(current_meas)
                    fprintf('  Step %d: %d measurements\n', i, size(current_meas, 2));
                end

                % Perform Timestep Update (no true_meas_flag needed for real data)
                current_class.timestep(current_meas, GT(:, i));

                %% --- Update Comprehensive Performance Metrics for RBPF ---
                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();

                % Store RBPF-specific metrics
                performance{i}.particles = current_class.particles; % Updated particle array
                performance{i}.N_p = current_class.N_p; % Number of particles

                % Calculate ESS from history (most recent timestep)
                if ~isempty(current_class.history)
                    performance{i}.ESS = current_class.history(end).ESS;
                else
                    performance{i}.ESS = current_class.N_p; % Full ESS if no history yet
                end

            case 'HMM_RBPF'
                %% --- HMM-RBPF Timestep ---
                current_meas = z{i}; % Detection measurements [2 x N_meas]

                % Store measurements for visualization
                performance{i}.measurements_original = current_meas;
                performance{i}.measurements_used = current_meas;

                if DEBUG && ~isempty(current_meas)
                    fprintf('  Step %d: %d measurements\n', i, size(current_meas, 2));
                end

                % Perform timestep (GT is 2D position for truth overlay)
                current_class.timestep(current_meas, GT(1:2, i));

                %% --- Update Performance Metrics for HMM-RBPF ---
                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();
                performance{i}.particles = current_class.particles;
                performance{i}.N_p = current_class.N_p;

                if ~isempty(current_class.history)
                    performance{i}.ESS = current_class.history(end).ESS;
                else
                    performance{i}.ESS = current_class.N_p;
                end

            case 'MC_PF'
                %% --- MC_PF (Hybrid SIR) Timestep ---
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
                measurement_struct.mag = current_signal; % Pre-calculated mWidar signal [128 x 128]

                % Debug: Print measurement filtering info
                if DEBUG && ~isempty(current_meas_raw)
                    fprintf('  Step %d: %d original measurements, %d used measurements, signal size %dx%d\n', ...
                        i, size(current_meas_raw, 2), size(measurements_used, 2), size(current_signal, 1), size(current_signal, 2));
                end

                % Store pre-timestep particle info for comparison
                particles_before = current_class.particles;
                weights_before = current_class.weights;

                % Perform Timestep Update with measurement struct (includes dynamic plotting if enabled)
                current_class.timestep(measurement_struct, GT(:, i)); % Pass ground truth for visualization

                %% --- Update Comprehensive Performance Metrics for MC_PF ---
                performance{i}.particles = current_class.particles; % Updated particle states
                performance{i}.weights = current_class.weights; % Updated particle weights

                % Calculate effective sample size
                performance{i}.N_eff = 1 / sum(current_class.weights .^ 2); % Effective sample size

                % Detect resampling (check if particles changed significantly)
                performance{i}.resampled = ~isequal(particles_before, current_class.particles);

                % Update Gaussian estimate
                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();

                % Store ESS from history
                if ~isempty(current_class.history)
                    performance{i}.ESS = current_class.history(end).ESS;
                else
                    performance{i}.ESS = current_class.N_p; % Full ESS if no history yet
                end

            otherwise
                error('Unknown filter type in timestep loop: %s', FILTER_TYPE);
        end

        %% --- Built-in DA_Track Animation ---
        % The DA_Track classes already handle their own dynamic plotting during timestep()
        % No additional animation code needed - the dynamic plotting is built-in!

    end % End of for loop

    %% ========== RBPF-SPECIFIC POST-PROCESSING VISUALIZATION ==========
    if FILTER_TYPE == "RBPF"
        fprintf('\n=== RBPF POST-PROCESSING VISUALIZATION ===\n');

        % Create RBPF-specific save directory with particle count
        rbpf_save_dir = fullfile("..", "figures", "DA_Track", "KFRBPF_tuning", sprintf('%dparticles', current_class.N_p));

        if ~exist(rbpf_save_dir, 'dir')
            mkdir(rbpf_save_dir);
            fprintf('Created RBPF tuning save directory: %s\n', rbpf_save_dir);
        end

        % Generate summary plot showing tracking performance
        fprintf('Creating RBPF tracking summary plot...\n');

        try
            figure('Name', 'RBPF Tracking Summary', 'Position', [100, 100, 1400, 900]);

            % Extract estimates and ground truth from performance
            % State is 6D: [x, y, vx, vy, ax, ay]
            N_states = size(performance{1}.x, 1);
            estimates = zeros(N_states, n_k);

            for i = 1:n_k
                estimates(:, i) = performance{i}.x;
            end

            % Compute errors (position and velocity only)
            position_errors = GT(1:2, :) - estimates(1:2, :);
            position_errors_norm = sqrt(sum(position_errors .^ 2, 1));
            position_rmse = sqrt(mean(sum(position_errors .^ 2, 1)));

            velocity_errors = GT(3:4, :) - estimates(3:4, :);
            velocity_errors_norm = sqrt(sum(velocity_errors .^ 2, 1));
            velocity_rmse = sqrt(mean(sum(velocity_errors .^ 2, 1)));

            % Use the dt from tuning_main (defined at line 132)
            time = (1:n_k) * dt;

            % Subplot 1: Trajectory
            subplot(2, 3, 1);
            plot(GT(1, :), GT(2, :), 'g-', 'LineWidth', 2, 'DisplayName', 'True');
            hold on;
            plot(estimates(1, :), estimates(2, :), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimate');
            xlabel('X (m)'); ylabel('Y (m)');
            title('Trajectory');
            legend('Location', 'best'); grid on; axis equal;

            % Subplot 2: Position Error
            subplot(2, 3, 2);
            plot(time, position_errors_norm, 'r-', 'LineWidth', 2);
            xlabel('Time (s)'); ylabel('Position Error (m)');
            title(sprintf('Position Error (RMSE=%.3f m)', position_rmse));
            grid on;

            % Subplot 3: Velocity Error
            subplot(2, 3, 3);
            plot(time, velocity_errors_norm, 'b-', 'LineWidth', 2);
            xlabel('Time (s)'); ylabel('Velocity Error (m/s)');
            title(sprintf('Velocity Error (RMSE=%.3f m/s)', velocity_rmse));
            grid on;

            % Subplot 4: X Position
            subplot(2, 3, 4);
            plot(time, GT(1, :), 'g-', 'LineWidth', 2, 'DisplayName', 'True');
            hold on;
            plot(time, estimates(1, :), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimate');
            xlabel('Time (s)'); ylabel('X Position (m)');
            title('X Position vs Time');
            legend; grid on;

            % Subplot 5: Y Position
            subplot(2, 3, 5);
            plot(time, GT(2, :), 'g-', 'LineWidth', 2, 'DisplayName', 'True');
            hold on;
            plot(time, estimates(2, :), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimate');
            xlabel('Time (s)'); ylabel('Y Position (m)');
            title('Y Position vs Time');
            legend; grid on;

            % Subplot 6: ESS over time
            subplot(2, 3, 6);

            if ~isempty(current_class.history)
                ESS_history = [current_class.history.ESS];
                time_history = (1:length(ESS_history)) * dt;
                plot(time_history, ESS_history, 'k-', 'LineWidth', 2);
                hold on;
                yline(current_class.N_p * current_class.ESS_threshold_percentage, 'r--', 'LineWidth', 1.5, ...
                    'Label', sprintf('Resample Threshold (%.0f%%)', current_class.ESS_threshold_percentage * 100));
                xlabel('Time (s)'); ylabel('Effective Sample Size');
                title('ESS History (Particle Diversity)');
                ylim([0, current_class.N_p]);
                grid on;
            end

            % Save summary figure
            summary_filename = fullfile(rbpf_save_dir, sprintf('RBPF_%s_summary.png', DATASET));
            saveas(gcf, summary_filename);
            fprintf('✓ Summary plot saved to: %s\n', summary_filename);

        catch ME
            warning('Failed to generate summary plot: %s', ME.message);
            fprintf('  Error in summary plot generation:\n');
            fprintf('    %s\n', ME.message);

            for j = 1:length(ME.stack)
                fprintf('    %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
            end

        end

        % Ask user if they want to generate individual GIF files
        fprintf('\n');
        fprintf('Generate and save individual GIF files for each subplot? (y/n): ');
        user_input = input('', 's');

        if strcmpi(user_input, 'y')

            try
                fprintf('Creating individual GIF files...\n');
                fprintf('This will generate separate GIFs for:\n');
                fprintf('  - position_weighted.gif\n');
                fprintf('  - velocity_weighted.gif\n');
                fprintf('  - acceleration_weighted.gif\n');
                fprintf('  - signal.gif\n');
                fprintf('  - association.gif\n');
                fprintf('  - trajectory.gif\n');
                fprintf('\n');

                % Use SaveIndividualGIFs parameter
                visualize_RBPF_history(current_class, 'Animate', true, ...
                    'SaveIndividualGIFs', true, ...
                    'GIFDirectory', rbpf_save_dir, ...
                    'AnimationSpeed', 0.1, ...
                    'PlotMargin', 0.05, ...
                    'SignalData', signal, ...
                    'PlotTrajectories', true, ...
                    'MaxParticlesToPlot', inf);

                fprintf('\n✓ All individual GIF files saved to: %s\n', rbpf_save_dir);
            catch ME
                warning('Failed to generate individual GIF files: %s', ME.message);
                fprintf('  Error message: %s\n', ME.message);
                fprintf('  Error stack:\n');

                for j = 1:length(ME.stack)
                    fprintf('    %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
                end

            end

        else
            fprintf('Skipping individual GIF generation.\n');
        end

        fprintf('==========================================\n');
    end

    %% ========== HMM-RBPF POST-PROCESSING VISUALIZATION ==========
    if FILTER_TYPE == "HMM_RBPF"
        fprintf('\n=== HMM-RBPF POST-PROCESSING VISUALIZATION ===\n');

        % Create save directory
        hmmrbpf_save_dir = fullfile("..", "figures", "DA_Track", "HMMRBPF_tuning", ...
            sprintf('%dparticles', current_class.N_p));

        if ~exist(hmmrbpf_save_dir, 'dir')
            mkdir(hmmrbpf_save_dir);
            fprintf('Created HMM-RBPF save directory: %s\n', hmmrbpf_save_dir);
        end

        % ------ Summary Plot: 2-state (position only) ------
        fprintf('Creating HMM-RBPF tracking summary plot...\n');

        try
            figure('Name', 'HMM-RBPF Tracking Summary', 'Position', [100, 100, 1400, 600]);

            estimates = zeros(2, n_k);
            for ii = 1:n_k
                estimates(:, ii) = performance{ii}.x(1:2);
            end

            position_errors      = GT(1:2, :) - estimates;
            position_errors_norm = sqrt(sum(position_errors .^ 2, 1));
            position_rmse        = sqrt(mean(sum(position_errors .^ 2, 1)));
            time                 = (1:n_k) * dt;

            % Subplot 1: Trajectory
            subplot(1, 3, 1);
            plot(GT(1, :), GT(2, :), 'g-', 'LineWidth', 2, 'DisplayName', 'True');
            hold on;
            plot(estimates(1, :), estimates(2, :), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimate');
            xlabel('X (m)'); ylabel('Y (m)');
            xlim([-2, 2]); ylim([0.5, 4]);
            title('Trajectory'); legend('Location', 'best'); grid on; axis square;

            % Subplot 2: Position Error over time
            subplot(1, 3, 2);
            plot(time, position_errors_norm, 'r-', 'LineWidth', 2);
            xlabel('Time (s)'); ylabel('Position Error (m)');
            title(sprintf('Position Error (RMSE=%.3f m)', position_rmse));
            grid on;

            % Subplot 3: ESS over time
            subplot(1, 3, 3);
            if ~isempty(current_class.history)
                ESS_history  = [current_class.history.ESS];
                time_history = (1:length(ESS_history)) * dt;
                plot(time_history, ESS_history, 'k-', 'LineWidth', 2);
                hold on;
                yline(current_class.N_p * current_class.ESS_threshold_percentage, 'r--', 'LineWidth', 1.5, ...
                    'Label', sprintf('Resample (%.0f%%)', current_class.ESS_threshold_percentage * 100));
                xlabel('Time (s)'); ylabel('ESS');
                title('ESS History');
                ylim([0, current_class.N_p]);
                grid on;
            end

            summary_filename = fullfile(hmmrbpf_save_dir, sprintf('HMMRBPF_%s_summary.png', DATASET));
            saveas(gcf, summary_filename);
            fprintf('✓ Summary plot saved to: %s\n', summary_filename);

        catch ME
            warning('Failed to generate HMM-RBPF summary plot: %s', ME.message);
            for j = 1:length(ME.stack)
                fprintf('    %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
            end
        end

        % ------ Animated history visualization ------
        fprintf('\nGenerate animated HMM-RBPF history? (y/n): ');
        user_input = input('', 's');

        if strcmpi(user_input, 'y')
            try
                visualize_RBPFHMM_history(current_class, ...
                    'Animate', true, ...
                    'AnimationSpeed', 0.15, ...
                    'SignalData', signal, ...
                    'PlotTrajectories', true, ...
                    'MaxParticlesToPlot', inf, ...
                    'SaveFinalFigure', fullfile(hmmrbpf_save_dir, ...
                        sprintf('HMMRBPF_%s_final.png', DATASET)));
                fprintf('✓ HMM-RBPF history visualization complete.\n');
            catch ME
                warning('Failed to generate HMM-RBPF history visualization: %s', ME.message);
                for j = 1:length(ME.stack)
                    fprintf('    %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
                end
            end
        else
            fprintf('Skipping animated history.\n');
        end

        fprintf('==========================================\n');
    end

    %% ========== MC_PF (PDA_PF) POST-PROCESSING VISUALIZATION ==========
    if FILTER_TYPE == "MC_PF" && isprop(current_class, 'history') && ~isempty(current_class.history)
        fprintf('\n=== PDA_PF POST-PROCESSING VISUALIZATION ===\n');

        % Create PDA_PF-specific save directory with particle count
        mcpf_save_dir = fullfile("..", "figures", "DA_Track", "PDA_PF_tuning", sprintf('%dparticles', current_class.N_p));

        if ~exist(mcpf_save_dir, 'dir')
            mkdir(mcpf_save_dir);
            fprintf('Created PDA_PF tuning save directory: %s\n', mcpf_save_dir);
        end

        % Get signal data for visualization
        signal = measurements.signal;

        % Generate summary plot showing tracking performance (matching RBPF format exactly)
        fprintf('Creating PDA_PF tracking summary plot...\n');

        try
            figure('Name', 'PDA_PF Tracking Summary', 'Position', [100, 100, 1400, 900]);

            % Extract estimates and ground truth from performance
            % State is 6D: [x, y, vx, vy, ax, ay]
            N_states = size(performance{1}.x, 1);
            estimates = zeros(N_states, n_k);

            for i = 1:n_k
                estimates(:, i) = performance{i}.x;
            end

            % Compute errors (position and velocity only)
            position_errors = GT(1:2, :) - estimates(1:2, :);
            position_errors_norm = sqrt(sum(position_errors .^ 2, 1));
            position_rmse = sqrt(mean(sum(position_errors .^ 2, 1)));

            velocity_errors = GT(3:4, :) - estimates(3:4, :);
            velocity_errors_norm = sqrt(sum(velocity_errors .^ 2, 1));
            velocity_rmse = sqrt(mean(sum(velocity_errors .^ 2, 1)));

            % Use the dt from tuning_main (defined at line 132)
            time = (1:n_k) * dt;

            % Subplot 1: Trajectory
            subplot(2, 3, 1);
            plot(GT(1, :), GT(2, :), 'g-', 'LineWidth', 2, 'DisplayName', 'True');
            hold on;
            plot(estimates(1, :), estimates(2, :), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimate');
            xlabel('X (m)'); ylabel('Y (m)');
            title('Trajectory');
            legend('Location', 'best'); grid on; axis equal;

            % Subplot 2: Position Error
            subplot(2, 3, 2);
            plot(time, position_errors_norm, 'r-', 'LineWidth', 2);
            xlabel('Time (s)'); ylabel('Position Error (m)');
            title(sprintf('Position Error (RMSE=%.3f m)', position_rmse));
            grid on;

            % Subplot 3: Velocity Error
            subplot(2, 3, 3);
            plot(time, velocity_errors_norm, 'b-', 'LineWidth', 2);
            xlabel('Time (s)'); ylabel('Velocity Error (m/s)');
            title(sprintf('Velocity Error (RMSE=%.3f m/s)', velocity_rmse));
            grid on;

            % Subplot 4: X Position
            subplot(2, 3, 4);
            plot(time, GT(1, :), 'g-', 'LineWidth', 2, 'DisplayName', 'True');
            hold on;
            plot(time, estimates(1, :), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimate');
            xlabel('Time (s)'); ylabel('X Position (m)');
            title('X Position vs Time');
            legend; grid on;

            % Subplot 5: Y Position
            subplot(2, 3, 5);
            plot(time, GT(2, :), 'g-', 'LineWidth', 2, 'DisplayName', 'True');
            hold on;
            plot(time, estimates(2, :), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimate');
            xlabel('Time (s)'); ylabel('Y Position (m)');
            title('Y Position vs Time');
            legend; grid on;

            % Subplot 6: ESS over time
            subplot(2, 3, 6);

            if ~isempty(current_class.history)
                ESS_history = [current_class.history.ESS];
                time_history = (1:length(ESS_history)) * dt;
                plot(time_history, ESS_history, 'k-', 'LineWidth', 2);
                hold on;
                yline(current_class.N_p * current_class.ESS_threshold_percentage, 'r--', 'LineWidth', 1.5, ...
                    'Label', sprintf('Resample Threshold (%.0f%%)', current_class.ESS_threshold_percentage * 100));
                xlabel('Time (s)'); ylabel('Effective Sample Size');
                title('ESS History (Particle Diversity)');
                ylim([0, current_class.N_p]);
                grid on;
            end

            % Save summary figure
            summary_filename = fullfile(mcpf_save_dir, sprintf('PDA_PF_%s_summary.png', DATASET));
            saveas(gcf, summary_filename);
            fprintf('✓ Summary plot saved to: %s\n', summary_filename);

        catch ME
            warning('Failed to generate summary plot: %s', ME.message);
            fprintf('  Error in summary plot generation:\n');
            fprintf('    %s\n', ME.message);

            for j = 1:length(ME.stack)
                fprintf('    %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
            end

        end

        % Ask user if they want to generate individual GIF files
        fprintf('\n');
        fprintf('Generate and save individual GIF files for each subplot? (y/n): ');
        user_input = input('', 's');

        if strcmpi(user_input, 'y')

            try
                fprintf('Creating individual GIF files...\n');
                fprintf('This will generate separate GIFs for:\n');
                fprintf('  - position_weighted.gif\n');
                fprintf('  - velocity_weighted.gif\n');
                fprintf('  - acceleration_weighted.gif\n');
                fprintf('  - signal.gif\n');
                fprintf('  - association.gif\n');
                fprintf('  - trajectory.gif\n');
                fprintf('\n');

                % Use SaveIndividualGIFs parameter (visualize_RBPF_history works with MC_PF!)
                visualize_RBPF_history(current_class, 'Animate', true, ...
                    'SaveIndividualGIFs', true, ...
                    'GIFDirectory', mcpf_save_dir, ...
                    'AnimationSpeed', 0.1, ...
                    'PlotMargin', 0.05, ...
                    'SignalData', signal, ...
                    'PlotTrajectories', true, ...
                    'MaxParticlesToPlot', inf);

                fprintf('\n✓ All individual GIF files saved to: %s\n', mcpf_save_dir);
            catch ME
                warning('Failed to generate individual GIF files: %s', ME.message);
                fprintf('  Error message: %s\n', ME.message);
                fprintf('  Error stack:\n');

                for j = 1:length(ME.stack)
                    fprintf('    %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
                end

            end

        else
            fprintf('Skipping individual GIF generation.\n');
        end

        fprintf('==========================================\n');
    end

    %% ========== HYBRID PF POST-PROCESSING VISUALIZATION ==========
    if FILTER_TYPE == "HybridPF" && isprop(current_class, 'history') && ~isempty(current_class.history)
        fprintf('\n=== HYBRID PF POST-PROCESSING VISUALIZATION ===\n');
        fprintf('Generate and save individual RBPF-style GIF files for this PF run? (y/n): ');
        user_input = input('', 's');

        if strcmpi(user_input, 'y')

            try
                pf_save_dir = fullfile("..", "figures", "DA_Track", "PF_tuning_history");

                if ~exist(pf_save_dir, 'dir')
                    mkdir(pf_save_dir);
                end

                visualize_PF_history(current_class, 'Animate', true, ...
                    'SaveIndividualGIFs', true, ...
                    'GIFDirectory', pf_save_dir, ...
                    'AnimationSpeed', 0.1, ...
                    'PlotMargin', 0.05, ...
                    'SignalData', signal, ...
                    'PlotTrajectories', true, ...
                    'MaxParticlesToPlot', inf);

                fprintf('✓ PF history GIF files saved to: %s\n', pf_save_dir);
            catch ME
                warning('Failed to generate PF history GIF files: %s', ME.message);
            end

        else
            fprintf('Skipping PF history GIF generation.\n');
        end

        fprintf('===============================================\n');
    end

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
