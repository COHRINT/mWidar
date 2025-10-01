function main(varargin)
    % MAIN - Run mWidar tracking analysis
    % Usage:
    %   main()                                    % Uses all defaults
    %   main('T2_far')                           % Uses specified dataset
    %   main('T3_border', 'KF')                  % Uses dataset and filter type
    %   main('T4_parab', 'HybridPF', 'DA', 'GNN', 'FinalPlot', 'interactive', 'Debug', true, 'DynamicPlot', false, 'InitializeTrue', true)
    %
    % Parameters:
    %   DATASET (positional 1)    - Dataset name (default: "T4_parab")
    %   FILTER_TYPE (positional 2) - Filter type: "KF", "HybridPF", or "HMM" (default: "HybridPF")
    %   DA (name-value)           - Data association: "PDA" or "GNN" (default: "PDA")
    %   FinalPlot (name-value)    - Plot mode: "interactive", "animation", or "none" (default: "none")
    %   Debug (name-value)        - Enable debug output: true or false (default: false)
    %   DynamicPlot (name-value)  - Enable dynamic plotting during filtering: true or false (default: false)
    %   InitializeTrue (name-value) - Enable true/uninformative initialization: true or false (default: true)

    %% ========== COMMAND LINE ARGUMENT PARSING ==========

    %% --- Parse Positional Arguments ---
    % Parse positional arguments first
    if nargin >= 1 && ~ischar(varargin{1}) && ~isstring(varargin{1})
        error('First argument (DATASET) must be a string');
    elseif nargin >= 1 && ~startsWith(varargin{1}, {'DA', 'FinalPlot', 'Debug', 'DynamicPlot'})
        DATASET = string(varargin{1});
        start_named_args = 2;
    else
        DATASET = "T4_parab"; % Default dataset
        start_named_args = 1;
    end

    %% --- Parse Filter Type Argument ---
    % Parse second positional argument (filter type)
    if nargin >= start_named_args && ~startsWith(varargin{start_named_args}, {'DA', 'FinalPlot', 'Debug', 'DynamicPlot'})
        filter_type = string(varargin{start_named_args});
        start_named_args = start_named_args + 1;
    else
        filter_type = "HybridPF"; % Default filter type
    end

    %% --- Parse Name-Value Pairs ---
    % Parse name-value pairs
    remaining_args = varargin(start_named_args:end);

    %% --- Parameter Defaults ---
    DA = "PDA";
    FinalPlot = "none";
    DEBUG = false;
    DYNAMIC_PLOT = false;
    INITIALIZE_TRUE = true;

    %% --- Process Name-Value Pairs ---
    % Parse name-value pairs
    for i = 1:2:length(remaining_args)

        if i + 1 > length(remaining_args)
            error('Name-value pairs must come in pairs. Missing value for "%s"', remaining_args{i});
        end

        param_name = remaining_args{i};
        param_value = remaining_args{i + 1};

        switch lower(param_name)
            case 'da'
                DA = string(param_value);
            case 'finalplot'
                FinalPlot = string(param_value);
            case 'debug'
                DEBUG = logical(param_value);
            case 'dynamicplot'
                DYNAMIC_PLOT = logical(param_value);
            case 'initializetrue'
                INITIALIZE_TRUE = logical(param_value);
            otherwise
                error('Unknown parameter: %s', param_name);
        end

    end

    %% --- Parameter Validation ---
    % Validate dataset
    valid_datasets = ["T1_near", "T2_far", "T3_border", "T4_parab", "T5_parab_noise"];

    if ~ismember(DATASET, valid_datasets)
        error('Invalid dataset. Options: %s', strjoin(valid_datasets, ', '));
    end

    % Validate filter type
    valid_filters = ["KF", "HybridPF", "HMM"];

    if ~ismember(filter_type, valid_filters)
        error('Invalid filter type. Options: %s', strjoin(valid_filters, ', '));
    end

    % Validate DA method
    valid_da_methods = ["PDA", "GNN"];

    if ~ismember(DA, valid_da_methods)
        error('Invalid DA method. Options: %s', strjoin(valid_da_methods, ', '));
    end

    % Validate FinalPlot mode
    valid_plot_modes = ["interactive", "animation", "none"];

    if ~ismember(FinalPlot, valid_plot_modes)
        error('Invalid FinalPlot mode. Options: %s', strjoin(valid_plot_modes, ', '));
    end

    %% --- Display Configuration ---
    fprintf('Using dataset: %s\n', DATASET);
    fprintf('Using filter type: %s\n', filter_type);
    fprintf('Using DA method: %s\n', DA);
    fprintf('Using plot mode: %s\n', FinalPlot);
    fprintf('Debug enabled: %s\n', string(DEBUG));
    fprintf('Dynamic plot enabled: %s\n', string(DYNAMIC_PLOT));

    %% ========== WORKSPACE SETUP ==========
    %% --- Environment Configuration ---
    clc; close all
    % Add paths for MATLAB functions
    addpath(fullfile('DA_Track'))
    addpath(fullfile('supplemental'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObj'))

    %% --- Load Supplemental Data ---
    % Load necessary supplemental data
    % load(fullfile('supplemental', 'recovery.mat'))
    % load(fullfile('supplemental', 'sampling.mat'))
    load(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObj', DATASET + '.mat'), 'Data');

    %% --- Configure Plotting Environment ---
    % Set default plotting settings -- in startup.m now
    % setDefaultPlotSettings();

    % Set plotting flags based on FinalPlot parameter
    INTERACTIVE = (FinalPlot == "interactive");
    ANIMATION = (FinalPlot == "animation");

    %% --- Initialize Variables ---
    % Load Data
    GT = Data.GT;
    GT_meas = GT(1:2, :);
    z = Data.y;
    signal = Data.signal;

    % Store original measurements for visualization (before any filtering)
    Data.y_original = Data.y; % Keep original measurements for plotting

    n_k = size(GT, 2);
    performance = cell(1, n_k);

    % Create measurements struct for composite likelihood (HybridPF)
    measurements = struct();
    measurements.z = Data.y;        % Cell array of detection measurements [2 x N_meas] for each timestep
    measurements.signal = signal;   % Cell array of mWidar signals [128 x 128] for each timestep

    % Initialize validation sigma parameter (will be set based on filter type)
    validation_sigma = 2; % Default value

    %% ========== FILTER PARAMETERS SETUP ==========
    %% --- Basic Parameters ---
    dt = 0.1; % sec

    %% --- Define Kalman Filter Matrices ---
    % Define KF Matrices state vector - {x,y,vx,vy,ax,ay}
    % Correct continuous-time dynamics matrix for constant acceleration model
    A = [0, 0, 1, 0, 0, 0;
         0, 0, 0, 1, 0, 0;
         0, 0, 0, 0, 1, 0;
         0, 0, 0, 0, 0, 1;
         0, 0, 0, 0, 0, 0;
         0, 0, 0, 0, 0, 0];

    F_KF = expm(A * dt);

    %% --- Define Particle Filter Matrices ---
    % Use the direct discrete-time formulation (matches test_hybrid_PF)
    F_PF = [1, 0, dt, 0, dt ^ 2/2, 0; % x
            0, 1, 0, dt, 0, dt ^ 2/2; % y
            0, 0, 1, 0, dt, 0; % vx
            0, 0, 0, 1, 0, dt; % vy
            0, 0, 0, 0, 1, 0; % ax
            0, 0, 0, 0, 0, 1]; % ay

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
    switch filter_type

        case "KF"
            %% --- Kalman Filter Setup ---
            fprintf("Using Kalman Filter ")

            % Handle true initialization for KF
            if INITIALIZE_TRUE
                % True initialization: zero mean with very large covariance
                initial_state = zeros(size(GT(:, 1))); % Zero mean for all states
                initial_covariance = diag([100, 100, 10, 10, 5, 5]); % Very large covariance

                if DEBUG
                    fprintf("with TRUE INITIALIZATION (zero mean, large covariance)\n");
                end

            else
                % Standard initialization using ground truth
                initial_state = GT(:, 1);
                initial_covariance = P0;

                if DEBUG
                    fprintf("with STANDARD INITIALIZATION (GT-based)\n");
                end

            end

            if DA == "PDA"
                current_class = PDA_KF(initial_state, initial_covariance, F_KF, Q, R, H, 'Debug', DEBUG, "DynamicPlot", DYNAMIC_PLOT);

            elseif DA == "GNN"
                current_class = GNN_KF(initial_state, initial_covariance, F_KF, Q, R, H, 'Debug', DEBUG, "DynamicPlot", DYNAMIC_PLOT);
            else
                error('Unknown data association method: %s', DA);
            end

            fprintf("with %s data association\n", DA);

            %% --- Initialize Performance Storage ---
            [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate(); % Initial Gaussian estimate

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

            if DA == "PDA"
                fprintf("with PDA data association\n");
                validation_sigma = 5; % Set validation sigma for PDA
                current_class = PDA_PF(GT(:, 1), 10000, F_PF, Q_PF, H, pointlikelihood_image, pointlikelihood_mag, "Debug", DEBUG, "DynamicPlot", DYNAMIC_PLOT, "ValidationSigma", validation_sigma, "UniformInit", INITIALIZE_TRUE);
                current_class.setDetectionModel(0.99, 0.2); % Set default detection model parameters
                current_class.ESS_threshold_percentage = .10;
                % Enable composite likelihood mode for comprehensive visualization
                current_class.composite_likelihood = true;
                % Set GIF filename if desired (uncomment to enable GIF output)
                % current_class.gif_filename = 'main_output.gif';

            elseif DA == "GNN"
                fprintf("with GNN data association\n");
                Q_PF = Q;
                validation_sigma = 5; % Set validation sigma for GNN
                current_class = GNN_PF(GT(:, 1), 10000, F_PF, Q_PF, H, pointlikelihood_image, "Debug", DEBUG, "DynamicPlot", DYNAMIC_PLOT, "ValidationSigma", validation_sigma, "UniformInit", INITIALIZE_TRUE);
            else
                error('Unknown data association method: %s', DA);
            end

            %% --- Initialize Performance Storage ---
            performance{1}.particles = current_class.particles; % Store initial particles
            performance{1}.weights = current_class.weights; % Store initial weights
            [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate(); % Initial Gaussian estimate
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

            if DA == "PDA"
                fprintf("with PDA data association\n");
                validation_sigma = 5; % Set validation sigma for PDA
                current_class = PDA_HMM(initial_position, A_slow, pointlikelihood_image, "Debug", DEBUG, "DynamicPlot", DYNAMIC_PLOT, "ValidationSigma", validation_sigma);

                current_class.setDetectionModel(0.99, 0.2); % Set default detection model parameters
                current_class.ESS_threshold_percentage = .10;
                
                % Enable composite likelihood mode for comprehensive visualization
                current_class.composite_likelihood = true;
                fprintf('-> Enabled composite likelihood mode for comprehensive visualization\n');
                current_class.gif_filename = 'coolgifTraj.gif'; % No GIF by default

            elseif DA == "GNN"
                fprintf("with GNN data association\n");
                % Note: GNN_HMM would need to be implemented if desired
                validation_sigma = 5; % Set validation sigma for GNN
                current_class = GNN_HMM(initial_position, A_slow, pointlikelihood_image, "Debug", DEBUG, "DynamicPlot", DYNAMIC_PLOT, "ValidationSigma", validation_sigma);

            else
                error('Unknown data association method: %s', DA);
            end

            % Override with uniform distribution if true initialization is requested
            if INITIALIZE_TRUE
                current_class.ptarget_prob = ones(current_class.npx2, 1) / current_class.npx2;

                if DEBUG
                    fprintf("-> Overrode with uniform distribution over entire field\n");
                end

            end

            %% --- Initialize Performance Storage ---
            [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate(); % Initial Gaussian estimate
            performance{1}.prior_prob = current_class.ptarget_prob; % Store initial distribution
            performance{1}.likelihood_prob = []; % No likelihood at initialization
            performance{1}.posterior_prob = current_class.ptarget_prob; % Same as initial distribution

            % Store initial measurements (empty for first timestep)
            performance{1}.measurements_original = [];
            performance{1}.measurements_used = [];

        otherwise
            error('Unknown filter type: %s', filter_type);

    end

    %% ========== STATE ESTIMATION LOOP ==========

    for i = 2:n_k
        fprintf('Processing time step %d/%d\n', i, n_k);

        %% --- Filter-Specific Timestep Processing ---
        switch filter_type
            case "KF"
                %% --- Kalman Filter Timestep ---
                current_meas = z{i}; % Use original measurements
                performance{i}.measurements_original = current_meas;
                performance{i}.measurements_used = current_meas; % KF uses all measurements
                current_class.timestep(current_meas, GT(:, i)); % Pass ground truth for visualization
                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();

            case 'HybridPF'
                %% --- Hybrid Particle Filter Timestep ---
                current_meas_raw = measurements.z{i};
                current_signal = measurements.signal{i};
                performance{i}.measurements_original = current_meas_raw;
                [measurements_used, ~] = current_class.Validation(current_meas_raw);
                performance{i}.measurements_used = measurements_used;
                measurement_struct = struct();
                measurement_struct.det = current_meas_raw;
                measurement_struct.mag = current_signal;
                if DEBUG && ~isempty(current_meas_raw)
                    fprintf('  Step %d: %d original measurements, %d used measurements, signal size %dx%d\n', ...
                        i, size(current_meas_raw, 2), size(measurements_used, 2), size(current_signal, 1), size(current_signal, 2));
                end
                current_class.timestep(measurement_struct, GT(:, i));
                performance{i}.particles = current_class.particles;
                performance{i}.weights = current_class.weights;
                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();

            case 'HMM'
                %% --- Hidden Markov Model Timestep ---
                current_meas = z{i};
                performance{i}.measurements_original = current_meas;
                [measurements_used, ~] = current_class.Validation(current_meas);
                performance{i}.measurements_used = measurements_used;
                if DEBUG && ~isempty(current_meas)
                    fprintf('  Step %d: %d original measurements, %d used measurements\n', ...
                        i, size(current_meas, 2), size(measurements_used, 2));
                end
                current_class.timestep(current_meas, GT(1:2, i));
                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();
                performance{i}.prior_prob = current_class.prior_prob;
                performance{i}.likelihood_prob = current_class.likelihood_prob;
                performance{i}.posterior_prob = current_class.posterior_prob;

            otherwise
                error('Unknown filter type in timestep loop: %s', filter_type);
        end
    end % End of for loop

    %% ========== FINAL PLOTTING AND VISUALIZATION ==========
    if INTERACTIVE
        %% --- Interactive Plotting ---
        fprintf('Generating interactive plot...\n');
        mWidar_FilterPlot_Interactive(performance, Data, 0:dt:10, filter_type, validation_sigma); % Interactive plotting function with slider
        fprintf('Interactive plot is ready! Use the slider and controls to navigate through timesteps.\n');
        fprintf('Close the figure window when you are done.\n');
    elseif ANIMATION
        %% --- Animation Plotting ---
        fprintf('Generating animation plot...\n');
        saveString = filter_type + "_" + DATASET + "_" + DA + ".gif";

        % Animation plotting function
        mWidar_FilterPlot_Distribution(performance, Data, 0:dt:10, filter_type, validation_sigma, fullfile("..", "figures", "DA_Track", saveString));
    else
        %% --- No Plotting Mode ---
        fprintf('No final plotting requested (FinalPlot = "none").\n');
    end

%% ========== FINAL SUMMARY ==========
    fprintf('\n=== EXPERIMENT COMPLETED ===\n');
    fprintf('Filter: %s-%s\n', filter_type, DA);
    fprintf('Dataset: %s\n', DATASET);
    fprintf('Timesteps processed: %d\n', n_k);
    fprintf('Performance metrics stored for all timesteps: Yes\n');

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
        gif_filename = sprintf('%s_%s_%s_captured.gif', filter_type, DATASET, DA);
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
        fprintf('Generating final animation...\n');
        
        % Create traj_plots directory if it doesn't exist
        traj_plots_dir = fullfile("..", "figures", "DA_Track", "traj_plots");
        if ~exist(traj_plots_dir, 'dir')
            mkdir(traj_plots_dir);
            fprintf('Created trajectory plots directory: %s\n', traj_plots_dir);
        end
        
        saveString = filter_type + "_" + DATASET + "_" + DA + ".gif";
        animation_path = fullfile(traj_plots_dir, saveString);

        % Animation plotting function - use actual simulation time vector
        time_vector = 0:dt:(n_k-1)*dt;
        mWidar_FilterPlot_Distribution(performance, Data, time_vector, filter_type, validation_sigma, animation_path);
        fprintf('Final animation saved to: %s\n', animation_path);
    end

    fprintf('===================================\n');



    %% --- Future Enhancements ---
    % NEES(current_class,initial_state,A,10,M,G)

end
