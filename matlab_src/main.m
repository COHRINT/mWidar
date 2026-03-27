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
    %   FILTER_TYPE (positional 2) - Filter type: "KF", "HybridPF", "HMM", or "RBPF" (default: "HybridPF")
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
        filter_type = "RBPF"; % Default filter type
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
    valid_datasets = ["T1_near", "T2_far", "T3_border", "T4_parab", "T5_parab_noise", "T1_mult_1", "T2_mult_2", "T3_mult_3"];

    if ~ismember(DATASET, valid_datasets)
        error('Invalid dataset. Options: %s', strjoin(valid_datasets, ', '));
    end

    % Validate filter type
    valid_filters = ["KF", "HybridPF", "HMM", "RBPF"];

    if ~ismember(filter_type, valid_filters)
        error('Invalid filter type. Options: %s', strjoin(valid_filters, ', '));
    end

    % Validate DA method
    valid_da_methods = ["PDA", "GNN", "MC"];

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
    addpath(fullfile('DA_Track'))                          % base: DA_Filter, KF, HMM
    addpath(fullfile('DA_Track', 'single'))                % single-target filters
    addpath(fullfile('supplemental'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObj'))

    %% --- Load Supplemental Data ---
    % Load necessary supplemental data
    % load(fullfile('supplemental', 'recovery.mat'))
    % load(fullfile('supplemental', 'sampling.mat'))

    load(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObj', DATASET + '.mat'), 'Data');
    % load(fullfile('supplemental', "Final_Test_Tracks", "MultiObj", 'JPDAF_test_traj.mat'), 'Data');

    %% --- Configure Plotting Environment ---
    % Set default plotting settings -- in startup.m now
    % setDefaultPlotSettings();

    % Set plotting flags based on FinalPlot parameter
    INTERACTIVE = (FinalPlot == "interactive");
    ANIMATION = (FinalPlot == "animation");

    %% --- Initialize Variables ---
    % Load Data
    GT = Data.GT;
    fprintf('Loaded dataset with %d objects.\n', size(GT, 2));
    size(GT)
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
    dt = 0.1; % sec

    % Discrete-time dynamics matrices (constant-acceleration model)
    F_KF = expm([0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0;
                 0 0 0 0 0 1; 0 0 0 0 0 0; 0 0 0 0 0 0] * dt);
    F_PF = [1 0 dt 0 dt^2/2 0; 0 1 0 dt 0 dt^2/2;
            0 0 1  0 dt     0; 0 0 0 1  0 dt;
            0 0 0  0 1      0; 0 0 0 0  0 1];

    % Per-family noise tuning (matches previous per-case values)
    Q_KF   = diag([1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2]);
    Q_PF   = diag([1e-3, 1e-3, 1e-2, 1e-2, 1e-1, 1e-1]);
    Q_RBPF = diag([1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3]);
    R      = 0.1 * eye(2);
    H      = [1 0 0 0 0 0; 0 1 0 0 0 0];
    P0     = diag([0.1 0.1 0.25 0.25 0.5 0.5]);

    %% ========== FILTER INITIALIZATION ==========

    % --- Map (filter_type, DA) to concrete FilterConfig name ---
    filter_family_map = containers.Map(...
        {'KF_PDA',       'KF_GNN',       ...
         'HybridPF_PDA', 'HybridPF_GNN', 'HybridPF_MC', ...
         'HMM_PDA',      'HMM_GNN',      ...
         'RBPF_PDA',     'RBPF_GNN'}, ...
        {'PDA_KF',  'GNN_KF', ...
         'PDA_PF',  'GNN_PF', 'MC_PF', ...
         'PDA_HMM', 'GNN_HMM', ...
         'KF_RBPF', 'KF_RBPF'});
    filter_config_name = filter_family_map(char(filter_type + "_" + DA));
    fprintf('Constructing %s filter...\n', filter_config_name);

    % --- Load supplemental matrices when needed ---
    A_transition = []; pointlikelihood_image = []; pointlikelihood_mag = [];
    if ismember(filter_config_name, {'PDA_HMM', 'GNN_HMM'})
        load(fullfile('supplemental', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');
        load(fullfile('supplemental', 'precalc_imagegridHMMSTMn15.mat'), 'A');
        A_transition = A; clear A;
    elseif ismember(filter_config_name, {'PDA_PF', 'MC_PF'})
        load(fullfile('supplemental', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');
        tmp = load(fullfile('supplemental', 'precalc_imagegridHMMEmLikeMag.mat'), 'pointlikelihood_image');
        pointlikelihood_mag = tmp.pointlikelihood_image;
    elseif strcmp(filter_config_name, 'GNN_PF')
        load(fullfile('supplemental', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');
    end

    % --- Choose initial state and covariance ---
    if INITIALIZE_TRUE
        x0 = zeros(6, 1);
        x0(1:2) = GT(1:2, 1);  % Known position, zero velocity/acceleration
        P0_init = diag([100, 100, 10, 10, 5, 5]);
    else
        x0 = GT(:, 1);
        P0_init = P0;
    end
    % HMM family operates on 2D position only
    if ismember(filter_config_name, {'PDA_HMM', 'GNN_HMM'})
        x0 = x0(1:2);
    end

    % --- Per-family parameters for FilterConfig ---
    N_particles_map = containers.Map(...
        {'PDA_KF','GNN_KF','PDA_HMM','GNN_HMM','GNN_PF','PDA_PF','MC_PF','KF_RBPF','HMM_RBPF'}, ...
        {NaN, NaN, NaN, NaN, 10000, 1000, 10000, 100, 100});
    Q_map = containers.Map(...
        {'PDA_KF','GNN_KF','PDA_HMM','GNN_HMM','GNN_PF','PDA_PF','MC_PF','KF_RBPF','HMM_RBPF'}, ...
        {Q_KF, Q_KF, [], [], Q_PF, Q_PF, Q_PF, Q_RBPF, Q_RBPF});
    F_map = containers.Map(...
        {'PDA_KF','GNN_KF','PDA_HMM','GNN_HMM','GNN_PF','PDA_PF','MC_PF','KF_RBPF','HMM_RBPF'}, ...
        {F_KF, F_KF, [], [], F_PF, F_PF, F_PF, F_KF, F_KF});
    validation_sigma = 5; % Broad gate for PF/HMM families; KF uses chi2 internally

    % --- Build FilterConfig and override matrices ---
    cfg = FilterConfig(filter_config_name, ...
        'dt', dt, 'Debug', DEBUG, 'DynamicPlot', DYNAMIC_PLOT, ...
        'ValidationSigma', validation_sigma, ...
        'store_full_history', false);
    % Override F and Q with tuned values if this filter uses them
    if isfield(cfg, 'F'), cfg.F = F_map(filter_config_name); end
    if isfield(cfg, 'Q'), cfg.Q = Q_map(filter_config_name); end
    if isfield(cfg, 'R'), cfg.R = R; end
    if isfield(cfg, 'H'), cfg.H = H; end
    if isfield(cfg, 'N_particles')
        cfg.N_particles = N_particles_map(filter_config_name);
    end
    cfg.ESS_threshold = 0.2;

    % --- Construct filter ---
    current_class = FilterFactory(cfg, x0, P0_init, ...
        A_transition, pointlikelihood_image, pointlikelihood_mag);

    % --- Post-construction filter-specific tuning ---
    if ismember(filter_config_name, {'PDA_PF', 'MC_PF'})
        current_class.setDetectionModel(0.99, 0.25);
        if strcmp(filter_config_name, 'PDA_PF')
            current_class.hybrid_resample_fraction = 0.9;
        else
            current_class.hybrid_resample_fraction = 0.99;
        end
        current_class.composite_likelihood = true;
    end
    if strcmp(filter_config_name, 'PDA_HMM')
        if ismethod(current_class, 'setDetectionModel')
            current_class.setDetectionModel(0.99, 0.2);
        end
        current_class.composite_likelihood = true;
    end
    if strcmp(filter_config_name, 'KF_RBPF')
        current_class.association_strategy = 'uniform';
    end
    % HMM uniform init override
    if ismember(filter_config_name, {'PDA_HMM', 'GNN_HMM'}) && INITIALIZE_TRUE
        current_class.ptarget_prob = ones(current_class.npx2, 1) / current_class.npx2;
    end

    % --- Initialize performance storage (filter-family-specific) ---
    [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate();
    performance{1}.measurements_original = [];
    performance{1}.measurements_used = [];
    if isprop(current_class, 'particles')
        performance{1}.particles = current_class.particles;
        performance{1}.weights   = current_class.weights;
        performance{1}.N_p       = current_class.N_p;
        performance{1}.ESS       = [];
        if ismethod(current_class, 'getAssociationDistribution')
            performance{1}.association_prob = current_class.getAssociationDistribution();
        end
    end
    if isprop(current_class, 'ptarget_prob')
        performance{1}.prior_prob     = current_class.ptarget_prob;
        performance{1}.likelihood_prob = [];
        performance{1}.posterior_prob = current_class.ptarget_prob;
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
                % current_class.timestep(measurements.z{i}, GT(:, i));
                performance{i}.particles = current_class.particles;
                performance{i}.weights = current_class.weights;
                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();
                performance{i}.association_prob = current_class.getAssociationDistribution();

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

            case 'RBPF'
                %% --- Rao-Blackwellized Particle Filter Timestep ---
                current_meas = z{i};

                performance{i}.measurements_original = current_meas;
                performance{i}.measurements_used = current_meas;

                if DEBUG && ~isempty(current_meas)
                    fprintf('  Step %d: %d measurements\n', i, size(current_meas, 2));
                end

                current_class.timestep(current_meas, GT(:, i));

                [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();
                performance{i}.particles = current_class.particles;
                performance{i}.N_p       = current_class.N_p;
                performance{i}.ESS       = current_class.current_ESS; % Set in timestep() before resampling

            otherwise
                error('Unknown filter type in timestep loop: %s', filter_type);
        end
    end % End of for loop

    %% ========== RBPF POST-PROCESSING VISUALIZATION ==========
    if filter_type == "RBPF"
        fprintf('\n=== RBPF POST-PROCESSING VISUALIZATION ===\n');
        
        % Create RBPF-specific save directory
        rbpf_save_dir = fullfile("..", "figures", "DA_Track", "KFRBPF_plots");
        if ~exist(rbpf_save_dir, 'dir')
            mkdir(rbpf_save_dir);
            fprintf('Created RBPF plots save directory: %s\n', rbpf_save_dir);
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
            
            % Subplot 6: ESS over time (collected from performance struct)
            subplot(2, 3, 6);
            ESS_vec = cellfun(@(p) p.ESS, performance(2:end), 'UniformOutput', true);
            ESS_vec = ESS_vec(~isnan(ESS_vec));
            if ~isempty(ESS_vec)
                plot((1:length(ESS_vec)) * dt, ESS_vec, 'k-', 'LineWidth', 2);
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
                                       'SignalData', measurements.signal, ...
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

    %% ========== PF HISTORY POST-PROCESSING VISUALIZATION ==========
    if filter_type == "HybridPF" && isprop(current_class, 'history') && ~isempty(current_class.history)
        fprintf('\n=== PF HISTORY VISUALIZATION ===\n');
        fprintf('Generate individual RBPF-style GIFs (position/velocity/acceleration/signal/association/trajectory)? (y/n): ');
        user_input = input('', 's');

        if strcmpi(user_input, 'y')
            try
                history_gif_dir = fullfile("..", "figures", "DA_Track", "pf_history_gifs");
                if ~exist(history_gif_dir, 'dir')
                    mkdir(history_gif_dir);
                end

                visualize_PF_history(current_class, ...
                    'Animate', true, ...
                    'SaveIndividualGIFs', true, ...
                    'GIFDirectory', history_gif_dir, ...
                    'AnimationSpeed', 0.1, ...
                    'PlotMargin', 0.05, ...
                    'SignalData', measurements.signal, ...
                    'PlotTrajectories', true, ...
                    'MaxParticlesToPlot', inf);

                fprintf('PF history GIFs saved to: %s\n', history_gif_dir);
            catch ME
                warning('Failed to generate PF history GIFs: %s', ME.message);
            end
        else
            fprintf('Skipping PF history GIF generation.\n');
        end
    end

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
