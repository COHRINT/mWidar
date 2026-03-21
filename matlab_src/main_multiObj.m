function main_multiObj(varargin)
    % MAIN for multiObj trackers
    %
    % USAGE:
    %   main_multiObj()                           % Default: JPDA on JPDAF_test_traj
    %   main_multiObj('Algorithm', 'RBPF')        % Run RBPF multi-target
    %   main_multiObj('Dataset', 'JPDAF_test_traj_2')
    %   main_multiObj('Algorithm', 'RBPF', 'NumParticles', 500)

    %% --- Parse Input Arguments ---
    p = inputParser;
    addParameter(p, 'Algorithm', 'JPDA', @(x) ismember(x, {'JPDA', 'RBPF'}));
    addParameter(p, 'Dataset', 'JPDAF_test_traj_2', @ischar);
    addParameter(p, 'NumParticles', 1000, @isnumeric); % For RBPF
    addParameter(p, 'Debug', false, @islogical);
    parse(p, varargin{:});
    
    ALGORITHM = p.Results.Algorithm;
    DATASET = p.Results.Dataset;
    N_PARTICLES = p.Results.NumParticles;
    DEBUG = p.Results.Debug;

    %% --- Environment Configuration ---
    clc; close all
    % Add paths for MATLAB functions
    addpath(fullfile('DA_Track'))
    addpath(fullfile('supplemental'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks', 'MultiObj'))

    fprintf('=== Multi-Object Tracker Test ===\n');
    fprintf('Algorithm: %s\n', ALGORITHM);
    fprintf('Dataset: %s\n', DATASET);
    if strcmp(ALGORITHM, 'RBPF')
        fprintf('Particles: %d\n', N_PARTICLES);
    end
    fprintf('================================\n\n');

    load(fullfile('supplemental', 'Final_Test_Tracks', 'MultiObj', [DATASET, '.mat']), 'Data');

     %% --- Initialize Variables ---
    % Load Data
    GT = Data.GT;
    z = Data.y;
    signal = Data.signal; % Required for JPDA

    dt = 0.1;

    n_k = size(GT{1}, 2);
    nt = size(GT,2);
    performance = cell(1, n_k);
    %% --- Define Kalman Filter Matrices ---
    % Define KF Matrices state vector - {x,y,vx,vy,ax,ay}
    A = [0 0 1 0 0 0;
         0 0 0 1 0 0;
         0 0 0 0 1 0;
         0 0 0 0 0 1;
         0 0 0 0 0 0;
         0 0 0 0 0 0];

    F_KF = expm(A * dt);

        %% --- Define Noise Matrices ---
    Q = 1e-2 * eye(6);
    Q(3,3) = 1e-6; % Set process noise for acceleration

    Q(6,6) = 1e-6; % Set process noise for acceleration

    R = 0.1 * eye(2);

    %% --- Define Observation Matrix ---
    H = [1 0 0 0 0 0;
         0 1 0 0 0 0];

    %% --- Define Initial Covariance ---
    P0 = diag([0.1 0.1 0.25 0.25 0.5 0.5]);

    %% --- Initialize Tracker Based on Algorithm ---
    switch ALGORITHM
        case 'JPDA'
            fprintf('Initializing JPDA-KF tracker...\n');
            initial_states = [];
            initial_covs = {};
            for t = 1:nt
                initial_states = [initial_states, GT{t}(:,1)];
                initial_covs = [initial_covs, P0];
            end
            
            % JPDA_KF requires pointliklihoodmag parameter (use [] for no magnitude likelihood)
            current_class = JPDA_KF(initial_states, initial_covs, F_KF, Q, R, H, nt, [], 'Debug', DEBUG);
            
        case 'RBPF'
            fprintf('Initializing KF-RBPF-Multi tracker with %d particles...\n', N_PARTICLES);
            % Create cell array of initial states
            x0_cell = cell(1, nt);
            for t = 1:nt
                x0_cell{t} = GT{t}(:, 1);
            end
            
            current_class = KF_RBPF_multi(x0_cell, N_PARTICLES, F_KF, Q, H, R, ...
                'Debug', DEBUG, ...
                'AssociationStrategy', 'optimal', ...
                'PD', 0.95, ...
                'PFA', 0.05, ...
                'ESSThreshold', 0.5);
            
        otherwise
            error('Unknown algorithm: %s', ALGORITHM);
    end

    [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate(); % Initial Gaussian estimate
    
    % For RBPF, save raw particle data for visualization before conversion
    if strcmp(ALGORITHM, 'RBPF')
        performance{1}.particles = current_class.particles;
        performance{1}.x = convertCellToMatrix(performance{1}.x);
    end

    %% --- Main Tracking Loop ---
    fprintf('\nStarting tracking loop...\n');
    for k = 2:n_k
        if mod(k, 10) == 0 || k == 2
            fprintf('Processing time step %d/%d\n', k, n_k);
        end

        current_meas = z{k};
        
        % Call timestep with appropriate parameters based on algorithm
        if strcmp(ALGORITHM, 'JPDA')
            % JPDA requires both measurements and signal
            current_class.timestep(current_meas, signal{k})
        else
            % RBPF only needs measurements
            current_class.timestep(current_meas)
        end
        
        [performance{k}.x, performance{k}.P] = current_class.getGaussianEstimate();
        
        % For RBPF, save raw particle data for visualization before conversion
        if strcmp(ALGORITHM, 'RBPF')
            performance{k}.particles = current_class.particles;
            performance{k}.x = convertCellToMatrix(performance{k}.x);
        end
    end

    fprintf('Tracking complete!\n\n');

    %% --- Plot Results ---
    fprintf('Generating plots...\n');
    
    % Simple scatter plot of position estimates
    plotSimpleTrajectories(performance, Data, ALGORITHM, DATASET);
    fprintf('Simple trajectory plot saved to: %s_%s_trajectories.png\n', DATASET, ALGORITHM);
    
    % Animated GIF with distribution/particle visualization
    gif_filename = sprintf('%s_%s_tracking.gif', DATASET, ALGORITHM);
    
    if strcmp(ALGORITHM, 'RBPF')
        % Use RBPF-specific visualization with particle scatter
        mWidar_FilterPlot_multiObj_RBPF(performance, Data, 0:dt:10, gif_filename);
        fprintf('RBPF particle animation GIF saved to: %s\n', gif_filename);
    else
        % Use standard Gaussian distribution visualization for JPDA
        mWidar_FilterPlot_multiObj_Distribution(performance, Data, 0:dt:10, gif_filename);
        fprintf('Distribution GIF saved to: %s\n', gif_filename);
    end
    
    fprintf('Done!\n');

end

%% Helper Functions

function plotSimpleTrajectories(performance, Data, algorithm_name, dataset_name)
    % PLOTSIMPLETRAJECTORIES Plot position estimates for all targets
    %
    % This creates a simple 2D scatter plot showing:
    % - Ground truth trajectories
    % - All position estimates (x,y) at each timestep
    % - No attempt to associate which KF tracks which target
    %
    % INPUTS:
    %   performance    - Cell array of filter outputs {1 x n_k}
    %   Data           - Data struct with GT, measurements, etc.
    %   algorithm_name - Name of algorithm for title
    %   dataset_name   - Name of dataset for saving
    
    GT = Data.GT;
    z = Data.y;
    n_k = length(performance);
    n_t = length(GT);
    
    figure('Position', [100, 100, 1200, 800]);
    hold on; grid on; axis equal;
    
    % Define colors
    colors = lines(n_t);
    
    % Plot ground truth trajectories
    for t = 1:n_t
        gt_traj = GT{t};
        plot(gt_traj(1, :), gt_traj(2, :), '-', 'Color', colors(t, :), ...
            'LineWidth', 2.5, 'DisplayName', sprintf('GT Target %d', t));
        
        % Mark start and end
        plot(gt_traj(1, 1), gt_traj(2, 1), 'o', 'Color', colors(t, :), ...
            'MarkerSize', 10, 'MarkerFaceColor', colors(t, :), 'HandleVisibility', 'off');
        plot(gt_traj(1, end), gt_traj(2, end), 's', 'Color', colors(t, :), ...
            'MarkerSize', 10, 'MarkerFaceColor', colors(t, :), 'HandleVisibility', 'off');
    end
    
    % Plot all position estimates (scatter - no trajectory connection)
    % Extract all estimates and plot as scatter
    all_est_x = [];
    all_est_y = [];
    
    for k = 1:n_k
        if isfield(performance{k}, 'x')
            x_est = performance{k}.x;
            
            % Handle both cell array (RBPF before conversion) and matrix format
            if iscell(x_est)
                % Cell array format: {[N_x x 1], [N_x x 1], ...}
                for t = 1:length(x_est)
                    all_est_x = [all_est_x; x_est{t}(1)];
                    all_est_y = [all_est_y; x_est{t}(2)];
                end
            else
                % Matrix format: [N_x x N_t]
                all_est_x = [all_est_x; x_est(1, :)'];
                all_est_y = [all_est_y; x_est(2, :)'];
            end
        end
    end
    
    % Plot all estimates as scatter
    scatter(all_est_x, all_est_y, 30, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.4, ...
        'DisplayName', sprintf('%s Estimates', algorithm_name));
    
    % Plot measurements (all)
    all_meas_x = [];
    all_meas_y = [];
    for k = 1:n_k
        if ~isempty(z{k})
            all_meas_x = [all_meas_x; z{k}(1, :)'];
            all_meas_y = [all_meas_y; z{k}(2, :)'];
        end
    end
    scatter(all_meas_x, all_meas_y, 20, 'r', '+', 'MarkerFaceAlpha', 0.3, ...
        'DisplayName', 'Measurements');
    
    xlabel('X Position (m)', 'FontSize', 12);
    ylabel('Y Position (m)', 'FontSize', 12);
    title(sprintf('Multi-Target Tracking: %s on %s', algorithm_name, dataset_name), ...
        'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    
    % Save figure
    filename = sprintf('%s_%s_trajectories.png', dataset_name, algorithm_name);
    saveas(gcf, filename);
    
    hold off;
end

function x_matrix = convertCellToMatrix(x_cell)
    % CONVERTCELLTOMATRIX Convert RBPF cell array output to matrix format
    %
    % RBPF returns: x_cell = {[N_x x 1], [N_x x 1], ...} (1 x N_t)
    % JPDA expects: x_matrix = [N_x x N_t]
    
    n_targets = length(x_cell);
    n_states = length(x_cell{1});
    
    x_matrix = zeros(n_states, n_targets);
    for t = 1:n_targets
        x_matrix(:, t) = x_cell{t};
    end
end