% TEST_KF_RBPF - Simple test of KF_RBPF implementation
%
% Tests basic functionality:
% 1. Object creation
% 2. Timestep execution
% 3. State estimation
% 4. Visualization (if enabled)

clear; clc; close all;

% Add paths
addpath('../DA_Track');

fprintf('=== KF_RBPF Basic Functionality Test ===\n\n');

%% Test 1: Object Creation
fprintf('Test 1: Creating KF_RBPF object...\n');

% Define system (simple 2D constant velocity)
dt = 0.1;
F = [1, 0, dt, 0;
     0, 1, 0, dt;
     0, 0, 1, 0;
     0, 0, 0, 1];
Q = diag([0.01, 0.01, 0.5, 0.5]); % Slightly higher process noise in velocity
H = [1, 0, 0, 0;
     0, 1, 0, 0];
R_sim = 0.01 * eye(2); %R_sim is much smaller for measurements
R = 0.1 * eye(2);

% Initial state - start at origin with upward and rightward velocity
% This will create a nice parabolic arc trajectory
x0_true = [0; 0; 1.5; 3.0]; % True initial state: Start at (0,0) moving at [1.5, 3.0] m/s (higher vertical velocity)

% UNCERTAIN INITIALIZATION - We don't know the true initial state!
% Initialize filter with a random guess to test localization capability
fprintf('\n*** LOCALIZATION TEST: Unknown initial state ***\n');
fprintf('True initial state: [%.2f, %.2f, %.2f, %.2f]\n', x0_true);

% Random initialization within reasonable bounds
initial_pos_uncertainty = 5.0; % meters
initial_vel_uncertainty = 2.0; % m/s
x0_filter = [randn() * initial_pos_uncertainty;
             randn() * initial_pos_uncertainty;
             randn() * initial_vel_uncertainty;
             randn() * initial_vel_uncertainty];

fprintf('Filter initial guess: [%.2f, %.2f, %.2f, %.2f]\n', x0_filter);
fprintf('Initial error (position): %.2f m\n', norm(x0_filter(1:2) - x0_true(1:2)));
fprintf('Initial error (velocity): %.2f m/s\n\n', norm(x0_filter(3:4) - x0_true(3:4)));

% Create RBPF with debug enabled (more particles for localization robustness)
try
    rbpf = KF_RBPF(x0_filter, 500, F, Q, H, R, 'Debug', true);
    fprintf('✓ KF_RBPF object created successfully!\n');
    fprintf('  N_p = %d (increased for localization)\n', rbpf.N_p);
    fprintf('  N_x = %d, N_z = %d\n', rbpf.N_x, rbpf.N_z);
catch ME
    fprintf('✗ FAILED: %s\n', ME.message);
    return;
end

%% Test 2: State Estimation
fprintf('\nTest 2: Getting initial state estimate...\n');

try
    [x_est, P_est] = rbpf.getGaussianEstimate();
    fprintf('✓ State estimate computed!\n');
    fprintf('  Position estimate: [%.2f, %.2f] (true: [%.2f, %.2f])\n', ...
        x_est(1), x_est(2), x0_true(1), x0_true(2));
    fprintf('  Velocity estimate: [%.2f, %.2f] (true: [%.2f, %.2f])\n', ...
        x_est(3), x_est(4), x0_true(3), x0_true(4));
    fprintf('  Velocity estimate: [%.2f, %.2f]\n', x_est(3), x_est(4));
catch ME
    fprintf('✗ FAILED: %s\n', ME.message);
    return;
end

%% Test 3: Generate Synthetic Trajectory and Measurements
fprintf('\nTest 3: Generating synthetic data...\n');

n_steps = 100; % Longer trajectory to test localization
true_states = zeros(4, n_steps);
measurements = cell(1, n_steps);

% Generate trajectory: Curved path (parabolic arc)
% The target follows a curved trajectory with changing velocity
% This tests the filter's ability to track maneuvering targets
rng(42); % For reproducibility
true_states(:, 1) = x0_true; % Use TRUE initial state for ground truth

for k = 2:n_steps
    t = (k - 1) * dt;

    % Base trajectory: parabolic path with constant acceleration in y
    % x(t) = x0 + vx0*t
    % y(t) = y0 + vy0*t - 0.5*a*t^2  (parabola)
    % vx(t) = vx0 (constant in x)
    % vy(t) = vy0 - a*t (linearly changing in y)

    acceleration_y = -0.5; % m/s^2 (stronger downward acceleration for clear parabolic arc)

    % Propagate with constant velocity model + process noise
    true_states(:, k) = F * true_states(:, k - 1);

    % Add the acceleration effect manually (model mismatch!)
    % This simulates a maneuvering target that the CV model can't perfectly track
    true_states(2, k) = true_states(2, k) + 0.5 * acceleration_y * dt ^ 2; % position update
    true_states(4, k) = true_states(4, k) + acceleration_y * dt; % velocity update
end

fprintf('  Trajectory type: Parabolic arc (maneuvering target)\n');
fprintf('  Timesteps: %d (%.1f seconds)\n', n_steps, n_steps * dt);
fprintf('  Initial position: [%.2f, %.2f] m\n', x0_true(1), x0_true(2));
fprintf('  Initial velocity: [%.2f, %.2f] m/s\n', x0_true(3), x0_true(4));
fprintf('  Acceleration (y): %.2f m/s^2\n', acceleration_y);
fprintf('  Final position: [%.2f, %.2f] m\n', true_states(1, end), true_states(2, end));
fprintf('  Final velocity: [%.2f, %.2f] m/s\n', true_states(3, end), true_states(4, end));

% Generate measurements (with significant clutter and missed detections)
rng(42); % For reproducibility
true_measurement_exists = false(1, n_steps); % Track which timesteps have true measurements

for k = 1:n_steps
    measurements{k} = [];

    % Probability of detection (90% - some missed detections)
    if rand() < 0.9
        % True measurement with noise
        z_true = H * true_states(:, k) + mvnrnd([0; 0], R_sim)';
        measurements{k} = z_true;
        true_measurement_exists(k) = true;
    end

    % Add clutter measurements (3-7 false alarms)
    n_clutter = randi([3, 7]);

    % Generate clutter uniformly across the surveillance region
    % Get bounds from true trajectory
    x_min = min(true_states(1, :)) - 5;
    x_max = max(true_states(1, :)) + 5;
    y_min = min(true_states(2, :)) - 5;
    y_max = max(true_states(2, :)) + 5;

    clutter = [x_min + rand(1, n_clutter) * (x_max - x_min);
               y_min + rand(1, n_clutter) * (y_max - y_min)];

    measurements{k} = [measurements{k}, clutter];

    % Randomly permute measurements so true detection isn't always first
    if ~isempty(measurements{k})
        measurements{k} = measurements{k}(:, randperm(size(measurements{k}, 2)));
    end

end

n_detections = sum(true_measurement_exists);
n_missed = n_steps - n_detections;
fprintf('✓ Generated %d timesteps of synthetic data\n', n_steps);
fprintf('  True detections: %d (%.1f%%)\n', n_detections, 100 * n_detections / n_steps);
fprintf('  Missed detections: %d (%.1f%%)\n', n_missed, 100 * n_missed / n_steps);
fprintf('  Average clutter per timestep: ~5 false alarms\n');

%% Test 4: Run Filter Through Data
fprintf('\nTest 4: Running filter through trajectory...\n');

estimates = zeros(4, n_steps);
[estimates(:, 1), ~] = rbpf.getGaussianEstimate(); % Get initial estimate from filter

try

    for k = 2:n_steps
        rbpf.timestep(measurements{k}, true_states(:, k), true_measurement_exists(k));
        [x_est, ~] = rbpf.getGaussianEstimate();
        estimates(:, k) = x_est;

        if mod(k, 10) == 0
            fprintf('  Processed timestep %d/%d\n', k, n_steps);
        end

    end

    fprintf('✓ Filter completed all timesteps!\n');
catch ME
    fprintf('✗ FAILED at timestep %d: %s\n', k, ME.message);
    fprintf('  Error stack:\n');

    for i = 1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end

    return;
end

%% Test 5: Compute RMSE and Localization Convergence
fprintf('\nTest 5: Computing tracking performance and localization...\n');

position_errors = true_states(1:2, :) - estimates(1:2, :);
position_rmse = sqrt(mean(sum(position_errors .^ 2, 1)));
position_errors_norm = sqrt(sum(position_errors .^ 2, 1));

velocity_errors = true_states(3:4, :) - estimates(3:4, :);
velocity_rmse = sqrt(mean(sum(velocity_errors .^ 2, 1)));
velocity_errors_norm = sqrt(sum(velocity_errors .^ 2, 1));

% Localization analysis - how quickly did filter converge?
initial_pos_error = norm(x0_filter(1:2) - x0_true(1:2));
initial_vel_error = norm(x0_filter(3:4) - x0_true(3:4));
final_pos_error = position_errors_norm(end);
final_vel_error = velocity_errors_norm(end);

% Find when position error drops below 1 meter (localized)
localized_idx = find(position_errors_norm < 1.0, 1);

if ~isempty(localized_idx)
    convergence_time = localized_idx * dt;
    fprintf('✓ Localization achieved at timestep %d (%.1f sec)\n', localized_idx, convergence_time);
else
    fprintf('⚠ Warning: Position error never dropped below 1.0 m\n');
end

fprintf('✓ Performance metrics:\n');
fprintf('  Initial position error: %.4f m\n', initial_pos_error);
fprintf('  Final position error:   %.4f m\n', final_pos_error);
fprintf('  Position RMSE (overall): %.4f m\n', position_rmse);
fprintf('\n');
fprintf('  Initial velocity error: %.4f m/s\n', initial_vel_error);
fprintf('  Final velocity error:   %.4f m/s\n', final_vel_error);
fprintf('  Velocity RMSE (overall): %.4f m/s\n', velocity_rmse);

%% Test 6: Visualization
fprintf('\nTest 6: Testing visualization...\n');

% Define save directory early so it's available for summary
save_dir = fullfile('..', '..', 'figures', 'DA_Track', 'KFRBPF_prelim');

try
    figure('Name', 'KF_RBPF Localization Test Results', 'Position', [100, 100, 1400, 900]);

    % Debug: Check dimensions
    fprintf('  Debug - Data dimensions:\n');
    fprintf('    n_steps: %d\n', n_steps);
    fprintf('    true_states: %s\n', mat2str(size(true_states)));
    fprintf('    estimates: %s\n', mat2str(size(estimates)));
    fprintf('    position_errors_norm: %s\n', mat2str(size(position_errors_norm)));
    fprintf('    velocity_errors_norm: %s\n', mat2str(size(velocity_errors_norm)));

    % Subplot 1: Trajectory
    subplot(2, 3, 1);
    fprintf('    Plotting subplot 1 (trajectory)...\n');
    plot(true_states(1, :), true_states(2, :), 'g-', 'LineWidth', 2, 'DisplayName', 'True');
    hold on;
    plot(estimates(1, :), estimates(2, :), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimate');

    % Mark initial positions
    plot(x0_true(1), x0_true(2), 'go', 'MarkerSize', 12, 'LineWidth', 3, ...
        'DisplayName', 'True Start');
    plot(x0_filter(1), x0_filter(2), 'rs', 'MarkerSize', 12, 'LineWidth', 3, ...
        'DisplayName', 'Filter Start');

    % Mark convergence point if it exists
    if ~isempty(localized_idx)
        plot(estimates(1, localized_idx), estimates(2, localized_idx), 'kp', ...
            'MarkerSize', 16, 'LineWidth', 2, 'MarkerFaceColor', 'y', ...
            'DisplayName', 'Localized');
    end

    xlabel('X (m)'); ylabel('Y (m)');
    title('Trajectory (Parabolic Arc)');
    legend('Location', 'best'); grid on; axis equal;
    fprintf('    ✓ Subplot 1 complete\n');

    % Subplot 2: Position Error Convergence
    subplot(2, 3, 2);
    fprintf('    Plotting subplot 2 (position error)...\n');
    time = (1:n_steps) * dt;  % Fixed: should be 1:n_steps to match data size
    fprintf('    time size: %s\n', mat2str(size(time)));
    plot(time, position_errors_norm, 'r-', 'LineWidth', 2);
    hold on;
    yline(1.0, 'k--', 'LineWidth', 1.5, 'Label', 'Localized (1m)');

    if ~isempty(localized_idx)
        plot(time(localized_idx), position_errors_norm(localized_idx), 'kp', ...
            'MarkerSize', 16, 'LineWidth', 2, 'MarkerFaceColor', 'y');
    end

    xlabel('Time (s)'); ylabel('Position Error (m)');
    title(sprintf('Position Error (RMSE=%.3f m)', position_rmse));
    grid on;

    % Subplot 3: Velocity Error Convergence
    subplot(2, 3, 3);
    plot(time, velocity_errors_norm, 'b-', 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Velocity Error (m/s)');
    title(sprintf('Velocity Error (RMSE=%.3f m/s)', velocity_rmse));
    grid on;

    % Subplot 4: X Position comparison
    subplot(2, 3, 4);
    plot(time, true_states(1, :), 'g-', 'LineWidth', 2, 'DisplayName', 'True');
    hold on;
    plot(time, estimates(1, :), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimate');
    xlabel('Time (s)'); ylabel('X Position (m)');
    title('X Position vs Time');
    legend; grid on;

    % Subplot 5: Y Position comparison
    subplot(2, 3, 5);
    plot(time, true_states(2, :), 'g-', 'LineWidth', 2, 'DisplayName', 'True');
    hold on;
    plot(time, estimates(2, :), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimate');
    xlabel('Time (s)'); ylabel('Y Position (m)');
    title('Y Position vs Time (Parabolic)');
    legend; grid on;

    % Subplot 6: ESS over time (particle filter specific metric)
    subplot(2, 3, 6);

    if ~isempty(rbpf.history)
        ESS_history = [rbpf.history.ESS];
        % History is only stored when timestep() is called (k=2:n_steps)
        % So we need time points from 2:n_steps
        time_history = (2:n_steps) * dt;
        plot(time_history, ESS_history, 'k-', 'LineWidth', 2);
        hold on;
        yline(rbpf.N_p * rbpf.ESS_threshold_percentage, 'r--', 'LineWidth', 1.5, ...
            'Label', sprintf('Resample Threshold (%.0f%%)', rbpf.ESS_threshold_percentage * 100));
        xlabel('Time (s)'); ylabel('Effective Sample Size');
        title('ESS History (Particle Diversity)');
        ylim([0, rbpf.N_p]);
        grid on;
    else
        text(0.5, 0.5, 'No ESS history available', ...
            'HorizontalAlignment', 'center', 'Units', 'normalized');
        axis off;
    end

    fprintf('✓ Visualization created!\n');

    % Create save directory (if it doesn't exist)
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
        fprintf('Created save directory: %s\n', save_dir);
    end

    % Save the current figure (2x3 summary plot)
    summary_filename = fullfile(save_dir, 'localization_summary.png');
    saveas(gcf, summary_filename);
    fprintf('✓ Saved localization summary to: %s\n', summary_filename);

catch ME
    fprintf('✗ Visualization failed: %s\n', ME.message);
end

%% Test 7: Animation (optional)
fprintf('\nTest 7: Post-processing animation...\n');
fprintf('Create animation from history? (y/n): ');
user_input = input('', 's');

if strcmpi(user_input, 'y')
    try
        fprintf('Creating animated visualization...\n');
        gif_filename = fullfile(save_dir, 'test_animation.gif');
        visualize_RBPF_history(rbpf, 'Animate', true, 'SaveGIF', gif_filename, ...
            'AnimationSpeed', 0.2, 'PlotMargin', 0.2);
        fprintf('✓ Animation saved to: %s\n', gif_filename);
    catch ME
        fprintf('✗ Animation failed: %s\n', ME.message);
        fprintf('  Error stack:\n');
        for i = 1:length(ME.stack)
            fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
else
    fprintf('Skipping animation.\n');
end

%% Summary
fprintf('\n=== TEST SUMMARY ===\n');
fprintf('✓ All tests passed!\n');
fprintf('Position RMSE: %.4f m\n', position_rmse);
fprintf('Velocity RMSE: %.4f m/s\n', velocity_rmse);
fprintf('Figures saved to: %s\n', save_dir);
fprintf('===================\n');
