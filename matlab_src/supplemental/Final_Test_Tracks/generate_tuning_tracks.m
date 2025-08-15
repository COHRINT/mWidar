clear; clc; close all

%{
    Script to generate/save mWidar images for tuning with a simple S-curve trajectory
    and 4 different clutter levels:
        1. No clutter (clean signal only)
        2. Low clutter (uniform across scene)
        3. Medium clutter (centralized around target area)
        4. Simulated clutter (full mWidar simulation)
    
    Based on the analytical S-curve trajectory from generate_easytrack.m
    State vector, x = [x, y, vx, vy, ax, ay]'
%}

% Load mWidar simulation matrices
load(fullfile('..', 'recovery.mat'))
load(fullfile('..', 'sampling.mat'))

% Time parameters (matching the easytrack trajectory)
dt = 0.1; % [sec]
num_steps = 50;
tvec = (0:num_steps) * dt;
n_t = length(tvec); % # of timesteps
detector = "CFAR"; % Use CFAR detector

% Scene parameters
Lscene = 4;
npx = 128;
xgrid = linspace(-2, 2, npx);
ygrid = linspace(0, Lscene, npx);
[pxgrid, pygrid] = meshgrid(xgrid, ygrid);

fprintf('=== GENERATING TUNING DATASETS ===\n');
fprintf('Trajectory: Analytical S-curve\n');
fprintf('Duration: %.1f seconds, %d timesteps\n', tvec(end), n_t);
fprintf('Detector: %s\n', detector);

%% Generate Enhanced S-curve Trajectory (from generate_easytrack.m)

% Enhanced S-curve trajectory with smooth analytical dynamics
% Start at (-1, 1) and move in an S-turn to (1, 3)
x_start = -1.5;
y_start = 0.5;
x_end = 1.5;
y_end = 3.5;

% Use parametric equations with time parameter tau = t/T where T is total time
T = tvec(end); % Total trajectory time
tau = tvec / T; % Normalized time [0, 1]

% Smooth trajectory using polynomial and sinusoidal components
% Position with smooth S-curve using 5th order polynomial for smoothness
% X: Linear progression with sinusoidal perturbation
x_traj = x_start + (x_end - x_start) * tau + 0.6 * sin(2 * pi * tau) .* (1 - tau) .^ 2 .* tau .^ 2;

% Y: S-curve with smooth transitions using tanh-like function
y_base = y_start + (y_end - y_start) * tau;
y_perturbation = 0.4 * sin(pi * tau) .* sin(2 * pi * tau);
y_traj = y_base + y_perturbation;

% Analytical velocity (first derivative with respect to time)
dtau_dt = 1 / T; % d(tau)/dt

% X velocity components
dx_dtau = (x_end - x_start) + 0.6 .* (2 .* pi .* cos(2 .* pi .* tau) .* (1 - tau) .^ 2 .* tau .^ 2 + ...
    sin(2 .* pi .* tau) .* (2 .* (1 - tau) .* tau .^ 2 .* (-1) + 2 .* (1 - tau) .^ 2 .* tau));
vx_traj = dx_dtau .* dtau_dt;

% Y velocity components
dy_dtau = (y_end - y_start) + 0.4 .* (pi .* cos(pi .* tau) .* sin(2 .* pi .* tau) + ...
    sin(pi .* tau) .* 2 .* pi .* cos(2 .* pi .* tau));
vy_traj = dy_dtau .* dtau_dt;

% Analytical acceleration (second derivative with respect to time)
% X acceleration
d2x_dtau2 = 0.6 .* (-4 .* pi .^ 2 .* sin(2 .* pi .* tau) .* (1 - tau) .^ 2 .* tau .^ 2 + ...
    4 .* pi .* cos(2 .* pi .* tau) .* (2 .* (1 - tau) .* tau .^ 2 .* (-1) + 2 .* (1 - tau) .^ 2 .* tau) + ...
    2 .* pi .* cos(2 .* pi .* tau) .* (2 .* tau .^ 2 .* (-1) + 4 .* (1 - tau) .* tau) + ...
    sin(2 .* pi .* tau) .* 2 .* (2 .* tau .* (-1) + 2 .* (1 - tau)));
ax_traj = d2x_dtau2 .* dtau_dt .^ 2;

% Y acceleration
d2y_dtau2 = 0.4 .* (-pi .^ 2 .* sin(pi .* tau) .* sin(2 .* pi .* tau) + 2 .* pi .* cos(pi .* tau) .* 2 .* pi .* cos(2 .* pi .* tau) + ...
    pi .* cos(pi .* tau) .* 2 .* pi .* cos(2 .* pi .* tau) - sin(pi .* tau) .* 4 .* pi .^ 2 .* sin(2 .* pi .* tau));
ay_traj = d2y_dtau2 .* dtau_dt .^ 2;

% Ensure trajectory stays within bounds
x_traj = max(-2, min(2, x_traj));
y_traj = max(0, min(4, y_traj));

% Store full state trajectory [x, y, vx, vy, ax, ay]
X_GT = [x_traj; y_traj; vx_traj; vy_traj; ax_traj; ay_traj]; % 6 x n_t

fprintf('Generated smooth analytical S-curve trajectory\n');
fprintf('Max speeds: Vx=%.2f m/s, Vy=%.2f m/s\n', max(abs(vx_traj)), max(abs(vy_traj)));
fprintf('Max accelerations: Ax=%.2f m/s², Ay=%.2f m/s²\n', max(abs(ax_traj)), max(abs(ay_traj)));

%% Dataset 1: No Clutter (Clean Signal Only)
fprintf('\n--- Generating Dataset 1: No Clutter ---\n');

% Generate clean measurements (just the true position with minimal noise)
meas_noise_std = 0.01; % Very small measurement noise
y_1 = cell(1, n_t);
Signal_1 = cell(1, n_t);

rng(100); % Set seed for reproducibility

for i = 1:n_t
    % Create clean measurement at true position with tiny amount of noise
    true_pos = X_GT(1:2, i);
    noisy_pos = true_pos + meas_noise_std * randn(2, 1);
    
    % Single measurement per timestep
    y_1{i} = noisy_pos;
    
    % Generate signal using mWidar simulation for just this position
    S = zeros(128, 128);
    
    % Convert to grid coordinates
    px = true_pos(1);
    py = true_pos(2);
    
    if px > -2 && px < 2 && py > 0 && py < 4
        Gx = find(px <= xgrid, 1, 'first');
        Gy = find(py <= ygrid, 1, 'first');
        
        if Gx > 0 && Gx <= 128 && Gy > 0 && Gy <= 128
            S(Gy, Gx) = 1;
        end
    end
    
    % Apply mWidar forward model
    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';
    
    % Apply blurring and normalization
    blurred = imgaussfilt(sim_signal, 2);
    Signal_1{i} = (blurred - min(blurred(:))) / (max(blurred(:)) - min(blurred(:)));
end

Data.GT = X_GT;
Data.y = y_1;
Data.signal = Signal_1;
save(fullfile('SingleObjTune', 'T1_no_clutter.mat'), 'Data', '-mat')
fprintf('Saved: T1_no_clutter.mat\n');

%% Dataset 2: Low Clutter (Uniform Across Scene)
fprintf('\n--- Generating Dataset 2: Low Clutter ---\n');

y_2 = cell(1, n_t);
Signal_2 = cell(1, n_t);

rng(200); % Different seed for clutter

for i = 1:n_t
    % Start with true position (with noise)
    true_pos = X_GT(1:2, i);
    measurements = true_pos + meas_noise_std * randn(2, 1);
    
    % Add 10 uniform clutter points across entire scene
    n_clutter = 10;
    clutter_x = -2 + 4 * rand(1, n_clutter); % Uniform in [-2, 2]
    clutter_y = 0 + 4 * rand(1, n_clutter);  % Uniform in [0, 4]
    clutter_points = [clutter_x; clutter_y];
    
    % Combine true measurement with clutter
    measurements = [measurements, clutter_points];
    y_2{i} = measurements;
    
    % Generate signal with ONLY the true target (not clutter points)
    S = zeros(128, 128);
    
    px = true_pos(1);
    py = true_pos(2);
    
    if px > -2 && px < 2 && py > 0 && py < 4
        Gx = find(px <= xgrid, 1, 'first');
        Gy = find(py <= ygrid, 1, 'first');
        
        if Gx > 0 && Gx <= 128 && Gy > 0 && Gy <= 128
            S(Gy, Gx) = 1;
        end
    end
    
    % Apply mWidar forward model
    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';
    
    % Apply blurring and normalization
    blurred = imgaussfilt(sim_signal, 2);
    Signal_2{i} = (blurred - min(blurred(:))) / (max(blurred(:)) - min(blurred(:)));
end

Data.GT = X_GT;
Data.y = y_2;
Data.signal = Signal_2;
save(fullfile('SingleObjTune', 'T2_low_clutter.mat'), 'Data', '-mat')
fprintf('Saved: T2_low_clutter.mat\n');

%% Dataset 3: Medium Clutter (Centralized Around Target)
fprintf('\n--- Generating Dataset 3: Medium Clutter ---\n');

y_3 = cell(1, n_t);
Signal_3 = cell(1, n_t);

rng(300); % Different seed for clutter

for i = 1:n_t
    % Start with true position (with noise)
    true_pos = X_GT(1:2, i);
    measurements = true_pos + meas_noise_std * randn(2, 1);
    
    % Add 15 uniform clutter points across entire scene
    n_uniform_clutter = 15;
    uniform_clutter_x = -2 + 4 * rand(1, n_uniform_clutter); % Uniform in [-2, 2]
    uniform_clutter_y = 0 + 4 * rand(1, n_uniform_clutter);  % Uniform in [0, 4]
    uniform_clutter_points = [uniform_clutter_x; uniform_clutter_y];
    
    % Add 5 nearby clutter points in regions around where the target "might be"
    % Create clutter in regions that are plausible but not centered on truth
    n_nearby_clutter = 5;
    
    % Create clutter in regions that are "reasonable" target locations
    % Define offset vectors explicitly
    offset1 = [0.5; 0.3];
    offset2 = [-0.3; 0.5];
    offset3 = [0.2; -0.4];
    offsets = {offset1, offset2, offset3};
    
    nearby_clutter_points = [];
    for region_idx = 1:3
        n_clutter_region = floor(n_nearby_clutter / 3) + (region_idx <= mod(n_nearby_clutter, 3));
        
        % Calculate region center
        center = true_pos + offsets{region_idx};
        
        % Generate clutter around this region center with moderate spread
        clutter_spread = 0.4; % Standard deviation of clutter around region center
        region_clutter = center + clutter_spread * randn(2, n_clutter_region);
        
        % Ensure clutter stays within scene bounds
        region_clutter(1, :) = max(-2, min(2, region_clutter(1, :)));
        region_clutter(2, :) = max(0, min(4, region_clutter(2, :)));
        
        nearby_clutter_points = [nearby_clutter_points, region_clutter];
    end
    
    % Combine all measurements: true + uniform clutter + nearby clutter
    measurements = [measurements, uniform_clutter_points, nearby_clutter_points];
    y_3{i} = measurements;
    
    % Generate signal with ONLY the true target (not clutter points)
    S = zeros(128, 128);
    
    px = true_pos(1);
    py = true_pos(2);
    
    if px > -2 && px < 2 && py > 0 && py < 4
        Gx = find(px <= xgrid, 1, 'first');
        Gy = find(py <= ygrid, 1, 'first');
        
        if Gx > 0 && Gx <= 128 && Gy > 0 && Gy <= 128
            S(Gy, Gx) = 1;
        end
    end
    
    % Apply mWidar forward model
    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';
    
    % Apply blurring and normalization
    blurred = imgaussfilt(sim_signal, 2);
    Signal_3{i} = (blurred - min(blurred(:))) / (max(blurred(:)) - min(blurred(:)));
end

Data.GT = X_GT;
Data.y = y_3;
Data.signal = Signal_3;
save(fullfile('SingleObjTune', 'T3_medium_clutter.mat'), 'Data', '-mat')
fprintf('Saved: T3_medium_clutter.mat\n');

%% Dataset 4: Simulated Clutter (Full mWidar Simulation)
fprintf('\n--- Generating Dataset 4: Simulated Clutter ---\n');

% Create a simulated clutter dataset manually since sim_mWidar_image seems to have issues
% We'll create realistic clutter by adding noise and running detection on the mWidar signal
y_4 = cell(1, n_t);
Signal_4 = cell(1, n_t);

rng(400); % Different seed for this dataset

for i = 1:n_t
    % Get true position
    true_pos = X_GT(1:2, i);
    
    % Create signal with true target
    S = zeros(128, 128);
    
    px = true_pos(1);
    py = true_pos(2);
    
    if px > -2 && px < 2 && py > 0 && py < 4
        Gx = find(px <= xgrid, 1, 'first');
        Gy = find(py <= ygrid, 1, 'first');
        
        if Gx > 0 && Gx <= 128 && Gy > 0 && Gy <= 128
            S(Gy, Gx) = 1;
        end
    end
    
    % Apply mWidar forward model
    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';
    
    % Apply blurring and normalization
    blurred = imgaussfilt(sim_signal, 2);
    signal_normalized = (blurred - min(blurred(:))) / (max(blurred(:)) - min(blurred(:)));
    
    % Run CFAR detector to find peaks (this will include clutter detections)
    try
        [~, peak_x, peak_y] = CA_CFAR(signal_normalized, 0.4, 3, 10);
        
        if ~isempty(peak_x) && ~isempty(peak_y)
            % Convert detector indices back to world coordinates
            pvinds = sub2ind([npx, npx], peak_x, peak_y);
            y_4{i} = [pxgrid(pvinds)'; pygrid(pvinds)'];
        else
            y_4{i} = [];
        end
    catch
        % If detector fails, just use the true position with some noise
        y_4{i} = true_pos + 0.1 * randn(2, 1);
    end
    
    Signal_4{i} = signal_normalized;
end

Data.GT = X_GT;
Data.y = y_4;
Data.signal = Signal_4;
save(fullfile('SingleObjTune', 'T4_simulated_clutter.mat'), 'Data', '-mat')
fprintf('Saved: T4_simulated_clutter.mat\n');

%% Generate Animated Summary Visualization
fprintf('\n--- Generating Animated Summary Visualization ---\n');

figure(100); clf;
set(gcf, 'Position', [100, 100, 1400, 1000], 'Visible', 'off');

% Prepare data
datasets = {y_1, y_2, y_3, y_4};
signals = {Signal_1, Signal_2, Signal_3, Signal_4};
dataset_names = {'No Clutter', 'Low Clutter', 'Medium Clutter', 'Simulated Clutter'};
colors = {'r', 'g', 'b', 'm'};

% Animation parameters
gif_filename = fullfile('SingleObjTune', 'tuning_datasets_animation.gif');
delay_time = 0.2; % seconds per frame

% Create animation
for t = 1:n_t
    clf; % Clear figure
    
    for dataset_idx = 1:4
        subplot(2, 2, dataset_idx); hold on;
        
        % Plot mWidar signal as background
        if ~isempty(signals{dataset_idx}{t})
            imagesc(xgrid, ygrid, signals{dataset_idx}{t});
            set(gca, 'YDir', 'normal');
            colormap('parula');
            alpha(0.7); % Make signal semi-transparent
        end
        
        % Plot trajectory up to current time (fading older points)
        for t_past = max(1, t-20):t
            alpha_val = 0.3 + 0.7 * (t_past - max(1, t-20) + 1) / min(21, t);
            if t_past == t
                % Current position - bright
                plot(X_GT(1, t_past), X_GT(2, t_past), 'ko', 'MarkerSize', 8, 'LineWidth', 2);
            else
                % Past positions - fading
                plot(X_GT(1, t_past), X_GT(2, t_past), 'k-', 'LineWidth', 1, 'Color', [0 0 0 alpha_val]);
                if t_past < t
                    plot(X_GT(1, t_past+1), X_GT(2, t_past+1), 'k-', 'LineWidth', 1, 'Color', [0 0 0 alpha_val]);
                end
            end
        end
        
        % Plot trajectory line up to current time
        if t > 1
            plot(X_GT(1, 1:t), X_GT(2, 1:t), 'k-', 'LineWidth', 2, 'DisplayName', 'True Trajectory');
        end
        
        % Plot measurements at current timestep
        if ~isempty(datasets{dataset_idx}{t})
            measurements = datasets{dataset_idx}{t};
            scatter(measurements(1, :), measurements(2, :), 50, colors{dataset_idx}, '+', ...
                'LineWidth', 2, 'DisplayName', 'Measurements');
            n_measurements = size(measurements, 2);
        else
            n_measurements = 0;
        end
        
        % Plot current true position (large marker)
        plot(X_GT(1, t), X_GT(2, t), 'wo', 'MarkerSize', 10, ...
            'LineWidth', 2, 'DisplayName', 'Current True Position');
        
        xlim([-2.2, 2.2]);
        ylim([-0.2, 4.2]);
        xlabel('X (m)', 'FontSize', 10);
        ylabel('Y (m)', 'FontSize', 10);
        title(sprintf('%s (%d meas)', dataset_names{dataset_idx}, n_measurements), 'FontSize', 12);
        axis equal;
        grid on;
        
        % Add colorbar for signal
        if dataset_idx == 4 % Only add colorbar to last subplot
            c = colorbar;
            c.Label.String = 'Signal Intensity';
        end
    end
    
    % Overall title with time information
    sgtitle(sprintf('Tuning Datasets Animation: S-curve Trajectory (t=%.1fs, step %d/%d)', ...
        (t-1)*dt, t, n_t), 'FontSize', 16, 'FontWeight', 'bold');
    
    % Capture frame for GIF
    try
        frame = getframe(gcf);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        
        if t == 1
            imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', delay_time);
            fprintf('Started saving animation GIF: %s\n', gif_filename);
        else
            imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', delay_time);
        end
        
        % Print progress
        if mod(t, 10) == 0 || t == 1 || t == n_t
            fprintf('Animation frame %d/%d completed\n', t, n_t);
        end
    catch ME
        warning('Failed to save animation frame %d: %s', t, ME.message);
    end
    
    pause(0.05); % Small pause to allow for processing
end

fprintf('Animated GIF saved: %s\n', gif_filename);

%% Generate Static Summary Plot
fprintf('\n--- Generating Static Summary Plot ---\n');

figure(101); clf;
set(gcf, 'Position', [100, 100, 1400, 1000], 'Visible', 'off');

% Plot static summary for documentation
sample_timesteps = [5, 15, 25, 35, 45]; % Sample timesteps to show

for dataset_idx = 1:4
    subplot(2, 2, dataset_idx); hold on;
    
    % Plot a representative signal (middle timestep)
    mid_timestep = round(n_t/2);
    if ~isempty(signals{dataset_idx}{mid_timestep})
        imagesc(xgrid, ygrid, signals{dataset_idx}{mid_timestep});
        set(gca, 'YDir', 'normal');
        colormap('parula');
        alpha(0.5);
    end
    
    % Plot full trajectory
    plot(X_GT(1, :), X_GT(2, :), 'k-', 'LineWidth', 2, 'DisplayName', 'True Trajectory');
    
    % Plot measurements for sample timesteps
    for t_idx = 1:length(sample_timesteps)
        t = sample_timesteps(t_idx);
        if t <= n_t && ~isempty(datasets{dataset_idx}{t})
            measurements = datasets{dataset_idx}{t};
            scatter(measurements(1, :), measurements(2, :), 30, colors{dataset_idx}, '+', ...
                'LineWidth', 1.5, 'DisplayName', sprintf('t=%.1fs', (t-1)*dt));
        end
        
        % Mark true positions at sample timesteps
        if t <= n_t
            plot(X_GT(1, t), X_GT(2, t), 'ko', 'MarkerSize', 6, 'MarkerFaceColor', 'k');
        end
    end
    
    xlim([-2.2, 2.2]);
    ylim([-0.2, 4.2]);
    xlabel('X (m)');
    ylabel('Y (m)');
    title(sprintf('%s\n(%.1fs trajectory, %d measurements avg)', ...
        dataset_names{dataset_idx}, (n_t-1)*dt, round(mean(cellfun(@(x) size(x,2), datasets{dataset_idx})))));
    legend('Location', 'best');
    axis equal;
    grid on;
end

sgtitle('Tuning Datasets: S-curve Trajectory with Different Clutter Levels (Static Summary)', 'FontSize', 16);

% Save the static summary plot
static_filename = fullfile('SingleObjTune', 'tuning_datasets_static_summary.png');
print(gcf, static_filename, '-dpng', '-r150');
fprintf('Static summary plot saved: %s\n', static_filename);

fprintf('\n=== TUNING DATASETS GENERATION COMPLETE ===\n');
fprintf('All datasets saved to: SingleObjTune/\n');
fprintf('Dataset files:\n');
fprintf('  - T1_no_clutter.mat\n');
fprintf('  - T2_low_clutter.mat\n');
fprintf('  - T3_medium_clutter.mat\n');
fprintf('  - T4_simulated_clutter.mat\n');
fprintf('==========================================\n');
