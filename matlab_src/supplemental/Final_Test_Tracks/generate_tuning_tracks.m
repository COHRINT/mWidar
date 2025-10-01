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
    Signal_1{i} = blurred;
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
    clutter_y = 0 + 4 * rand(1, n_clutter); % Uniform in [0, 4]
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
    Signal_2{i} = blurred;
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
    uniform_clutter_y = 0 + 4 * rand(1, n_uniform_clutter); % Uniform in [0, 4]
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
    Signal_3{i} = blurred;
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

% Setup figure for CA_CFAR tuning visualization
figure(1); clf;
set(gcf, 'Position', [100, 100, 1200, 800]);

% CA-CFAR parameters
Pfa = 0.365; % Probability of false alarm -- 36 WORKS THE BEST
Ng = 5;    % Guard cells
Nr = 20;   % Training cells

% Initialize animation saving
animation_filename = fullfile('SingleObjTune', 'CA_CFAR_Tuning_Animation.gif');
frame_delay = 0.5; % Delay between frames in seconds

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
    signal_normalized = blurred;

    % Run CFAR detector to find peaks (this will include clutter detections)
    try
        [~, peak_x, peak_y] = CA_CFAR(signal_normalized, Pfa, Ng, Nr);

        if ~isempty(peak_x) && ~isempty(peak_y)
            % Convert detector indices back to world coordinates
            pvinds = sub2ind([npx, npx], peak_x, peak_y);
            y_4{i} = [pxgrid(pvinds)'; pygrid(pvinds)'];

            % Remove any detections that are in the lower .5 (y < 0.5 m)
            valid_idx = y_4{i}(2, :) >= 0.5;
            y_4{i} = y_4{i}(:, valid_idx);
        
        else
            fprintf('No detections found\n');
            y_4{i} = [];
        end

    catch
        % If detector fails, just use the true position with some noise
        fprintf('CA_CFAR failed, using noisy true position\n');
        y_4{i} = true_pos + 0.1 * randn(2, 1);
    end

    Signal_4{i} = signal_normalized;

    % Plot for CA_CFAR tuning (show all frames)
    clf;

        % Plot signal
        subplot(2, 2, 1);
        imagesc(xgrid, ygrid, signal_normalized);
        set(gca, 'YDir', 'normal');
        colorbar;
        title(sprintf('mWidar Signal (t=%.1fs)', (i - 1) * dt));
        xlabel('X (m)'); ylabel('Y (m)');

        % Plot true position
        hold on;
        plot(true_pos(1), true_pos(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

        % Plot detected measurements
        subplot(2, 2, 2);
        imagesc(xgrid, ygrid, signal_normalized); hold on;
        set(gca, 'YDir', 'normal');

        if ~isempty(y_4{i})
            scatter(y_4{i}(1, :), y_4{i}(2, :), 50, 'r+', 'LineWidth', 2);
        end

        plot(true_pos(1), true_pos(2), 'wo', 'MarkerSize', 8, 'LineWidth', 2);
        title(sprintf('CA_CFAR Detections (%d found)', size(y_4{i}, 2)));
        xlabel('X (m)'); ylabel('Y (m)');

        % Plot trajectory so far
        subplot(2, 2, 3);
        plot(X_GT(1, 1:i), X_GT(2, 1:i), 'k-', 'LineWidth', 2); hold on;

        if ~isempty(y_4{i})
            scatter(y_4{i}(1, :), y_4{i}(2, :), 30, 'r+', 'LineWidth', 1.5);
        end

        plot(true_pos(1), true_pos(2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
        xlim([-2.2, 2.2]); ylim([-0.2, 4.2]);
        grid on; axis equal;
        title('Trajectory with Detections');
        xlabel('X (m)'); ylabel('Y (m)');

        % Plot detection statistics
        subplot(2, 2, 4);
        n_detections = cellfun(@(x) size(x, 2), y_4(1:i));
        plot(1:i, n_detections, 'b-o', 'LineWidth', 2);
        xlabel('Time Step'); ylabel('Number of Detections');
        title('Detection Count Over Time');
        grid on;

        sgtitle(sprintf('CA_CFAR Tuning: Step %d/%d (Pfa=%.2f, guard=%d, train=%d)', i, n_t, Pfa, Ng, Nr));
        drawnow;
        
        % Capture frame for animation
        frame = getframe(gcf);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        
        % Write to GIF file
        if i == 1
            imwrite(imind, cm, animation_filename, 'gif', 'Loopcount', inf, 'DelayTime', frame_delay);
        else
            imwrite(imind, cm, animation_filename, 'gif', 'WriteMode', 'append', 'DelayTime', frame_delay);
        end
        
        pause(0.1);

end

Data.GT = X_GT;
Data.y = y_4;
Data.signal = Signal_4;
save(fullfile('SingleObjTune', 'T4_simulated_clutter.mat'), 'Data', '-mat')
fprintf('Saved: T4_simulated_clutter.mat\n');

%% Create Comprehensive Animation Showing All Four Detector Scenarios
fprintf('\n--- Creating Comprehensive Tuning Datasets Animation ---\n');

% Setup figure for comprehensive animation
figure(2); clf;
set(gcf, 'Position', [150, 150, 1400, 900]);

% Initialize comprehensive animation saving
comprehensive_animation_filename = fullfile('SingleObjTune', 'Tuning_Datasets_Animation.gif');
comp_frame_delay = 0.8; % Slightly slower for better visibility

for i = 1:n_t
    clf;
    
    % Get current true position
    true_pos = X_GT(1:2, i);
    current_time = (i-1) * dt;
    
    %% Subplot 1: No Clutter
    subplot(2, 2, 1);
    imagesc(xgrid, ygrid, Signal_1{i});
    set(gca, 'YDir', 'normal');
    colormap(gca, parula);
    hold on;
    
    % Plot true position
    plot(true_pos(1), true_pos(2), 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
    % Plot measurements if any
    if ~isempty(y_1{i})
        scatter(y_1{i}(1, :), y_1{i}(2, :), 40, 'w+', 'LineWidth', 2);
    end
    
    xlim([-3, 3]); ylim([0, 4]);
    xlabel('X (m)'); ylabel('Y (m)');
    title(sprintf('No Clutter (%d meas)', size(y_1{i}, 2)));
    grid on; set(gca, 'GridAlpha', 0.3);
    
    %% Subplot 2: Low Clutter
    subplot(2, 2, 2);
    imagesc(xgrid, ygrid, Signal_2{i});
    set(gca, 'YDir', 'normal');
    colormap(gca, parula);
    hold on;
    
    % Plot true position
    plot(true_pos(1), true_pos(2), 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
    % Plot measurements if any
    if ~isempty(y_2{i})
        scatter(y_2{i}(1, :), y_2{i}(2, :), 40, 'g+', 'LineWidth', 2);
    end
    
    xlim([-3, 3]); ylim([0, 4]);
    xlabel('X (m)'); ylabel('Y (m)');
    title(sprintf('Low Clutter (%d meas)', size(y_2{i}, 2)));
    grid on; set(gca, 'GridAlpha', 0.3);
    
    %% Subplot 3: Medium Clutter
    subplot(2, 2, 3);
    imagesc(xgrid, ygrid, Signal_3{i});
    set(gca, 'YDir', 'normal');
    colormap(gca, parula);
    hold on;
    
    % Plot true position
    plot(true_pos(1), true_pos(2), 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
    % Plot measurements if any
    if ~isempty(y_3{i})
        scatter(y_3{i}(1, :), y_3{i}(2, :), 40, 'b+', 'LineWidth', 2);
    end
    
    xlim([-3, 3]); ylim([0, 4]);
    xlabel('X (m)'); ylabel('Y (m)');
    title(sprintf('Medium Clutter (%d meas)', size(y_3{i}, 2)));
    grid on; set(gca, 'GridAlpha', 0.3);
    
    %% Subplot 4: Simulated Clutter
    subplot(2, 2, 4);
    imagesc(xgrid, ygrid, Signal_4{i});
    set(gca, 'YDir', 'normal');
    colorbar;
    hold on;
    
    % Plot true position
    plot(true_pos(1), true_pos(2), 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
    
    % Plot measurements if any
    if ~isempty(y_4{i})
        scatter(y_4{i}(1, :), y_4{i}(2, :), 40, 'r+', 'LineWidth', 2);
    end
    
    xlim([-3, 3]); ylim([0, 4]);
    xlabel('X (m)'); ylabel('Y (m)');
    title(sprintf('Simulated Clutter (%d meas)', size(y_4{i}, 2)));
    grid on; set(gca, 'GridAlpha', 0.3);
    
    % Add overall title
    sgtitle(sprintf('Tuning Datasets Animation: S-curve Trajectory (t=%.1fs, step %d/%d)', current_time, i, n_t), 'FontSize', 14, 'FontWeight', 'bold');
    
    % Ensure tight layout
    drawnow;
    
    % Capture frame for comprehensive animation
    frame = getframe(gcf);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    % Write to GIF file
    if i == 1
        imwrite(imind, cm, comprehensive_animation_filename, 'gif', 'Loopcount', inf, 'DelayTime', comp_frame_delay);
    else
        imwrite(imind, cm, comprehensive_animation_filename, 'gif', 'WriteMode', 'append', 'DelayTime', comp_frame_delay);
    end
    
    % Small pause for display
    pause(0.05);
end

fprintf('Comprehensive animation saved to: %s\n', comprehensive_animation_filename);

fprintf('\n=== TUNING DATASETS GENERATION COMPLETE ===\n');
fprintf('All datasets saved to: SingleObjTune/\n');
fprintf('Dataset files:\n');
fprintf('  - T1_no_clutter.mat\n');
fprintf('  - T2_low_clutter.mat\n');
fprintf('  - T3_medium_clutter.mat\n');
fprintf('  - T4_simulated_clutter.mat\n');
fprintf('Animations saved:\n');
fprintf('  - %s (CA-CFAR tuning)\n', animation_filename);
fprintf('  - %s (comprehensive datasets)\n', comprehensive_animation_filename);
fprintf('==========================================\n');
