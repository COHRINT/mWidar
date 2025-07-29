% TEST_HYBRID_PF Hybrid Particle Filter for Single Target Tracking
%   Combines continuous state space (Kalman dynamics) with discrete HMM likelihood
%
%   FILTER STRUCTURE:
%   1. Particles represent continuous state [x, y, vx, vy, ax, ay]
%   2. PREDICTION: Kalman filter dynamics with process noise
%   3. MEASUREMENT UPDATE: HMM likelihood on discretized position
%   4. RESAMPLING: Bootstrap resampling at beginning of each time step
%
%   ASSUMPTIONS:
%   - Measurements are from data association algorithm
%   - Bootstrap particle filter (sample from prior, weight by likelihood)
%   - Position measurements only (velocity/acceleration not observed)
%
%   STATE REPRESENTATION:
%   - Continuous particle states [x, y, vx, vy, ax, ay]
%   - Enhanced S-curve trajectory with analytical derivatives
%   - Kalman dynamics with process noise
%
%   ENVIRONMENTAL VARIABLES:
%   - PLOT_FLAG: Set to 1 to show plots, 0 to hide
%   - SAVE_FLAG: Set to 1 to save figures, 0 to disable saving
%   - SAVE_PATH: Directory path for saving figures
%
%   See also TEST_HMM, PFRESAMPLE

% Author: Anthony La Barca
% Date: 2025-06-17
clc, clear, close all

%% Environmental Variables

% Control plotting and saving behavior
PLOT_FLAG = 0; % Set to 1 to show plots, 0 to hide (but still create for saving)
SAVE_FLAG = 1; % Set to 1 to save figures, 0 to disable saving
SAVE_PATH = fullfile('..', '..', 'figures'); % Default save path for figures

% Override with environment variables if they exist
if exist('PLOT_FLAG_ENV', 'var')
    PLOT_FLAG = PLOT_FLAG_ENV;
end

if exist('SAVE_FLAG_ENV', 'var')
    SAVE_FLAG = SAVE_FLAG_ENV;
end

if exist('SAVE_PATH_ENV', 'var')
    SAVE_PATH = SAVE_PATH_ENV;
end

% Create save directory if saving is enabled
if SAVE_FLAG && ~exist(SAVE_PATH, 'dir')
    mkdir(SAVE_PATH);
    fprintf('Created save directory: %s\n', SAVE_PATH);
end

% Set figure visibility based on PLOT_FLAG
if PLOT_FLAG
    fig_visible = 'on';
else
    fig_visible = 'off';
end

fprintf('Configuration: PLOT_FLAG=%d, SAVE_FLAG=%d, SAVE_PATH=%s\n', PLOT_FLAG, SAVE_FLAG, SAVE_PATH);

%% Plotting settings

% LaTeX interpreter for text
set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex');
% Default figure properties
set(0, 'DefaultFigureColor', 'w'); % White background
set(0, 'DefaultAxesColor', 'none'); % Transparent axes background
set(0, 'DefaultAxesBox', 'on'); % Box around axes
set(0, 'DefaultLineLineWidth', 2); % Default line width
set(0, 'DefaultLineMarkerSize', 6); % Default marker size
set(0, 'DefaultAxesGridLineStyle', '--'); % Dashed grid lines
set(0, 'DefaultAxesGridAlpha', 0.5); % Grid transparency
set(0, 'DefaultAxesMinorGridAlpha', 0.3); % Minor grid transparency
set(0, 'DefaultAxesMinorGridLineStyle', ':'); % Dotted minor grid lines
% Text Size - Increased for better visibility
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultTextFontSize', 14);
set(0, 'DefaultAxesLineWidth', 1.2);
set(0, 'DefaultLegendFontSize', 12);
set(0, 'DefaultColorbarFontSize', 12);
set(0, 'DefaultAxesTitleFontSizeMultiplier', 1.1); % Smaller title font

%% Define State Space

STATE_STRING = {'x', 'y', 'vx', 'vy', 'ax', 'ay'}; % State variables
STATE_DIM = length(STATE_STRING); % Number of state variables
fprintf('State space defined with %d dimensions: %s\n', STATE_DIM, strjoin(STATE_STRING, ', '));

%% Define scene parameters

Xbounds = [-2 2]; %X bounds of scene
Ybounds = [0 4]; %Y bounds of scene
Xblind = []; %no blind zone
Yblind = []; %no blind zone

%% Generate enhanced S-curve trajectory with position, velocity, and acceleration

% Time parameters
num_steps = 50;
dt = 0.1; % Time step in seconds
t = (0:num_steps) * dt;

% Enhanced S-curve trajectory with smooth analytical dynamics
% Create a parametric trajectory that ensures smooth derivatives
% Start at (-1, 1) and move in an S-turn to (1, 3)
x_start = -1.5;
y_start = 0.5;
x_end = 1.5;
y_end = 3.5;

% Use parametric equations with time parameter tau = t/T where T is total time
T = t(end); % Total trajectory time
tau = t / T; % Normalized time [0, 1]

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
x_traj = max(Xbounds(1), min(Xbounds(2), x_traj));
y_traj = max(Ybounds(1), min(Ybounds(2), y_traj));

% Store full state trajectory [x, y, vx, vy, ax, ay]
state_traj = [x_traj; y_traj; vx_traj; vy_traj; ax_traj; ay_traj]; % 6 x (num_steps+1)

fprintf('Generated smooth analytical S-curve trajectory\n');
fprintf('Trajectory duration: %.1f seconds with %d time steps\n', T, num_steps + 1);
fprintf('Max speeds: Vx=%.2f m/s, Vy=%.2f m/s\n', max(abs(vx_traj)), max(abs(vy_traj)));
fprintf('Max accelerations: Ax=%.2f m/s², Ay=%.2f m/s²\n', max(abs(ax_traj)), max(abs(ay_traj)));

%% Generate simulated measurements with noise (position only)

% Add measurement noise to the true position trajectory
meas_noise_std = 0.01; % Standard deviation of measurement noise
measurements = state_traj(1:2, 1:end - 1) + meas_noise_std * randn(2, num_steps);

fprintf('Generated %d noisy position measurements\n', num_steps);

%% Load HMM model parameters

% Load likelihood model
load(fullfile('..', 'data', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');

%% Validate loaded parameters

if size(pointlikelihood_image, 1) ~= 128 ^ 2 || size(pointlikelihood_image, 2) ~= 128 ^ 2
    error('Likelihood model dimensions (%dx%d) do not match grid size (%d)', ...
        size(pointlikelihood_image, 1), size(pointlikelihood_image, 2), 128 ^ 2);
end

fprintf('Successfully loaded HMM matrices and validated dimensions.\n');

%% Define spatial grid

Lscene = 4; %physical length of scene in m (square shape)
npx = 128; %number of pixels in image (same in x&y dims)
npx2 = npx ^ 2;

xgrid = linspace(-2, 2, npx);
ygrid = linspace(0, Lscene, npx);
[pxgrid, pygrid] = meshgrid(xgrid, ygrid);
pxyvec = [pxgrid(:), pygrid(:)];
dx = xgrid(2) - xgrid(1);
dy = ygrid(2) - ygrid(1);

%% Initialize Particle Filter

N_particles = 10000; % Number of particles
state_dim = 6; % [x, y, vx, vy, ax, ay]

% Initialize particles around first measurement with uncertainty
init_pos_std = 0.2; % Position uncertainty
init_vel_std = 0.5; % Velocity uncertainty
init_acc_std = 0.1; % Acceleration uncertainty

% Initialize particle states (N_states x N_particles format)
particles = zeros(state_dim, N_particles);
particles(1, :) = measurements(1, 1); %+ init_pos_std * randn(N_particles, 1); % x
particles(2, :) = measurements(2, 1); %+ init_pos_std * randn(N_particles, 1); % y
particles(3, :) = state_traj(3, 1); %+init_vel_std* randn(N_particles, 1); % vx
particles(4, :) = state_traj(4, 1); %+init_vel_std* randn(N_particles, 1); % vy
particles(5, :) = state_traj(5, 1); %+init_acc_std* randn(N_particles, 1); % ax
particles(6, :) = state_traj(6, 1); %+init_acc_std* randn(N_particles, 1); % ay

% Initialize weights (uniform) - column vector
weights = ones(N_particles, 1) / N_particles;

% Kalman dynamics matrices for each particle
F = [1, 0, dt, 0, dt ^ 2/2, 0;
     0, 1, 0, dt, 0, dt ^ 2/2;
     0, 0, 1, 0, dt, 0;
     0, 0, 0, 1, 0, dt;
     0, 0, 0, 0, 1, 0;
     0, 0, 0, 0, 0, 1]; % State transition matrix

% Process noise covariance
% q_pos = 0.004; % Position process noise
% q_vel = 0.005; % Velocity process noise
% q_acc = 0.01; % Acceleration process noise

% Q = diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc]); % Process noise covariance
Q = 1e-2*eye(6); % Default process noise covariance

% Storage for results using cell arrays
particle_history = cell(1, num_steps + 1);
weight_history = cell(1, num_steps + 1);
mean_state_history = zeros(state_dim, num_steps + 1);
cov_history = zeros(state_dim, state_dim, num_steps + 1);

% Store initial state
particle_history{1} = particles;
weight_history{1} = weights;
mean_state_history(:, 1) = particles * weights;  % Weighted mean: [state_dim x 1]

fprintf('Initialized %d particles for hybrid PF\n', N_particles);

%% Setup visualization

fig = figure(1);
set(fig, 'Visible', fig_visible, 'Position', [100, 100, 1400, 600]);
hold on

% Initialize GIF saving if requested
if SAVE_FLAG
    gif_filename = fullfile(SAVE_PATH, 'hybrid_pf_animation.gif');
end

%% Apply Hybrid Particle Filter Updates

for kk = 1:num_steps
    fprintf('Processing time step %d/%d\n', kk, num_steps);

    % ========== RESAMPLING STEP (at beginning of each timestep) ==========
    if kk > 1 % Skip resampling at first step
        fprintf('\t-> Resampling step\n');
        % PFResample expects [state_dim x N_particles], which matches our format
        [particles, ~] = PFResample(particles, weights); 
    end

    % ========== PREDICTION STEP (Kalman dynamics) ==========
    fprintf('\t-> Prediction step (Kalman dynamics)\n');

    % Vectorized Kalman dynamics for all particles at once
    % Apply deterministic dynamics: particles = F * particles 
    particles = F * particles;
    
    % Add process noise to all particles at once
    process_noise = mvnrnd(zeros(1, state_dim), Q, N_particles)'; % [state_dim x N_particles]
    particles = particles + process_noise;
    
    % Vectorized bounds enforcement for position (rows 1 and 2)
    particles(1, :) = max(Xbounds(1), min(Xbounds(2), particles(1, :))); % x bounds
    particles(2, :) = max(Ybounds(1), min(Ybounds(2), particles(2, :))); % y bounds

    % ========== MEASUREMENT UPDATE STEP (HMM likelihood) ==========
    fprintf('\t-> Measurement update step (HMM likelihood)\n');

    % Get current measurement (assuming data association already done)
    current_meas = measurements(:, kk);
    fprintf('\t  Measurement: [%.3f, %.3f]\n', current_meas(1), current_meas(2));

    % Vectorized likelihood computation for all particles
    
    % Find measurement grid point (computed once, used for all particles)
    [~, meas_x_idx] = min(abs(xgrid - current_meas(1)));
    [~, meas_y_idx] = min(abs(ygrid - current_meas(2)));
    meas_linear_idx = sub2ind([npx, npx], meas_y_idx, meas_x_idx);
    
    % Vectorized grid point finding for all particles
    % Find closest x-grid indices for all particles at once
    % Use explicit broadcasting: particles(1,:) is [1 x N_particles], xgrid is [1 x npx]
    [~, px_indices] = min(abs(particles(1, :)' - xgrid), [], 2); % [N_particles x 1]
    [~, py_indices] = min(abs(particles(2, :)' - ygrid), [], 2); % [N_particles x 1]
    
    % Enforce boundary constraints vectorized
    px_indices = max(1, min(npx, px_indices));
    py_indices = max(1, min(npx, py_indices));
    
    % Convert to linear indices for all particles at once
    particle_linear_indices = sub2ind([npx, npx], py_indices, px_indices); % [N_particles x 1]
    
    % Vectorized likelihood lookup from pre-computed model
    likelihood_raw = pointlikelihood_image(meas_linear_idx, particle_linear_indices); % Should be [N_particles x 1]
    
    % Ensure likelihood_raw is a column vector
    if size(likelihood_raw, 1) == 1 && size(likelihood_raw, 2) == N_particles
        likelihood_raw = likelihood_raw'; % Transpose to column vector
    end
    
    % Vectorized Gaussian weighting for improved localization
    sf = 0.15;
    % Compute squared distances for all particles at once
    dx = particles(1, :) - current_meas(1); % [1 x N_particles] 
    dy = particles(2, :) - current_meas(2); % [1 x N_particles]
    dist_sq = dx.^2 + dy.^2; % [1 x N_particles]
    gauss_weights = exp(-dist_sq / (2 * sf^2)); % [1 x N_particles]
    
    % Combine likelihoods and normalize
    new_weights = (likelihood_raw .* gauss_weights') + eps; % Add small epsilon
    weights = new_weights / sum(new_weights);

    % Store results in cell arrays
    particle_history{kk + 1} = particles;
    weight_history{kk + 1} = weights;

    % Compute weighted statistics
    mean_state = particles * weights;  % [state_dim x 1] = [state_dim x N_particles] * [N_particles x 1]
    mean_state_history(:, kk + 1) = mean_state;

    % Vectorized weighted covariance computation
    particles_centered = particles - mean_state;  % Center particles around mean: [state_dim x N_particles] - [state_dim x 1]
    
    % Efficient vectorized covariance: C = (particles_centered * diag(weights) * particles_centered')
    % This is equivalent to: sum over p of weights(p) * particles_centered(:,p) * particles_centered(:,p)'
    weighted_particles = particles_centered .* sqrt(weights');  % [state_dim x N_particles]
    cov_weighted = weighted_particles * weighted_particles';   % [state_dim x state_dim]

    cov_history(:, :, kk + 1) = cov_weighted;

    %% ========== PLOTTING ==========

    % Plot particles and estimates (always create, but control visibility)
    figure(1); % Make sure we're on the right figure

    % Subplot 1: Position with full scene view and inset zoom
    ax1 = subplot(1, 3, 1); cla
    h_scatter = scatter(particles(1, :), particles(2, :), 20, weights, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on
    plot(mean_state(1), mean_state(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3)
    plot(state_traj(1, kk), state_traj(2, kk), 'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', [0.7 0.7 0.7])
    plot(current_meas(1), current_meas(2), '+', 'Color', [0.5 0.5 0.5], 'MarkerSize', 6, 'LineWidth', 1.5)

    % Draw rectangle indicating inset zoom region
    zoom_rect = rectangle('Position', [current_meas(1) - 0.3, current_meas(2) - 0.3, 0.6, 0.6], ...
        'EdgeColor', 'k', 'LineWidth', 1, 'LineStyle', '--');

    title(['Particles at $k=', num2str(kk), '$'])
    xlabel('$X$ (m)'), ylabel('$Y$ (m)')
    xlim([-2, 2]), ylim([0, 4]) % Fixed bounds as requested
    axis square % Make subplot square
    legend('Particles', 'PF Estimate', 'True Position', 'Measurement', 'Location', 'northwest')

    % Create inset zoom window positioned within the main subplot using data coordinates
    % Define inset location in data coordinates (within the main plot)
    inset_x_range = [- .5, 1.3]; % X position on the main plot where inset appears
    inset_y_range = [.1, 1.3]; % Y position on the main plot where inset appears

    % Convert data coordinates to normalized figure coordinates
    ax1_pos = get(ax1, 'Position'); % [left, bottom, width, height] in figure units

    % Map from data coordinates to axes coordinates (0 to 1)
    xlims = get(ax1, 'XLim'); % [-2, 2]
    ylims = get(ax1, 'YLim'); % [0, 4]

    % Normalize inset position within the main subplot
    inset_left_norm = (inset_x_range(1) - xlims(1)) / (xlims(2) - xlims(1));
    inset_bottom_norm = (inset_y_range(1) - ylims(1)) / (ylims(2) - ylims(1));
    inset_width_norm = (inset_x_range(2) - inset_x_range(1)) / (xlims(2) - xlims(1));
    inset_height_norm = (inset_y_range(2) - inset_y_range(1)) / (ylims(2) - ylims(1));

    % Convert to figure coordinates
    inset_pos = [ax1_pos(1) + inset_left_norm * ax1_pos(3), ... % Left edge
                     ax1_pos(2) + inset_bottom_norm * ax1_pos(4), ... % Bottom edge
                     inset_width_norm * ax1_pos(3), ... % Width
                     inset_height_norm * ax1_pos(4)]; % Height

    inset_ax = axes('Position', inset_pos);
    scatter(particles(1, :), particles(2, :), 15, weights, 'filled', 'MarkerFaceAlpha', 0.7);
    hold on
    plot(mean_state(1), mean_state(2), 'ro', 'MarkerSize', 8, 'LineWidth', 2)
    plot(state_traj(1, kk), state_traj(2, kk), 'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 5, 'LineWidth', 1.5, 'MarkerFaceColor', [0.7 0.7 0.7])
    plot(current_meas(1), current_meas(2), '+', 'Color', [0.5 0.5 0.5], 'MarkerSize', 6, 'LineWidth', 2)
    xlim([current_meas(1) - 0.3, current_meas(1) + 0.3])
    ylim([current_meas(2) - 0.3, current_meas(2) + 0.3])
    axis square % Make inset square
    set(inset_ax, 'FontSize', 8, 'Box', 'on', 'XTick', [], 'YTick', [])

    % Return focus to main subplot for proper legend placement
    axes(ax1);

    % Subplot 2: Velocity estimates
    subplot(1, 3, 2), cla
    scatter(particles(3, :), particles(4, :), 20, weights, 'filled', 'MarkerFaceAlpha', 0.6)
    hold on
    plot(mean_state(3), mean_state(4), 'ro', 'MarkerSize', 10, 'LineWidth', 3)
    plot(state_traj(3, kk), state_traj(4, kk), 'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', [0.7 0.7 0.7])
    title(['Velocity at $k=', num2str(kk), '$'])
    xlabel('$V_x$ (m/s)'), ylabel('$V_y$ (m/s)')
    axis square % Make subplot square
    legend('Particles', 'PF Estimate', 'True Velocity', 'Location', 'northwest')

    % Subplot 3: Acceleration estimates
    subplot(1, 3, 3), cla
    scatter(particles(5, :), particles(6, :), 20, weights, 'filled', 'MarkerFaceAlpha', 0.6)
    hold on
    plot(mean_state(5), mean_state(6), 'ro', 'MarkerSize', 10, 'LineWidth', 3)
    plot(state_traj(5, kk), state_traj(6, kk), 'o', 'Color', [0.7 0.7 0.7], 'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', [0.7 0.7 0.7])
    title(['Acceleration at $k=', num2str(kk), '$'])
    xlabel('$A_x$ (m/s$^2$)'), ylabel('$A_y$ (m/s$^2$)')
    axis square % Make subplot square
    legend('Particles', 'PF Estimate', 'True Acceleration', 'Location', 'northwest')

    % Add single colorbar on the right spanning full height
    cb = colorbar('Position', [0.92, 0.15, 0.02, 0.7]);
    cb.Label.String = 'Particle Weight';
    cb.Label.Interpreter = 'latex';

    % Save frame to GIF if requested
    if SAVE_FLAG
        frame = getframe(gcf);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);

        if kk == 1
            imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
        else
            imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end

    end

    % Pause only if plots are visible
    if PLOT_FLAG
        pause(0.001); % Small pause to see animation
    end

    % Compute estimation errors
    pos_error = mean_state(1:2) - state_traj(1:2, kk);
    vel_error = mean_state(3:4) - state_traj(3:4, kk);
    acc_error = mean_state(5:6) - state_traj(5:6, kk);

    fprintf('\t  Position error: [%.3f, %.3f] m\n', pos_error(1), pos_error(2));
    fprintf('\t  Velocity error: [%.3f, %.3f] m/s\n', vel_error(1), vel_error(2));
    fprintf('\t  Acceleration error: [%.3f, %.3f] m/s²\n', acc_error(1), acc_error(2));
end

return
%% Plot results summary

fig2 = figure(2);
set(fig2, 'Visible', fig_visible, 'Position', [200, 100, 1200, 800]);
clf(2)

% Plot trajectory comparison
subplot(2, 3, 1)
plot(state_traj(1, 1:num_steps), state_traj(2, 1:num_steps), 'b-', 'LineWidth', 2)
hold on
plot(measurements(1, :), measurements(2, :), 'ro', 'MarkerSize', 4)
plot(mean_state_history(1, 1:num_steps), mean_state_history(2, 1:num_steps), 'g--', 'LineWidth', 2)
xlabel('$X$ position (m)')
ylabel('$Y$ position (m)')
title('Trajectory Comparison')
legend('True trajectory', 'Measurements', 'PF estimate', 'Location', 'best')
grid on
axis equal
xlim(Xbounds)
ylim(Ybounds)

% Plot position errors
subplot(2, 3, 2)
tvec = 1:num_steps;
pos_errors = mean_state_history(1:2, 1:num_steps) - state_traj(1:2, 1:num_steps);
plot(tvec, pos_errors(1, :), 'r-o', tvec, pos_errors(2, :), 'b-s')
xlabel('Time step')
ylabel('Position error (m)')
title('Position Estimation Errors')
legend('$X$ error', '$Y$ error', 'Location', 'best')
grid on

% Plot velocity errors
subplot(2, 3, 3)
vel_errors = mean_state_history(3:4, 1:num_steps) - state_traj(3:4, 1:num_steps);
plot(tvec, vel_errors(1, :), 'r-o', tvec, vel_errors(2, :), 'b-s')
xlabel('Time step')
ylabel('Velocity error (m/s)')
title('Velocity Estimation Errors')
legend('$V_x$ error', '$V_y$ error', 'Location', 'best')
grid on

% Plot acceleration errors
subplot(2, 3, 4)
acc_errors = mean_state_history(5:6, 1:num_steps) - state_traj(5:6, 1:num_steps);
plot(tvec, acc_errors(1, :), 'r-o', tvec, acc_errors(2, :), 'b-s')
xlabel('Time step')
ylabel('Acceleration error (m/s$^2$)')
title('Acceleration Estimation Errors')
legend('$A_x$ error', '$A_y$ error', 'Location', 'best')
grid on

% Plot effective sample size
subplot(2, 3, 5)
Neff = zeros(1, num_steps + 1);

for k = 1:num_steps + 1
    Neff(k) = 1 / sum(weight_history{k} .^ 2);
end

plot(0:num_steps, Neff, 'k-', 'LineWidth', 2)
hold on
plot([0, num_steps], [N_particles / 2, N_particles / 2], 'r--', 'LineWidth', 2)
xlabel('Time step')
ylabel('Effective sample size')
title('Particle Filter Health')
legend('$N_{\mathrm{eff}}$', '$N/2$ threshold', 'Location', 'best')
grid on

% Plot RMSE evolution
subplot(2, 3, 6)
rmse_pos = sqrt(pos_errors(1, :) .^ 2 + pos_errors(2, :) .^ 2);
rmse_vel = sqrt(vel_errors(1, :) .^ 2 + vel_errors(2, :) .^ 2);
rmse_acc = sqrt(acc_errors(1, :) .^ 2 + acc_errors(2, :) .^ 2);
plot(tvec, rmse_pos, 'r-', tvec, rmse_vel, 'g--', tvec, rmse_acc, 'b:')
xlabel('Time step')
ylabel('RMSE')
title('Root Mean Square Errors')
legend({'Position', 'Velocity', 'Acceleration'}, 'Location', 'best')
grid on

% Save figure if requested
if SAVE_FLAG
    print(gcf, fullfile(SAVE_PATH, 'hybrid_pf_results_summary.png'), '-dpng', '-r600');
end

%% Estimation Error Visualization

fig3 = figure(3);
set(fig3, 'Visible', fig_visible, 'Position', [300, 100, 1200, 600]);
clf(3)
idx = 1;

for state = [1, 3, 5, 2, 4, 6]

    covariance = cov_history(state, state, 1:num_steps);
    std2 = sqrt(squeeze(covariance));
    subplot(2, 3, idx)
    hold on
    % Compute mean state for this variable
    mean_state_vec = mean_state_history(state, 1:num_steps);
    % Upper and lower bounds for ±2 std
    upper = 2 * std2';
    lower =- 2 * std2';
    % Fill area between upper and lower bounds
    fill([tvec, fliplr(tvec)], [upper, fliplr(lower)], ...
        'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    % Plot estimation error
    plot(tvec, mean_state_vec - state_traj(state, 1:num_steps), 'b-', 'LineWidth', 2);

    ylabel([STATE_STRING{state}, ' Error'])
    xlabel('Time step')
    xlim([0, tvec(end)])
    grid on;
    idx = idx + 1; % Increment index for next subplot

end

sgtitle('Estimation Errors with ±2 Std Dev Bounds')

% Save figure if requested
if SAVE_FLAG
    print(gcf, fullfile(SAVE_PATH, 'hybrid_pf_error_visualization.png'), '-dpng', '-r600');
end

%% State Trajectory Visualization

fig4 = figure(4);
set(fig4, 'Visible', fig_visible, 'Position', [500, 100, 1200, 600]);
clf(4)
% Plot the true trajectory and estimated trajectory
h_true = gobjects(STATE_DIM, 1);
h_est = gobjects(STATE_DIM, 1);
idx = 1; % Time index for plotting

for state = [1, 3, 5, 2, 4, 6]
    subplot(2, 3, idx)
    hold on
    % Plot true trajectory for this state
    h_true(state) = plot(t, state_traj(state, :), 'k-', 'LineWidth', 2);
    % Plot estimated trajectory for this state
    h_est(state) = plot(t, mean_state_history(state, :), 'b--', 'LineWidth', 2);

    xlabel('Time (s)')
    xlim([0, t(end)])
    ylabel(STATE_STRING{state})
    title(['State: ', STATE_STRING{state}])
    grid on;
    idx = idx + 1; % Increment index for next subplot
end

sgtitle('State Trajectories: True vs Estimated')

% Add legend to the first subplot
subplot(2, 3, 1)
legend([h_true(1), h_est(1)], {'True Trajectory', 'Estimated Trajectory'}, ...
    'Location', 'best', 'Interpreter', 'latex');

% Save figure if requested
if SAVE_FLAG
    print(gcf, fullfile(SAVE_PATH, 'hybrid_pf_state_trajectories.png'), '-dpng', '-r600');
end

%% Weight history visualization

% Weight history plot can be useful to see how particle weights evolve
% Not useful because all information is encapsulated in effective sample size
fig5 = figure(5); % Always hidden since it's for analysis only
set(fig5, 'Visible', fig_visible, 'Position', [400, 100, 1200, 800]);
clf(5)

% Create a 2x2 subplot layout for comprehensive weight analysis
subplot(2, 2, 1)
% Weight distribution statistics over time
weight_stats = zeros(5, num_steps + 1); % [min, 25th, median, 75th, max]

for k = 1:num_steps + 1
    weight_stats(:, k) = prctile(weight_history{k}, [0, 25, 50, 75, 100]);
end

fill([0:num_steps, fliplr(0:num_steps)], ...
    [weight_stats(1, :), fliplr(weight_stats(5, :))], ...
    [0.8 0.8 0.8], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on
fill([0:num_steps, fliplr(0:num_steps)], ...
    [weight_stats(2, :), fliplr(weight_stats(4, :))], ...
    [0.6 0.6 0.6], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
plot(0:num_steps, weight_stats(3, :), 'k-', 'LineWidth', 2);
xlabel('Time step');
ylabel('Particle weight');
title('Weight Distribution Over Time');
legend({'Min-Max Range', '25th-75th Percentile', 'Median'}, 'Location', 'best');
grid on

subplot(2, 2, 2)
% Effective sample size
plot(0:num_steps, Neff, 'b-', 'LineWidth', 2);
hold on
plot([0, num_steps], [N_particles / 2, N_particles / 2], 'r--', 'LineWidth', 2);
xlabel('Time step');
ylabel('Effective sample size');
title('Effective Sample Size Over Time');
legend({'$N_{\mathrm{eff}}$', '$N/2$ threshold'}, 'Location', 'best');
grid on
ylim([0 N_particles]);

subplot(2, 2, 3)
% Weight concentration (entropy-based measure)
weight_entropy = zeros(1, num_steps + 1);

for k = 1:num_steps + 1
    w = weight_history{k};
    w = w(w > 0); % Remove zero weights for log calculation
    weight_entropy(k) = -sum(w .* log(w));
end

plot(0:num_steps, weight_entropy, 'g-', 'LineWidth', 2);
xlabel('Time step');
ylabel('Weight entropy');
title('Weight Diversity (Higher = More Diverse)');
grid on

subplot(2, 2, 4)
% Histogram of final weights
histogram(weight_history{end}, 50, 'FaceColor', [0.7 0.7 0.9], 'EdgeColor', 'k');
xlabel('Final particle weight');
ylabel('Number of particles');
title('Final Weight Distribution');
grid on

% Save figure if requested
if SAVE_FLAG
    print(gcf, fullfile(SAVE_PATH, 'hybrid_pf_weight_analysis.png'), '-dpng', '-r600');
end

%% Print performance summary

final_rmse_pos = sqrt(mean(pos_errors(1, :) .^ 2 + pos_errors(2, :) .^ 2));
final_rmse_vel = sqrt(mean(vel_errors(1, :) .^ 2 + vel_errors(2, :) .^ 2));
final_rmse_acc = sqrt(mean(acc_errors(1, :) .^ 2 + acc_errors(2, :) .^ 2));

fprintf('\n=== Hybrid Particle Filter Performance Summary ===\n');
fprintf('Number of particles: %d\n', N_particles);
fprintf('RMSE Position error: %.4f m\n', final_rmse_pos);
fprintf('RMSE Velocity error: %.4f m/s\n', final_rmse_vel);
fprintf('RMSE Acceleration error: %.4f m/s²\n', final_rmse_acc);
fprintf('Mean measurement noise std: %.4f m\n', meas_noise_std);
fprintf('Trajectory duration: %.1f seconds (%d steps)\n', t(end), num_steps);
fprintf('Average effective sample size: %.1f\n', mean(Neff));
fprintf('Minimum effective sample size: %.1f\n', min(Neff));
