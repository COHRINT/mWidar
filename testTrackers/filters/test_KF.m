% TEST_KF Kalman Filter for Single Target Tracking
%   Linear Gaussian state space model with continuous approximation
%
%   FILTER STRUCTURE:
%   1. PREDICTION STEP: Apply linear dynamics with Gaussian noise
%   2. MEASUREMENT UPDATE STEP: Linear observation model with Gaussian noise
%   3. STATE ESTIMATION: Gaussian posterior (analytical solution)
%
%   ASSUMPTIONS:
%   - Measurements are already from a data association algorithm
%   - Each measurement corresponds to the single target at each time step
%   - No clutter measurements or missed detections
%   - Linear dynamics and observation models
%   - Gaussian noise
%
%   STATE REPRESENTATION:
%   - Continuous Gaussian distribution over state [x, y, vx, vy]
%   - Kalman dynamics with constant velocity model
%
%   ENVIRONMENTAL VARIABLES:
%   - PLOT_FLAG: Set to 1 to show plots, 0 to hide
%   - SAVE_FLAG: Set to 1 to save figures, 0 to disable saving
%   - SAVE_PATH: Directory path for saving figures
%
%   See also TEST_HMM, TEST_HYBRID_PF

% Author: Anthony La Barca
% Date: 2025-02-08

clc, clear, close all

%% Environmental Variables
% Control plotting and saving behavior
PLOT_FLAG = 1; % Set to 1 to show plots, 0 to hide (but still create for saving)
SAVE_FLAG = 1; % Set to 1 to save figures, 0 to disable saving
SAVE_PATH = fullfile('..', '..', 'figures', 'TESTINGSCRIPTS'); % Default save path for figures

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

%% Add path to KF class
addpath(fullfile('..', '..', 'matlab_src', 'DA_Track'));

%% Define scene parameters
Xbounds = [-2 2]; %X bounds of scene
Ybounds = [0 4]; %Y bounds of scene

%% Generate enhanced S-curve trajectory (position and velocity for KF)
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

% Ensure trajectory stays within bounds
x_traj = max(Xbounds(1), min(Xbounds(2), x_traj));
y_traj = max(Ybounds(1), min(Ybounds(2), y_traj));

% Store single target trajectory (position and velocity for KF)
state_traj = [x_traj; y_traj; vx_traj; vy_traj]; % 4 x (num_steps+1) array

fprintf('Generated smooth analytical S-curve trajectory\n');
fprintf('Trajectory duration: %.1f seconds with %d time steps\n', T, num_steps + 1);

%% Generate simulated measurements with noise (position only)
% Add measurement noise to the true position trajectory
meas_noise_std = 0.01; % Standard deviation of measurement noise
measurements = state_traj(1:2, 1:end-1) + meas_noise_std * randn(2, num_steps);

fprintf('Generated %d noisy position measurements\n', num_steps);

%% Define spatial grid for visualization
Lscene = 4; %physical length of scene in m (square shape)
npx = 128; %number of pixels in image (same in x&y dims)
npx2 = npx ^ 2;

xgrid = linspace(-2, 2, npx);
ygrid = linspace(0, Lscene, npx);
[pxgrid, pygrid] = meshgrid(xgrid, ygrid);
pxyvec = [pxgrid(:), pygrid(:)];
dx = xgrid(2) - xgrid(1);
dy = ygrid(2) - ygrid(1);

fprintf('Spatial grid: %dx%d pixels, %.4fm resolution\n', npx, npx, dx);

%% Initialize Kalman Filter
% State: [x, y, vx, vy]
% Dynamics: constant velocity model
F = [1, 0, dt, 0;
     0, 1, 0, dt;
     0, 0, 1, 0;
     0, 0, 0, 1]; % State transition matrix

% Process noise covariance (tuned for trajectory)
q_pos = 0.01; % Position process noise
q_vel = 0.05; % Velocity process noise
Q = diag([q_pos, q_pos, q_vel, q_vel]); % Process noise covariance

% Observation model (position only)
H = [1, 0, 0, 0;
     0, 1, 0, 0]; % Observation matrix

% Measurement noise covariance
R = meas_noise_std^2 * eye(2); % Measurement noise covariance

% Initial state estimate (from first measurement)
x0 = [measurements(1, 1); measurements(2, 1); 0; 0]; % Initial state
P0 = diag([0.1, 0.1, 0.5, 0.5]); % Initial covariance

% Create Kalman Filter instance
kf = KF(x0, P0, F, Q, H, R);

% Storage for results
x_hist = zeros(4, num_steps+1);
P_hist = zeros(4, 4, num_steps+1);
x_hist(:, 1) = x0;
P_hist(:, :, 1) = P0;

fprintf('Initialized Kalman Filter\n');

%% Setup visualization
fig = figure(1);
set(fig, 'Visible', fig_visible, 'Position', [100, 100, 600, 500]);
% Pause to make sure it draws
fprintf("Pausing to generate figure\n");
pause(2);

% Initialize GIF saving if requested
if SAVE_FLAG
    gif_filename = fullfile(SAVE_PATH, 'kf_animation.gif');
end

%% Apply Kalman Filter Updates
for kk = 1:num_steps
    fprintf('Processing time step %d/%d\n', kk, num_steps);

    %% ========== PREDICTION STEP ==========
    fprintf('\t-> Prediction step\n');
    kf.prediction();
    x_pred = kf.x;
    P_pred = kf.P;

    %% ========== MEASUREMENT UPDATE STEP ==========
    fprintf('\t-> Measurement update step\n');
    current_meas = measurements(:, kk);
    fprintf('\t  Measurement: [%.3f, %.3f]\n', current_meas(1), current_meas(2));

    kf.measurement_update(current_meas);
    x_post = kf.x;
    P_post = kf.P;

    % Store results
    x_hist(:, kk+1) = x_post;
    P_hist(:, :, kk+1) = P_post;

    %% ========== PLOTTING ==========
    % Plot KF - just show the Gaussian posterior evolution
    figure(1);
    clf;
    
    % Single plot showing the Gaussian posterior
    % Convert Kalman Gaussian to grid representation for visualization
    posterior_density = mvnpdf(pxyvec, x_post(1:2)', P_post(1:2, 1:2));
    posterior_density = reshape(posterior_density, [npx, npx]);
    
    surf(xgrid, ygrid, posterior_density, 'EdgeColor', 'none'), view(2)
    hold on
    
    % Plot ground truth
    plot3(state_traj(1, kk), state_traj(2, kk), max(posterior_density(:))*1.1, 'p', ...
        'Color', [0.7 0.7 0.7], 'MarkerSize', 20, 'LineWidth', 2)
    
    % Plot measurement
    plot3(current_meas(1), current_meas(2), max(posterior_density(:))*1.1, '+', ...
        'Color', 'c', 'MarkerSize', 10, 'LineWidth', 2)
    
    % Plot KF estimate (mean)
    plot3(x_post(1), x_post(2), max(posterior_density(:))*1.1, 'ro', ...
        'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r')
    
    % Plot trajectory history
    if kk > 1
        plot3(x_hist(1, 1:kk+1), x_hist(2, 1:kk+1), ...
            max(posterior_density(:))*1.1*ones(1,kk+1), 'r-', 'LineWidth', 1)
        plot3(state_traj(1, 1:kk), state_traj(2, 1:kk), ...
            max(posterior_density(:))*1.1*ones(1,kk), 'Color', [0.7 0.7 0.7], 'LineWidth', 1)
    end
    
    title(sprintf('Kalman Filter Posterior: $k=%d$', kk), 'Interpreter', 'latex', 'FontSize', 16)
    xlabel('$X$ (m)'), ylabel('$Y$ (m)')
    xlim([-2, 2]), ylim([0, 4])
    axis square
    colormap('hot')
    c = colorbar;
    c.TickLabelInterpreter = 'latex';
    legend('', 'True State', 'Measurement', 'KF Estimate', 'Location', 'southeast')
    
    % Add text with innovation info
    innovation = kf.z;
    innov_cov = kf.S;
    text(-1.8, 3.7, sprintf('Innovation: [%.3f, %.3f]', innovation(1), innovation(2)), ...
        'FontSize', 10, 'BackgroundColor', 'white')
    text(-1.8, 3.4, sprintf('$||\\nu||$: %.3f m', norm(innovation)), ...
        'Interpreter', 'latex', 'FontSize', 10, 'BackgroundColor', 'white')

    % Save frame to GIF if requested
    if SAVE_FLAG
        frame = getframe(gcf);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);

        if kk == 1
            imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.2);
        else
            imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2);
        end
    end

    % Pause only if plots are visible
    if PLOT_FLAG
        pause(0.2); % Small pause to see animation
    end

    fprintf('\t  Position error: [%.3f, %.3f] m\n', x_post(1) - state_traj(1, kk), x_post(2) - state_traj(2, kk));
end

%% Performance summary
pos_errors = x_hist(1:2, 2:end) - state_traj(1:2, 1:end-1);
rmse_pos = sqrt(mean(pos_errors(1, :).^2 + pos_errors(2, :).^2));

fprintf('\n=== Kalman Filter Single Target Tracker Performance Summary ===\n');
fprintf('RMSE position error: %.4f m\n', rmse_pos);
fprintf('Mean measurement noise std: %.4f m\n', meas_noise_std);
fprintf('Trajectory duration: %.1f seconds (%d steps)\n', t(end), num_steps);
fprintf('Final position error: %.4f m\n', norm(pos_errors(:, end)));
