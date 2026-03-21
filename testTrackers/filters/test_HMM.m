% TEST_HMM Hidden Markov Model for Single Target Tracking
%   Discrete state space Bayesian filter using pre-computed transition and likelihood models
%
%   FILTER STRUCTURE:
%   1. PREDICTION STEP: Apply state transition model P(x_k | x_{k-1})
%   2. MEASUREMENT UPDATE STEP: Apply likelihood P(z_k | x_k) to get posterior
%   3. STATE ESTIMATION: Compute MMSE and MAP estimates from posterior distribution
%
%   ASSUMPTIONS:
%   - Measurements are already from a data association algorithm
%   - Each measurement corresponds to the single target at each time step
%   - No clutter measurements or missed detections
%   - State space is discretized on a regular grid
%
%   STATE REPRESENTATION:
%   - Discrete probability distribution over 128x128 spatial grid
%   - Position-only tracking (no velocity/acceleration states)
%   - Grid covers scene bounds: X\in[-2,2]m, Y\in[0,4]m
%
%   ENVIRONMENTAL VARIABLES:
%   - PLOT_FLAG: Set to 1 to show plots, 0 to hide
%   - SAVE_FLAG: Set to 1 to save figures, 0 to disable saving
%   - SAVE_PATH: Directory path for saving figures
%
%   See also TEST_KF, HMM, PDA_HMM

% Author: Anthony La Barca
% Date: 2026-03-02

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

%% Add path to HMM class
addpath(fullfile('..', '..', 'matlab_src', 'DA_Track'));

%% Define scene parameters
Xbounds = [-2 2]; % X bounds of scene
Ybounds = [0 4];  % Y bounds of scene

%% Generate enhanced S-curve trajectory (position only for HMM)
% Time parameters
num_steps = 50;
dt = 0.1; % Time step in seconds
t = (0:num_steps) * dt;

% Enhanced S-curve trajectory with smooth analytical dynamics
% Start at (-1.5, 0.5) and move in an S-turn to (1.5, 3.5)
x_start = -1.5;
y_start = 0.5;
x_end = 1.5;
y_end = 3.5;

% Use parametric equations with time parameter tau = t/T
T = t(end); % Total trajectory time
tau = t / T; % Normalized time [0, 1]

% X: Linear progression with sinusoidal perturbation
x_traj = x_start + (x_end - x_start) * tau + 0.6 * sin(2 * pi * tau) .* (1 - tau) .^ 2 .* tau .^ 2;

% Y: S-curve with smooth transitions
y_base = y_start + (y_end - y_start) * tau;
y_perturbation = 0.4 * sin(pi * tau) .* sin(2 * pi * tau);
y_traj = y_base + y_perturbation;

% Clamp to scene bounds
x_traj = max(Xbounds(1), min(Xbounds(2), x_traj));
y_traj = max(Ybounds(1), min(Ybounds(2), y_traj));

% Position-only trajectory (HMM has no velocity state)
pttraj = [x_traj; y_traj]; % 2 x (num_steps+1)

fprintf('Generated smooth analytical S-curve trajectory\n');
fprintf('Trajectory duration: %.1f seconds with %d time steps\n', T, num_steps + 1);

%% Generate simulated measurements with noise (position only)
meas_noise_std = 0.01; % Standard deviation of measurement noise (m)
measurements = pttraj(:, 1:end-1) + meas_noise_std * randn(2, num_steps);

fprintf('Generated %d noisy position measurements\n', num_steps);

%% Load HMM model parameters
load(fullfile('..', 'data', 'precalc_imagegridHMMSTMn15.mat'), 'A');
A_transition = A; clear A;

load(fullfile('..', 'data', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');

fprintf('Loaded HMM matrices.\n');
fprintf('  A_transition:          %dx%d\n', size(A_transition, 1), size(A_transition, 2));
fprintf('  pointlikelihood_image: %dx%d\n', size(pointlikelihood_image, 1), size(pointlikelihood_image, 2));

%% Initialize HMM filter
% Use first measurement as initial position estimate
x0 = measurements(:, 1); % [2 x 1] position

hmm = HMM(x0, A_transition, pointlikelihood_image);

% Storage for results
x_hist    = nan(2, num_steps + 1); % MMSE position history
P_hist    = nan(2, 2, num_steps + 1); % Covariance history
[x_hist(:, 1), P_hist(:, :, 1)] = hmm.getGaussianEstimate();

fprintf('Initialized HMM filter at [%.3f, %.3f]\n', x0(1), x0(2));

%% Setup visualization
fig = figure(1);
set(fig, 'Visible', fig_visible, 'Position', [100, 100, 1600, 500]);
fprintf('Pausing to generate figure\n');
pause(2);

% Initialize GIF saving if requested
if SAVE_FLAG
    gif_filename = fullfile(SAVE_PATH, 'hmm_animation.gif');
end

%% Apply HMM Filter Updates
for kk = 1:num_steps
    fprintf('Processing time step %d/%d\n', kk, num_steps);

    current_meas = measurements(:, kk);
    fprintf('\t  Measurement: [%.3f, %.3f]\n', current_meas(1), current_meas(2));

    %% ========== PREDICTION STEP ==========
    fprintf('\t-> Prediction step\n');
    hmm.prediction();

    %% ========== MEASUREMENT UPDATE STEP ==========
    fprintf('\t-> Measurement update step\n');
    hmm.measurement_update(current_meas);

    % Extract estimates
    [x_post, P_post] = hmm.getGaussianEstimate();
    [x_map, ~]       = hmm.getMAPEstimate();

    % Store results
    x_hist(:, kk + 1)       = x_post;
    P_hist(:, :, kk + 1)    = P_post;

    %% ========== PLOTTING ==========
    figure(1);
    clf;

    tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    % ---- Tile 1: Prior P(x_k | z_{1:k-1}) ----
    nexttile(1);
    prior_grid = reshape(full(hmm.prior_prob), [hmm.grid_size, hmm.grid_size]);
    surf(hmm.xgrid, hmm.ygrid, prior_grid, 'EdgeColor', 'none');
    view(2);
    hold on;
    % True state overlay
    plot3(pttraj(1, kk), pttraj(2, kk), max(prior_grid(:)) * 1.1, ...
        'p', 'Color', [0.7 0.7 0.7], 'MarkerSize', 20, 'LineWidth', 0.5);
    hold off;
    xlabel('$X$ (m)'), ylabel('$Y$ (m)');
    title('Prior $P(x_k \mid z_{1:k-1})$');
    xlim(Xbounds), ylim(Ybounds);
    axis square;
    colormap(gca, 'hot');
    c = colorbar; c.TickLabelInterpreter = 'latex';

    % ---- Tile 2: Likelihood P(z_k | x_k) ----
    nexttile(2);
    like_grid = reshape(full(hmm.likelihood_prob), [hmm.grid_size, hmm.grid_size]);
    surf(hmm.xgrid, hmm.ygrid, like_grid, 'EdgeColor', 'none');
    view(2);
    hold on;
    % Measurement overlay
    plot3(current_meas(1), current_meas(2), max(like_grid(:)) * 1.1, ...
        '+', 'Color', 'c', 'MarkerSize', 10, 'LineWidth', 2);
    % True state overlay
    plot3(pttraj(1, kk), pttraj(2, kk), max(like_grid(:)) * 1.1, ...
        'p', 'Color', [0.7 0.7 0.7], 'MarkerSize', 20, 'LineWidth', 0.5);
    hold off;
    xlabel('$X$ (m)'), ylabel('$Y$ (m)');
    title('Likelihood $P(z_k \mid x_k)$');
    xlim(Xbounds), ylim(Ybounds);
    axis square;
    colormap(gca, 'hot');
    c = colorbar; c.TickLabelInterpreter = 'latex';

    % ---- Tile 3: Posterior P(x_k | z_{1:k}) ----
    nexttile(3);
    post_grid = reshape(full(hmm.posterior_prob), [hmm.grid_size, hmm.grid_size]);
    surf(hmm.xgrid, hmm.ygrid, post_grid, 'EdgeColor', 'none');
    view(2);
    hold on;
    z_top = max(post_grid(:)) * 1.1;
    % True state
    plot3(pttraj(1, kk), pttraj(2, kk), z_top, ...
        'p', 'Color', [0.7 0.7 0.7], 'MarkerSize', 20, 'LineWidth', 0.5);
    % Measurement
    plot3(current_meas(1), current_meas(2), z_top, ...
        '+', 'Color', 'c', 'MarkerSize', 10, 'LineWidth', 2);
    % MMSE estimate
    plot3(x_post(1), x_post(2), z_top, ...
        'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
    % MAP estimate
    plot3(x_map(1), x_map(2), z_top, ...
        'bs', 'MarkerSize', 8, 'LineWidth', 2);
    % Trajectory history
    if kk > 1
        plot3(x_hist(1, 2:kk+1), x_hist(2, 2:kk+1), z_top * ones(1, kk), ...
            'r-', 'LineWidth', 1.5);
        plot3(pttraj(1, 1:kk), pttraj(2, 1:kk), z_top * ones(1, kk), ...
            '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5);
    end
    hold off;
    xlabel('$X$ (m)'), ylabel('$Y$ (m)');
    title('Posterior $P(x_k \mid z_{1:k})$');
    legend('', 'True State', 'Measurement', 'MMSE', 'MAP', 'Location', 'southeast');
    xlim(Xbounds), ylim(Ybounds);
    axis square;
    colormap(gca, 'hot');
    c = colorbar; c.TickLabelInterpreter = 'latex';

    % Add entropy and error info as text on posterior tile
    pos_err = norm(x_post - pttraj(:, kk));
    H_val   = hmm.getEntropy();
    text(Xbounds(1) + 0.05, Ybounds(2) - 0.15, ...
        sprintf('$||e||$: %.3f m', pos_err), ...
        'Interpreter', 'latex', 'FontSize', 10, 'BackgroundColor', 'white');
    text(Xbounds(1) + 0.05, Ybounds(2) - 0.45, ...
        sprintf('$H$: %.2f', H_val), ...
        'Interpreter', 'latex', 'FontSize', 10, 'BackgroundColor', 'white');

    sgtitle(sprintf('Grid-Based HMM: $k = %d$', kk), 'Interpreter', 'latex', 'FontSize', 16);

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
        pause(0.2);
    end

    fprintf('\t  Position error: [%.3f, %.3f] m\n', ...
        x_post(1) - pttraj(1, kk), x_post(2) - pttraj(2, kk));
end

%% Performance summary
pos_errors = x_hist(1:2, 2:end) - pttraj(1:2, 1:end-1);
rmse_pos   = sqrt(mean(pos_errors(1, :) .^ 2 + pos_errors(2, :) .^ 2));

fprintf('\n=== HMM Single Target Tracker Performance Summary ===\n');
fprintf('RMSE position error: %.4f m\n', rmse_pos);
fprintf('Mean measurement noise std: %.4f m\n', meas_noise_std);
fprintf('Trajectory duration: %.1f seconds (%d steps)\n', t(end), num_steps);
fprintf('Final position error: %.4f m\n', norm(pos_errors(:, end)));
fprintf('Grid: %dx%d, %.4f m/cell\n', hmm.grid_size, hmm.grid_size, hmm.dx);

% y_traj = max(Ybounds(1), min(Ybounds(2), y_traj));

% % Store single target trajectory (position only for HMM)
% pttraj = [x_traj; y_traj]; % 2 x (num_steps+1) array

% fprintf('Generated smooth analytical S-curve trajectory\n');
% fprintf('Trajectory duration: %.1f seconds with %d time steps\n', T, num_steps + 1);

% %% Generate simulated measurements with noise (position only)
% % Add measurement noise to the true position trajectory
% meas_noise_std = 0.01; % Standard deviation of measurement noise
% measurements = pttraj(:, 1:end-1) + meas_noise_std * randn(2, num_steps);

% fprintf('Generated %d noisy position measurements\n', num_steps);

% %% load HMM model params
% load(fullfile('..', 'data', 'precalc_imagegridHMMSTMn15.mat'), 'A');
% A_slow = A; clear A
% load(fullfile('..', 'data', 'precalc_imagegridHMMSTMn30.mat'), 'A');
% A_fast = A; clear A

% % Load likelihood models (both detection and magnitude)
% load(fullfile('..', 'data', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');
% det_likelihood_lookup = pointlikelihood_image;
% load(fullfile('..', 'data', 'precalc_imagegridHMMEmLikeMag.mat'), 'pointlikelihood_image');
% mag_likelihood_lookup = pointlikelihood_image;
% clear pointlikelihood_image;

% fprintf("Loaded Likelihood maps: Sizes are as follows:\n")
% fprintf("Detection Likelihood Lookup Table: ")
% disp(size(det_likelihood_lookup))
% fprintf("Magnitude Likelihood Lookup Table: ")
% disp(size(mag_likelihood_lookup))

% %% Validate loaded parameters
% if size(A_slow, 1) ~= 128 ^ 2 || size(A_fast, 1) ~= 128 ^ 2
%     error('Transition matrix dimensions (%dx%d) do not match grid size (%d)', ...
%         size(A_slow, 1), size(A_slow, 2), 128 ^ 2);
% end

% if size(det_likelihood_lookup, 1) ~= 128 ^ 2 || size(det_likelihood_lookup, 2) ~= 128 ^ 2
%     error('Detection likelihood model dimensions (%dx%d) do not match grid size (%d)', ...
%         size(det_likelihood_lookup, 1), size(det_likelihood_lookup, 2), 128 ^ 2);
% end

% if size(mag_likelihood_lookup, 1) ~= 128 ^ 2 || size(mag_likelihood_lookup, 2) ~= 2
%     error('Magnitude likelihood model dimensions (%dx%d) do not match expected size (%dx2)', ...
%         size(mag_likelihood_lookup, 1), size(mag_likelihood_lookup, 2), 128 ^ 2);
% end

% fprintf('Successfully loaded HMM matrices and validated dimensions.\n');

% %% Load mWidar model parameters
% % Load mWidar simulation matrices
% load(fullfile('..', '..', 'matlab_src', 'supplemental', 'recovery.mat'))
% load(fullfile('..', '..', 'matlab_src', 'supplemental', 'sampling.mat'))

% % Put into mWidar params struct
% mWidarParams.sampling = M;
% mWidarParams.recovery = G;
% clear M G;

% % Validate loaded matrices
% fprintf('Successfully loaded mWidar Signal Generation Matrices and validated dimensions. Testing Generation\n');

% try
%     test = zeros(128, 128);
%     test(30, 30) = 1;
%     % Test using genmWidarImage function - reshape to 1x128x128 for function input
%     test_input = reshape(test, 1, 128, 128);
%     test_output = genmWidarImage(test_input, mWidarParams);
%     test_new = squeeze(test_output(1, :, :)); % Extract the 128x128 result
%     fprintf('Successfully generated test mWidar image with dimensions %dx%d using genmWidarImage function\n', size(test_new, 1), size(test_new, 2));
% catch ME
%     warning('Failed to generate test mWidar image: %s', ME.message);
%     fprintf('Continuing without image generation test...\n');
% end

% fprintf('Successful validation of mWidar matrices. Continuing iteration\n');

% %% Define spatial grid
% Lscene = 4; %physical length of scene in m (square shape)
% npx = 128; %number of pixels in image (same in x&y dims)
% npx2 = npx ^ 2;

% xgrid = linspace(-2, 2, npx);
% ygrid = linspace(0, Lscene, npx);
% [pxgrid, pygrid] = meshgrid(xgrid, ygrid);
% pxyvec = [pxgrid(:), pygrid(:)];
% dx = xgrid(2) - xgrid(1);
% dy = ygrid(2) - ygrid(1);

% fprintf('Spatial grid: %dx%d pixels, %.4fm resolution\n', npx, npx, dx);

% %% Initialize HMM state probability distribution
% ptarget_Hist = cell(1, num_steps+1);

% % Initialize target probability near first measurement
% % Find grid point closest to first measurement
% [~, init_x_idx] = min(abs(xgrid - measurements(1,1)));
% [~, init_y_idx] = min(abs(ygrid - measurements(2,1)));
% init_linear_idx = sub2ind([npx, npx], init_y_idx, init_x_idx);

% % Initialize with Gaussian distribution around first measurement
% init_prob = sparse(npx2, 1);
% sigma_init = 0.3; % Initial uncertainty
% for i = 1:npx2
%     [row, col] = ind2sub([npx, npx], i);
%     x_pos = xgrid(col);
%     y_pos = ygrid(row);
%     dist_sq = (x_pos - measurements(1,1))^2 + (y_pos - measurements(2,1))^2;
%     init_prob(i) = exp(-dist_sq / (2 * sigma_init^2));
% end
% init_prob = init_prob / sum(init_prob); % Normalize
% ptarget_Hist{1} = init_prob;

% % Storage for results
% mutarget_Hist = nan(2, num_steps+1);
% sig2target_Hist = nan(2, 2, num_steps+1);
% mmse_errtarget_Hist = nan(2, num_steps+1);
% map_errtarget_Hist = nan(2, num_steps+1);

% fprintf('Initialized HMM state probability distribution\n');

% %% Setup visualization
% fig = figure(1);
% set(fig, 'Visible', fig_visible, 'Position', [100, 100, 2400, 600]);
% % Pause to make sure it draws
% fprintf("Pausing to generate figure\n");
% pause(2);

% % Initialize GIF saving if requested
% if SAVE_FLAG
%     gif_filename = fullfile(SAVE_PATH, 'hmm_animation.gif');
% end

% %% Apply HMM Bayesian Filter Updates
% for kk = 1:num_steps
%     fprintf('Processing time step %d/%d\n', kk, num_steps);

%     %% ========== PREDICTION STEP (TIME UPDATE) ==========
%     fprintf('\t-> Prediction step\n');

%     % Get previous posterior
%     ptargetkkm1 = ptarget_Hist{kk};

%     % Apply state transition model (dynamics)
%     ptargetkk_pred = A_slow * ptargetkkm1; % Use slow model for demonstration
%     % ptargetkk_pred = A_fast * ptargetkkm1; % Use fast model for demonstration

%     %% ========== MEASUREMENT UPDATE STEP ==========
%     fprintf('\t-> Measurement update step\n');

%     % Get current measurement (assuming data association already done)
%     current_meas = measurements(:, kk);
%     fprintf('\t  Measurement: [%.3f, %.3f]\n', current_meas(1), current_meas(2));

%     % Find closest grid point to measurement for likelihood lookup
%     [~, meas_x_idx] = min(abs(xgrid - current_meas(1)));
%     [~, meas_y_idx] = min(abs(ygrid - current_meas(2)));
%     meas_linear_idx = sub2ind([npx, npx], meas_y_idx, meas_x_idx);

%     % Get detection likelihood function from pre-computed model
%     likeframekk_det_raw = det_likelihood_lookup(meas_linear_idx, :)';

%     % Apply Gaussian mask around measurement for improved localization
%     sf = 0.15; % scaling factor for Gaussian mask
%     meas_pos = [current_meas(1), current_meas(2)];
%     gaussmask = mvnpdf(pxyvec, meas_pos, sf * eye(2));
%     gaussmask(gaussmask < 0.1 * max(gaussmask)) = 0; % threshold small values
%     likeframekk_det = likeframekk_det_raw .* gaussmask;

%     % mWIDAR SIGNAL CREATION (following test_hybrid_PF approach)
%     curr_signal = zeros(128,128);
%     curr_signal(meas_linear_idx) = 1;
%     curr_signal = reshape(curr_signal, 1, 128, 128);
%     curr_signal = genmWidarImage(curr_signal, mWidarParams);
%     curr_signal = squeeze(curr_signal(1, :, :)); % Extract the 128x128 result
    
%     % For HMM case: evaluate magnitude likelihood for all grid points
%     % Get all grid point linear indices
%     all_grid_indices = (1:npx*npx)';
    
%     % Get magnitude likelihood parameters for all grid points
%     mag_likelihood_values = mag_likelihood_lookup(all_grid_indices, :); % [N_grid x 2]
    
%     % Calculate magnitude likelihood for each grid point
%     % For each grid point, get signal value and compute likelihood
%     likeframekk_mag = zeros(npx*npx, 1);
%     for grid_idx = 1:npx*npx
%         signal_value = curr_signal(grid_idx);
%         % likelihood_mag = Normal(signal_value, mean=likelihood_mag_values(1), var=likelihood_mag_values(2)^2)
%         likeframekk_mag(grid_idx) = 0.1 * normpdf(signal_value, mag_likelihood_values(grid_idx, 1), mag_likelihood_values(grid_idx, 2));
%     end

%     % Combine detection and magnitude likelihoods (composite likelihood)
%     likeframekk_combined = likeframekk_det .* likeframekk_mag;
    
%     % Normalize combined likelihood
%     likeframekk = likeframekk_combined / sum(likeframekk_combined);

%     % Compute posterior: P(x_k | z_1:k) ∝ P(z_k | x_k) * P(x_k | z_1:k-1)
%     ptargetkk_post = ptargetkk_pred .* likeframekk;
%     ptargetkk_post = ptargetkk_post / sum(ptargetkk_post); % normalize

%     % Store posterior for next iteration
%     ptarget_Hist{kk+1} = ptargetkk_post;

%     %% ========== COMPUTE STATISTICS ==========
%     % Compute MMSE estimate (mean of posterior)
%     mutarget_Hist(:, kk) = sum(pxyvec .* repmat(ptarget_Hist{kk+1}, [1, 2]), 1)';

%     % Compute posterior covariance
%     sig2target_Hist(:, :, kk) = reshape(sum([pxyvec(:, 1).^2, pxyvec(:, 1).*pxyvec(:, 2), ...
%         pxyvec(:, 2).*pxyvec(:, 1), pxyvec(:, 2).^2] .* ...
%         repmat(ptarget_Hist{kk+1}, [1, 4]), 1), [2 2]) ...
%         - mutarget_Hist(:, kk) * (mutarget_Hist(:, kk))';

%     % Compute estimation errors w.r.t. ground truth
%     mmse_errtarget_Hist(:, kk) = mutarget_Hist(:, kk) - pttraj(:, kk);
%     [~, indmapxy] = max(ptarget_Hist{kk+1});
%     map_errtarget_Hist(:, kk) = pxyvec(indmapxy, :)' - pttraj(:, kk);

%     %% ========== PLOTTING ==========
%     % Plot HMM states and estimates (always create, but control visibility)
%     figure(1); % Make sure we're on the right figure
%     clf; % Clear figure for fresh tiledlayout
    
%     % Create tiled layout (1 row, 3 columns)
%     t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%     % Tile 1: Prior/Prediction
%     nexttile(1); cla
%     surf(xgrid, ygrid, reshape(ptargetkk_pred, [npx, npx]), 'EdgeColor', 'none'), view(2)
%     title('Prior')
%     xlabel('$X$ (m)'), ylabel('$Y$ (m)')
%     xlim([-2, 2]), ylim([0, 4])
%     axis square
%     colormap(gca, 'hot')
%     c = colorbar;
%     c.TickLabelInterpreter = 'latex';

%     % Tile 2: Combined Likelihood
%     nexttile(2); cla
%     surf(xgrid, ygrid, reshape(likeframekk, [npx, npx]), 'EdgeColor', 'none'), view(2)
%     hold on
%     plot3(current_meas(1), current_meas(2), max(likeframekk)*1.1, '+', 'Color', 'c', 'MarkerSize', 8, 'LineWidth', 2)
%     title('Combined Likelihood')
%     xlabel('$X$ (m)'), ylabel('$Y$ (m)')
%     xlim([-2, 2]), ylim([0, 4])
%     axis square
%     colormap(gca, 'hot')
%     c = colorbar;
%     c.TickLabelInterpreter = 'latex';

%     % Tile 3: Posterior
%     nexttile(3); cla
%     surf(xgrid, ygrid, reshape(ptargetkk_post, [npx, npx]), 'EdgeColor', 'none'), view(2)
%     hold on
%     plot3(pttraj(1, kk), pttraj(2, kk), max(ptargetkk_post)*1.1, 'p', 'Color', [0.7 0.7 0.7], 'MarkerSize', 20, 'LineWidth', .5)
%     plot3(mutarget_Hist(1, kk), mutarget_Hist(2, kk), max(ptargetkk_post)*1.1, 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r')
    
%     % Plot trajectory history
%     if kk > 1
%         plot3(mutarget_Hist(1, 1:kk), mutarget_Hist(2, 1:kk), max(ptargetkk_post)*1.1*ones(1,kk), 'r-', 'LineWidth', 2)
%         plot3(pttraj(1, 1:kk), pttraj(2, 1:kk), max(ptargetkk_post)*1.1*ones(1,kk), 'Color', [0.7 0.7 0.7], 'LineWidth', 2)
%     end
    
%     title('Posterior')
%     xlabel('$X$ (m)'), ylabel('$Y$ (m)')
%     xlim([-2, 2]), ylim([0, 4])
%     axis square
%     colormap(gca, 'hot')
%     c = colorbar;
%     c.TickLabelInterpreter = 'latex';

%     % Add suptitle with timestep
%     sgtitle(sprintf('Grid-Based Hidden Markov Model: $k=%d$', kk), "Interpreter", "latex", 'FontSize', 16);

%     % Save frame to GIF if requested
%     if SAVE_FLAG
%         frame = getframe(gcf);
%         im = frame2im(frame);
%         [imind, cm] = rgb2ind(im, 256);

%         if kk == 1
%             imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.2);
%         else
%             imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.2);
%         end
%     end

%     % Pause only if plots are visible
%     if PLOT_FLAG
%         pause(0.2); % Small pause to see animation
%     end

%     fprintf('\t  MMSE error: [%.3f, %.3f] m\n', mmse_errtarget_Hist(1, kk), mmse_errtarget_Hist(2, kk));
% end
% return 
% %% Plot results summary
% fig2 = figure(2);
% set(fig2, 'Visible', fig_visible, 'Position', [200, 100, 1200, 800]);

% % Plot trajectory comparison
% subplot(2,2,1)
% plot(pttraj(1, 1:num_steps), pttraj(2, 1:num_steps), 'b-', 'LineWidth', 2)
% hold on
% plot(measurements(1, :), measurements(2, :), 'ro', 'MarkerSize', 4)
% plot(mutarget_Hist(1, 1:num_steps), mutarget_Hist(2, 1:num_steps), 'g--', 'LineWidth', 2)
% xlabel('$X$ position (m)')
% ylabel('$Y$ position (m)')
% title('Single Target Trajectory')
% legend('True trajectory', 'Measurements', 'HMM estimate', 'Location', 'best')
% grid on
% axis equal
% xlim(Xbounds)
% ylim(Ybounds)

% % Plot MMSE errors vs. time
% % Use only the valid data indices (1:num_steps)
% tvec = 1:num_steps;
% mmse_x_valid = mmse_errtarget_Hist(1, 1:num_steps);
% mmse_y_valid = mmse_errtarget_Hist(2, 1:num_steps);
% map_x_valid = map_errtarget_Hist(1, 1:num_steps);
% map_y_valid = map_errtarget_Hist(2, 1:num_steps);
% sig2_x_valid = squeeze(sig2target_Hist(1, 1, 1:num_steps));
% sig2_y_valid = squeeze(sig2target_Hist(2, 2, 1:num_steps));

% subplot(2,2,2)
% plot(tvec, mmse_x_valid, 'r-o'), hold on
% plot(tvec, 2 * sqrt(sig2_x_valid), 'r--')
% plot(tvec, -2 * sqrt(sig2_x_valid), 'r--'), hold off
% xlabel('Time step, $k$')
% ylabel('$X$ position error (m)')
% title('MMSE $X$ Estimation Error')
% grid on

% subplot(2,2,3)
% plot(tvec, mmse_y_valid, 'r-o'), hold on
% plot(tvec, 2 * sqrt(sig2_y_valid), 'r--')
% plot(tvec, -2 * sqrt(sig2_y_valid), 'r--'), hold off
% xlabel('Time step, $k$')
% ylabel('$Y$ position error (m)')
% title('MMSE $Y$ Estimation Error')
% grid on

% % Plot MAP errors vs. time
% subplot(2,2,4)
% plot(tvec, sqrt(map_x_valid.^2 + map_y_valid.^2), 'b-o')
% hold on
% plot(tvec, sqrt(mmse_x_valid.^2 + mmse_y_valid.^2), 'r-o')
% xlabel('Time step, $k$')
% ylabel('Position error magnitude (m)')
% title('Position Error Comparison')
% legend('MAP error', 'MMSE error', 'Location', 'best')
% grid on

% % Save figure if requested
% if SAVE_FLAG
%     print(gcf, fullfile(SAVE_PATH, 'hmm_results_summary.png'), '-dpng', '-r600');
% end

% %% Additional analysis plots
% fig3 = figure(3);
% set(fig3, 'Visible', fig_visible, 'Position', [300, 100, 1200, 600]);

% % Plot estimation errors with uncertainty bounds
% subplot(1,2,1)
% % Ensure all vectors are row vectors for proper concatenation
% tvec_row = tvec(:)';
% sig2_x_row = sig2_x_valid(:)';
% upper_bound = 2*sqrt(sig2_x_row);
% lower_bound = -2*sqrt(sig2_x_row);
% fill([tvec_row, fliplr(tvec_row)], [upper_bound, fliplr(lower_bound)], ...
%     'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
% hold on
% plot(tvec, mmse_x_valid, 'r-', 'LineWidth', 2)
% plot(tvec, map_x_valid, 'b--', 'LineWidth', 2)
% xlabel('Time step, $k$')
% ylabel('$X$ Error (m)')
% title('$X$ Position Estimation Errors')
% legend('$\pm2\sigma$ bounds', 'MMSE error', 'MAP error', 'Location', 'best')
% grid on

% subplot(1,2,2)
% % Ensure all vectors are row vectors for proper concatenation
% sig2_y_row = sig2_y_valid(:)';
% upper_bound_y = 2*sqrt(sig2_y_row);
% lower_bound_y = -2*sqrt(sig2_y_row);
% fill([tvec_row, fliplr(tvec_row)], [upper_bound_y, fliplr(lower_bound_y)], ...
%     'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
% hold on
% plot(tvec, mmse_y_valid, 'r-', 'LineWidth', 2)
% plot(tvec, map_y_valid, 'b--', 'LineWidth', 2)
% xlabel('Time step, $k$')
% ylabel('$Y$ Error (m)')
% title('$Y$ Position Estimation Errors')
% legend('$\pm2\sigma$ bounds', 'MMSE error', 'MAP error', 'Location', 'best')
% grid on

% % Save figure if requested
% if SAVE_FLAG
%     print(gcf, fullfile(SAVE_PATH, 'hmm_error_analysis.png'), '-dpng', '-r600');
% end

% %% Performance summary
% rmse_mmse = sqrt(mean(mmse_x_valid.^2 + mmse_y_valid.^2));
% rmse_map = sqrt(mean(map_x_valid.^2 + map_y_valid.^2));

% fprintf('\n=== HMM Single Target Tracker Performance Summary ===\n');
% fprintf('RMSE MMSE error: %.4f m\n', rmse_mmse);
% fprintf('RMSE MAP error:  %.4f m\n', rmse_map);
% fprintf('Mean measurement noise std: %.4f m\n', meas_noise_std);
% fprintf('Trajectory duration: %.1f seconds (%d steps)\n', t(end), num_steps);
% fprintf('Grid resolution: %.4f m/pixel\n', dx);
% fprintf('Scene bounds: X[%.1f, %.1f], Y[%.1f, %.1f] m\n', Xbounds(1), Xbounds(2), Ybounds(1), Ybounds(2));

% % Display final position estimate
% final_pos_true = pttraj(:, end-1);
% final_pos_est = mutarget_Hist(:, num_steps);
% final_pos_error = norm(final_pos_est - final_pos_true);
% fprintf('Final position error: %.4f m\n', final_pos_error);
