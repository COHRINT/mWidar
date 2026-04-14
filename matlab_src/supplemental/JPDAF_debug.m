clear; clc; close all;

addpath(fullfile('DA_Track','multi/'));
addpath(fullfile('supplemental'));
addpath(fullfile('supplemental', 'Final_Test_Tracks'));

rng(42);

%% Configuration
n_t = 20;
dt = 0.1;
n_track = 2;

npx = 128;
xgrid = linspace(-2, 2, npx);
ygrid = linspace(0, 4, npx);
[pxgrid, pygrid] = meshgrid(xgrid, ygrid);

Pfa = 0.367;
Ng = 5;
Nr = 20;

true_meas_sigma = 0.05;
init_sigma_pos = 0.08;
init_sigma_vel = 0.10;
init_sigma_acc = 0.05;
save_beta_debug_figures = true;
beta_debug_dir = fullfile('supplemental', 'JPDAF_debug_outputs', 'beta_debug');

%% Load mWidar forward model
load(fullfile('supplemental', 'sampling.mat'), 'M');
load(fullfile('supplemental', 'recovery.mat'), 'G');

%% Generate two simple trajectories
tvec = dt:dt:n_t * dt;
GT = cell(1, n_track);
GT{1} = generate_line_track(tvec, [-1.4, 1.0], [1.4, 2.6]);
GT{2} = generate_line_track(tvec, [ 1.2, 3.2], [-1.2, 2.0]);

%% Generate signal and measurements
signal = cell(1, n_t);
z = cell(1, n_t);
true_meas = cell(1, n_t);

for k = 1:n_t
    S = zeros(npx, npx);

    for t = 1:n_track
        true_pos = GT{t}(1:2, k);
        [grid_x, grid_y] = clamp_to_grid(true_pos, xgrid, ygrid);
        S(grid_y, grid_x) = 1;
    end

    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, npx, npx)';
    blurred_signal = imgaussfilt(sim_signal, 2);
    signal{k} = blurred_signal;

    [~, peak_y, peak_x] = CA_CFAR(blurred_signal, Pfa, Ng, Nr);
    if ~isempty(peak_x)
        peak_inds = sub2ind([npx, npx], peak_y, peak_x);
        cfar_meas = [pxgrid(peak_inds)'; pygrid(peak_inds)'];
    else
        cfar_meas = zeros(2, 0);
    end

    forced_meas = zeros(2, n_track);
    for t = 1:n_track
        forced_meas(:, t) = GT{t}(1:2, k) + true_meas_sigma * randn(2, 1);
    end

    true_meas{k} = forced_meas;
    z{k} = [cfar_meas, forced_meas];
end

save("supplemental\JPDAF_debug_outputs\true_meas.mat",'true_meas','-mat')

Data = struct();
Data.GT = GT;
Data.y = z;
Data.signal = signal;
Data.true_measurements = true_meas;
Data.params = struct( ...
    'dt', dt, ...
    'n_t', n_t, ...
    'Pfa', Pfa, ...
    'Ng', Ng, ...
    'Nr', Nr, ...
    'rng_seed', 42, ...
    'xgrid_range', [-2, 2], ...
    'ygrid_range', [0, 4], ...
    'npx', npx);

%% JPDA_KF parameters (mirrors main_multiObj.m structure)
A = [0 0 1 0 0 0;
     0 0 0 1 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 1;
     0 0 0 0 0 0;
     0 0 0 0 0 0];
F = expm(A * dt);

Q = 1e-2 * eye(6);
Q(3,3) = 1e-6;
Q(6,6) = 1e-6;

R = 1 * eye(2);
H = [1 0 0 0 0 0;
     0 1 0 0 0 0];
P0 = diag([0.1 0.1 0.25 0.25 0.5 0.5]);

%% Initialize JPDA_KF at GT + noise
x0 = zeros(6, n_track);
P0_cell = cell(1, n_track);
for t = 1:n_track
    x0(:, t) = GT{t}(:, 1) + [ ...
        init_sigma_pos * randn(2, 1); ...
        init_sigma_vel * randn(2, 1); ...
        init_sigma_acc * randn(2, 1)];
    P0_cell{t} = P0;
end

jpda = JPDA_KF(x0, P0_cell, F, Q, R, H, n_track, [], 'Debug', false, 'DynamicPlot', false);
jpda.lambda_clutter = 2.5;
jpda.measurement_space_area = 16;
jpda.gate_probability = 0.95;
jpda.PD = 0.95;

if save_beta_debug_figures && ~exist(beta_debug_dir, 'dir')
    mkdir(beta_debug_dir);
end

%% Run tracker
performance = cell(1, n_t);
[performance{1}.x, performance{1}.P] = jpda.getGaussianEstimate();

for k = 1:n_t
    if k > 1
        jpda.timestep(z{k}, signal{k});
        [performance{k}.x, performance{k}.P] = jpda.getGaussianEstimate();

        if save_beta_debug_figures && ...
                ~isempty(jpda.association_debug_figure_handle) && ...
                isgraphics(jpda.association_debug_figure_handle)
            drawnow;
            frame = getframe(jpda.association_debug_figure_handle);
            imwrite(frame.cdata, ...
                fullfile(beta_debug_dir, sprintf('beta_debug_step_%02d.png', k)));
        end
    end

    performance{k}.measurements = z{k};
    performance{k}.true_measurements = true_meas{k};
    performance{k}.signal = signal{k};
end

%% Plot summary
plot_jpdaf_debug_summary(performance, Data);


function X_GT = generate_line_track(tvec, start_xy, stop_xy)
    n_t_local = numel(tvec);
    T = tvec(end);
    delta_pos = [stop_xy(1) - start_xy(1); stop_xy(2) - start_xy(2)];

    a_progress = -0.01 + 0.02 * rand();
    v0_progress = (1 - 0.5 * a_progress * T^2) / T;

    progress = v0_progress .* tvec + 0.5 .* a_progress .* tvec.^2;
    progress_dot = v0_progress + a_progress .* tvec;
    progress_ddot = a_progress .* ones(1, n_t_local);

    x_traj = start_xy(1) + delta_pos(1) .* progress;
    y_traj = start_xy(2) + delta_pos(2) .* progress;
    vx_traj = delta_pos(1) .* progress_dot;
    vy_traj = delta_pos(2) .* progress_dot;
    ax_traj = delta_pos(1) .* progress_ddot;
    ay_traj = delta_pos(2) .* progress_ddot;

    x_traj = max(-2, min(2, x_traj));
    y_traj = max(0, min(4, y_traj));

    X_GT = [x_traj; y_traj; vx_traj; vy_traj; ax_traj; ay_traj];
end


function [grid_x, grid_y] = clamp_to_grid(true_pos, xgrid, ygrid)
    [~, grid_x] = min(abs(xgrid - true_pos(1)));
    [~, grid_y] = min(abs(ygrid - true_pos(2)));
    grid_x = min(max(grid_x, 1), numel(xgrid));
    grid_y = min(max(grid_y, 1), numel(ygrid));
end


function plot_jpdaf_debug_summary(perf, Data)
    n_t_local = numel(perf);
    n_track_local = numel(Data.GT);
    colors = lines(n_track_local);

    figure('Color', 'w', 'Position', [100, 100, 1200, 700]);
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax1 = nexttile;
    hold(ax1, 'on');
    for t = 1:n_track_local
        gt_xy = Data.GT{t}(1:2, :);
        est_xy = zeros(2, n_t_local);
        for k = 1:n_t_local
            est_xy(:, k) = perf{k}.x(1:2, t);
        end
        plot(ax1, gt_xy(1, :), gt_xy(2, :), '-', 'Color', colors(t, :), 'LineWidth', 2.0);
        plot(ax1, est_xy(1, :), est_xy(2, :), '--', 'Color', colors(t, :), 'LineWidth', 1.5);
    end
    xlabel(ax1, 'x (m)');
    ylabel(ax1, 'y (m)');
    title(ax1, 'GT vs JPDA Estimates');
    axis(ax1, 'equal');
    xlim(ax1, [-2, 2]);
    ylim(ax1, [0, 4]);
    grid(ax1, 'on');

    ax2 = nexttile;
    hold(ax2, 'on');
    all_meas = [];
    all_true_meas = [];
    for k = 1:n_t_local
        all_meas = [all_meas, perf{k}.measurements];
        all_true_meas = [all_true_meas, perf{k}.true_measurements];
    end
    if ~isempty(all_meas)
        scatter(ax2, all_meas(1, :), all_meas(2, :), 18, [0.65 0.65 0.65], 'filled');
    end
    if ~isempty(all_true_meas)
        scatter(ax2, all_true_meas(1, :), all_true_meas(2, :), 28, [0.85 0.15 0.15], 'filled');
    end
    for t = 1:n_track_local
        gt_xy = Data.GT{t}(1:2, :);
        plot(ax2, gt_xy(1, :), gt_xy(2, :), '-', 'Color', colors(t, :), 'LineWidth', 2.0);
    end
    xlabel(ax2, 'x (m)');
    ylabel(ax2, 'y (m)');
    title(ax2, 'Measurements (Gray) and Forced True Measurements (Red)');
    axis(ax2, 'equal');
    xlim(ax2, [-2, 2]);
    ylim(ax2, [0, 4]);
    grid(ax2, 'on');
end
