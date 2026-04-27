% test_dynamic_targets.m
%
% Monte Carlo end-to-end test of variable-cardinality multi-target
% tracking with TrackManager + ProbabilisticEstimator.
%
% Compares filters: PDA_PF_multi and KF_RBPF_multi.
%
% Per trial:
%   - Same GT spawn schedule (T1, T2 from k=1..60/T; T3 from k=31..T)
%   - Fresh process noise on GT trajectories (drawn from Q)
%   - Fresh measurement noise + clutter realization
%   - Both filters run on the same trial's measurements -> apples-to-apples
%
% Outputs:
%   - Per-filter aggregate figure (mean +/- std curves over MC trials)
%   - Side-by-side comparison figure (OSPA, cardinality, per-target RMSE)
%   - Numerical results saved to .mat
%
% Run from matlab_src/.

clear; clc; close all;
script_dir     = fileparts(mfilename('fullpath'));     % .../matlab_src/tests
matlab_src_dir = fileparts(script_dir);                 % .../matlab_src
addpath(matlab_src_dir);
addpath(fullfile(matlab_src_dir, 'DA_Track'));
addpath(fullfile(matlab_src_dir, 'DA_Track', 'multi'));
addpath(fullfile(matlab_src_dir, 'supplemental'));
addpath(fullfile(matlab_src_dir, 'supplemental', 'track_init'));
addpath(fullfile(matlab_src_dir, 'supplemental', 'multitarget_metrics'));
addpath(fullfile(matlab_src_dir, 'supplemental', 'Final_Test_Tracks'));

%% --- MC + scenario configuration --------------------------------------
N_MC       = 100;            % Monte Carlo trials per filter (set to 1 for debug)
base_seed  = 42;            % rng(base_seed + mc) per trial -> reproducible

T  = 90;                    % total frames
dt = 0.1;
F = [eye(2), dt*eye(2); zeros(2), eye(2)];      % 4-state CV
Q = blkdiag(1e-4*eye(2), 1e-3*eye(2));           % process noise (used for GT + filter)
H = [eye(2), zeros(2)];
R_meas = 0.05^2 * eye(2);

PD_true = 0.95;
n_clutter_lambda = 1.5;
clutter_box = [-2 2; 0 4];

% Spawn / death schedule (FIXED across MC trials).
schedule(1).x0      = [-1.0; 1.0;  0.05; 0.10];
schedule(1).k_start = 1;
schedule(1).k_end   = 60;

schedule(2).x0      = [ 1.0; 1.0; -0.04; 0.10];
schedule(2).k_start = 1;
schedule(2).k_end   = T;

schedule(3).x0      = [ 0.0; 1.5;  0.00; 0.08];
schedule(3).k_start = 31;
schedule(3).k_end   = T;

N_gt_targets = numel(schedule);

%% --- Build the count-estimator likelihood table (shared) --------------
N_vals = 0:4;
m_axis = 0:6;
n_states = numel(N_vals);
n_obs    = numel(m_axis);
P_m_given_N = zeros(n_states, n_obs);
for n = 1:n_states
    Nn = N_vals(n);
    mean_m = PD_true * Nn + n_clutter_lambda;
    for j = 1:n_obs
        m = m_axis(j);
        P_m_given_N(n, j) = exp(-mean_m + m * log(max(mean_m, eps)) - gammaln(m + 1));
    end
    P_m_given_N(n, :) = P_m_given_N(n, :) / sum(P_m_given_N(n, :));
end

%% --- Filter specs: list of (name, builder) ----------------------------
filter_specs(1).name  = 'PDA_PF_multi';
filter_specs(1).build = @(x0_cell) PDA_PF_multi(x0_cell, 500, F, Q, H, R_meas, ...
    'NMax', 4, 'PD', PD_true, 'LambdaClutter', n_clutter_lambda, ...
    'InitSigmaPos', 0.10, 'InitSigmaVel', 0.05, ...
    'ValidationSigma', 5, 'Debug', false);

filter_specs(2).name  = 'KF_RBPF_multi';
filter_specs(2).build = @(x0_cell) KF_RBPF_multi(x0_cell, 50, F, Q, H, R_meas, ...
    'NMax', 4, 'PD', PD_true, ...
    'InitSigmaPos', 0.10, 'InitSigmaVel', 0.05, 'Debug', false);

n_filters = numel(filter_specs);

%% --- Output directory --------------------------------------------------
out_dir = fullfile(matlab_src_dir, 'tests', 'figures', 'multitarget_testing');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% --- Monte Carlo loop --------------------------------------------------
results = struct();
filter_colors = [0.10 0.45 0.85;     % PDA blue
                 0.85 0.32 0.10];    % RBPF red/orange

for f = 1:n_filters
    fname = filter_specs(f).name;
    fprintf('\n=================== %s (%d MC trials) ===================\n', ...
            fname, N_MC);

    cardinality_mae   = zeros(N_MC, 1);
    cardinality_exact = zeros(N_MC, 1);
    rmse_per_target   = nan(N_MC, N_gt_targets);
    ospa_seq          = nan(N_MC, T);
    ospa_loc_seq      = nan(N_MC, T);
    ospa_card_seq     = nan(N_MC, T);
    err_by_gt_seq     = nan(N_MC, N_gt_targets, T);   % Hungarian-aligned per-frame
    N_est_seq_mc      = nan(N_MC, T);
    N_gt_seq_mc       = nan(N_MC, T);
    runtime_sec       = zeros(N_MC, 1);
    GT_trial1         = [];          % store first trial for plotting
    est_xy_trial1     = [];          % candidate trial estimates (for trajectory figure)
    meas_trial1       = [];          % candidate trial measurements

    for mc = 1:N_MC
        rng(base_seed + mc - 1);

        % ---- Stochastic GT (process noise) ----
        GT = simulate_GT(schedule, F, Q, T);
        N_gt_seq = compute_N_gt(GT, T);
        gt_xy_seq = build_gt_xy_seq(GT, T);
        if mc == 1, GT_trial1 = GT; end

        % ---- Measurements ----
        measurements = generate_measurements( ...
            GT, H, R_meas, PD_true, n_clutter_lambda, clutter_box, T);

        % ---- Build estimator + filter + TrackManager (fresh per trial) ----
        est_prob = ProbabilisticEstimator(P_m_given_N, ...
            'NVals', N_vals, 'MAxis', m_axis, ...
            'LambdaArrival', 0.05, 'LambdaDepart', 0.05);

        x0_cell = {GT{1}(:, 1), GT{2}(:, 1)};
        filt = filter_specs(f).build(x0_cell);
        tm = TrackManager(filt, est_prob, ...
            'KHyst', 3, ...
            'PInitDiag', [0.3, 0.3, 0.5, 0.5].^2, ...
            'NMin', 1, 'NMax', 4);

        % ---- Run loop ----
        est_xy_seq = cell(1, T);
        N_est_loc  = zeros(1, T);
        t_start = tic;
        for k = 1:T
            tm.step([], measurements{k});
            [xc, ~, ~] = filt.getGaussianEstimate();
            xy = zeros(2, numel(xc));
            for a = 1:numel(xc)
                xy(:, a) = xc{a}(1:2);
            end
            est_xy_seq{k} = xy;
            N_est_loc(k)  = filt.N_t;
        end
        runtime_sec(mc) = toc(t_start);

        if mc == 1
            est_xy_trial1 = est_xy_seq;
            meas_trial1   = measurements;
        end

        % ---- Metrics ----
        metrics = compute_multitarget_metrics(est_xy_seq, gt_xy_seq, ...
            'OSPACutoff', 1.0, 'Gate', 1.0);

        cardinality_mae(mc)   = metrics.cardinality.mae;
        cardinality_exact(mc) = metrics.cardinality.exact_pct;
        rmse_per_target(mc, 1:numel(metrics.per_target_rmse)) = ...
            metrics.per_target_rmse(:).';
        ospa_seq(mc, :)      = metrics.ospa;
        ospa_loc_seq(mc, :)  = metrics.ospa_loc;
        ospa_card_seq(mc, :) = metrics.ospa_card;
        N_est_seq_mc(mc, :)  = N_est_loc;
        N_gt_seq_mc(mc, :)   = N_gt_seq;

        % Per-frame Hungarian-aligned per-target error
        gt_id_at_frame = gt_id_per_frame(GT, T);
        for k = 1:T
            pairs = metrics.assignment_seq{k};
            for r = 1:size(pairs, 1)
                i_est   = pairs(r, 1);
                j_local = pairs(r, 2);
                gt_id   = gt_id_at_frame{k}(j_local);
                err_by_gt_seq(mc, gt_id, k) = norm( ...
                    est_xy_seq{k}(:, i_est) - gt_xy_seq{k}(:, j_local));
            end
        end

        if mod(mc, max(1, round(N_MC / 5))) == 0 || mc == N_MC
            fprintf('  trial %3d/%d done (%.2fs)\n', mc, N_MC, runtime_sec(mc));
        end
    end

    % ---- Aggregate ----
    R = struct();
    R.cardinality_mae   = struct('mean', mean(cardinality_mae), ...
                                 'std',  std(cardinality_mae));
    R.cardinality_exact = struct('mean', mean(cardinality_exact), ...
                                 'std',  std(cardinality_exact));
    R.rmse_per_target   = struct( ...
        'mean', mean(rmse_per_target, 1, 'omitnan'), ...
        'std',  std(rmse_per_target, 0, 1, 'omitnan'));
    R.ospa_mean       = mean(ospa_seq, 1, 'omitnan');
    R.ospa_std        = std(ospa_seq, 0, 1, 'omitnan');
    R.ospa_loc_mean   = mean(ospa_loc_seq, 1, 'omitnan');
    R.ospa_card_mean  = mean(ospa_card_seq, 1, 'omitnan');
    R.err_by_gt_mean  = squeeze(mean(err_by_gt_seq, 1, 'omitnan'));   % [N_gt x T]
    R.err_by_gt_std   = squeeze(std(err_by_gt_seq, 0, 1, 'omitnan')); % [N_gt x T]
    R.N_est_mean      = mean(N_est_seq_mc, 1);
    R.N_gt_mean       = mean(N_gt_seq_mc, 1);
    R.runtime_sec     = struct('mean', mean(runtime_sec), 'std', std(runtime_sec));
    % Raw per-trial arrays (for re-plotting later)
    R.raw.cardinality_mae   = cardinality_mae;
    R.raw.rmse_per_target   = rmse_per_target;
    R.raw.ospa_seq          = ospa_seq;
    R.raw.N_est_seq         = N_est_seq_mc;
    R.raw.N_gt_seq          = N_gt_seq_mc;
    R.raw.runtime_sec       = runtime_sec;

    R.cand_est_xy = est_xy_trial1;
    R.cand_meas   = meas_trial1;
    R.cand_GT     = GT_trial1;
    R.cand_N_est  = N_est_seq_mc(1, :);
    R.cand_N_gt   = N_gt_seq_mc(1, :);

    results.(fname) = R;

    % ---- Print summary ----
    fprintf('\n--- %s summary (N_MC=%d) ---\n', fname, N_MC);
    fprintf('  Cardinality MAE : %.3f +/- %.3f  (exact %% mean = %.1f)\n', ...
        R.cardinality_mae.mean, R.cardinality_mae.std, ...
        R.cardinality_exact.mean);
    fprintf('  Mean OSPA       : %.3f m  (loc %.3f, card %.3f)\n', ...
        mean(R.ospa_mean, 'omitnan'), ...
        mean(R.ospa_loc_mean, 'omitnan'), ...
        mean(R.ospa_card_mean, 'omitnan'));
    fprintf('  Per-target RMSE : %s (m, mean over MC)\n', ...
        mat2str(R.rmse_per_target.mean, 3));
    fprintf('  Per-target std  : %s (m, std over MC)\n', ...
        mat2str(R.rmse_per_target.std,  3));
    fprintf('  Runtime / trial : %.2f +/- %.2f s\n', ...
        R.runtime_sec.mean, R.runtime_sec.std);

    % ---- Per-filter aggregate figure -------------------------------------
    fig = figure('Color', 'w', 'Position', [60 60 1300 850]);
    color = filter_colors(f, :);

    % (1) Trajectories: GT(trial 1) + scatter of all estimated positions
    subplot(2, 2, 1); hold on; box on; grid on;
    gt_colors = lines(N_gt_targets);
    for tt = 1:N_gt_targets
        plot(GT_trial1{tt}(1, :), GT_trial1{tt}(2, :), '-', ...
             'Color', [0.4 0.4 0.4], 'LineWidth', 2.0, ...
             'DisplayName', sprintf('GT T%d (trial 1)', tt));
    end
    % Overlay mean estimated trajectory per GT (Hungarian-aligned mean of x,y)
    % Reconstruct from err_by_gt is not enough -> overlay as point cloud:
    % we don't store full est_xy_by_gt across MC trials, so just show the
    % mean per-target error in panel (2) and skip trajectory overlay here.
    xlabel('x [m]'); ylabel('y [m]'); axis equal;
    title(sprintf('GT trajectories (trial 1) - %s', fname), 'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none', 'FontSize', 8);

    % (2) Per-target position error: mean +/- std band
    subplot(2, 2, 2); hold on; grid on; box on;
    for tt = 1:N_gt_targets
        m = R.err_by_gt_mean(tt, :);
        s = R.err_by_gt_std(tt, :);
        valid = ~isnan(m);
        if any(valid)
            kk = find(valid);
            plot_band(kk, m(valid), s(valid), gt_colors(tt, :), ...
                sprintf('T%d mean +/- std', tt));
        end
    end
    xlabel('Frame'); ylabel('Position error [m]', 'Interpreter', 'none');
    title('Per-target Hungarian-aligned position error', 'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none', 'FontSize', 8);

    % (3) Cardinality: GT mean + per-trial est (low alpha) + mean est
    subplot(2, 2, 3); hold on; grid on; box on;
    for mc = 1:N_MC
        plot(1:T, N_est_seq_mc(mc, :), '-', 'Color', [color, 0.15], ...
             'HandleVisibility', 'off');
    end
    plot(1:T, R.N_gt_mean,  'k-',  'LineWidth', 2.0, 'DisplayName', 'GT (mean)');
    plot(1:T, R.N_est_mean, '--', 'Color', color, 'LineWidth', 1.8, ...
         'DisplayName', 'Est (mean over MC)');
    xlabel('Frame'); ylabel('Target count');
    title(sprintf('Cardinality - %s', fname), 'Interpreter', 'none');
    legend('Location', 'best');
    ylim([-0.5, max([R.N_gt_mean, R.N_est_mean]) + 1]);

    % (4) OSPA: mean +/- std, plus loc/card decomposition
    subplot(2, 2, 4); hold on; grid on; box on;
    plot_band(1:T, R.ospa_mean, R.ospa_std, color, 'OSPA mean +/- std');
    plot(1:T, R.ospa_loc_mean,  '-',  'Color', [0.2 0.6 0.2], ...
         'LineWidth', 1.2, 'DisplayName', 'OSPA loc (mean)');
    plot(1:T, R.ospa_card_mean, '-',  'Color', [0.85 0.4 0.1], ...
         'LineWidth', 1.2, 'DisplayName', 'OSPA card (mean)');
    xlabel('Frame'); ylabel('OSPA [m]');
    title('OSPA breakdown', 'Interpreter', 'none');
    legend('Location', 'best');

    sgtitle(sprintf('%s + ProbabilisticEstimator (N_{MC} = %d)', fname, N_MC), ...
            'Interpreter', 'tex');

    out_path = fullfile(out_dir, sprintf('test_dynamic_targets_%s_mc%d.png', ...
                        fname, N_MC));
    exportgraphics(fig, out_path, 'Resolution', 200);
    fprintf('  Saved figure: %s\n', out_path);
end

%% --- Comparison figure -------------------------------------------------
fig_cmp = figure('Color', 'w', 'Position', [80 80 1300 800]);

% (1) OSPA mean +/- std for both filters
subplot(2, 2, 1); hold on; grid on; box on;
for f = 1:n_filters
    R = results.(filter_specs(f).name);
    plot_band(1:T, R.ospa_mean, R.ospa_std, filter_colors(f, :), ...
              sprintf('%s', filter_specs(f).name));
end
xlabel('Frame'); ylabel('OSPA [m]');
title('OSPA: filter comparison', 'Interpreter', 'none');
legend('Location', 'best', 'Interpreter', 'none');

% (2) Cardinality MAE bar chart
subplot(2, 2, 2); hold on; grid on; box on;
mae_means = zeros(1, n_filters);
mae_stds  = zeros(1, n_filters);
for f = 1:n_filters
    mae_means(f) = results.(filter_specs(f).name).cardinality_mae.mean;
    mae_stds(f)  = results.(filter_specs(f).name).cardinality_mae.std;
end
b = bar(1:n_filters, mae_means, 0.5, 'FaceColor', 'flat');
for f = 1:n_filters, b.CData(f, :) = filter_colors(f, :); end
errorbar(1:n_filters, mae_means, mae_stds, 'k.', 'LineWidth', 1.2);
set(gca, 'XTick', 1:n_filters, ...
    'XTickLabel', {filter_specs.name}, 'TickLabelInterpreter', 'none');
ylabel('Cardinality MAE');
title('Cardinality MAE (mean +/- std over MC)', 'Interpreter', 'none');

% (3) Per-target RMSE grouped bar
subplot(2, 2, 3); hold on; grid on; box on;
rmse_grp = zeros(N_gt_targets, n_filters);
rmse_err = zeros(N_gt_targets, n_filters);
for f = 1:n_filters
    rmse_grp(:, f) = results.(filter_specs(f).name).rmse_per_target.mean(1:N_gt_targets).';
    rmse_err(:, f) = results.(filter_specs(f).name).rmse_per_target.std(1:N_gt_targets).';
end
b = bar(1:N_gt_targets, rmse_grp, 'grouped');
for f = 1:n_filters, b(f).FaceColor = filter_colors(f, :); end
% Place error bars at grouped centers
ngroups = N_gt_targets; nbars = n_filters;
gw = min(0.8, nbars / (nbars + 1.5));
for f = 1:nbars
    xc = (1:ngroups) - gw/2 + (2*f-1) * gw / (2*nbars);
    errorbar(xc, rmse_grp(:, f), rmse_err(:, f), 'k.', 'LineWidth', 1.0);
end
set(gca, 'XTick', 1:N_gt_targets, ...
    'XTickLabel', arrayfun(@(t) sprintf('T%d', t), 1:N_gt_targets, 'Uni', 0));
ylabel('RMSE [m]');
title('Per-target RMSE (mean +/- std over MC)', 'Interpreter', 'none');
legend({filter_specs.name}, 'Interpreter', 'none', 'Location', 'best');

% (4) Runtime per trial
subplot(2, 2, 4); hold on; grid on; box on;
rt_means = zeros(1, n_filters);
rt_stds  = zeros(1, n_filters);
for f = 1:n_filters
    rt_means(f) = results.(filter_specs(f).name).runtime_sec.mean;
    rt_stds(f)  = results.(filter_specs(f).name).runtime_sec.std;
end
b = bar(1:n_filters, rt_means, 0.5, 'FaceColor', 'flat');
for f = 1:n_filters, b.CData(f, :) = filter_colors(f, :); end
errorbar(1:n_filters, rt_means, rt_stds, 'k.', 'LineWidth', 1.2);
set(gca, 'XTick', 1:n_filters, ...
    'XTickLabel', {filter_specs.name}, 'TickLabelInterpreter', 'none');
ylabel('Runtime per trial [s]');
title('Runtime', 'Interpreter', 'none');

sgtitle(sprintf('Filter comparison (N_{MC} = %d, T = %d frames)', ...
                N_MC, T), 'Interpreter', 'tex');

cmp_path = fullfile(out_dir, sprintf('test_dynamic_targets_comparison_mc%d.png', N_MC));
exportgraphics(fig_cmp, cmp_path, 'Resolution', 200);
fprintf('\nSaved comparison figure: %s\n', cmp_path);

%% --- Candidate trajectory figures (per filter, trial 1) ---------------
% 2x2 per-filter figure:  (1,1) 2D trajectories   (1,2) per-target error
%                         (2,1) cardinality        (2,2) OSPA

for f = 1:n_filters
    fname  = filter_specs(f).name;
    R      = results.(fname);
    color  = filter_colors(f, :);
    GT_c   = R.cand_GT;
    est_c  = R.cand_est_xy;
    meas_c = R.cand_meas;

    if isempty(GT_c) || isempty(est_c), continue; end

    fig_c = figure('Color', 'w', 'Position', [70 70 1200 850]);
    gt_colors_c = lines(N_gt_targets);

    % (1,1) 2D trajectories
    subplot(2, 2, 1); hold on; box on; grid on;
    all_z_c = cell2mat(meas_c);
    scatter(all_z_c(1,:), all_z_c(2,:), 8, [0.80 0.80 0.80], '+', ...
            'LineWidth', 0.8, 'HandleVisibility', 'off');
    for tt = 1:N_gt_targets
        gt_traj = GT_c{tt};
        valid_k = find(~any(isnan(gt_traj), 1));
        if isempty(valid_k), continue; end
        plot(gt_traj(1, valid_k), gt_traj(2, valid_k), '-', ...
             'Color', [0.35 0.35 0.35], 'LineWidth', 2.2, ...
             'DisplayName', sprintf('GT T%d', tt));
        plot(gt_traj(1, valid_k(1)), gt_traj(2, valid_k(1)), 'o', ...
             'Color', [0.35 0.35 0.35], 'MarkerFaceColor', gt_colors_c(tt,:), ...
             'MarkerSize', 7, 'HandleVisibility', 'off');
    end
    % Estimated trajectories (color per target using Hungarian assignment at each frame)
    for tt = 1:N_gt_targets
        ex_c = nan(1, T); ey_c = nan(1, T);
        for k = 1:T
            if ~isempty(est_c{k}) && size(est_c{k}, 2) >= tt
                ex_c(k) = est_c{k}(1, tt);
                ey_c(k) = est_c{k}(2, tt);
            end
        end
        plot(ex_c, ey_c, '--', 'Color', gt_colors_c(tt,:), 'LineWidth', 1.5, ...
             'DisplayName', sprintf('Est T%d', tt));
    end
    xlabel('x [m]'); ylabel('y [m]'); axis equal;
    title('Trajectories (candidate trial)', 'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none', 'FontSize', 8);

    % (1,2) Per-target position error (candidate trial, Hungarian-aligned)
    subplot(2, 2, 2); hold on; grid on; box on;
    for tt = 1:N_gt_targets
        gt_traj = GT_c{tt};
        valid_k = find(~any(isnan(gt_traj), 1));
        err_k = nan(1, T);
        for k = valid_k
            if ~isempty(est_c{k}) && size(est_c{k}, 2) >= 1
                % Simple nearest-GT assignment for this display
                dists = sqrt(sum((est_c{k} - gt_traj(1:2, k)).^2, 1));
                err_k(k) = min(dists);
            end
        end
        plot(1:T, err_k, '-', 'Color', gt_colors_c(tt,:), 'LineWidth', 1.5, ...
             'DisplayName', sprintf('T%d', tt));
    end
    xlabel('Frame'); ylabel('Position error [m]');
    title('Per-target position error (candidate trial)', 'Interpreter', 'none');
    legend('Location', 'best', 'Interpreter', 'none', 'FontSize', 8);

    % (2,1) Cardinality
    subplot(2, 2, 3); hold on; grid on; box on;
    plot(1:T, R.cand_N_gt, 'k-', 'LineWidth', 2.0, 'DisplayName', 'GT count');
    plot(1:T, R.cand_N_est, '--', 'Color', color, 'LineWidth', 1.8, ...
         'DisplayName', 'Estimated count');
    xlabel('Frame'); ylabel('Target count');
    title('Cardinality (candidate trial)', 'Interpreter', 'none');
    legend('Location', 'best');
    ylim([-0.5, max([R.cand_N_gt, R.cand_N_est]) + 1]);

    % (2,2) OSPA (candidate trial)
    subplot(2, 2, 4); hold on; grid on; box on;
    gt_xy_c = build_gt_xy_seq(GT_c, T);
    ospa_c = zeros(1, T);
    for k = 1:T
        est_k = est_c{k};
        gt_k  = gt_xy_c{k};
        if isempty(est_k) || isempty(gt_k)
            ospa_c(k) = 1.0;
        else
            [ospa_c(k), ~, ~] = compute_ospa(est_k, gt_k, 1.0, 2);
        end
    end
    plot(1:T, ospa_c, '-', 'Color', color, 'LineWidth', 1.7, 'DisplayName', 'OSPA');
    xlabel('Frame'); ylabel('OSPA [m]');
    title('OSPA (candidate trial)', 'Interpreter', 'none');
    legend('Location', 'best');

    sgtitle(sprintf('%s + ProbabilisticEstimator — Candidate Trajectory (trial 1)', fname), ...
            'Interpreter', 'none', 'FontSize', 12);

    cand_path = fullfile(out_dir, sprintf('test_dynamic_targets_%s_candidate.png', fname));
    exportgraphics(fig_c, cand_path, 'Resolution', 200);
    fprintf('  Saved candidate figure: %s\n', cand_path);
end

%% --- Save numerical results -------------------------------------------
results_meta = struct('N_MC', N_MC, 'T', T, 'dt', dt, ...
    'PD_true', PD_true, 'lambda_clutter', n_clutter_lambda, ...
    'base_seed', base_seed, 'schedule', schedule, ...
    'filter_names', {{filter_specs.name}});
mat_path = fullfile(out_dir, sprintf('test_dynamic_targets_results_mc%d.mat', N_MC));
save(mat_path, 'results', 'results_meta', '-v7');
fprintf('Saved results: %s\n', mat_path);

%% --- Pass / fail across filters ---------------------------------------
fprintf('\n==================== PASS/FAIL ====================\n');
all_pass = true;
for f = 1:n_filters
    R = results.(filter_specs(f).name);
    pass_f = R.cardinality_mae.mean <= 0.5;
    all_pass = all_pass && pass_f;
    if pass_f
        fprintf('  %-16s PASS  (cardinality MAE %.3f +/- %.3f)\n', ...
            filter_specs(f).name, R.cardinality_mae.mean, R.cardinality_mae.std);
    else
        fprintf('  %-16s FAIL  (cardinality MAE %.3f > 0.5)\n', ...
            filter_specs(f).name, R.cardinality_mae.mean);
    end
end
if all_pass
    fprintf('\n*** test_dynamic_targets PASSED across all filters ***\n');
else
    fprintf('\n!!! test_dynamic_targets had failures !!!\n');
end


%% =====================================================================
%  Local helpers
%% =====================================================================

function GT = simulate_GT(schedule, F, Q, T)
    n  = numel(schedule);
    Nx = size(F, 1);
    L  = chol(Q, 'lower');
    GT = cell(n, 1);
    for i = 1:n
        GT{i} = nan(Nx, T);
        x = schedule(i).x0;
        for k = schedule(i).k_start:schedule(i).k_end
            GT{i}(:, k) = x;
            x = F * x + L * randn(Nx, 1);
        end
    end
end

function N_gt_seq = compute_N_gt(GT, T)
    N_gt_seq = zeros(1, T);
    for k = 1:T
        N_gt_seq(k) = sum(cellfun(@(g) ~any(isnan(g(:, k))), GT));
    end
end

function gt_xy_seq = build_gt_xy_seq(GT, T)
    gt_xy_seq = cell(1, T);
    for k = 1:T
        cols = [];
        for tt = 1:numel(GT)
            if ~any(isnan(GT{tt}(:, k))), cols(end+1) = tt; end %#ok<AGROW>
        end
        xy = zeros(2, numel(cols));
        for a = 1:numel(cols)
            xy(:, a) = GT{cols(a)}(1:2, k);
        end
        gt_xy_seq{k} = xy;
    end
end

function ids = gt_id_per_frame(GT, T)
    ids = cell(1, T);
    for k = 1:T
        v = [];
        for tt = 1:numel(GT)
            if ~any(isnan(GT{tt}(:, k))), v(end+1) = tt; end %#ok<AGROW>
        end
        ids{k} = v;
    end
end

function measurements = generate_measurements(GT, H, R, PD, lambda, box, T)
    measurements = cell(1, T);
    Nz = size(H, 1);
    Lr = chol(R, 'lower');
    n_gt = numel(GT);
    for k = 1:T
        z = zeros(Nz, 0);
        for tt = 1:n_gt
            if any(isnan(GT{tt}(:, k))), continue; end
            if rand() <= PD
                z(:, end+1) = H * GT{tt}(:, k) + Lr * randn(Nz, 1); %#ok<AGROW>
            end
        end
        n_c = poissrnd(lambda);
        for c = 1:n_c
            z(:, end+1) = [box(1,1) + rand()*(box(1,2)-box(1,1)); ...
                           box(2,1) + rand()*(box(2,2)-box(2,1))]; %#ok<AGROW>
        end
        if ~isempty(z)
            ord = randperm(size(z, 2));
            z = z(:, ord);
        end
        measurements{k} = z;
    end
end

function plot_band(x, m, s, color, label)
    % Mean line plus +/- std shaded band.
    x = x(:).'; m = m(:).'; s = s(:).';
    s(isnan(s)) = 0;
    upper = m + s;
    lower = m - s;
    fill([x, fliplr(x)], [upper, fliplr(lower)], color, ...
         'FaceAlpha', 0.20, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(x, m, '-', 'Color', color, 'LineWidth', 1.7, 'DisplayName', label);
end
