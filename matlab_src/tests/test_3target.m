% test_3target.m  —  Monte Carlo 3-target tracking test
%
% Runs three multi-target filters on a synthetic 3-target scenario with
% crossing trajectories.  GT is fixed (one deterministic realisation);
% each MC trial draws fresh measurement noise + clutter so results are
% reproducible and the candidate trajectory is meaningful for all trials.
%
% SCENARIO:
%   State model  : constant-velocity 4-state filter, 6-state truth propagation
%   Duration     : 60 timesteps (6 s)
%   3 targets with crossing geometry:
%     T1: (0.5, 0.5)  vx= 0.30  vy= 0.40  (rightward-upward)
%     T2: (3.5, 0.5)  vx=-0.30  vy= 0.40  (converging toward T1)
%     T3: (2.0, 4.0)  vx= 0.00  vy=-0.50  (descending, cuts across)
%   sigma_z = 0.05 m,  PD = 0.9,  1-3 clutter/step
%
% FILTERS:
%   JPDA-KF, KF-RBPF-multi, PDA-PF-multi
%
% OUTPUTS  (all in tests/figures/multitarget_testing/):
%   test_3target_GT_meas.gif              — candidate trial GT + measurements
%   test_3target_{FILTER}_candidate.png   — per-filter candidate trajectory + error
%   test_3target_all_filters_candidate.gif — combined animation (candidate trial)
%   test_3target_mc_comparison.png        — MC aggregate comparison
%   test_3target_results_mc{N}.mat        — numerical results
%   test_3target_perf_mc{N}.tex           — LaTeX performance table
%
% Run from matlab_src/.

clear; clc; close all;

%% -----------------------------------------------------------------------
%% 0. Paths + output directory
%% -----------------------------------------------------------------------
script_dir    = fileparts(mfilename('fullpath'));
repo_root     = fullfile(script_dir, '..');

addpath(fullfile(repo_root, 'DA_Track'));
addpath(fullfile(repo_root, 'DA_Track', 'single'));
addpath(fullfile(repo_root, 'DA_Track', 'multi'));
addpath(fullfile(repo_root, 'supplemental'));
addpath(fullfile(repo_root, 'supplemental', 'multitarget_metrics'));

out_dir = fullfile(script_dir, 'figures', 'multitarget_testing');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

fprintf('=== 3-Target Synthetic Tracking Test (MC) ===\n\n');

%% -----------------------------------------------------------------------
%% 1. Simulation parameters
%% -----------------------------------------------------------------------
N_MC      = 50;
base_seed = 42;
dt        = 0.1;
T         = 60;
N_t       = 3;
sigma_z   = 0.05;
PD_true   = 0.9;

x_bounds   = [-0.5, 5.0];
y_bounds   = [-0.5, 5.0];
scene_area = diff(x_bounds) * diff(y_bounds);

F_true = [1, 0, dt, 0, 0.5*dt^2, 0;
          0, 1, 0, dt, 0, 0.5*dt^2;
          0, 0, 1,  0, dt, 0;
          0, 0, 0,  1,  0, dt;
          0, 0, 0,  0,  1,  0;
          0, 0, 0,  0,  0,  1];

F_model = [1, 0, dt,  0;
           0, 1,  0, dt;
           0, 0,  1,  0;
           0, 0,  0,  1];

H      = [1, 0, 0, 0; 0, 1, 0, 0];
H_true = [1, 0, 0, 0, 0, 0; 0, 1, 0, 0, 0, 0];

Q  = diag([1e-4, 1e-4, 1e-3, 1e-3]);
R  = sigma_z^2 * eye(2);
P0 = diag([0.1, 0.1, 0.25, 0.25]);

% Different crossing geometry from original scenario
x0_true = {[0.5; 0.5;  0.30;  0.40; 0; 0], ...   % T1 right+up
           [3.5; 0.5; -0.30;  0.40; 0; 0], ...   % T2 left+up  (meets T1 mid-scene)
           [2.0; 4.0;  0.00; -0.50; 0; 0]};       % T3 descending (cuts across)

Q_traj = diag([1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6]);

%% -----------------------------------------------------------------------
%% 2. Fixed GT (generated once with base_seed)
%% -----------------------------------------------------------------------
rng(base_seed);
GT = cell(1, N_t);
for t = 1:N_t
    GT{t} = zeros(6, T);
    GT{t}(:, 1) = x0_true{t};
    for k = 2:T
        GT{t}(:, k) = F_true * GT{t}(:, k-1) + sqrtm(Q_traj) * randn(6, 1);
    end
end

% Initial filter states: 4-state GT + small perturbation
x0_cell = cell(1, N_t);
rng(base_seed + 1000);
for t = 1:N_t
    x0_cell{t} = GT{t}(1:4, 1) + [0.15*randn(2,1); 0.10*randn(2,1)];
end

fprintf('GT generated (%d targets, %d steps).\n', N_t, T);

%% -----------------------------------------------------------------------
%% 3. Visual constants (consistent across all figures/GIFs)
%% -----------------------------------------------------------------------
tgt_markers  = {'o', 's', '^'};
tgt_msize_sm = 5;
tgt_msize_lg = 9;
mark_idx   = round(linspace(1, T, 10));
gt_color   = [0.15, 0.15, 0.15];
gt_lw      = 2.5;
est_lw     = 1.5;
meas_color = [0.72, 0.72, 0.72];
tgt_colors = lines(N_t);
t_axis     = (0:T-1) * dt;

filter_colors = [0.12, 0.47, 0.71;   % JPDA-KF    blue
                 0.89, 0.47, 0.10;   % KF-RBPF    orange
                 0.17, 0.63, 0.17];  % PDA-PF     green

filter_names_short = {'JPDA-KF', 'KF-RBPF', 'PDA-PF'};
filter_names_full  = {'JPDA-KF (baseline)', 'KF-RBPF-multi', 'PDA-PF-multi'};
filter_file_tags   = {'JPDA_KF', 'KF_RBPF_multi', 'PDA_PF_multi'};
filter_names_tex   = {'JPDA-KF (baseline)', 'KF-RBPF-multi', 'PDA-PF-multi'};

%% -----------------------------------------------------------------------
%% 4. MC loop — per filter
%% -----------------------------------------------------------------------
mc_results = struct();

for fi = 1:3
    fname = filter_names_short{fi};
    fprintf('\n=== %s (%d MC trials) ===\n', fname, N_MC);

    rmse_trials = zeros(N_MC, N_t);   % mean-over-time per-target RMSE
    ospa_trials = zeros(N_MC, 1);
    runtime_sec = zeros(N_MC, 1);

    cand_est  = [];
    cand_Pest = [];
    cand_meas = [];

    for mc = 1:N_MC
        rng(base_seed + mc);   % fresh measurements, same GT

        meas = gen_measurements_3t(GT, H_true, R, PD_true, x_bounds, y_bounds, T);

        % Build fresh filter
        switch fi
            case 1   % JPDA-KF
                x0_jpda = cell2mat(x0_cell);
                P0_jpda = repmat({P0}, 1, N_t);
                filt = JPDA_KF(x0_jpda, P0_jpda, F_model, Q, R, H, N_t, []);
                filt.PD                     = PD_true;
                filt.lambda_clutter         = 2.0;
                filt.measurement_space_area = scene_area;
            case 2   % KF-RBPF-multi
                filt = KF_RBPF_multi(x0_cell, 200, F_model, Q, H, R, ...
                    'PD', 0.9, 'PFA', 0.05, 'ESSThreshold', 0.5, ...
                    'AssociationStrategy', 'optimal', 'Debug', false, ...
                    'InitSigmaPos', 0.03, 'InitSigmaVel', 0.02);
            case 3   % PDA-PF-multi
                filt = PDA_PF_multi(x0_cell, 1000, F_model, Q, H, R, ...
                    'PD', 0.9, 'PFA', 0.05, 'LambdaClutter', 3.0/scene_area, ...
                    'ValidationSigma', 5, 'ESSThreshold', 0.5, 'Debug', false, ...
                    'InitSigmaPos', 0.03, 'InitSigmaVel', 0.05);
        end

        % Initial estimate
        est_hist  = cell(1, T);
        Pest_hist = cell(1, T);
        [est_hist{1}, Pest_hist{1}] = get_estimate(filt, fi, N_t);

        % Run
        t0 = tic;
        for k = 2:T
            step_filter(filt, fi, meas{k});
            [est_hist{k}, Pest_hist{k}] = get_estimate(filt, fi, N_t);
        end
        runtime_sec(mc) = toc(t0);

        % Per-target RMSE (mean over frames)
        for t = 1:N_t
            err = arrayfun(@(k) norm(est_hist{k}{t}(1:2) - GT{t}(1:2, k)), 1:T);
            rmse_trials(mc, t) = mean(err);
        end

        % Mean OSPA (fixed cardinality -> localization only)
        ospa_k = zeros(1, T);
        for k = 1:T
            gt_xy  = cell2mat(cellfun(@(g) g(1:2,k), GT, 'Uni', 0));
            est_xy = cell2mat(cellfun(@(e) e(1:2),   est_hist{k}, 'Uni', 0));
            [ospa_k(k), ~, ~] = compute_ospa(est_xy, gt_xy, 1.0, 2);
        end
        ospa_trials(mc) = mean(ospa_k);

        if mc == 1
            cand_est  = est_hist;
            cand_Pest = Pest_hist;
            cand_meas = meas;
        end

        if mod(mc, max(1, round(N_MC / 5))) == 0 || mc == N_MC
            fprintf('  trial %3d/%d  (%.2f s)  -- current mean OSPA: %.3f\n', ...
            mc, N_MC, runtime_sec(mc), mean(ospa_trials(1:mc)));
        end
    end

    % Aggregate
    S.rmse_mean = mean(rmse_trials, 1);
    S.rmse_std  = std(rmse_trials,  0, 1);
    S.ospa_mean = mean(ospa_trials);
    S.ospa_std  = std(ospa_trials);
    S.runtime   = struct('mean', mean(runtime_sec), 'std', std(runtime_sec));
    S.cand_est  = cand_est;
    S.cand_Pest = cand_Pest;
    S.cand_meas = cand_meas;
    S.raw_rmse  = rmse_trials;

    mc_results.(filter_file_tags{fi}) = S;

    fprintf('  RMSE: T1=%.3f  T2=%.3f  T3=%.3f  OSPA=%.3f  Runtime=%.2fs\n', ...
        S.rmse_mean(1), S.rmse_mean(2), S.rmse_mean(3), S.ospa_mean, S.runtime.mean);
end

%% -----------------------------------------------------------------------
%% 5. GT + measurements animation  (candidate trial / trial 1)
%% -----------------------------------------------------------------------
fprintf('\nGenerating GT+measurements animation...\n');
meas_cand = mc_results.(filter_file_tags{1}).cand_meas;
gif_gt    = fullfile(out_dir, 'test_3target_GT_meas.gif');
fig_gt    = figure('Name', 'GT + Measurements', 'Position', [50 50 620 540]);

for k = 1:T
    clf; hold on; grid on; axis equal;
    title(sprintf('GT + Measurements  |  t = %.1f s', (k-1)*dt), 'FontSize', 11);
    xlabel('x (m)'); ylabel('y (m)');
    xlim(x_bounds + [-0.3, 0.3]); ylim(y_bounds + [-0.3, 0.3]);

    zk = meas_cand{k};
    scatter(zk(1,:), zk(2,:), 45, meas_color, '+', 'LineWidth', 1.4, ...
            'DisplayName', 'Measurements');

    for t = 1:N_t
        plot(GT{t}(1,1:k), GT{t}(2,1:k), '-', 'Color', tgt_colors(t,:), ...
             'LineWidth', gt_lw, 'HandleVisibility', 'off');
        plot(GT{t}(1,k), GT{t}(2,k), tgt_markers{t}, ...
             'Color', tgt_colors(t,:), 'MarkerSize', tgt_msize_lg, ...
             'MarkerFaceColor', tgt_colors(t,:), 'DisplayName', sprintf('GT T%d', t));
    end

    legend('Location', 'northeast', 'FontSize', 9, 'Box', 'on');
    drawnow;
    write_gif(fig_gt, gif_gt, k, dt);
end
close(fig_gt);
fprintf('  Saved: %s\n', gif_gt);

%% -----------------------------------------------------------------------
%% 6. Per-filter candidate trajectory figures
%%    Row 1: 2D trajectory (one subplot per target)
%%    Row 2: x-error +/-2sigma
%%    Row 3: y-error +/-2sigma
%% -----------------------------------------------------------------------
fprintf('Generating per-filter candidate trajectory figures...\n');
all_z_cand = cell2mat(meas_cand);   % all candidate measurements concatenated

for fi = 1:3
    S    = mc_results.(filter_file_tags{fi});
    est  = S.cand_est;
    Pest = S.cand_Pest;
    fc   = filter_colors(fi, :);

    fig = figure('Position', [50+fi*25, 50+fi*25, 1200, 900]);

    %% Row 1 — 2D trajectory per target
    for t = 1:N_t
        subplot(3, N_t, t); hold on; grid on; axis equal;
        title(sprintf('Target %d — Trajectory', t), 'FontSize', 10);
        xlabel('x (m)'); ylabel('y (m)');

        scatter(all_z_cand(1,:), all_z_cand(2,:), 6, meas_color, '+', ...
                'HandleVisibility', 'off');

        plot(GT{t}(1,:), GT{t}(2,:), '-', 'Color', tgt_colors(t,:), ...
             'LineWidth', gt_lw, 'DisplayName', 'GT');
        plot(GT{t}(1,1), GT{t}(2,1), tgt_markers{t}, ...
             'Color', tgt_colors(t,:), 'MarkerSize', tgt_msize_lg, ...
             'MarkerFaceColor', tgt_colors(t,:), 'HandleVisibility', 'off');

        ex = cellfun(@(e) e{t}(1), est);
        ey = cellfun(@(e) e{t}(2), est);
        ec = tgt_colors(t,:) * 0.6 + 0.1;
        plot(ex, ey, '--', 'Color', ec, 'LineWidth', est_lw, 'HandleVisibility', 'off');
        plot(ex(mark_idx), ey(mark_idx), tgt_markers{t}, ...
             'Color', ec, 'MarkerSize', tgt_msize_sm, 'MarkerFaceColor', ec, ...
             'LineStyle', 'none', 'DisplayName', 'Estimate');

        legend('Location', 'best', 'FontSize', 9, 'Box', 'on');
        xlim(x_bounds + [-0.3 0.3]); ylim(y_bounds + [-0.3 0.3]);
    end

    %% Row 2 — x-error +/-2sigma
    for t = 1:N_t
        subplot(3, N_t, N_t+t); hold on; grid on;
        title(sprintf('T%d  —  x-error  \\pm2\\sigma_x', t), 'FontSize', 10);
        xlabel('Time (s)'); ylabel('x error (m)');

        err_x  = cellfun(@(e) e{t}(1), est) - GT{t}(1,:);
        sig2_x = 2 * sqrt(cellfun(@(P) P{t}(1,1), Pest));

        fill([t_axis, fliplr(t_axis)], [sig2_x, fliplr(-sig2_x)], ...
             tgt_colors(t,:), 'FaceAlpha', 0.20, 'EdgeColor', 'none', 'DisplayName', '\pm2\sigma');
        plot(t_axis, err_x, '-', 'Color', tgt_colors(t,:), 'LineWidth', 1.5, ...
             'DisplayName', 'error');
        yline(0, '-k', 'LineWidth', 0.5, 'HandleVisibility', 'off');
        legend('Location', 'best', 'FontSize', 9);
    end

    %% Row 3 — y-error +/-2sigma
    for t = 1:N_t
        subplot(3, N_t, 2*N_t+t); hold on; grid on;
        title(sprintf('T%d  —  y-error  \\pm2\\sigma_y', t), 'FontSize', 10);
        xlabel('Time (s)'); ylabel('y error (m)');

        err_y  = cellfun(@(e) e{t}(2), est) - GT{t}(2,:);
        sig2_y = 2 * sqrt(cellfun(@(P) P{t}(2,2), Pest));

        fill([t_axis, fliplr(t_axis)], [sig2_y, fliplr(-sig2_y)], ...
             tgt_colors(t,:), 'FaceAlpha', 0.20, 'EdgeColor', 'none', 'DisplayName', '\pm2\sigma');
        plot(t_axis, err_y, '-', 'Color', tgt_colors(t,:), 'LineWidth', 1.5, ...
             'DisplayName', 'error');
        yline(0, '-k', 'LineWidth', 0.5, 'HandleVisibility', 'off');
        legend('Location', 'best', 'FontSize', 9);
    end

    sgtitle(sprintf('%s — Candidate Trajectory (trial 1)', filter_names_full{fi}), ...
            'FontSize', 13, 'FontWeight', 'bold');

    out_path = fullfile(out_dir, sprintf('test_3target_%s_candidate.png', filter_file_tags{fi}));
    exportgraphics(fig, out_path, 'Resolution', 150);
    fprintf('  Saved: %s\n', out_path);
end

%% -----------------------------------------------------------------------
%% 7. Combined multi-filter animation  (candidate trial)
%% -----------------------------------------------------------------------
fprintf('Generating combined animation (candidate trial)...\n');

% Re-run candidate trial to collect Pest for all filters simultaneously
cand_est_all  = cell(1, 3);
cand_Pest_all = cell(1, 3);
for fi = 1:3
    cand_est_all{fi}  = mc_results.(filter_file_tags{fi}).cand_est;
    cand_Pest_all{fi} = mc_results.(filter_file_tags{fi}).cand_Pest;
end

gif_all = fullfile(out_dir, 'test_3target_all_filters_candidate.gif');
fig_all = figure('Name', 'All Filters (candidate)', 'Position', [120 80 720 620]);

for k = 1:T
    clf; hold on; grid on; axis equal;
    title(sprintf('All Filters  |  t = %.1f s', (k-1)*dt), 'FontSize', 11);
    xlabel('x (m)'); ylabel('y (m)');
    xlim(x_bounds + [-0.35, 0.35]); ylim(y_bounds + [-0.35, 0.35]);

    for fi = 1:3
        plot(NaN, NaN, '-o', 'Color', filter_colors(fi,:), ...
             'MarkerFaceColor', filter_colors(fi,:), 'LineWidth', est_lw, ...
             'MarkerSize', tgt_msize_sm+1, 'DisplayName', filter_names_short{fi});
    end
    plot(NaN, NaN, '-', 'Color', gt_color, 'LineWidth', gt_lw, 'DisplayName', 'GT');
    for t = 1:N_t
        plot(NaN, NaN, tgt_markers{t}, 'Color', [0.45 0.45 0.45], ...
             'MarkerFaceColor', [0.45 0.45 0.45], 'MarkerSize', tgt_msize_sm+1, ...
             'LineStyle', 'none', 'DisplayName', sprintf('T%d', t));
    end
    scatter(NaN, NaN, 45, meas_color, '+', 'LineWidth', 1.4, 'DisplayName', 'Meas');

    zk = meas_cand{k};
    scatter(zk(1,:), zk(2,:), 45, meas_color, '+', 'LineWidth', 1.4, ...
            'HandleVisibility', 'off');

    for t = 1:N_t
        plot(GT{t}(1,1:k), GT{t}(2,1:k), '-', 'Color', gt_color, ...
             'LineWidth', gt_lw, 'HandleVisibility', 'off');
        plot(GT{t}(1,k), GT{t}(2,k), tgt_markers{t}, 'Color', gt_color, ...
             'MarkerSize', tgt_msize_lg, 'MarkerFaceColor', gt_color, ...
             'HandleVisibility', 'off');
    end

    for fi = 1:3
        fc   = filter_colors(fi,:);
        est  = cand_est_all{fi};
        Pest = cand_Pest_all{fi};
        for t = 1:N_t
            ex = cellfun(@(e) e{t}(1), est);
            ey = cellfun(@(e) e{t}(2), est);
            plot(ex(1:k), ey(1:k), '-', 'Color', [fc, 0.55], ...
                 'LineWidth', est_lw, 'HandleVisibility', 'off');
            P_pos = Pest{k}{t}(1:2, 1:2);
            exy   = cov_ellipse([ex(k); ey(k)], P_pos, 2);
            patch(exy(1,:), exy(2,:), fc, 'FaceAlpha', 0.12, 'EdgeColor', fc, ...
                  'LineWidth', 0.9, 'HandleVisibility', 'off');
            plot(ex(k), ey(k), tgt_markers{t}, 'Color', fc, ...
                 'MarkerSize', tgt_msize_lg, 'MarkerFaceColor', fc, ...
                 'LineWidth', 1.2, 'HandleVisibility', 'off');
        end
    end

    legend('Location', 'northeast', 'FontSize', 8, 'Box', 'on', 'NumColumns', 2);
    drawnow;
    write_gif(fig_all, gif_all, k, dt);
end
close(fig_all);
fprintf('  Saved: %s\n', gif_all);

%% -----------------------------------------------------------------------
%% 8. MC comparison figure
%% -----------------------------------------------------------------------
fprintf('Generating MC comparison figure...\n');
fig_cmp = figure('Color', 'w', 'Position', [100 100 1100 800]);

% Panel 1: Per-target RMSE grouped bar (targets as groups)
subplot(2, 2, 1); hold on; grid on; box on;
rmse_grp = zeros(N_t, 3);
rmse_err = zeros(N_t, 3);
for fi = 1:3
    rmse_grp(:, fi) = mc_results.(filter_file_tags{fi}).rmse_mean.';
    rmse_err(:, fi) = mc_results.(filter_file_tags{fi}).rmse_std.';
end
b = bar(1:N_t, rmse_grp, 'grouped');
for fi = 1:3, b(fi).FaceColor = filter_colors(fi,:); end
ngroups = N_t; nbars = 3;
gw = min(0.8, nbars / (nbars + 1.5));
for fi = 1:nbars
    xc = (1:ngroups) - gw/2 + (2*fi-1) * gw / (2*nbars);
    errorbar(xc, rmse_grp(:, fi), rmse_err(:, fi), 'k.', 'LineWidth', 1.0);
end
set(gca, 'XTick', 1:N_t, 'XTickLabel', {'T1','T2','T3'});
ylabel('Mean RMSE [m]');
title('Per-target RMSE (mean \pm std, MC)', 'Interpreter', 'tex');
legend(filter_names_short, 'Interpreter', 'none', 'Location', 'best');

% Panel 2: Mean OSPA bar
subplot(2, 2, 2); hold on; grid on; box on;
ospa_m = arrayfun(@(fi) mc_results.(filter_file_tags{fi}).ospa_mean, 1:3);
ospa_s = arrayfun(@(fi) mc_results.(filter_file_tags{fi}).ospa_std,  1:3);
b2 = bar(1:3, ospa_m, 0.5, 'FaceColor', 'flat');
for fi = 1:3, b2.CData(fi,:) = filter_colors(fi,:); end
errorbar(1:3, ospa_m, ospa_s, 'k.', 'LineWidth', 1.2);
set(gca, 'XTick', 1:3, 'XTickLabel', filter_names_short, 'TickLabelInterpreter', 'none');
ylabel('Mean OSPA [m]');
title('Mean OSPA (mean \pm std, MC)', 'Interpreter', 'tex');

% Panel 3: Per-target RMSE distribution (box plot)
subplot(2, 2, 3); hold on; grid on; box on;
boxplot_data = cell(3, N_t);
for fi = 1:3
    raw = mc_results.(filter_file_tags{fi}).raw_rmse;   % [N_MC x N_t]
    for t = 1:N_t
        boxplot_data{fi, t} = raw(:, t);
    end
end
grp_gap = 1;  % gap between target groups
grp_size = 3; % one box per filter
grp_centers = zeros(1, N_t);
for t = 1:N_t
    grp_start = (t-1) * (grp_size + grp_gap) + 1;
    grp_centers(t) = grp_start + (grp_size - 1) / 2;
    for fi = 1:3
        bx_x = grp_start + (fi - 1);
        d = boxplot_data{fi, t};
        q25 = prctile(d, 25); q75 = prctile(d, 75); q50 = median(d);
        iqr_d = q75 - q25;
        whi_lo = max(min(d), q25 - 1.5*iqr_d);
        whi_hi = min(max(d), q75 + 1.5*iqr_d);
        rectangle('Position', [bx_x-0.3, q25, 0.6, q75-q25], ...
                  'FaceColor', [filter_colors(fi,:), 0.35], 'EdgeColor', filter_colors(fi,:));
        plot([bx_x-0.3, bx_x+0.3], [q50, q50], '-', 'Color', filter_colors(fi,:), 'LineWidth', 2);
        plot([bx_x, bx_x], [q25, whi_lo], '-', 'Color', filter_colors(fi,:));
        plot([bx_x, bx_x], [q75, whi_hi], '-', 'Color', filter_colors(fi,:));
    end
end
set(gca, 'XTick', grp_centers, 'XTickLabel', {'T1','T2','T3'}, 'FontSize', 9);
ylabel('RMSE [m]');
title('RMSE distribution (MC trials)');

% Panel 4: Runtime
subplot(2, 2, 4); hold on; grid on; box on;
rt_m = arrayfun(@(fi) mc_results.(filter_file_tags{fi}).runtime.mean, 1:3);
rt_s = arrayfun(@(fi) mc_results.(filter_file_tags{fi}).runtime.std,  1:3);
b4 = bar(1:3, rt_m, 0.5, 'FaceColor', 'flat');
for fi = 1:3, b4.CData(fi,:) = filter_colors(fi,:); end
errorbar(1:3, rt_m, rt_s, 'k.', 'LineWidth', 1.2);
set(gca, 'XTick', 1:3, 'XTickLabel', filter_names_short, 'TickLabelInterpreter', 'none');
ylabel('Runtime per trial [s]');
title('Runtime (mean \pm std, MC)', 'Interpreter', 'tex');

sgtitle(sprintf('3-Target MC Comparison  (N_{MC} = %d,  T = %d frames)', N_MC, T), ...
        'Interpreter', 'tex', 'FontSize', 13);

cmp_path = fullfile(out_dir, sprintf('test_3target_mc_comparison_mc%d.png', N_MC));
exportgraphics(fig_cmp, cmp_path, 'Resolution', 150);
fprintf('  Saved: %s\n', cmp_path);

%% -----------------------------------------------------------------------
%% 9. LaTeX performance table
%% -----------------------------------------------------------------------
tex_path = fullfile(out_dir, sprintf('test_3target_perf_mc%d.tex', N_MC));
fid = fopen(tex_path, 'w');
fprintf(fid, '%% Auto-generated by test_3target.m  (N_MC = %d, T = %d)\n', N_MC, T);
fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '  \\centering\n');
fprintf(fid, '  \\caption{3-Target Tracking Performance ($N_{\\mathrm{MC}} = %d$, $T = %d$ frames, $\\sigma_z = %.2f$\\,m)}\n', N_MC, T, sigma_z);
fprintf(fid, '  \\label{tab:3target_mc}\n');
fprintf(fid, '  \\begin{tabular}{lcccccc}\n');
fprintf(fid, '    \\toprule\n');
fprintf(fid, '    Filter & RMSE$_{T1}$ (m) & RMSE$_{T2}$ (m) & RMSE$_{T3}$ (m) & OSPA (m) & Runtime (s) \\\\\n');
fprintf(fid, '    \\midrule\n');
for fi = 1:3
    S = mc_results.(filter_file_tags{fi});
    fprintf(fid, '    %s & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & $%.2f \\pm %.2f$ \\\\\n', ...
        filter_names_tex{fi}, ...
        S.rmse_mean(1), S.rmse_std(1), ...
        S.rmse_mean(2), S.rmse_std(2), ...
        S.rmse_mean(3), S.rmse_std(3), ...
        S.ospa_mean, S.ospa_std, ...
        S.runtime.mean, S.runtime.std);
end
fprintf(fid, '    \\bottomrule\n');
fprintf(fid, '  \\end{tabular}\n');
fprintf(fid, '\\end{table}\n');
fclose(fid);
fprintf('  Saved: %s\n', tex_path);

%% -----------------------------------------------------------------------
%% 10. Save numerical results
%% -----------------------------------------------------------------------
mat_path = fullfile(out_dir, sprintf('test_3target_results_mc%d.mat', N_MC));
meta = struct('N_MC', N_MC, 'T', T, 'dt', dt, 'sigma_z', sigma_z, ...
              'PD_true', PD_true, 'base_seed', base_seed, ...
              'filter_names', {filter_names_full});
save(mat_path, 'mc_results', 'meta', '-v7');
fprintf('  Saved: %s\n', mat_path);

%% -----------------------------------------------------------------------
%% 11. Console summary
%% -----------------------------------------------------------------------
fprintf('\n=== Mean RMSE (mean over MC trials and time, position) ===\n');
fprintf('%-22s  %6s  %6s  %6s  %6s\n', 'Filter', 'T1', 'T2', 'T3', 'OSPA');
fprintf('%s\n', repmat('-', 1, 55));
for fi = 1:3
    S = mc_results.(filter_file_tags{fi});
    fprintf('%-22s  %6.3f  %6.3f  %6.3f  %6.3f\n', filter_names_full{fi}, ...
        S.rmse_mean(1), S.rmse_mean(2), S.rmse_mean(3), S.ospa_mean);
end
fprintf('\nAll outputs in: %s\n', out_dir);


%% =======================================================================
%%  Local helpers
%% =======================================================================

function meas = gen_measurements_3t(GT, H_true, R, PD, x_bounds, y_bounds, T)
    meas = cell(1, T);
    Lr   = chol(R, 'lower');
    N_t  = numel(GT);
    for k = 1:T
        z = zeros(2, 0);
        for t = 1:N_t
            if rand() < PD
                z(:, end+1) = H_true * GT{t}(:,k) + Lr * randn(2,1); %#ok<AGROW>
            end
        end
        n_c = randi([1, 3]);
        for c = 1:n_c
            z(:, end+1) = [x_bounds(1)+diff(x_bounds)*rand(); ...
                           y_bounds(1)+diff(y_bounds)*rand()]; %#ok<AGROW>
        end
        perm = randperm(size(z,2));
        meas{k} = z(:, perm);
    end
end

function step_filter(filt, fi, z)
    if fi == 1
        filt.timestep(z, []);
    else
        filt.timestep(z);
    end
end

function [xc, Pc] = get_estimate(filt, fi, N_t)
    if fi == 1
        [X, Pc] = filt.getGaussianEstimate();
        xc = num2cell(X, 1);
    else
        [xc, Pc] = filt.getGaussianEstimate();
    end
end

function write_gif(fig, filename, frame_idx, dt)
    frame = getframe(fig);
    [imind, cm] = rgb2ind(frame2im(frame), 256);
    if frame_idx == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', dt);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', dt);
    end
end

function pts = cov_ellipse(center, P, n_sigma)
    theta  = linspace(0, 2*pi, 64);
    circle = [cos(theta); sin(theta)];
    [V, D] = eig(P);
    radii  = n_sigma * sqrt(max(diag(D), 0));
    pts    = V * diag(radii) * circle + center;
end
