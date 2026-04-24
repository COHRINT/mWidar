% TEST_3TARGET  Self-contained synthetic 3-target tracking test
%
% DESCRIPTION:
%   Runs three multi-target filters on a synthetic 3-target scenario with
%   deliberately crossing trajectories.  No mWidar signal or external .mat
%   files are required for the KF_RBPF_multi and PDA_PF_multi tests.
%   HMM_RBPF_multi requires the supplemental lookup tables; if they are not
%   found the test is skipped with a warning rather than erroring out.
%
% SCENARIO:
%   State model  : constant-velocity, [x, y, vx, vy]', dt = 0.1 s
%   Duration     : 60 timesteps (6 s)
%   3 targets with crossing geometry to stress data association:
%     T1: (0,0)   -> vx=0.5, vy= 0.5  (right-upward drift)
%     T2: (0,2)   -> vx=0.4, vy= 0.4  (left-ward, crosses T1)
%     T3: (1.5,3) -> vx=0,   vy=-0.4  (descending, passes near T1/T2 crossing)
%   Measurement noise : sigma = 0.05 m
%   Detection prob    : PD = 0.9
%   Clutter           : 1-3 uniform measurements per step in the scene bounds
%
% FILTERS TESTED:
%   0. JPDA_KF        (KF baseline)
%   1. KF_RBPF_multi  (500 particles, optimal association)
%   2. PDA_PF_multi   (1000 particles)
%   3. HMM_RBPF_multi (50 particles) -- skipped if lookup tables absent
%
% OUTPUTS:
%   test_3target_GT_meas.gif         -- GT + measurements animation
%   test_3target_JPDA_KF.png         -- per-filter trajectory + error plots
%   test_3target_KF_RBPF.png
%   test_3target_PDA_PF.png
%   test_3target_rmse_comparison.png -- all-filter RMSE overlay
%   test_3target_ess.png             -- PDA-PF ESS over time
%   test_3target_all_filters.gif     -- combined multi-filter animation
%
% UNIFIED VISUAL VOCABULARY (consistent across every figure / GIF):
%   Filter identity  --> color  (blue / orange / green / purple)
%   Target identity  --> marker (T1=circle, T2=square, T3=triangle)
%   Ground truth     --> dark-gray thick solid line
%   Measurements     --> light-gray plus scatter
%
% USAGE:
%   cd matlab_src
%   run tests/test_3target

clear; clc; close all;

%% -----------------------------------------------------------------------
%% 0. Paths
%% -----------------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
repo_root  = fullfile(script_dir, '..');

addpath(fullfile(repo_root, 'DA_Track'));
addpath(fullfile(repo_root, 'DA_Track', 'single'));
addpath(fullfile(repo_root, 'DA_Track', 'multi'));
addpath(fullfile(repo_root, 'supplemental'));

fprintf('=== 3-Target Synthetic Tracking Test ===\n\n');

%% -----------------------------------------------------------------------
%% 1. Simulation parameters
%% -----------------------------------------------------------------------
dt       = 0.1;
T        = 60;
N_t      = 3;
sigma_z  = 0.05;
PD_true  = 0.9;
rng(42);

x_bounds   = [-0.5, 5.0];
y_bounds   = [-0.5, 5.0];
scene_area = diff(x_bounds) * diff(y_bounds);

% 6-state constant-acceleration truth model
F_true = [1, 0, dt, 0, 0.5*dt^2, 0;
          0, 1, 0, dt, 0, 0.5*dt^2;
          0, 0, 1,  0, dt, 0;
          0, 0, 0,  1,  0, dt;
          0, 0, 0,  0,  1,  0;
          0, 0, 0,  0,  0,  1];

% 4-state constant-velocity filter model
F_model = [1, 0, dt,  0;
           0, 1,  0, dt;
           0, 0,  1,  0;
           0, 0,  0,  1];

H = [1, 0, 0, 0;
     0, 1, 0, 0];

H_true = [1, 0, 0, 0, 0, 0;
          0, 1, 0, 0, 0, 0];

x0_true = {[0.0; 0.0; 0.5; 0.5; 0.0;  0.0], ...  % T1: right-upward  (blue)
           [0.0; 2.0; 0.4; 0.4; 0.0; -0.1], ...  % T2: drifting       (red)
           [1.5; 3.0; 0.0;-0.4; 0.0;  0.0]};      % T3: descending     (yellow)

Q_trajectory = diag([1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6]);

%% -----------------------------------------------------------------------
%% 2. Ground truth trajectories
%% -----------------------------------------------------------------------
GT = cell(1, N_t);
for t = 1:N_t
    GT{t} = zeros(6, T);
    GT{t}(:, 1) = x0_true{t};
    for k = 2:T
        GT{t}(:, k) = F_true * GT{t}(:, k-1) + sqrtm(Q_trajectory)*randn(6,1);
    end
end
fprintf('Ground truth generated (%d targets, %d steps).\n', N_t, T);

%% -----------------------------------------------------------------------
%% 3. Noisy measurements
%% -----------------------------------------------------------------------
measurements = cell(1, T);
gt_assoc     = cell(1, T);

for k = 1:T
    z_list = zeros(2, 0);
    assoc  = zeros(1, 0);

    for t = 1:N_t
        if rand() < PD_true
            z_list = [z_list, H_true*GT{t}(:,k) + sigma_z*randn(2,1)]; %#ok<AGROW>
            assoc  = [assoc, t]; %#ok<AGROW>
        end
    end

    n_clutter = randi([1, 3]);
    for c = 1:n_clutter
        z_list = [z_list, [x_bounds(1)+diff(x_bounds)*rand(); ...
                            y_bounds(1)+diff(y_bounds)*rand()]]; %#ok<AGROW>
        assoc  = [assoc, 0]; %#ok<AGROW>
    end

    perm = randperm(size(z_list, 2));
    measurements{k} = z_list(:, perm);
    gt_assoc{k}     = assoc(perm);
end

n_meas_avg = mean(cellfun(@(z) size(z,2), measurements));
fprintf('Measurements generated. Average per step: %.1f\n\n', n_meas_avg);

%% -----------------------------------------------------------------------
%% 4. Filter configuration — shared matrices
%% -----------------------------------------------------------------------
Q  = diag([1e-4, 1e-4, 1e-3, 1e-3]);
R  = sigma_z^2 * eye(2);
P0 = diag([0.1, 0.1, 0.25, 0.25]);

x0_cell = cell(1, N_t);
for t = 1:N_t
    gt_state   = GT{t}(1:4, 1);
    x0_cell{t} = gt_state + [0.15*randn(2,1); 0.10*randn(2,1)];
end

%% -----------------------------------------------------------------------
%% 5. Run JPDA_KF  (baseline)
%% -----------------------------------------------------------------------
fprintf('--- Filter 0: JPDA_KF ---\n');

x0_jpda  = cell2mat(x0_cell);
P0_jpda  = repmat({P0}, 1, N_t);
filt_jpda = JPDA_KF(x0_jpda, P0_jpda, F_model, Q, R, H, N_t, []);
filt_jpda.PD                   = PD_true;
filt_jpda.lambda_clutter       = 2.0;
filt_jpda.measurement_space_area = scene_area;

jpda_to_cell = @(X) num2cell(X, 1);
est_jpda  = cell(1, T);
Pest_jpda = cell(1, T);
[X_est, P_est]  = filt_jpda.getGaussianEstimate();
est_jpda{1}     = jpda_to_cell(X_est);
Pest_jpda{1}    = P_est;

for k = 2:T
    filt_jpda.timestep(measurements{k}, []);
    [X_est, P_est] = filt_jpda.getGaussianEstimate();
    est_jpda{k}    = jpda_to_cell(X_est);
    Pest_jpda{k}   = P_est;
end

rmse_jpda = computeRMSE(est_jpda, GT, N_t, T);
fprintf('  RMSE: T1=%.3f  T2=%.3f  T3=%.3f\n\n', ...
    rmse_jpda(1,end), rmse_jpda(2,end), rmse_jpda(3,end));

%% -----------------------------------------------------------------------
%% 6. Run KF_RBPF_multi
%% -----------------------------------------------------------------------
fprintf('--- Filter 1: KF_RBPF_multi (200 particles) ---\n');

filt1 = KF_RBPF_multi(x0_cell, 200, F_model, Q, H, R, ...
    'PD', 0.9, 'PFA', 0.05, 'ESSThreshold', 0.5, ...
    'AssociationStrategy', 'optimal', 'Debug', false, ...
    'InitSigmaPos', 0.03, 'InitSigmaVel', 0.02);

est1  = cell(1, T);
Pest1 = cell(1, T);
[x_est, P_est] = filt1.getGaussianEstimate();
est1{1}  = x_est;
Pest1{1} = P_est;

for k = 2:T
    filt1.timestep(measurements{k});
    [x_est, P_est] = filt1.getGaussianEstimate();
    est1{k}  = x_est;
    Pest1{k} = P_est;
end

rmse1 = computeRMSE(est1, GT, N_t, T);
fprintf('  RMSE: T1=%.3f  T2=%.3f  T3=%.3f\n\n', ...
    rmse1(1,end), rmse1(2,end), rmse1(3,end));

%% -----------------------------------------------------------------------
%% 7. Run PDA_PF_multi
%% -----------------------------------------------------------------------
fprintf('--- Filter 2: PDA_PF_multi (1000 particles) ---\n');

filt2 = PDA_PF_multi(x0_cell, 1000, F_model, Q, H, R, ...
    'PD', 0.9, 'PFA', 0.05, 'LambdaClutter', 3.0/scene_area, ...
    'ValidationSigma', 5, 'ESSThreshold', 0.5, 'Debug', false, ...
    'InitSigmaPos', 0.03, 'InitSigmaVel', 0.05);

est2  = cell(1, T);
Pest2 = cell(1, T);
[x_est, P_est] = filt2.getGaussianEstimate();
est2{1}  = x_est;
Pest2{1} = P_est;

for k = 2:T
    filt2.timestep(measurements{k});
    [x_est, P_est] = filt2.getGaussianEstimate();
    est2{k}  = x_est;
    Pest2{k} = P_est;
end

rmse2 = computeRMSE(est2, GT, N_t, T);
fprintf('  RMSE: T1=%.3f  T2=%.3f  T3=%.3f\n\n', ...
    rmse2(1,end), rmse2(2,end), rmse2(3,end));

%% -----------------------------------------------------------------------
%% 8. Run HMM_RBPF_multi (optional — requires lookup tables)
%% -----------------------------------------------------------------------
run_hmm = false;

% hmm_like_file  = fullfile(repo_root, 'supplemental', 'precalc_imagegridHMMEmLike.mat');
% hmm_trans_file = fullfile(repo_root, 'supplemental', 'precalc_imagegridHMMSTMn15.mat');
%
% if isfile(hmm_like_file) && isfile(hmm_trans_file)
%     fprintf('--- Filter 3: HMM_RBPF_multi (50 particles) ---\n');
%     load(hmm_like_file,  'pointlikelihood_image');
%     load(hmm_trans_file, 'A');
%     A_transition  = A; clear A;
%     x0_cell_2d = cellfun(@(x) x(1:2), x0_cell, 'UniformOutput', false);
%
%     filt3 = HMM_RBPF_multi(x0_cell_2d, 50, A_transition, pointlikelihood_image, ...
%         'PD', 0.9, 'PFA', 0.05, 'ESSThreshold', 0.5, ...
%         'AssociationStrategy', 'optimal', 'Debug', false);
%
%     est3  = cell(1, T);
%     Pest3 = cell(1, T);
%     [x_est, P_est] = filt3.getGaussianEstimate();
%     est3{1}  = x_est;
%     Pest3{1} = P_est;
%     for k = 2:T
%         filt3.timestep(measurements{k});
%         [x_est, P_est] = filt3.getGaussianEstimate();
%         est3{k} = x_est; Pest3{k} = P_est;
%     end
%
%     GT_2d = cellfun(@(g) g(1:2,:), GT, 'UniformOutput', false);
%     rmse3 = zeros(N_t, T);
%     for k = 1:T
%         for t = 1:N_t
%             err = est3{k}{t}(1:2) - GT_2d{t}(:,k);
%             rmse3(t,k) = norm(err);
%         end
%     end
%     fprintf('  RMSE: T1=%.3f  T2=%.3f  T3=%.3f\n\n', ...
%         rmse3(1,end), rmse3(2,end), rmse3(3,end));
%     run_hmm = true;
% else
%     fprintf('--- Filter 3: HMM_RBPF_multi SKIPPED ---\n\n');
% end

%% -----------------------------------------------------------------------
%% 9. Summary table
%% -----------------------------------------------------------------------
filter_names       = {'JPDA-KF (baseline)', 'KF-RBPF-multi', 'PDA-PF-multi'};
filter_names_short = {'JPDA-KF',            'KF-RBPF',       'PDA-PF'};
filter_file_tags   = {'JPDA_KF',            'KF_RBPF',       'PDA_PF'};
est_all            = {est_jpda,  est1,  est2};
Pest_all           = {Pest_jpda, Pest1, Pest2};
rmse_all           = {rmse_jpda, rmse1, rmse2};

if run_hmm
    filter_names{end+1}       = 'HMM-RBPF-multi';
    filter_names_short{end+1} = 'HMM-RBPF';
    filter_file_tags{end+1}   = 'HMM_RBPF';
    est_all{end+1}            = est3;
    Pest_all{end+1}           = Pest3;
    rmse_all{end+1}           = rmse3;
end

n_f    = length(filter_names);
t_axis = (0:T-1) * dt;
all_z  = cell2mat(measurements);

fprintf('=== Mean RMSE (all timesteps, position only) ===\n');
fprintf('%-22s  %6s  %6s  %6s\n', 'Filter', 'T1', 'T2', 'T3');
fprintf('%s\n', repmat('-', 1, 44));
for fi = 1:n_f
    mr = mean(rmse_all{fi}, 2);
    fprintf('%-22s  %6.3f  %6.3f  %6.3f\n', filter_names{fi}, mr(1), mr(2), mr(3));
end
fprintf('\n');

%% -----------------------------------------------------------------------
%% 10. Unified visual design constants
%% -----------------------------------------------------------------------
% Filter identity --> color  (blue / orange / green / purple)
filter_colors = [0.12, 0.47, 0.71;   % JPDA-KF       blue
                 0.89, 0.47, 0.10;   % KF-RBPF-multi orange
                 0.17, 0.63, 0.17];  % PDA-PF-multi   green
if run_hmm
    filter_colors(end+1,:) = [0.58, 0.40, 0.74];  % HMM-RBPF purple
end

% Target identity --> marker  (circle / square / triangle)
% These marker shapes are used in EVERY figure and GIF so T2 is always a
% square whether you're looking at a static PNG or the combined animation.
tgt_markers  = {'o', 's', '^'};
tgt_msize_sm = 5;    % on static trajectory lines
tgt_msize_lg = 9;    % on current-frame dots in animations

% Marker spacing on static plots (avoid overplotting 60 markers)
mark_idx = round(linspace(1, T, 10));

% Ground truth: dark gray, not assigned to any filter
gt_color = [0.15, 0.15, 0.15];
gt_lw    = 2.5;
est_lw   = 1.5;
meas_color = [0.72, 0.72, 0.72];

% Target colors for per-filter static figures (color encodes target there,
% since only one filter is shown per figure)
tgt_colors = lines(N_t);

%% -----------------------------------------------------------------------
%% 11. GT + measurements animation
%% -----------------------------------------------------------------------
fprintf('Generating GT+measurements animation...\n');
gif_gt  = 'test_3target_GT_meas.gif';
fig_gt  = figure('Name','GT + Measurements','Position',[50 50 620 540]);

for k = 1:T
    clf; hold on; grid on; axis equal;
    title(sprintf('Ground Truth + Measurements  |  t = %.1f s', (k-1)*dt), ...
          'FontSize', 11);
    xlabel('x (m)'); ylabel('y (m)');
    xlim(x_bounds + [-0.3, 0.3]); ylim(y_bounds + [-0.3, 0.3]);

    % Current measurements
    zk = measurements{k};
    scatter(zk(1,:), zk(2,:), 45, meas_color, '+', 'LineWidth', 1.4, ...
            'DisplayName', 'Measurements');

    % GT trails + current-frame marker (target marker encodes target identity)
    for t = 1:N_t
        plot(GT{t}(1,1:k), GT{t}(2,1:k), '-', 'Color', tgt_colors(t,:), ...
             'LineWidth', gt_lw, 'HandleVisibility','off');
        plot(GT{t}(1,k), GT{t}(2,k), tgt_markers{t}, ...
             'Color', tgt_colors(t,:), 'MarkerSize', tgt_msize_lg, ...
             'MarkerFaceColor', tgt_colors(t,:), ...
             'DisplayName', sprintf('GT T%d', t));
    end

    legend('Location','northeast','FontSize',9,'Box','on');
    drawnow;
    writeGIF(fig_gt, gif_gt, k, dt);
end
close(fig_gt);
fprintf('  Saved: %s\n', gif_gt);

%% -----------------------------------------------------------------------
%% 12. Per-filter static PNG figures
%%     Color = target (best encoding when only one filter is shown)
%%     Marker = target (shared vocabulary with all other figures)
%% -----------------------------------------------------------------------
fprintf('Generating per-filter PNG figures...\n');

for fi = 1:n_f
    fig = figure('Name', filter_names{fi}, ...
                 'Position', [50+fi*25, 50+fi*25, 1200, 900]);
    est  = est_all{fi};
    Pest = Pest_all{fi};

    %% Row 1 — 2D trajectory (one subplot per target)
    for t = 1:N_t
        subplot(3, N_t, t); hold on; grid on; axis equal;
        title(sprintf('Target %d — Trajectory', t), 'FontSize', 10);
        xlabel('x (m)'); ylabel('y (m)');

        % All measurements faint background
        scatter(all_z(1,:), all_z(2,:), 6, meas_color, '+', ...
                'HandleVisibility','off');

        % GT: thick solid + start marker
        plot(GT{t}(1,:), GT{t}(2,:), '-', 'Color', tgt_colors(t,:), ...
             'LineWidth', gt_lw, 'DisplayName','GT');
        plot(GT{t}(1,1), GT{t}(2,1), tgt_markers{t}, ...
             'Color', tgt_colors(t,:), 'MarkerSize', tgt_msize_lg, ...
             'MarkerFaceColor', tgt_colors(t,:), 'HandleVisibility','off');

        % Estimate: dashed, same target color (darker shade), with target markers
        ex = cellfun(@(e) e{t}(1), est);
        ey = cellfun(@(e) e{t}(2), est);
        ec = tgt_colors(t,:) * 0.6 + 0.1;   % slightly darker
        plot(ex, ey, '--', 'Color', ec, 'LineWidth', est_lw, ...
             'HandleVisibility','off');
        plot(ex(mark_idx), ey(mark_idx), tgt_markers{t}, ...
             'Color', ec, 'MarkerSize', tgt_msize_sm, ...
             'MarkerFaceColor', ec, 'LineStyle','none', ...
             'DisplayName','Estimate');

        legend('Location','best','FontSize',9,'Box','on');
        xlim(x_bounds+[-0.3 0.3]); ylim(y_bounds+[-0.3 0.3]);
    end

    %% Row 2 — x-error ±2σ
    for t = 1:N_t
        subplot(3, N_t, N_t+t); hold on; grid on;
        title(sprintf('T%d  —  x-error  ±2σ_x', t), 'FontSize', 10);
        xlabel('Time (s)'); ylabel('x error (m)');

        err_x  = cellfun(@(e) e{t}(1), est) - GT{t}(1,:);
        sig2_x = 2 * sqrt(cellfun(@(P) P{t}(1,1), Pest));

        fill([t_axis, fliplr(t_axis)], [sig2_x, fliplr(-sig2_x)], ...
             tgt_colors(t,:), 'FaceAlpha', 0.20, 'EdgeColor','none', ...
             'DisplayName','±2σ');
        plot(t_axis, err_x, '-', 'Color', tgt_colors(t,:), ...
             'LineWidth', 1.5, 'DisplayName','error');
        yline(0, '-k', 'LineWidth', 0.5, 'HandleVisibility','off');
        legend('Location','best','FontSize',9);
    end

    %% Row 3 — y-error ±2σ
    for t = 1:N_t
        subplot(3, N_t, 2*N_t+t); hold on; grid on;
        title(sprintf('T%d  —  y-error  ±2σ_y', t), 'FontSize', 10);
        xlabel('Time (s)'); ylabel('y error (m)');

        err_y  = cellfun(@(e) e{t}(2), est) - GT{t}(2,:);
        sig2_y = 2 * sqrt(cellfun(@(P) P{t}(2,2), Pest));

        fill([t_axis, fliplr(t_axis)], [sig2_y, fliplr(-sig2_y)], ...
             tgt_colors(t,:), 'FaceAlpha', 0.20, 'EdgeColor','none', ...
             'DisplayName','±2σ');
        plot(t_axis, err_y, '-', 'Color', tgt_colors(t,:), ...
             'LineWidth', 1.5, 'DisplayName','error');
        yline(0, '-k', 'LineWidth', 0.5, 'HandleVisibility','off');
        legend('Location','best','FontSize',9);
    end

    sgtitle(filter_names{fi}, 'FontSize', 13, 'FontWeight','bold');
    fname = sprintf('test_3target_%s.png', filter_file_tags{fi});
    exportgraphics(fig, fname, 'Resolution', 150);
    fprintf('  Saved: %s\n', fname);
end

%% -----------------------------------------------------------------------
%% 13. RMSE comparison — all filters on the same axes, one panel per target
%%     Color = filter (shared with combined animation)
%% -----------------------------------------------------------------------
fprintf('Generating RMSE comparison figure...\n');
fig_rmse = figure('Name','RMSE Comparison','Position',[100 100 1100 380]);

for t = 1:N_t
    subplot(1, N_t, t); hold on; grid on;
    title(sprintf('Target %d — Position RMSE', t), 'FontSize', 10);
    xlabel('Time (s)'); ylabel('RMSE (m)');

    for fi = 1:n_f
        plot(t_axis, rmse_all{fi}(t,:), '-', ...
             'Color', filter_colors(fi,:), 'LineWidth', 1.8, ...
             'DisplayName', filter_names_short{fi});
    end
    legend('Location','best','FontSize',9,'Box','on');
    ylim([0, inf]);
end

sgtitle('Position RMSE — All Filters', 'FontSize', 13, 'FontWeight','bold');
exportgraphics(fig_rmse, 'test_3target_rmse_comparison.png', 'Resolution', 150);
fprintf('  Saved: test_3target_rmse_comparison.png\n');

%% -----------------------------------------------------------------------
%% 14. ESS over time — PDA_PF_multi
%%     Shows particle degeneracy and resampling behavior.
%%     ESS = 1/sum(w_i^2); drops between resamples, jumps after.
%% -----------------------------------------------------------------------
if isfield(filt2, 'history') && ~isempty(filt2.history) && ...
   isfield(filt2.history, 'ESS')

    fprintf('Generating ESS figure...\n');
    ess_vals = [filt2.history.ESS];           % [1 x (T-1)]
    ess_axis = t_axis(2:1+length(ess_vals));  % ESS starts at k=2

    fig_ess = figure('Name','ESS','Position',[100 520 820 340]);
    hold on; grid on;
    title('Effective Sample Size — PDA-PF-multi', 'FontSize', 11);
    xlabel('Time (s)'); ylabel('ESS');

    % Shade resampling events (ESS jumped up to N_p = uniform reset)
    N_p_pda = filt2.N_p;
    resample_thresh = filt2.ESS_threshold_percentage * N_p_pda;

    plot(ess_axis, ess_vals, '-', 'Color', filter_colors(3,:), ...
         'LineWidth', 2.0, 'DisplayName', 'ESS');
    yline(N_p_pda, '--k', sprintf('N_p = %d', N_p_pda), ...
          'LineWidth', 0.9, 'LabelHorizontalAlignment','left', ...
          'HandleVisibility','off');
    yline(resample_thresh, ':k', sprintf('Resample threshold (%.0f%%)', ...
          filt2.ESS_threshold_percentage*100), ...
          'LineWidth', 0.9, 'LabelHorizontalAlignment','left', ...
          'HandleVisibility','off');

    legend('Location','best','FontSize',10,'Box','on');
    ylim([0, N_p_pda * 1.12]);
    xlim([ess_axis(1), ess_axis(end)]);

    exportgraphics(fig_ess, 'test_3target_ess.png', 'Resolution', 150);
    fprintf('  Saved: test_3target_ess.png\n');
end

%% -----------------------------------------------------------------------
%% 15. Combined multi-filter animation GIF
%%
%%     COLOR  = filter identity  (consistent with RMSE comparison figure)
%%     MARKER = target identity  (consistent with all static PNG figures)
%%     GT     = dark-gray thick solid + target-marker dot
%%     Meas   = light-gray plus scatter (current timestep only)
%%     Ellipse= 2-sigma position covariance patch per filter per target
%%
%%     Legend blocks:
%%       Filters: one colored circle-line entry per filter
%%       Targets: one gray marker entry per target (T1=o, T2=s, T3=^)
%%       + GT line entry + Meas entry
%% -----------------------------------------------------------------------
fprintf('Generating combined animation (this may take a moment)...\n');
gif_all = 'test_3target_all_filters.gif';
fig_all = figure('Name','All Filters','Position',[120 80 720 620]);

for k = 1:T
    clf; hold on; grid on; axis equal;
    title(sprintf('All Filters  |  t = %.1f s', (k-1)*dt), 'FontSize', 11);
    xlabel('x (m)'); ylabel('y (m)');
    xlim(x_bounds + [-0.35, 0.35]); ylim(y_bounds + [-0.35, 0.35]);

    % ------ Legend dummy entries (drawn first so legend is ordered) ------
    % Filters (colored circle-line)
    for fi = 1:n_f
        plot(NaN, NaN, '-o', 'Color', filter_colors(fi,:), ...
             'MarkerFaceColor', filter_colors(fi,:), ...
             'LineWidth', est_lw, 'MarkerSize', tgt_msize_sm+1, ...
             'DisplayName', filter_names_short{fi});
    end
    % Ground truth
    plot(NaN, NaN, '-', 'Color', gt_color, 'LineWidth', gt_lw, ...
         'DisplayName','GT');
    % Target shapes (neutral gray)
    for t = 1:N_t
        plot(NaN, NaN, tgt_markers{t}, 'Color', [0.45 0.45 0.45], ...
             'MarkerFaceColor', [0.45 0.45 0.45], 'MarkerSize', tgt_msize_sm+1, ...
             'LineStyle','none', 'DisplayName', sprintf('T%d', t));
    end
    % Measurements
    scatter(NaN, NaN, 45, meas_color, '+', 'LineWidth', 1.4, ...
            'DisplayName','Meas');

    % ------ Current measurements ------
    zk = measurements{k};
    scatter(zk(1,:), zk(2,:), 45, meas_color, '+', 'LineWidth', 1.4, ...
            'HandleVisibility','off');

    % ------ GT trails + current marker ------
    for t = 1:N_t
        plot(GT{t}(1,1:k), GT{t}(2,1:k), '-', 'Color', gt_color, ...
             'LineWidth', gt_lw, 'HandleVisibility','off');
        plot(GT{t}(1,k), GT{t}(2,k), tgt_markers{t}, ...
             'Color', gt_color, 'MarkerSize', tgt_msize_lg, ...
             'MarkerFaceColor', gt_color, 'HandleVisibility','off');
    end

    % ------ Filter estimate trails + 2-sigma ellipses + current markers ------
    for fi = 1:n_f
        fc   = filter_colors(fi,:);
        est  = est_all{fi};
        Pest = Pest_all{fi};

        for t = 1:N_t
            ex = cellfun(@(e) e{t}(1), est);
            ey = cellfun(@(e) e{t}(2), est);

            % Estimate trail (semi-transparent)
            plot(ex(1:k), ey(1:k), '-', 'Color', [fc, 0.55], ...
                 'LineWidth', est_lw, 'HandleVisibility','off');

            % 2-sigma covariance ellipse at current step
            P_pos = Pest{k}{t}(1:2, 1:2);
            exy   = covEllipse([ex(k); ey(k)], P_pos, 2);
            patch(exy(1,:), exy(2,:), fc, ...
                  'FaceAlpha', 0.12, 'EdgeColor', fc, 'LineWidth', 0.9, ...
                  'HandleVisibility','off');

            % Current-frame position dot (filter color + target marker)
            plot(ex(k), ey(k), tgt_markers{t}, ...
                 'Color', fc, 'MarkerSize', tgt_msize_lg, ...
                 'MarkerFaceColor', fc, 'LineWidth', 1.2, ...
                 'HandleVisibility','off');
        end
    end

    legend('Location','northeast','FontSize', 8, 'Box','on', 'NumColumns', 2);
    drawnow;
    writeGIF(fig_all, gif_all, k, dt);
end
close(fig_all);
fprintf('  Saved: %s\n', gif_all);

fprintf('\nAll outputs complete.\n');

%% -----------------------------------------------------------------------
%% Local helper functions
%% -----------------------------------------------------------------------

function rmse = computeRMSE(est_hist, GT, N_t, T)
    % COMPUTERMSE  Per-target position RMSE from a cell estimate history.
    rmse = zeros(N_t, T);
    for k = 1:T
        for t = 1:N_t
            err = est_hist{k}{t}(1:2) - GT{t}(1:2, k);
            rmse(t, k) = norm(err);
        end
    end
end

function writeGIF(fig, filename, frame_idx, dt)
    % WRITEGIF  Append one frame to an animated GIF file.
    frame = getframe(fig);
    [imind, cm] = rgb2ind(frame2im(frame), 256);
    if frame_idx == 1
        imwrite(imind, cm, filename, 'gif', 'Loopcount', inf, 'DelayTime', dt);
    else
        imwrite(imind, cm, filename, 'gif', 'WriteMode', 'append', 'DelayTime', dt);
    end
end

function pts = covEllipse(center, P, n_sigma)
    % COVELLIPSE  Points on a 2-sigma covariance ellipse.
    %   center  - [2x1] ellipse center
    %   P       - [2x2] position covariance sub-block
    %   n_sigma - number of standard deviations
    theta  = linspace(0, 2*pi, 64);
    circle = [cos(theta); sin(theta)];
    [V, D] = eig(P);
    radii  = n_sigma * sqrt(max(diag(D), 0));   % clamp negatives from float noise
    pts    = V * diag(radii) * circle + center;
end
