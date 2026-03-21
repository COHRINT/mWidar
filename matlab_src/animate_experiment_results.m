function animate_experiment_results(result_file, varargin)
% ANIMATE_EXPERIMENT_RESULTS  Create 3 GIF animations from a run_experiment result.
%
% USAGE:
%   animate_experiment_results                          % interactive file picker
%   animate_experiment_results('data/TUNING_DATASET1/HMM_RBPF_N150.mat')
%   animate_experiment_results(..., 'delay',      0.15)
%   animate_experiment_results(..., 'output_dir', 'figures/animations')
%   animate_experiment_results(..., 'gif_prefix', 'run1')
%
% OUTPUT — 3 GIF files (written to output_dir, or same dir as result_file):
%   <prefix>_scene.gif  — GT trail, estimate, measurements, winner position
%   <prefix>_state.gif  — filter-type-specific internal state
%                           KF          : covariance ellipse
%                           PF (PDA/MC/GNN) : particle scatter coloured by weight
%                           KF_RBPF     : all particles (faded) + winner ellipse
%                           HMM_RBPF    : weighted mixture grid (imagesc)
%                           HMM (PDA/GNN): per-step HMM grid
%   <prefix>_error.gif  — position error + ESS building up over time
%   <prefix>_statetraj.gif — per-particle state trajectory trees (RBPF only)
%                            shows ancestral paths and resampling collapse/diverge
%
% REQUIREMENTS:
%   Result must have been saved by the current version of run_experiment.m
%   (which stores particle_pos_all, particle_w_all, particle_cov_all,
%    hmm_mixture, hmm_standalone_grid, winner_idx).

%% ---- Parse arguments -------------------------------------------------------
p = inputParser;
addOptional(p,  'result_file_arg', '',   @ischar);
addParameter(p, 'delay',      0.18,      @(x) isnumeric(x) && x > 0);
addParameter(p, 'output_dir', '',        @ischar);
addParameter(p, 'gif_prefix', '',        @ischar);
parse(p, varargin{:});
opt = p.Results;

if nargin >= 1 && ~isempty(result_file)
    mat_path = result_file;
else
    [f, d] = uigetfile('*.mat', 'Select run_experiment result');
    if isequal(f, 0), fprintf('No file selected.\n'); return; end
    mat_path = fullfile(d, f);
end

%% ---- Load ------------------------------------------------------------------
tmp = load(mat_path, 'R');
R   = tmp.R;

GT          = R.GT;
x_est       = R.x_est;
z_all       = R.measurements;
n_k         = size(GT, 2);
dt          = R.params.dt;
tvec        = (0:n_k-1) * dt;
filter_name = R.filter_name;
has_map     = isfield(R,'x_map') && ~all(isnan(R.x_map(:)));

% Particle data (may be absent in old result files)
pp_all   = get_field(R, 'particle_pos_all',  []);  % [2 x N_p x n_k]
pw_all   = get_field(R, 'particle_w_all',    []);  % [N_p x n_k]
pc_all   = get_field(R, 'particle_cov_all',  []);  % [4 x N_p x n_k]  KF_RBPF
hmm_mix  = get_field(R, 'hmm_mixture',       []);  % [128x128xn_k]    HMM_RBPF
hmm_sa   = get_field(R, 'hmm_standalone_grid',[]);% [npx x npx x n_k] PDA/GNN HMM
widx     = get_field(R, 'winner_idx',        []);  % scalar
pfinal_traj = get_field(R, 'particle_final_traj', {});  % {1 x N_p} each [2 x n_k]

%% ---- Output paths ----------------------------------------------------------
[res_dir, res_name] = fileparts(mat_path);
if isempty(opt.output_dir)
    out_dir = res_dir;
else
    out_dir = opt.output_dir;
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
end
prefix = opt.gif_prefix;
if isempty(prefix), prefix = res_name; end

gif_scene    = fullfile(out_dir, [prefix '_scene.gif']);
gif_state    = fullfile(out_dir, [prefix '_state.gif']);
gif_error    = fullfile(out_dir, [prefix '_error.gif']);
gif_statetraj = fullfile(out_dir, [prefix '_statetraj.gif']);

%% ---- Classify filter -------------------------------------------------------
is_hmm_rbpf      = strcmp(filter_name, 'HMM_RBPF');
is_kf_rbpf       = strcmp(filter_name, 'KF_RBPF');
is_pf_matrix     = ismember(filter_name, {'PDA_PF','GNN_PF','MC_PF'}) && ~isempty(pp_all);
is_kf_plain      = ismember(filter_name, {'PDA_KF','GNN_KF'});
is_hmm_plain     = ismember(filter_name, {'PDA_HMM','GNN_HMM'});
has_particles    = ~isempty(pp_all);

% Colour scheme (consistent across 3 GIFs)
C_gt     = [0.10 0.65 0.10];   % green
C_est    = [0.15 0.45 0.85];   % blue
C_map    = [0.80 0.20 0.80];   % magenta
C_meas   = [0.85 0.40 0.00];   % orange
C_winner = [1.00 0.85 0.00];   % gold

fprintf('Animating %s  (%d steps)  →  %s\n', filter_name, n_k, out_dir);

%% ============================================================================
%% GIF 1 — SCENE
%% ============================================================================
fig1 = figure('Color','w','Position',[50 50 560 520],'Visible','off');
ax1  = axes(fig1);
hold(ax1,'on'); grid(ax1,'on'); axis(ax1,'equal');
xlim(ax1,[-2.3 2.3]); ylim(ax1,[-0.3 4.3]);
xlabel(ax1,'X (m)'); ylabel(ax1,'Y (m)');

for k = 1:n_k
    cla(ax1); hold(ax1,'on'); grid(ax1,'on');
    xlim(ax1,[-2.3 2.3]); ylim(ax1,[-0.3 4.3]);

    % Full GT in grey
    plot(ax1, GT(1,:), GT(2,:), '-', 'Color',[0.8 0.8 0.8], 'LineWidth',1);
    % GT trail up to k
    plot(ax1, GT(1,1:k), GT(2,1:k), '-', 'Color',C_gt, 'LineWidth',2);
    plot(ax1, GT(1,1),   GT(2,1),   's', 'Color',C_gt, 'MarkerSize',8, 'MarkerFaceColor',C_gt);

    % Estimate trail
    plot(ax1, x_est(1,1:k), x_est(2,1:k), '--', 'Color',C_est, 'LineWidth',1.8);
    plot(ax1, x_est(1,k),   x_est(2,k),   'o',  'Color',C_est, 'MarkerSize',8, 'MarkerFaceColor',C_est);

    % MAP trail
    if has_map
        plot(ax1, R.x_map(1,1:k), R.x_map(2,1:k), ':', 'Color',C_map, 'LineWidth',1.5);
        plot(ax1, R.x_map(1,k),   R.x_map(2,k),   'd', 'Color',C_map, 'MarkerSize',6, 'MarkerFaceColor',C_map);
    end

    % Measurements at step k
    z_k = z_all{k};
    if ~isempty(z_k)
        plot(ax1, z_k(1,:), z_k(2,:), 'x', 'Color',C_meas, 'MarkerSize',7, 'LineWidth',1.5);
    end

    % Winner particle position
    if has_particles && ~isempty(widx)
        wx = pp_all(1, widx, k);
        wy = pp_all(2, widx, k);
        plot(ax1, wx, wy, 'p', 'Color',C_winner, 'MarkerSize',12, 'LineWidth',1.5, 'MarkerFaceColor',C_winner);
    end

    title(ax1, sprintf('%s  |  k=%d  t=%.2fs', strrep(filter_name,'_','\_'), k, (k-1)*dt), ...
          'FontSize',10);

    % Legend only on first frame
    if k == 1
        legend_entries = {'GT','GT trail','Filter MMSE','Current MMSE'};
        if has_map, legend_entries{end+1} = 'MAP'; end
        legend_entries{end+1} = 'Measurements';
        if has_particles && ~isempty(widx), legend_entries{end+1} = 'Winner'; end
    end

    drawnow;
    write_gif_frame(fig1, gif_scene, k, opt.delay);
end
close(fig1);
fprintf('  scene  → %s\n', gif_scene);

%% ============================================================================
%% GIF 2 — FILTER STATE
%% ============================================================================
fig2 = figure('Color','w','Position',[620 50 560 520],'Visible','off');
ax2  = axes(fig2);

% Grid coordinates for HMM plots
xgrid = linspace(-2, 2, 128);
ygrid = linspace( 0, 4, 128);

% Colormap: white(0) → orange → red → dark red(1)
% High probability shows as saturated red/orange, zero probability = white (bg)
cm_thermal = thermal_white(256);

% Particle scatter: max weight for normalisation
if is_pf_matrix && ~isempty(pw_all)
    max_w = max(pw_all(:));
    if max_w == 0, max_w = 1; end
end

for k = 1:n_k
    cla(ax2); hold(ax2,'on');

    if is_hmm_rbpf && ~isempty(hmm_mix)
        %------------------------------------------------------------------
        % HMM_RBPF: weighted mixture grid
        % Normalise to [0,1] then apply sqrt for local contrast in the peak.
        %------------------------------------------------------------------
        g     = hmm_mix(:,:,k);
        g_n   = g / max(g(:) + eps);
        g_disp = sqrt(g_n);          % sqrt stretches low end, reveals peak structure
        colormap(fig2, cm_thermal);
        h_im = imagesc(ax2, xgrid, ygrid, g_disp);
        h_im.Interpolation = 'bilinear';
        set(ax2,'YDir','normal'); clim(ax2,[0 1]);
        axis(ax2,'tight');
        set(ax2,'Color','w','XColor','k','YColor','k');

        plot(ax2, GT(1,k),    GT(2,k),    'g+', 'MarkerSize',10,'LineWidth',2.5);
        plot(ax2, x_est(1,k), x_est(2,k), 'bo', 'MarkerSize',8, 'LineWidth',2);
        if has_map
            plot(ax2, R.x_map(1,k), R.x_map(2,k), 'ms', 'MarkerSize',8,'LineWidth',2);
        end
        z_k = z_all{k};
        if ~isempty(z_k)
            plot(ax2, z_k(1,:), z_k(2,:), 'kx','MarkerSize',7,'LineWidth',1.5);
        end
        title(ax2, sprintf('Mixture grid  k=%d', k), 'FontSize',10);
        xlabel(ax2,'X (m)'); ylabel(ax2,'Y (m)');

    elseif is_hmm_plain && ~isempty(hmm_sa)
        %------------------------------------------------------------------
        % Standalone HMM (PDA_HMM, GNN_HMM): direct grid
        %------------------------------------------------------------------
        npx_sa = size(hmm_sa, 1);
        xg_sa  = linspace(-2, 2, npx_sa);
        yg_sa  = linspace( 0, 4, npx_sa);
        g     = hmm_sa(:,:,k);
        g_n   = g / max(g(:) + eps);
        g_disp = sqrt(g_n);
        colormap(fig2, cm_thermal);
        h_im = imagesc(ax2, xg_sa, yg_sa, g_disp);
        h_im.Interpolation = 'bilinear';
        set(ax2,'YDir','normal'); clim(ax2,[0 1]);
        axis(ax2,'tight');
        set(ax2,'Color','w','XColor','k','YColor','k');

        plot(ax2, GT(1,k),    GT(2,k),    'g+','MarkerSize',10,'LineWidth',2.5);
        plot(ax2, x_est(1,k), x_est(2,k), 'bo','MarkerSize',8, 'LineWidth',2);
        z_k = z_all{k};
        if ~isempty(z_k)
            plot(ax2, z_k(1,:), z_k(2,:), 'kx','MarkerSize',7,'LineWidth',1.5);
        end
        title(ax2, sprintf('HMM grid  k=%d', k), 'FontSize',10);
        xlabel(ax2,'X (m)'); ylabel(ax2,'Y (m)');

    elseif is_kf_rbpf && has_particles
        %------------------------------------------------------------------
        % KF_RBPF: all particles as light dots + winner ellipse
        %------------------------------------------------------------------
        set(ax2,'Color','w','XColor','k','YColor','k');
        xlim(ax2,[-2.3 2.3]); ylim(ax2,[-0.3 4.3]);
        grid(ax2,'on'); ax2.GridColor = [0.85 0.85 0.85];

        % Particles: light blue dots, weight → opacity not possible in scatter,
        % so map weight to marker darkness instead
        w_k = pw_all(:,k);
        w_k = w_k / (max(w_k) + eps);
        scatter(ax2, pp_all(1,:,k), pp_all(2,:,k), 18, ...
                repmat([0.65 0.80 1.0], size(w_k,1), 1) .* (1 - 0.5*w_k), ...
                'filled', 'MarkerFaceAlpha', 0.6);

        % Winner ellipse
        if ~isempty(widx) && ~isempty(pc_all)
            P_w = reshape(pc_all(:, widx, k), 2, 2);
            [ex, ey] = cov_ellipse(pp_all(:, widx, k), P_w, 2);
            plot(ax2, ex, ey, '-', 'Color',[0.85 0.60 0.00], 'LineWidth',2.5);
            plot(ax2, pp_all(1,widx,k), pp_all(2,widx,k), 'p', ...
                 'Color',[0.85 0.60 0.00], 'MarkerSize',12, 'MarkerFaceColor',[0.85 0.60 0.00]);
        end

        plot(ax2, GT(1,k),    GT(2,k),    'g+','MarkerSize',12,'LineWidth',2.5);
        plot(ax2, x_est(1,k), x_est(2,k), 'bo','MarkerSize',8,'LineWidth',2,'MarkerFaceColor',C_est);
        title(ax2, sprintf('KF\_RBPF particles  k=%d', k), 'FontSize',10);
        xlabel(ax2,'X (m)'); ylabel(ax2,'Y (m)');

    elseif is_pf_matrix && has_particles
        %------------------------------------------------------------------
        % PF (PDA/GNN/MC): particle scatter, winner highlighted
        %------------------------------------------------------------------
        set(ax2,'Color','w','XColor','k','YColor','k');
        xlim(ax2,[-2.3 2.3]); ylim(ax2,[-0.3 4.3]);
        grid(ax2,'on'); ax2.GridColor = [0.85 0.85 0.85];

        w_k = pw_all(:,k) / (max_w + eps);
        scatter(ax2, pp_all(1,:,k), pp_all(2,:,k), 20, ...
                repmat([0.65 0.80 1.0], size(w_k,1), 1) .* (1 - 0.5*w_k), ...
                'filled', 'MarkerFaceAlpha', 0.65);

        if ~isempty(widx)
            plot(ax2, pp_all(1,widx,k), pp_all(2,widx,k), 'p', ...
                 'Color',[0.85 0.60 0.00], 'MarkerSize',12, 'MarkerFaceColor',[0.85 0.60 0.00]);
        end
        plot(ax2, GT(1,k),    GT(2,k),    'g+','MarkerSize',12,'LineWidth',2.5);
        plot(ax2, x_est(1,k), x_est(2,k), 'bo','MarkerSize',8,'LineWidth',2,'MarkerFaceColor',C_est);
        title(ax2, sprintf('%s particles  k=%d', strrep(filter_name,'_','\_'), k), 'FontSize',10);
        xlabel(ax2,'X (m)'); ylabel(ax2,'Y (m)');

    elseif is_kf_plain
        %------------------------------------------------------------------
        % KF: covariance ellipse on white background
        %------------------------------------------------------------------
        set(ax2,'Color','w','XColor','k','YColor','k');
        xlim(ax2,[-2.3 2.3]); ylim(ax2,[-0.3 4.3]);
        grid(ax2,'on'); axis(ax2,'equal');

        if ~isempty(R.P_est{k})
            P2 = R.P_est{k}(1:2, 1:2);
            [ex, ey] = cov_ellipse(x_est(1:2,k), P2, 2);
            fill(ax2, ex, ey, C_est, 'FaceAlpha',0.20, 'EdgeColor',C_est, 'LineWidth',2);
            [ex3, ey3] = cov_ellipse(x_est(1:2,k), P2, 3);
            plot(ax2, ex3, ey3, '--', 'Color',C_est*0.7, 'LineWidth',1);
        end
        plot(ax2, GT(1,k),    GT(2,k),    '+', 'Color',C_gt,  'MarkerSize',12,'LineWidth',2.5);
        plot(ax2, x_est(1,k), x_est(2,k), 'o', 'Color',C_est, 'MarkerSize',8,'MarkerFaceColor',C_est);
        title(ax2, sprintf('%s covariance  k=%d', strrep(filter_name,'_','\_'), k), 'FontSize',10);
        xlabel(ax2,'X (m)'); ylabel(ax2,'Y (m)');

    else
        %------------------------------------------------------------------
        % Fallback
        %------------------------------------------------------------------
        set(ax2,'Color','w');
        text(0.5, 0.5, sprintf('No state data\nfor %s', strrep(filter_name,'_','\_')), ...
             'Units','normalized','HorizontalAlignment','center','FontSize',12);
    end

    drawnow;
    write_gif_frame(fig2, gif_state, k, opt.delay);
end
close(fig2);
fprintf('  state  → %s\n', gif_state);

%% ============================================================================
%% GIF 3 — ERROR + ESS
%% ============================================================================
has_ess = any(~isnan(R.ESS));
pos_err = vecnorm(x_est(1:2,:) - GT(1:2,:));

if has_ess
    fig3 = figure('Color','w','Position',[50 590 560 420],'Visible','off');
    ax_err = subplot(2,1,1); hold(ax_err,'on'); grid(ax_err,'on');
    ax_ess = subplot(2,1,2); hold(ax_ess,'on'); grid(ax_ess,'on');
    err_max = max(pos_err(~isnan(pos_err))) * 1.15;
    ess_max = R.cfg.N_particles * 1.05;
else
    fig3 = figure('Color','w','Position',[50 590 560 280],'Visible','off');
    ax_err = axes(fig3); hold(ax_err,'on'); grid(ax_err,'on');
    ax_ess = [];
    err_max = max(pos_err(~isnan(pos_err))) * 1.15;
end

for k = 1:n_k
    cla(ax_err);
    % Full error in grey
    plot(ax_err, tvec, pos_err, '-', 'Color',[0.8 0.8 0.8], 'LineWidth',1);
    % Error trail up to k
    plot(ax_err, tvec(1:k), pos_err(1:k), '-', 'Color',C_est, 'LineWidth',2);
    % Current point
    if ~isnan(pos_err(k))
        plot(ax_err, tvec(k), pos_err(k), 'o', 'Color',C_est, 'MarkerSize',8, 'MarkerFaceColor',C_est);
    end
    xlabel(ax_err,'Time (s)'); ylabel(ax_err,'Position error (m)');
    title(ax_err, sprintf('%s  |  err=%.3fm', strrep(filter_name,'_','\_'), pos_err(k)));
    ylim(ax_err, [0, err_max]);

    if has_ess && ~isempty(ax_ess)
        cla(ax_ess);
        valid = ~isnan(R.ESS);
        plot(ax_ess, tvec(valid), R.ESS(valid), '-', 'Color',[0.8 0.8 0.8], 'LineWidth',1);
        plot(ax_ess, tvec(1:k), R.ESS(1:k), '-', 'Color',[0.85 0.40 0.00], 'LineWidth',2);
        if ~isnan(R.ESS(k))
            plot(ax_ess, tvec(k), R.ESS(k), 'o', 'Color',[0.85 0.40 0.00], ...
                 'MarkerSize',8,'MarkerFaceColor',[0.85 0.40 0.00]);
        end
        ess_thresh = get_field(R.cfg, 'ESS_threshold', 0.5);
        yline(ax_ess, R.cfg.N_particles * ess_thresh, 'r--', 'LineWidth',1.2);
        xlabel(ax_ess,'Time (s)'); ylabel(ax_ess,'ESS');
        ylim(ax_ess, [0, ess_max]);
        title(ax_ess, sprintf('ESS (N_p=%d)', R.cfg.N_particles));
    end

    drawnow;
    write_gif_frame(fig3, gif_error, k, opt.delay);
end
close(fig3);
fprintf('  error  → %s\n', gif_error);

%% ============================================================================
%% GIF 4 — STATE TRAJECTORY TREES (RBPF only)
%% ============================================================================
has_statetraj = ~isempty(pfinal_traj) && iscell(pfinal_traj) && ~isempty(pfinal_traj{1});
if ~has_statetraj
    fprintf('  statetraj → skipped (no particle trajectory data; re-run run_experiment)\n');
else
    N_p_traj = numel(pfinal_traj);
    % Determine axis bounds from all trajectories
    all_pts = cell2mat(pfinal_traj);   % [2 x (N_p * n_k)]
    x_lo = min(all_pts(1,:)) - 0.3;   x_hi = max(all_pts(1,:)) + 0.3;
    y_lo = min(all_pts(2,:)) - 0.3;   y_hi = max(all_pts(2,:)) + 0.3;
    % Widen to at least the scene bounds
    x_lo = min(x_lo, -2.3);  x_hi = max(x_hi, 2.3);
    y_lo = min(y_lo, -0.3);  y_hi = max(y_hi, 4.3);

    ess_thresh_st = get_field(R.cfg, 'ESS_threshold', 0.5);
    N_p_cfg = get_field(R.cfg, 'N_particles', N_p_traj);

    C_ptraj  = [0.70 0.78 0.90];   % light steel-blue for background particles
    C_winner_traj = [0.90 0.60 0.00];  % amber for winner

    fig4 = figure('Color','w','Position',[620 590 580 540],'Visible','off');
    ax4  = axes(fig4);

    for k = 1:n_k
        cla(ax4); hold(ax4,'on'); grid(ax4,'on');
        set(ax4,'Color','w','XColor','k','YColor','k');
        xlim(ax4,[x_lo x_hi]); ylim(ax4,[y_lo y_hi]);
        xlabel(ax4,'X (m)'); ylabel(ax4,'Y (m)');

        % Draw winner first as a faint base so non-winner lines appear on top
        if ~isempty(widx)
            traj_w0 = pfinal_traj{widx};
            cols_w0 = min(k, size(traj_w0, 2));
            if cols_w0 >= 2
                plot(ax4, traj_w0(1,1:cols_w0), traj_w0(2,1:cols_w0), '-', ...
                     'Color', C_winner_traj * 0.5 + 0.5, 'LineWidth', 3.5);
            end
        end

        % Draw all particle trajectory tails up to step k (thin lines, on top of winner base)
        for ii = 1:N_p_traj
            if ii == widx, continue; end
            traj_ii = pfinal_traj{ii};
            cols = min(k, size(traj_ii, 2));
            if cols >= 2
                plot(ax4, traj_ii(1,1:cols), traj_ii(2,1:cols), '-', ...
                     'Color', C_ptraj, 'LineWidth', 0.8);
            end
            % Current particle position
            plot(ax4, traj_ii(1,max(cols,1)), traj_ii(2,max(cols,1)), '.', ...
                 'Color', C_ptraj * 0.8, 'MarkerSize', 6);
        end

        % GT trail (green, building up)
        plot(ax4, GT(1,1:k), GT(2,1:k), '-', 'Color',C_gt, 'LineWidth',.4);
        plot(ax4, GT(1,1),   GT(2,1),   's', 'Color',C_gt, 'MarkerSize',8, 'MarkerFaceColor',C_gt);
        plot(ax4, GT(1,k),   GT(2,k),   '+', 'Color',C_gt, 'MarkerSize',10,'LineWidth',2);

        % Measurements at step k
        z_k = z_all{k};
        if ~isempty(z_k)
            plot(ax4, z_k(1,:), z_k(2,:), 'x', 'Color',C_meas, 'MarkerSize',7,'LineWidth',1.5);
        end

        % Winner trajectory (bold amber, drawn last so it sits on top)
        if ~isempty(widx)
            traj_w = pfinal_traj{widx};
            cols_w = min(k, size(traj_w, 2));
            if cols_w >= 2
                plot(ax4, traj_w(1,1:cols_w), traj_w(2,1:cols_w), '-', ...
                     'Color', C_winner_traj, 'LineWidth', 2.5);
            end
            plot(ax4, traj_w(1,max(cols_w,1)), traj_w(2,max(cols_w,1)), 'p', ...
                 'Color', C_winner_traj, 'MarkerSize', 12, 'LineWidth', 1.5, ...
                 'MarkerFaceColor', C_winner_traj);
        end

        % Title — flag resample events via ESS
        resampled = ~isnan(R.ESS(k)) && (R.ESS(k) < ess_thresh_st * N_p_cfg);
        if resampled
            title_str = sprintf('%s  |  k=%d  t=%.2fs  [RESAMPLE]', ...
                strrep(filter_name,'_','\_'), k, (k-1)*dt);
        else
            title_str = sprintf('%s  |  k=%d  t=%.2fs', ...
                strrep(filter_name,'_','\_'), k, (k-1)*dt);
        end
        title(ax4, title_str, 'FontSize', 10);

        drawnow;
        write_gif_frame(fig4, gif_statetraj, k, opt.delay);
    end
    close(fig4);
    fprintf('  statetraj → %s\n', gif_statetraj);
end

fprintf('Done.\n');

end % main function

%% ============================================================================
%% LOCAL HELPERS
%% ============================================================================

function write_gif_frame(fig, filepath, k, delay)
    frame = getframe(fig);
    im    = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    if k == 1
        imwrite(imind, cm, filepath, 'gif', 'Loopcount', inf, 'DelayTime', delay);
    else
        imwrite(imind, cm, filepath, 'gif', 'WriteMode', 'append', 'DelayTime', delay);
    end
end

function [ex, ey] = cov_ellipse(mu, P, n_sigma)
    % n_sigma standard-deviation ellipse for a 2D Gaussian
    theta   = linspace(0, 2*pi, 120);
    circle  = [cos(theta); sin(theta)];
    [V, D]  = eig(P);
    D(D < 0) = 0;  % guard against tiny negative eigenvalues
    ellipse = mu(:) + n_sigma * V * sqrt(D) * circle;
    ex = ellipse(1,:);
    ey = ellipse(2,:);
end

function cm = thermal_white(n)
    % Colormap: white(0) → light yellow → orange → red → dark red(1)
    % Designed for white-background probability density displays.
    if nargin < 1, n = 256; end
    t  = linspace(0, 1, n)';
    r  = ones(n, 1);                          % red: always 1
    g  = max(0, 1 - 1.8 * t);                % green: 1 → 0 (fast)
    b  = max(0, 1 - 3.5 * t);                % blue:  1 → 0 (faster)
    % Darken the very top end toward deep red
    dark = max(0, (t - 0.75) / 0.25);        % 0 until 75%, then ramp to 1
    r  = r  .* (1 - 0.45 * dark);
    cm = [r, g, b];
end

function v = get_field(S, fname, default)
    if isfield(S, fname)
        v = S.(fname);
    else
        v = default;
    end
end
