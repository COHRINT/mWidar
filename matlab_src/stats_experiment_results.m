%% stats_experiment_results.m
%
% Computes tracking statistics across filters and (optionally) datasets.
% Loads experiment result .mat files saved by run_experiment / run_all_experiments.
%
% Run from matlab_src/ after startup.m:
%   stats_experiment_results                           % auto-discover all data/TUNING_DATASET* dirs
%   stats_experiment_results('data/TUNING_DATASET1')   % specific dataset directory
%   stats_experiment_results('data/D1', 'data/D2')     % multiple datasets → cross-dataset table
%   stats_experiment_results('data/D1/KF_RBPF_N100.mat', 'data/D1/GNN_KF.mat')  % specific files
%
% Named options (append after paths):
%   'warmup'      - Steps to skip before computing steady-state metrics  (default: 10)
%   'conv_thresh' - Position error threshold for convergence (m)          (default: 0.09)
%   'conv_hold'   - Consecutive steps below threshold → converged         (default: 5)
%   'loss_thresh' - Position error threshold for track loss (m)           (default: 0.50)
%   'plot'        - Show summary figures                                   (default: true)
%
% Metrics computed per filter per dataset:
%   Full RMSE     - RMSE over all timesteps (standard baseline)
%   SS RMSE       - Steady-state RMSE (post-warmup steps only)
%   SS Max        - Max position error post-warmup
%   Conv @ k      - First step where error < conv_thresh for conv_hold consecutive steps
%   Retain %      - % of post-convergence steps with error < loss_thresh
%   NEES          - Mean normalised estimation error squared (pos only, chi2_2 → expect 2.0)

function stats_experiment_results(varargin)

%% ---- Separate positional paths from named options -----------------------
known_keys = {'warmup','conv_thresh','conv_hold','loss_thresh','plot'};
path_args = {};
opt_args  = {};
i = 1;
while i <= nargin
    if ischar(varargin{i}) && any(strcmpi(varargin{i}, known_keys))
        opt_args{end+1} = varargin{i};   %#ok<AGROW>
        opt_args{end+1} = varargin{i+1}; %#ok<AGROW>
        i = i + 2;
    else
        path_args{end+1} = varargin{i};  %#ok<AGROW>
        i = i + 1;
    end
end

p = inputParser;
addParameter(p, 'warmup',      10,   @(x) isnumeric(x) && x >= 0);
addParameter(p, 'conv_thresh', 0.2, @(x) isnumeric(x) && x > 0);
addParameter(p, 'conv_hold',   5,    @(x) isnumeric(x) && x >= 1);
addParameter(p, 'loss_thresh', 0.50, @(x) isnumeric(x) && x > 0);
addParameter(p, 'plot',        true, @islogical);
parse(p, opt_args{:});
opt = p.Results;

script_dir = fileparts(mfilename('fullpath'));

%% ---- Canonical filter registry ------------------------------------------
all_filters = {'GNN_KF','PDA_KF','GNN_HMM','PDA_HMM', ...
               'GNN_PF','PDA_PF','MC_PF','KF_RBPF','HMM_RBPF'};
pf_filters  = {'GNN_PF','PDA_PF','MC_PF','KF_RBPF','HMM_RBPF'};
canonical_N = containers.Map( ...
    {'GNN_PF','PDA_PF','MC_PF','KF_RBPF','HMM_RBPF'}, ...
    { 10000,   1000,    10000,  100,       100});

%% ---- Resolve inputs into datasets or file lists -------------------------
% Three modes:
%   A) No args          → auto-discover all data/TUNING_DATASET* directories
%   B) Dir path(s)      → load canonical filter files from each dir
%   C) .mat file path(s)→ load those files directly (single implicit dataset)

mode = 'dirs';
dataset_dirs  = {};
explicit_files = {};

if isempty(path_args)
    % Mode A: auto-discover
    hits = dir(fullfile(script_dir, 'data', 'TUNING_DATASET*'));
    hits = hits([hits.isdir]);
    if isempty(hits)
        error('stats_experiment_results:noDirs', ...
            'No TUNING_DATASET* directories found under %s/data/', script_dir);
    end
    dataset_dirs = arrayfun(@(h) fullfile(h.folder, h.name), hits, 'UniformOutput', false);
elseif all(cellfun(@(s) endsWith_local(s, '.mat'), path_args))
    % Mode C: explicit .mat files
    mode = 'files';
    explicit_files = path_args;
else
    % Mode B: directory paths
    dataset_dirs = path_args;
end

%% ---- Build list of (dataset_label, result_file) pairs ------------------
entries = {};   % each row: {label, abs_path}

if strcmp(mode, 'files')
    for k = 1:numel(explicit_files)
        fp = explicit_files{k};
        if ~isabsolute_local(fp), fp = fullfile(script_dir, fp); end
        [~, fname] = fileparts(fp);
        entries{end+1,1} = 'files'; %#ok<AGROW>
        entries{end,  2} = fp;
    end
else
    for d = 1:numel(dataset_dirs)
        dpath = dataset_dirs{d};
        if ~isabsolute_local(dpath), dpath = fullfile(script_dir, dpath); end
        [~, dlabel] = fileparts(dpath);

        for f = 1:numel(all_filters)
            fname = all_filters{f};
            if ismember(fname, pf_filters)
                matname = sprintf('%s_N%d.mat', fname, canonical_N(fname));
            else
                matname = sprintf('%s.mat', fname);
            end
            fullp = fullfile(dpath, matname);

            % Fallback: any .mat for this filter with a different N
            if ~exist(fullp, 'file')
                hits = dir(fullfile(dpath, [fname '_N*.mat']));
                if ~isempty(hits)
                    fullp = fullfile(hits(end).folder, hits(end).name);
                else
                    continue;  % filter not run yet for this dataset
                end
            end

            entries{end+1, 1} = dlabel; %#ok<AGROW>
            entries{end,   2} = fullp;
        end
    end
end

if isempty(entries)
    fprintf('No result files found.\n');
    return;
end

%% ---- Load all results ---------------------------------------------------
n_ent   = size(entries, 1);
all_R   = cell(n_ent, 1);
all_lbl = entries(:, 1);   % dataset label per entry
all_fp  = entries(:, 2);

fprintf('Loading %d result file(s)...\n', n_ent);
for k = 1:n_ent
    if ~exist(all_fp{k}, 'file')
        error('stats_experiment_results:missing', 'File not found: %s', all_fp{k});
    end
    tmp = load(all_fp{k}, 'R');
    all_R{k} = tmp.R;
    [~, fn] = fileparts(all_fp{k});
    fprintf('  [%d/%d] %s / %s\n', k, n_ent, all_lbl{k}, fn);
end

%% ---- Compute statistics per result ------------------------------------
all_stats = cell(n_ent, 1);
for k = 1:n_ent
    all_stats{k} = compute_stats(all_R{k}, opt.warmup, ...
                                  opt.conv_thresh, opt.conv_hold, opt.loss_thresh);
end

%% ---- Print per-dataset tables ------------------------------------------
unique_datasets = unique(all_lbl, 'stable');
n_ds = numel(unique_datasets);

sep1 = repmat('=', 1, 80);
sep2 = repmat('-', 1, 80);
hdr  = sprintf(' %-14s  %5s  %9s  %8s  %7s  %7s  %8s  %6s', ...
    'Filter', 'N_p', 'FullRMSE', 'SS_RMSE', 'SS_Max', 'Conv@k', 'Retain%', 'NEES');

for d = 1:n_ds
    ds = unique_datasets{d};
    idx = find(strcmp(all_lbl, ds));

    % Pull N_k and dt from first valid entry
    R0  = all_R{idx(1)};
    N_k = size(R0.GT, 2);
    dt  = R0.params.dt;

    fprintf('\n%s\n', sep1);
    fprintf('  STATS: %s   (%d steps, dt=%.2fs)\n', ds, N_k, dt);
    fprintf('  Warmup: %d steps | Conv thresh: %.3fm (~%.0fpx) hold %d | Loss: %.2fm\n', ...
        opt.warmup, opt.conv_thresh, opt.conv_thresh / (4/128), ...
        opt.conv_hold, opt.loss_thresh);
    fprintf('%s\n', sep1);
    fprintf('%s\n', hdr);
    fprintf('%s\n', sep2);

    for ki = 1:numel(idx)
        k  = idx(ki);
        R  = all_R{k};
        S  = all_stats{k};

        % N_particles
        if isfield(R, 'cfg') && isfield(R.cfg, 'N_particles') && ~isnan(R.cfg.N_particles)
            np_str = sprintf('%5d', R.cfg.N_particles);
        else
            np_str = '    —';
        end

        % Conv@k
        if isnan(S.conv_step)
            conv_str = '  never';
        else
            conv_str = sprintf('  k=%3d', S.conv_step);
        end

        % NEES
        if isnan(S.mean_nees)
            nees_str = '   N/A';
        else
            nees_str = sprintf('%6.2f', S.mean_nees);
        end

        fprintf(' %-14s  %s  %9.4f  %8.4f  %7.4f  %s  %7.1f%%  %s\n', ...
            R.filter_name, np_str, ...
            S.full_rmse, S.ss_rmse, S.ss_max, ...
            conv_str, S.retention, nees_str);
    end

    fprintf('%s\n', sep2);
    fprintf('  NEES: chi2(2) → expected mean = 2.0  (>>2 = overconfident, <<2 = underconfident)\n');
    fprintf('  SS = steady-state (steps %d–%d)  |  Conv = localization step  |  Retain = %% steps < %.2fm post-conv\n', ...
        opt.warmup+1, N_k, opt.loss_thresh);
    fprintf('%s\n', sep1);
end

%% ---- Cross-dataset SS RMSE table (if multiple datasets) -----------------
if n_ds > 1
    % Build matrix: rows = filters (union), cols = datasets
    all_filt_names = cellfun(@(R) R.filter_name, all_R, 'UniformOutput', false);
    unique_filts   = unique(all_filt_names, 'stable');
    n_f = numel(unique_filts);

    ss_mat = NaN(n_f, n_ds);
    for d = 1:n_ds
        idx = find(strcmp(all_lbl, unique_datasets{d}));
        for ki = 1:numel(idx)
            k  = idx(ki);
            fi = find(strcmp(unique_filts, all_R{k}.filter_name));
            if ~isempty(fi)
                ss_mat(fi, d) = all_stats{k}.ss_rmse;
            end
        end
    end

    fprintf('\n%s\n', sep1);
    fprintf('  CROSS-DATASET: Steady-State RMSE (m) — post-%d-step warmup\n', opt.warmup);
    fprintf('%s\n', sep1);

    % Header row
    ds_hdrs = cellfun(@(s) sprintf('%12s', truncate_label(s, 12)), ...
        unique_datasets, 'UniformOutput', false);
    fprintf(' %-14s  %s  %12s\n', 'Filter', strjoin(ds_hdrs, '  '), '        Mean');
    fprintf('%s\n', sep2);

    for fi = 1:n_f
        row = ss_mat(fi, :);
        mean_val = nanmean(row);
        row_strs = arrayfun(@(x) iif_str(isnan(x), '           —', sprintf('%12.4f', x)), ...
            row, 'UniformOutput', false);
        fprintf(' %-14s  %s  %12.4f\n', unique_filts{fi}, strjoin(row_strs, '  '), mean_val);
    end
    fprintf('%s\n', sep1);
end

%% ---- Figures -----------------------------------------------------------
if ~opt.plot, return; end

colors = lines(n_ent);

% ---- Figure 1: Full RMSE vs SS RMSE bar chart (per dataset) -------------
for d = 1:n_ds
    ds  = unique_datasets{d};
    idx = find(strcmp(all_lbl, ds));
    n_f = numel(idx);

    full_rmse_vals = cellfun(@(S) S.full_rmse, all_stats(idx));
    ss_rmse_vals   = cellfun(@(S) S.ss_rmse,   all_stats(idx));
    filt_lbls      = cellfun(@(R) R.filter_name, all_R(idx), 'UniformOutput', false);

    figure('Name', sprintf('RMSE: %s', ds), 'Position', [100 100 900 420]);
    b = bar([full_rmse_vals(:), ss_rmse_vals(:)]);
    b(1).FaceColor = [0.4 0.6 0.9];
    b(2).FaceColor = [0.2 0.8 0.4];
    set(gca, 'XTick', 1:n_f, 'XTickLabel', filt_lbls, 'XTickLabelRotation', 30);
    ylabel('Position RMSE (m)');
    title(sprintf('RMSE by Filter — %s', strrep(ds,'_','\_')));
    legend({'Full RMSE','Steady-State RMSE'}, 'Location','northwest');
    yline(opt.conv_thresh, 'r--', sprintf('%.2fm conv.', opt.conv_thresh), 'LineWidth', 1.2);
    grid on; box on;
end

% ---- Figure 2: Error-over-time overlay (per dataset) --------------------
for d = 1:n_ds
    ds  = unique_datasets{d};
    idx = find(strcmp(all_lbl, ds));
    n_f = numel(idx);

    R0  = all_R{idx(1)};
    N_k = size(R0.GT, 2);
    dt  = R0.params.dt;
    tvec = (0:N_k-1) * dt;

    figure('Name', sprintf('Error over time: %s', ds), 'Position', [150 150 900 420]);
    hold on; grid on;
    col = lines(n_f);

    for ki = 1:n_f
        k  = idx(ki);
        S  = all_stats{k};
        R  = all_R{k};
        plot(tvec, S.pos_err, '-', 'Color', col(ki,:), 'LineWidth', 1.5, ...
            'DisplayName', R.filter_name);
    end

    xline(opt.warmup * dt, 'k--', 'LineWidth', 1, 'HandleVisibility','off');
    yline(opt.conv_thresh, 'r--', 'LineWidth', 1, 'HandleVisibility','off');
    text(opt.warmup * dt + 0.02, opt.conv_thresh * 1.15, ...
        sprintf('conv=%.2fm', opt.conv_thresh), 'FontSize', 8, 'Color', 'r');
    text(opt.warmup * dt + 0.02, max(ylim)*0.97, 'warmup', 'FontSize', 8);

    xlabel('Time (s)'); ylabel('Position error (m)');
    title(sprintf('Position Error vs Time — %s', strrep(ds,'_','\_')));
    legend('Location','northeast', 'FontSize', 7);
end

% ---- Figure 3: Localization time + retention scatter --------------------
for d = 1:n_ds
    ds  = unique_datasets{d};
    idx = find(strcmp(all_lbl, ds));
    n_f = numel(idx);

    conv_steps = cellfun(@(S) iif_val(isnan(S.conv_step), NaN, S.conv_step), all_stats(idx));
    retentions = cellfun(@(S) S.retention, all_stats(idx));
    filt_lbls  = cellfun(@(R) R.filter_name, all_R(idx), 'UniformOutput', false);

    figure('Name', sprintf('Acquisition & Retention: %s', ds), 'Position', [200 200 900 400]);

    subplot(1,2,1);
    col = lines(n_f);
    for ki = 1:n_f
        cv = conv_steps(ki);
        if ~isnan(cv)
            bar(ki, cv, 'FaceColor', col(ki,:));
        else
            bar(ki, size(all_R{idx(1)}.GT,2), 'FaceColor', [0.8 0.2 0.2]);
        end
        hold on;
    end
    set(gca, 'XTick', 1:n_f, 'XTickLabel', filt_lbls, 'XTickLabelRotation', 30);
    ylabel('Step'); title('Localization Step k*');
    text(0.5, 0.95, sprintf('thresh=%.2fm, hold=%d steps', opt.conv_thresh, opt.conv_hold), ...
        'Units','normalized','FontSize',8,'HorizontalAlignment','left');
    grid on; box on;

    subplot(1,2,2);
    for ki = 1:n_f
        bar(ki, retentions(ki), 'FaceColor', col(ki,:));
        hold on;
    end
    set(gca, 'XTick', 1:n_f, 'XTickLabel', filt_lbls, 'XTickLabelRotation', 30);
    ylim([0 105]); ylabel('%'); title(sprintf('Track Retention  (loss > %.2fm)', opt.loss_thresh));
    yline(100, 'k--', 'LineWidth', 0.8);
    grid on; box on;

    sgtitle(strrep(ds,'_','\_'));
end

end % function stats_experiment_results

%% =========================================================================
%  Local helpers
%% =========================================================================

function S = compute_stats(R, warmup, conv_thresh, conv_hold, loss_thresh)
% Compute all tracking metrics from a result struct R.

GT_pos  = R.GT(1:2, :);        % [2 x N_k]
x_est   = R.x_est(1:2, :);     % [2 x N_k]
N_k     = size(GT_pos, 2);
pos_err = vecnorm(x_est - GT_pos);  % [1 x N_k]
valid   = ~isnan(pos_err);

% ---- Full RMSE -----------------------------------------------------------
S.full_rmse = sqrt(mean(pos_err(valid).^2));
S.full_mean = mean(pos_err(valid));
S.full_max  = max(pos_err(valid));

% ---- Steady-state (post warmup) ------------------------------------------
ss_mask  = false(1, N_k);
ss_mask(min(warmup+1, N_k):end) = true;
ss_err   = pos_err(ss_mask & valid);

if ~isempty(ss_err)
    S.ss_rmse = sqrt(mean(ss_err.^2));
    S.ss_mean = mean(ss_err);
    S.ss_max  = max(ss_err);
else
    S.ss_rmse = NaN;  S.ss_mean = NaN;  S.ss_max = NaN;
end

% ---- Localization time ---------------------------------------------------
% First step k where pos_err(k:k+hold-1) all < conv_thresh
S.conv_step = NaN;
for k = 1:N_k - conv_hold + 1
    window = pos_err(k : k + conv_hold - 1);
    if all(~isnan(window)) && all(window < conv_thresh)
        S.conv_step = k;
        break;
    end
end

% ---- Track retention -----------------------------------------------------
% % of post-convergence steps where error < loss_thresh
if ~isnan(S.conv_step)
    post_err = pos_err(S.conv_step:end);
    post_err = post_err(~isnan(post_err));
    S.retention = 100 * mean(post_err < loss_thresh);
else
    S.retention = 0;
end

% ---- NEES (position, 2-DOF chi2 → expected = 2.0) -----------------------
S.mean_nees = NaN;
if isfield(R, 'P_est') && iscell(R.P_est)
    nees_vals = NaN(1, N_k);
    ss_steps  = find(ss_mask & valid);
    for k = ss_steps
        Pk = R.P_est{k};
        if isempty(Pk) || size(Pk,1) < 2, continue; end
        P_pos = Pk(1:2, 1:2);
        e = GT_pos(:,k) - x_est(:,k);
        if any(isnan(e)) || rcond(P_pos) < 1e-12, continue; end
        nees_vals(k) = e' * (P_pos \ e);
    end
    valid_nees = nees_vals(~isnan(nees_vals));
    if ~isempty(valid_nees)
        S.mean_nees = mean(valid_nees);
    end
end

% Keep full error series for plotting
S.pos_err = pos_err;

end % compute_stats

%% ---- Utilities ----------------------------------------------------------

function tf = endsWith_local(s, suf)
    tf = numel(s) >= numel(suf) && strcmp(s(end-numel(suf)+1:end), suf);
end

function tf = isabsolute_local(s)
    tf = ~isempty(s) && (s(1) == '/' || s(1) == '\' || (numel(s)>=2 && s(2)==':'));
end

function s = truncate_label(s, maxlen)
    if numel(s) > maxlen, s = s(end-maxlen+1:end); end
end

function v = iif_val(cond, a, b)
    if cond, v = a; else, v = b; end
end

function s = iif_str(cond, a, b)
    if cond, s = a; else, s = b; end
end

function m = nanmean(v)
    v = v(~isnan(v));
    if isempty(v), m = NaN; else, m = mean(v); end
end
