%% plot_experiment_results.m
%
% Loads one or more experiment result files from a TUNING_DATASET directory
% and generates trajectory and accuracy plots.
%
% Run from matlab_src/ after startup.m:
%   plot_experiment_results                          % interactive file picker
%   plot_experiment_results('data/TUNING_DATASET1/KF_RBPF_N300.mat')
%   plot_experiment_results('data/TUNING_DATASET1/KF_RBPF_N300.mat', ...
%                           'data/TUNING_DATASET1/PDA_KF.mat')
%   plot_experiment_results('data/TUNING_DATASET1/KF_RBPF_N300.mat', ...
%                           'output_dir', 'figures/', 'prefix', 'run1')
%
% Named options (append after file paths):
%   'output_dir'  - Directory to save PNGs (default: same dir as first result)
%   'prefix'      - Filename prefix (default: first result filename)
%
% Figures produced (saved as <prefix>_trajectory.png etc. when output_dir set):
%   1) Trajectory: GT vs estimate (X-Y plane)
%   2) Position error over time + RMSE
%   3) Velocity error over time  (if state has vx/vy)
%   4) ESS over time             (particle filters only)

function plot_experiment_results(varargin)

%% ---- Separate file paths from named options -----------------------------
known_keys = {'output_dir', 'prefix'};
file_args = {};
opt_args  = {};
i = 1;
while i <= nargin
    if ischar(varargin{i}) && any(strcmpi(varargin{i}, known_keys))
        opt_args{end+1} = varargin{i};   %#ok<AGROW>
        opt_args{end+1} = varargin{i+1}; %#ok<AGROW>
        i = i + 2;
    else
        file_args{end+1} = varargin{i};  %#ok<AGROW>
        i = i + 1;
    end
end

p = inputParser;
addParameter(p, 'output_dir', '', @ischar);
addParameter(p, 'prefix',     '', @ischar);
parse(p, opt_args{:});
opt = p.Results;

%% ---- Collect result files -----------------------------------------------
if isempty(file_args)
    % Interactive picker
    [files, path] = uigetfile('*.mat', 'Select result file(s)', ...
        fullfile('data', 'TUNING_DATASET1'), 'MultiSelect', 'on');
    if isequal(files, 0)
        fprintf('No file selected. Exiting.\n');
        return;
    end
    if ischar(files), files = {files}; end
    result_files = cellfun(@(f) fullfile(path, f), files, 'UniformOutput', false);
else
    result_files = file_args;
    % Allow relative paths from matlab_src
    script_dir = fileparts(mfilename('fullpath'));
    for k = 1:numel(result_files)
        if ~isabs_path(result_files{k})
            result_files{k} = fullfile(script_dir, result_files{k});
        end
    end
end

%% ---- Load results -------------------------------------------------------
n_res = numel(result_files);
results = cell(n_res, 1);
labels  = cell(n_res, 1);

for k = 1:n_res
    if ~exist(result_files{k}, 'file')
        error('plot_experiment_results:missing', 'File not found: %s', result_files{k});
    end
    tmp = load(result_files{k}, 'R');
    results{k} = tmp.R;
    [~, fname] = fileparts(result_files{k});
    labels{k}  = strrep(fname, '_', '\_');  % escape underscores for LaTeX
    fprintf('Loaded: %s  (%s, %d steps)\n', fname, results{k}.filter_name, size(results{k}.GT,2));
end

% Use ground truth from first result
GT   = results{1}.GT;
n_k  = size(GT, 2);
tvec = (0:n_k-1) * results{1}.params.dt;

%% ---- Colour cycle -------------------------------------------------------
colors = lines(n_res);

%% ---- Output path setup --------------------------------------------------
[first_dir, first_name] = fileparts(result_files{1});
if isempty(opt.output_dir)
    out_dir = first_dir;
else
    out_dir = opt.output_dir;
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
end
pfx = opt.prefix;
if isempty(pfx), pfx = first_name; end
do_save = ~isempty(opt.output_dir) || ~isempty(opt.prefix);

%% ========================================================================
%  Figure 1: Trajectory (X-Y plane)
%% ========================================================================
fig1 = figure('Name', 'Trajectory', 'Position', [50 50 700 600]);
hold on; grid on; axis equal;

% Ground truth
plot(GT(1,:), GT(2,:), 'k-', 'LineWidth', 2.5, 'DisplayName', 'Ground truth');
plot(GT(1,1), GT(2,1), 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
plot(GT(1,end), GT(2,end), 'k^', 'MarkerSize', 10, 'MarkerFaceColor', 'k');

for k = 1:n_res
    R = results{k};
    plot(R.x_est(1,:), R.x_est(2,:), '--', ...
        'Color', colors(k,:), 'LineWidth', 1.8, 'DisplayName', labels{k});
    % Mark first estimate
    plot(R.x_est(1,1), R.x_est(2,1), 'o', ...
        'Color', colors(k,:), 'MarkerSize', 8, 'MarkerFaceColor', colors(k,:), ...
        'HandleVisibility', 'off');
end

xlabel('X (m)'); ylabel('Y (m)');
title('Trajectory: Ground Truth vs Estimate');
legend('Location', 'best');
xlim([-2.2, 2.2]); ylim([-0.2, 4.2]);

% Annotate start/end
text(GT(1,1),   GT(2,1)-0.15,   'Start', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(GT(1,end), GT(2,end)+0.12, 'End',   'HorizontalAlignment', 'center', 'FontSize', 9);

%% ========================================================================
%  Figure 2: Position error over time
%% ========================================================================
fig2 = figure('Name', 'Position Error', 'Position', [760 50 700 400]);
hold on; grid on;

for k = 1:n_res
    R = results{k};
    pos_err = vecnorm(R.x_est(1:2,:) - GT(1:2,:));  % [1 x n_k]
    rmse_val = sqrt(mean(pos_err(~isnan(pos_err)).^2));
    lbl = sprintf('%s MMSE (RMSE=%.3f m)', labels{k}, rmse_val);
    plot(tvec, pos_err, '-', 'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', lbl);

    % MAP estimate if available
    if isfield(R, 'x_map') && any(~isnan(R.x_map(1,:)))
        map_err = vecnorm(R.x_map(1:2,:) - GT(1:2,:));
        rmse_map = sqrt(mean(map_err(~isnan(map_err)).^2));
        lbl_map = sprintf('%s MAP  (RMSE=%.3f m)', labels{k}, rmse_map);
        plot(tvec, map_err, '--', 'Color', colors(k,:)*0.7, 'LineWidth', 1.2, 'DisplayName', lbl_map);
    end
end

xlabel('Time (s)'); ylabel('Position error (m)');
title('Position Error vs Time');
legend('Location', 'best');
ylim([0, max(ylim)*1.1]);

%% ========================================================================
%  Figure 3: Velocity error over time  (if state has vx/vy at indices 3:4)
%% ========================================================================
N_x = size(results{1}.x_est, 1);
if N_x >= 4
    fig3 = figure('Name', 'Velocity Error', 'Position', [50 500 700 350]);
    hold on; grid on;

    for k = 1:n_res
        R = results{k};
        if size(R.x_est, 1) >= 4
            vel_err = vecnorm(R.x_est(3:4,:) - GT(3:4,:));
            rmse_vel = sqrt(mean(vel_err(~isnan(vel_err)).^2));
            lbl = sprintf('%s  (RMSE=%.3f m/s)', labels{k}, rmse_vel);
            plot(tvec, vel_err, '-', 'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', lbl);
        end
    end

    xlabel('Time (s)'); ylabel('Velocity error (m/s)');
    title('Velocity Error vs Time');
    legend('Location', 'best');
end

%% ========================================================================
%  Figure 4: ESS over time (particle filters only)
%% ========================================================================
has_ess = false;
for k = 1:n_res
    R = results{k};
    if any(~isnan(R.ESS))
        has_ess = true;
        break;
    end
end

if has_ess
    fig4 = figure('Name', 'Effective Sample Size', 'Position', [760 500 700 350]);
    hold on; grid on;

    for k = 1:n_res
        R = results{k};
        if any(~isnan(R.ESS))
            valid = ~isnan(R.ESS);
            % Also look up N_particles from cfg if available
            if isfield(R.cfg, 'N_particles')
                N_p = R.cfg.N_particles;
                yyaxis left;
                plot(tvec(valid), R.ESS(valid), '-', ...
                    'Color', colors(k,:), 'LineWidth', 1.5, ...
                    'DisplayName', [labels{k} ' ESS']);
                ylabel('ESS (particles)');
                yyaxis right;
                ess_frac = R.ESS(valid) ./ N_p;
                plot(tvec(valid), ess_frac, '--', ...
                    'Color', colors(k,:)*0.7, 'LineWidth', 1, ...
                    'DisplayName', [labels{k} ' ESS/N']);
                ylabel('ESS / N_p');
            else
                plot(tvec(valid), R.ESS(valid), '-', ...
                    'Color', colors(k,:), 'LineWidth', 1.5, 'DisplayName', labels{k});
                ylabel('ESS');
            end
        end
    end

    xlabel('Time (s)');
    title('Effective Sample Size vs Time');
    legend('Location', 'best');
end

%% ---- Save PNGs ----------------------------------------------------------
if do_save
    save_png(fig1, fullfile(out_dir, [pfx '_trajectory.png']));
    save_png(fig2, fullfile(out_dir, [pfx '_error.png']));
    if N_x >= 4 && exist('fig3','var')
        save_png(fig3, fullfile(out_dir, [pfx '_velocity.png']));
    end
    if has_ess && exist('fig4','var')
        save_png(fig4, fullfile(out_dir, [pfx '_ess.png']));
    end
end

%% ---- Print table --------------------------------------------------------
fprintf('\n%-28s %-5s %8s %8s %8s\n', 'Filter', 'Est', 'RMSE(m)', 'Max(m)', 'Mean(m)');
fprintf('%s\n', repmat('-', 1, 62));
for k = 1:n_res
    R = results{k};
    pos_err = vecnorm(R.x_est(1:2,:) - GT(1:2,:));
    pos_err = pos_err(~isnan(pos_err));
    fprintf('%-28s %-5s %8.4f %8.4f %8.4f\n', results{k}.filter_name, 'MMSE', ...
        sqrt(mean(pos_err.^2)), max(pos_err), mean(pos_err));
    if isfield(R, 'x_map') && any(~isnan(R.x_map(1,:)))
        map_err = vecnorm(R.x_map(1:2,:) - GT(1:2,:));
        map_err = map_err(~isnan(map_err));
        fprintf('%-28s %-5s %8.4f %8.4f %8.4f\n', '', 'MAP', ...
            sqrt(mean(map_err.^2)), max(map_err), mean(map_err));
    end
end
fprintf('\n');

end % function

%% ---- Helper: platform-agnostic absolute path check ----------------------
function tf = isabs_path(p)
    tf = ~isempty(p) && (p(1) == '/' || p(1) == '\' || ...
         (numel(p) >= 2 && p(2) == ':'));
end

%% ---- Helper: save figure as PNG -----------------------------------------
function save_png(fig, filepath)
    try
        exportgraphics(fig, filepath, 'Resolution', 150);
    catch
        print(fig, filepath, '-dpng', '-r150');
    end
    fprintf('  saved → %s\n', filepath);
end
