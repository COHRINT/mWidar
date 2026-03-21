%% run_all_experiments.m
%
% Runs run_experiment for every single-target DA filter and summarises
% RMSE results in a table.
%
% Run from matlab_src/ after startup.m:
%   run_all_experiments
%   run_all_experiments('dataset_dir', 'data/TUNING_DATASET1')
%   run_all_experiments('dataset_dir', 'data/TUNING_DATASET1', 'N_particles', 300)
%   run_all_experiments('N_particles', 500, 'diary_dir', 'diary_logs')
%
% Optional name-value pairs:
%   'dataset_dir'  - path to dataset (default: 'data/TUNING_DATASET1')
%   'N_particles'  - particle count for PF/RBPF families (default: 150)
%   'diary_dir'    - directory to save diary logs (default: 'diary_logs')

function run_all_experiments(varargin)

%% ---- Parse arguments -------------------------------------------------------
p = inputParser;
addParameter(p, 'dataset_dir', fullfile('data', 'TUNING_DATASET1'), @ischar);
addParameter(p, 'N_particles', NaN, @(x) (isnumeric(x) && x > 0) || isnan(x));
addParameter(p, 'diary_dir', 'diary_logs', @ischar);
parse(p, varargin{:});
opt = p.Results;

%% ---- Filter list -----------------------------------------------------------
% KF family (no particles)
kf_filters = {'GNN_KF', 'PDA_KF'};

% HMM family (no particles)
hmm_filters = {'GNN_HMM', 'PDA_HMM'};

% PF family (needs N_particles)
pf_filters = {'GNN_PF', 'PDA_PF', 'MC_PF', 'KF_RBPF', 'HMM_RBPF'};

all_filters = [kf_filters, hmm_filters, pf_filters];
n_total     = numel(all_filters);

% Per-filter particle counts (tuned values from main.m).
% Override with 'N_particles' arg to use a uniform count instead.
default_N_particles = containers.Map( ...
    {'GNN_KF','PDA_KF','GNN_HMM','PDA_HMM','GNN_PF','PDA_PF','MC_PF','KF_RBPF','HMM_RBPF'}, ...
    {NaN,      NaN,     NaN,      NaN,      10000,   1000,    10000,  100,      100});
use_per_filter_N = isnan(opt.N_particles);

%% ---- Results storage -------------------------------------------------------
results_table = struct();
results_table.filter  = all_filters;
results_table.rmse    = nan(1, n_total);
results_table.mean_err= nan(1, n_total);
results_table.max_err = nan(1, n_total);
results_table.status  = repmat({''}, 1, n_total);

%% ---- Setup diary logging ------------------------------------------------------
if ~exist(opt.diary_dir, 'dir')
    mkdir(opt.diary_dir);
end
diary_file = fullfile(opt.diary_dir, sprintf('run_all_experiments_%s_%s.log', ...
    strrep(opt.dataset_dir, filesep, '_'), ...
    datestr(now, 'yyyymmdd_HHMMSS')));
diary(diary_file);
fprintf('Running all experiments on dataset: %s\n', opt.dataset_dir);
if use_per_filter_N
    fprintf('Using per-filter default N_particles from main.m\n');
else
    fprintf('Using uniform N_particles=%d for all PF/RBPF filters\n', opt.N_particles);
end
fprintf('Diary log: %s\n', diary_file);

%% ---- Run each filter -------------------------------------------------------
if use_per_filter_N
    n_desc = 'per-filter defaults (main.m)';
else
    n_desc = sprintf('%d (uniform override)', opt.N_particles);
end
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('  Running %d filters on: %s\n', n_total, opt.dataset_dir);
fprintf('  N_particles: %s\n', n_desc);
fprintf('%s\n\n', repmat('=', 1, 60));

t_total = tic;

for k = 1:n_total
    fname = all_filters{k};
    is_pf = ismember(fname, pf_filters);

    % Determine N_particles for this filter
    if is_pf
        if use_per_filter_N
            n_p = default_N_particles(fname);
        else
            n_p = opt.N_particles;
        end
    end

    fprintf('[%d/%d] %-12s  ', k, n_total, fname);

    t_k = tic;
    try
        if is_pf
            run_experiment('filter', fname, ...
                           'dataset_dir', opt.dataset_dir, ...
                           'N_particles', n_p);
        else
            run_experiment('filter', fname, ...
                           'dataset_dir', opt.dataset_dir);
        end

        % Load just-saved result to pull RMSE
        script_dir = fileparts(mfilename('fullpath'));
        if is_pf
            mat_name = sprintf('%s_N%d.mat', fname, n_p);
        else
            mat_name = sprintf('%s.mat', fname);
        end
        mat_path = fullfile(script_dir, opt.dataset_dir, mat_name);

        tmp = load(mat_path, 'R');
        R   = tmp.R;
        GT  = R.GT;
        pos_err = vecnorm(R.x_est(1:2,:) - GT(1:2,:));
        pos_err = pos_err(~isnan(pos_err));

        results_table.rmse(k)     = sqrt(mean(pos_err.^2));
        results_table.mean_err(k) = mean(pos_err);
        results_table.max_err(k)  = max(pos_err);
        results_table.status{k}   = 'OK';

        fprintf('RMSE=%.4f m  (%.1f s)\n', results_table.rmse(k), toc(t_k));

    catch ME
        results_table.status{k} = 'ERROR';
        fprintf('FAILED  (%.1f s)\n', toc(t_k));
        fprintf('  !! %s\n', ME.message);
        % Print each stack frame so we know exactly where it died
        for si = 1:numel(ME.stack)
            fprintf('     in %s (line %d)\n', ME.stack(si).name, ME.stack(si).line);
        end
    end
end

%% ---- Summary table ---------------------------------------------------------
elapsed = toc(t_total);

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('  SUMMARY  (total: %.1f s)\n', elapsed);
fprintf('%s\n', repmat('=', 1, 60));
fprintf('%-14s  %-6s  %10s  %10s  %10s\n', ...
    'Filter', 'Status', 'RMSE (m)', 'Mean (m)', 'Max (m)');
fprintf('%s\n', repmat('-', 1, 58));

for k = 1:n_total
    if strcmp(results_table.status{k}, 'OK')
        fprintf('%-14s  %-6s  %10.4f  %10.4f  %10.4f\n', ...
            all_filters{k}, results_table.status{k}, ...
            results_table.rmse(k), results_table.mean_err(k), results_table.max_err(k));
    else
        fprintf('%-14s  %-6s  %10s  %10s  %10s\n', ...
            all_filters{k}, results_table.status{k}, '—', '—', '—');
    end
end

fprintf('%s\n\n', repmat('-', 1, 58));

end
