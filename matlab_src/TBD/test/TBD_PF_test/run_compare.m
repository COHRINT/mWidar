%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% run_compare %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compare results from RBPF, TBD-PF: Gauss, TBD-PF: Model
% Comparison metrics:
% RMSE
% Runtime
% OSPA
% Ask around for more metrics 
% We could do missed associations for RBPF, & missed detection for TBD_PF

% Must run from matlab_src: environment adds supplemental/Final_Test_Tracks
% (CA_CFAR) and the simulator loads supplemental/*.mat by relative path.
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));       % .../matlab_src
addpath(fullfile(root, 'TBD', 'Includes'), fullfile(here, 'core'), fullfile(here, 'config'));
addpath(root, fullfile(root, 'DA_Track'), fullfile(root, 'DA_Track', 'single')); % FilterConfig, KF_RBPF
cd(root);

rng(10); % only so the trail_seed #'s are reproducable

M = 1; % Number of mc runs (One for now, change to 10 when done)
trial_seeds = randi(676742067,1,M);

noise_var = 10; % Static variance on all trails
kEnd = 100;     % same for all trials

% CA-CFAR settings for the RBPF measurements (see environment.run_detector).
% Blur only smooths the detector's copy, TBD-PF still gets the raw image.
detector_args = {'Pfa', 0.365, 'Blur', 2, 'YMin', 0.5};

%%% Build M environments for each type of environment

% Lists to hold env objects
line_envs   = [];
parab_envs  = [];
scurve_envs = [];
rw_envs     = [];

trajs = ["line", "parabola", "scurve", "rw"]; % rw will be a parabola w rw on
for m = 1:M
    rng(trial_seeds(m))
    for traj = trajs
        kb = randi(15);
        kd = kEnd - randi(15);

        % "rw" is not a generator trajectory: it is a parabola with rw on
        if traj == "rw"
            gen_traj = 'parabola';
            on = true;
        else
            gen_traj = char(traj);
            on = false;
        end

        e = environment('objs', 1, 'trajectories', {gen_traj}, 'rw', on, 'kEnd', kEnd, ...
            'kBirth', kb, 'kDeath', kd, 'Seed', trial_seeds(m), ...
            'Noise', true, 'Var', noise_var, 'Detector', true, 'DetectorArgs', detector_args);

        switch traj
            case "line"
                line_envs = [line_envs, e];
            case "parabola"
                parab_envs = [parab_envs, e];
            case "scurve"
                scurve_envs = [scurve_envs, e];
            case "rw"
                rw_envs = [rw_envs, e];
        end
    end
end

%%% Run each filter on every environment in parallel

%% Filter settings (edit these)
filters = ["KF_RBPF", "TBD_PF_gauss", "TBD_PF_model"];   % drop entries to skip a filter

% Particle counts. Each can be a vector, every value becomes its own job,
% e.g. N_TBD = [1000 2500 5000] for a particle-count study.
N_TBD  = 5000;
N_RBPF = 500;    % FilterHyperParams('KF_RBPF') default

tbd_ESS = 0.3;   % TBD-PF resample threshold (run.ESS)

% TBD-PF NoiseSTD per variant and trajectory.
% TODO: replace with the tuned values from the NoiseSTD sweeps (T7-T10).
% Placeholders: gauss = sqrt(noise_var) (T3 convention), model = default_cfg (10).

NoiseSTD.pf_gauss = struct('line', sqrt(noise_var), 'parabola', sqrt(noise_var), ...
                           'scurve', sqrt(noise_var), 'rw', sqrt(noise_var));
NoiseSTD.pf_model = struct('line', 10, 'parabola', 10, 'scurve', 10, 'rw', 10);

% KF_RBPF association sampling: 'gating' = optimal importance distribution.
% ('likelihood' currently falls back to uniform, 'optimal' is not implemented in timestep)
rbpf_assoc = 'gating';

%% Generate every scenario once on the client, so all filters see identical data
env_groups = {line_envs, parab_envs, scurve_envs, rw_envs};   % same order as trajs
envs      = {};
scenarios = {};
env_info  = struct('trial', {}, 'traj', {}, 'seed', {});
for t = 1:numel(trajs)
    for m = 1:M
        e = env_groups{t}(m);
        fprintf("generating scenario: %s, trial %d\n", trajs(t), m);
        envs{end+1}      = e;           %#ok<SAGROW>
        scenarios{end+1} = e.setup();   %#ok<SAGROW>
        env_info(end+1)  = struct('trial', m, 'traj', trajs(t), 'seed', trial_seeds(m)); %#ok<SAGROW>
    end
end

%% Flat job list: every (scenario, filter, N)
variant_of = struct('KF_RBPF', "", 'TBD_PF_gauss', "pf_gauss", 'TBD_PF_model', "pf_model");
jobs = {};
for i = 1:numel(scenarios)
    for filt_name = filters
        variant = variant_of.(filt_name);
        if filt_name == "KF_RBPF"
            Ns = N_RBPF;
            nstd = NaN;
        else
            Ns = N_TBD;
            nstd = NoiseSTD.(variant).(env_info(i).traj);
        end
        for N = Ns
            jobs{end+1} = struct( ...   %#ok<SAGROW>
                'envIdx',    i, ...
                'trial',     env_info(i).trial, ...
                'traj',      env_info(i).traj, ...
                'seed',      env_info(i).seed, ...
                'filter',    filt_name, ...
                'variant',   variant, ...
                'N',         N, ...
                'noiseSTD',  nstd, ...
                'tbdESS',    tbd_ESS, ...
                'rbpfAssoc', rbpf_assoc, ...
                'env',       envs{i}, ...
                'scenario',  scenarios{i});
        end
    end
end
nJobs = numel(jobs);
fprintf("\n%d jobs (%d scenarios x %d filters, over N values)\n", nJobs, numel(scenarios), numel(filters));

%% Pool. Workers need matlab_src as their current folder for the relative
%% loads (CA_CFAR path, TargetContribution.mat). parfor runs serially
%% without the Parallel Computing Toolbox.
if ~isempty(ver('parallel'))
    pool = gcp();
    wait(parfevalOnAll(pool, @cd, 0, root));
end

%% Run. runtime is timed per job on its worker (filtering loop only).
%% Workers share the CPU, so runtimes are comparable across filters but are
%% not single-job timings.
results = cell(1, nJobs);
parfor j = 1:nJobs
    r = run_compare_job(jobs{j});
    fprintf("done: %-13s %-9s trial %d  N=%-5d  %.1f s\n", r.filter, r.traj, r.trial, r.N, r.runtime);
    results{j} = r;
end
results = [results{:}];   % 1 x nJobs struct array, results(j).envIdx -> scenarios{...}

%% Runtime summary
runtimes = struct2table(rmfield(results, 'R'));
disp(runtimes(:, {'filter', 'traj', 'trial', 'N', 'runtime'}));
disp(groupsummary(runtimes, {'filter', 'N'}, {'mean', 'std'}, 'runtime'));

%%% Metrics and plots: RMSE, OSPA, runtime, every filter on the same axes

ospa_c = 0.5;   % OSPA cutoff [m], order p = 2
save_R = false; % also save every filter's full R in results.mat (TBD-PF R is ~35 MB per run)

addpath(fullfile(root, 'supplemental', 'multitarget_metrics'));   % compute_ospa
[metrics, series] = compare_metrics(results, scenarios, 'OSPACutoff', ospa_c, 'OSPAOrder', 2);

disp(metrics);
disp(groupsummary(metrics, {'filter', 'N', 'traj'}, {'mean', 'std'}, ...
    {'rmse', 'ospa', 'detect', 'false_alarm', 'runtime'}));

out_dir = fullfile(here, 'results', "run_compare_" + string(datetime('now', 'Format', 'yyyyMMdd_HHmmss')));
figs = plot_compare(metrics, series, 'OSPACutoff', ospa_c, 'SaveDir', fullfile(out_dir, 'figures'));

writetable(metrics, fullfile(out_dir, 'metrics.csv'));
settings = struct('M', M, 'trial_seeds', trial_seeds, 'noise_var', noise_var, 'kEnd', kEnd, ...
    'detector_args', {detector_args}, 'filters', filters, 'N_TBD', N_TBD, 'N_RBPF', N_RBPF, ...
    'tbd_ESS', tbd_ESS, 'NoiseSTD', NoiseSTD, 'rbpf_assoc', rbpf_assoc, 'ospa_c', ospa_c);
saved = results;
if ~save_R, saved = rmfield(saved, 'R'); end
save(fullfile(out_dir, 'results.mat'), 'settings', 'metrics', 'series', 'saved', '-v7.3');
fprintf("\nsaved metrics and figures to %s\n", out_dir);


