%% run_experiment.m
%
% Loads a pre-generated tuning dataset, runs a single filter through it
% using FilterConfig / FilterFactory, and saves per-timestep results.
%
% Run from matlab_src/ after startup.m:
%   run_experiment                   % default: KF_RBPF on TUNING_DATASET1
%   run_experiment('filter', 'PDA_KF')
%   run_experiment('filter', 'KF_RBPF', 'N_particles', 200)
%   run_experiment('filter', 'PDA_PF',  'dataset_dir', 'data/TUNING_DATASET1')
%
% Saved output:
%   data/TUNING_DATASET1/<filter_name>_N<N_particles>.mat  (PF/RBPF families)
%   data/TUNING_DATASET1/<filter_name>.mat                 (KF family)
%
% The result struct 'R' contains:
%   R.filter_name   - string
%   R.GT            - [6 x N_k] ground truth
%   R.x_est         - [N_x x N_k] state estimates
%   R.P_est         - {N_k x 1}   covariance matrices
%   R.measurements  - {N_k x 1}   raw measurements per step
%   R.ESS           - [1 x N_k]   effective sample size (PF/RBPF only, else NaN)
%   R.cfg           - FilterConfig struct used
%   R.params        - generation parameters from dataset

function run_experiment(varargin)

%% ---- Parse arguments ----------------------------------------------------
p = inputParser;
addParameter(p, 'filter',      'HMM_RBPF', @ischar);
addParameter(p, 'dataset_dir', fullfile('data', 'TUNING_DATASET1'), @ischar);
addParameter(p, 'N_particles', 150,  @(x) isnumeric(x) && x > 0);
addParameter(p, 'Debug',       false, @islogical);
parse(p, varargin{:});
opt = p.Results;

filter_name = opt.filter;
dataset_dir = opt.dataset_dir;

%% ---- Paths / addpath ----------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'DA_Track'));                  % base: DA_Filter, KF, HMM
addpath(fullfile(script_dir, 'DA_Track', 'single'));         % single-target filters
addpath(fullfile(script_dir, 'supplemental'));
addpath(fullfile(script_dir, 'supplemental', 'Final_Test_Tracks'));

%% ---- Load dataset -------------------------------------------------------
data_file = fullfile(script_dir, dataset_dir, 'data.mat');
if ~exist(data_file, 'file')
    error('run_experiment:NoDataset', ...
        'Dataset not found: %s\n  Run generate_tuning_dataset.m first.', data_file);
end
fprintf('Loading dataset: %s\n', data_file);
load(data_file, 'Data');

GT     = Data.GT;            % [6 x N_k]
z_all  = Data.y;             % {1 x N_k} cell of [2 x N_det]
signal = Data.signal;        % {1 x N_k} cell of [128 x 128]
n_k    = size(GT, 2);
dt     = Data.params.dt;

fprintf('Dataset: %d timesteps, dt=%.2f s\n', n_k, dt);

%% ---- Filter parameters --------------------------------------------------
PF_families  = {'PDA_PF','GNN_PF','MC_PF','KF_RBPF','HMM_RBPF'};
HMM_families = {'GNN_HMM','PDA_HMM'};
KF_families  = {'PDA_KF','GNN_KF'};

% Constant-acceleration continuous dynamics matrix (shared convenience variable)
Fc_cont = [0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0; ...
           0 0 0 0 0 1; 0 0 0 0 0 0; 0 0 0 0 0 0];

% Constant-acceleration discrete F (ZOH exact) — used by KF and RBPF families
F_zoh = expm(Fc_cont * dt);

% Constant-acceleration discrete F (analytic) — used by PF families
F_disc = [1 0 dt 0 dt^2/2 0; 0 1 0 dt 0 dt^2/2; ...
          0 0 1  0 dt     0; 0 0 0 1  0 dt; ...
          0 0 0  0 1      0; 0 0 0 0  0 1];

% -----------------------------------------------------------------------
% Per-filter process noise Q  [diag entries: x, y, vx, vy, ax, ay]
%
% TUNING GUIDE:
%   Increase Q(i,i) to let the filter track faster state changes in dim i.
%   Decrease Q(i,i) to smooth out noisy state estimates in dim i.
%   Rule of thumb: start at (expected_rms_acceleration * dt)^2 for acc dims.
%   HMM-family filters do not use Q (grid-based, not state-vector based).
% -----------------------------------------------------------------------
switch filter_name

    %% -- Kalman Filter family (ZOH F) ----------------------------------
    case 'GNN_KF'
        % TUNE Q: larger position/velocity noise → filter trusts measurements more
        %        and can track maneuvers. Too small → Kalman gain collapses,
        %        filter ignores measurements and drifts on curves.
        Q = diag([2e-4, 2e-4, ...   % x, y     (position)
                  2e-3, 2e-3, ...   % vx, vy   (velocity)
                  2e-2, 2e-2]);     % ax, ay   (acceleration)
        % TUNE R: reduce to trust pixel measurements more (0.1 → 0.05)
        R = 0.05 * eye(2);
        F = F_zoh;

    case 'PDA_KF'
        % DIAGNOSIS: was diverging by step 10 with original Q.
        % Root cause: Q_pos=1e-4 + R=0.1 → K ≈ 0.03 (nearly ignores measurements).
        % Fix: moderate Q so Kalman gain opens up; keep R=0.1 to avoid
        %   the over-corrected Q+small-R → non-PD covariance crash (mvnpdf).
        % Also tune lambda_clutter post-construction (below).
        Q = diag([5e-4, 5e-4, ...   % x, y
                  2e-3, 2e-3, ...   % vx, vy
                  5e-3, 5e-3]);     % ax, ay
        % TUNE R: keep at 0.1 — smaller R amplifies Kalman gain, risks S not PD
        R = 0.1 * eye(2);
        F = F_zoh;

    %% -- Particle Filter family (analytic F) ---------------------------
    case 'GNN_PF'
        % DIAGNOSIS: tightening Q to 5e-4 + setting validation_sigma_bounds=2
        %   caused total track loss (0.46→3.48m) — gate rejected all measurements
        %   once particles drifted, no recovery possible.
        % Fix: revert Q to original spread; gate stays at cfg default (5-sigma).
        Q = diag([1e-3, 1e-3, ...   % x, y
                  1e-2, 1e-2, ...   % vx, vy
                  1e-1, 1e-1]);     % ax, ay
        R = 0.1 * eye(2);   % not used in PF weight update; kept for consistency
        F = F_disc;

    case 'PDA_PF'
        % TUNE: working well at 0.11m RMSE — keep as reference baseline
        Q = diag([1e-3, 1e-3, ...   % x, y
                  1e-2, 1e-2, ...   % vx, vy
                  1e-1, 1e-1]);     % ax, ay
        R = 0.1 * eye(2);
        F = F_disc;

    case 'MC_PF'
        % TUNE: same Q as PDA_PF — resampling fraction tuned post-construction
        Q = diag([1e-3, 1e-3, ...   % x, y
                  1e-2, 1e-2, ...   % vx, vy
                  1e-1, 1e-1]);     % ax, ay
        R = 0.1 * eye(2);
        F = F_disc;

    %% -- RBPF family ---------------------------------------------------
    case 'KF_RBPF'
        % TUNE: embedded KF inner filter — open Q slightly so inner KFs
        %       can follow the S-curve; also tuned via N_particles (see
        %       run_all_experiments default_N_particles for KF_RBPF)
        Q = diag([5e-4, 5e-4, ...   % x, y
                  2e-3, 2e-3, ...   % vx, vy
                  2e-3, 2e-3]);     % ax, ay
        % TUNE R: trust pixel measurements more in the embedded KF
        R = 0.05 * eye(2);
        F = F_zoh;

    %% -- HMM and HMM-RBPF families (no Q/F — grid-based) --------------
    case {'GNN_HMM', 'PDA_HMM', 'HMM_RBPF'}
        Q = [];
        R = [];
        F = [];

    otherwise
        Q = [];
        R = 0.1 * eye(2);
        F = [];
        warning('run_experiment:UnknownFilter', ...
            'No Q/F tuning defined for filter ''%s'' — using empty matrices.', filter_name);
end
H  = [1 0 0 0 0 0; 0 1 0 0 0 0];
P0 = diag([0.1 0.1 0.25 0.25 0.5 0.5]);
% R is now set per-filter inside the switch above.

%% ---- Build FilterConfig -------------------------------------------------
cfg = FilterConfig(filter_name, ...
    'dt',               dt, ...
    'N_particles',      opt.N_particles, ...
    'Debug',            opt.Debug, ...
    'ValidationSigma',  5, ...
    'ESS_threshold',    0.2, ...
    'store_full_history', false);
if isfield(cfg, 'F') && ~isempty(F), cfg.F = F; end
if isfield(cfg, 'Q') && ~isempty(Q), cfg.Q = Q; end
if isfield(cfg, 'R'),                cfg.R = R; end
if isfield(cfg, 'H'),                cfg.H = H; end

%% ---- Load supplemental matrices if needed --------------------------------
A_transition          = [];
pointlikelihood_image = [];
pointlikelihood_mag   = [];

if ismember(filter_name, HMM_families) || strcmp(filter_name, 'HMM_RBPF')
    load(fullfile(script_dir, 'supplemental', 'precalc_imagegridHMMEmLike.mat'),  'pointlikelihood_image');
    tmp = load(fullfile(script_dir, 'supplemental', 'precalc_imagegridHMMSTMn15.mat'), 'A');
    A_transition = tmp.A;
elseif ismember(filter_name, {'PDA_PF','MC_PF'})
    load(fullfile(script_dir, 'supplemental', 'precalc_imagegridHMMEmLike.mat'),  'pointlikelihood_image');
    tmp = load(fullfile(script_dir, 'supplemental', 'precalc_imagegridHMMEmLikeMag.mat'), 'pointlikelihood_image');
    pointlikelihood_mag = tmp.pointlikelihood_image;
elseif strcmp(filter_name, 'GNN_PF')
    load(fullfile(script_dir, 'supplemental', 'precalc_imagegridHMMEmLike.mat'),  'pointlikelihood_image');
end

%% ---- Initial state -------------------------------------------------------
x0 = zeros(6, 1);
x0(1:2) = GT(1:2, 1);          % known position, zero velocity/acceleration
P0_init  = diag([100 100 10 10 5 5]);

% HMM family uses 2D state only
if ismember(filter_name, HMM_families) || strcmp(filter_name, 'HMM_RBPF')
    x0 = x0(1:2);
end

%% ---- Construct filter ---------------------------------------------------
fprintf('Constructing %s filter...\n', filter_name);
filt = FilterFactory(cfg, x0, P0_init, A_transition, pointlikelihood_image, pointlikelihood_mag);

% Post-construction tuning
% (Applied after FilterFactory so these override constructor/cfg defaults.)

% --- PDA_KF: loosen clutter assumption so beta0 doesn't dominate ---------
% DIAGNOSIS: lambda_clutter=2.5 was making the clutter hypothesis dominate
%   even when a real measurement was nearby, causing the update to barely move.
% TUNE: decrease lambda_clutter to assume fewer false alarms per unit area.
if strcmp(filter_name, 'PDA_KF')
    filt.lambda_clutter = 0.5;   % TUNE: [0.1 — 5.0]; lower = fewer expected clutter hits
end

% --- PF/MC-PF families ---------------------------------------------------
if ismember(filter_name, {'PDA_PF','MC_PF'})
    filt.setDetectionModel(0.99, 0.25);
    filt.composite_likelihood = true;
    if strcmp(filter_name, 'PDA_PF')
        filt.hybrid_resample_fraction = 0.9;    % TUNE: [0.5 — 1.0]
    else
        % MC_PF: was 0.99 (floods with uniform particles every step → no memory).
        % Reduce to 0.9 so 90% resample from posterior, 10% uniform exploration.
        filt.hybrid_resample_fraction = 0.9;    % TUNE: [0.5 — 1.0]
    end
end

% --- GNN_PF: validation gate -------------------------------------------------
% DIAGNOSIS: 2-sigma gate caused total track loss when particles drifted —
%   tight gate rejected all measurements, no recovery possible.
% Decision: let the cfg default ValidationSigma=5 stand (set in FilterConfig call above).
% Do NOT override validation_sigma_bounds here unless you understand the drift risk.

% --- KF_RBPF: use likelihood-guided association --------------------------
% DIAGNOSIS: 'uniform' association wastes particles on low-probability hypotheses,
%   causing rapid ESS collapse (100→15 in 50 steps).
% Fix 1: switch to 'likelihood' (OIS) for smarter association sampling.
% Fix 2: increase N_particles in run_all_experiments.m (100 → 300).
if strcmp(filter_name, 'KF_RBPF')
    filt.association_strategy = 'likelihood';   % 'uniform' | 'likelihood' | 'optimal'
end

% --- HMM families: keep constructor Gaussian init (don't reset to uniform) ---
% The constructor initialises ptarget_prob as a Gaussian centred on x0 = GT(:,1).
% Resetting to uniform here discards that good initialisation, causing the filter
% to start from the grid mean (0, 2) instead of the true start position and
% producing max-error = sqrt(4.5) ≈ 2.12 m at early timesteps.
% NOTE: for blind-init Monte Carlo runs, re-enable the uniform reset below.
% if ismember(filter_name, HMM_families)
%     filt.ptarget_prob = ones(filt.npx2, 1) / filt.npx2;
% end

%% ---- Print config summary -----------------------------------------------
print_filter_config(filter_name, cfg, Q, R, F, dt, filt);

%% ---- Determine N_x (state dimension) ------------------------------------
[x0_est, ~] = filt.getGaussianEstimate();
N_x = numel(x0_est);

%% ---- Pre-allocate results -----------------------------------------------
x_est_all = NaN(N_x, n_k);
P_est_all = cell(n_k, 1);
ESS_all   = NaN(1, n_k);
has_map   = ismethod(filt, 'getMAPEstimate');
x_map_all = NaN(2, n_k);  % MAP estimate (position only, HMM-based filters)

[x_est_all(:,1), P_est_all{1}] = filt.getGaussianEstimate();
if has_map
    x_map_all(:,1) = filt.getMAPEstimate();
end

has_grid  = ismethod(filt, 'getMixtureGrid');
hmm_grids = [];
if has_grid
    hmm_grids        = zeros(128, 128, n_k);
    hmm_grids(:,:,1) = filt.getMixtureGrid();
end

% Standalone HMM filters (PDA_HMM, GNN_HMM): save grid directly from ptarget_prob
has_standalone_hmm = ismember(filter_name, HMM_families) && isprop(filt, 'ptarget_prob');
hmm_standalone_grid = [];
if has_standalone_hmm
    npx = filt.grid_size;
    hmm_standalone_grid        = zeros(npx, npx, n_k);
    hmm_standalone_grid(:,:,1) = reshape(full(filt.ptarget_prob), npx, npx);
end

% Per-particle snapshots (PF and RBPF families)
has_matrix_particles = isprop(filt, 'particles') && ~isempty(filt.particles) && isnumeric(filt.particles);
has_cell_particles   = isprop(filt, 'particles') && ~isempty(filt.particles) && iscell(filt.particles);
has_any_particles    = has_matrix_particles || has_cell_particles;
is_kf_rbpf_flag      = strcmp(filter_name, 'KF_RBPF');

particle_pos_all = [];
particle_w_all   = [];
particle_cov_all = [];
if has_any_particles
    N_p_snap         = filt.N_p;
    particle_pos_all = zeros(2, N_p_snap, n_k);
    particle_w_all   = zeros(N_p_snap, n_k);
    particle_pos_all(:,:,1) = re_extract_pos(filt, filter_name);
    particle_w_all(:,1)     = re_extract_w(filt);
    if is_kf_rbpf_flag
        particle_cov_all        = zeros(4, N_p_snap, n_k);
        particle_cov_all(:,:,1) = re_extract_cov_kfrbpf(filt);
    end
end

%% ---- Main tracking loop -------------------------------------------------
fprintf('Running filter over %d timesteps...\n', n_k);

for i = 2:n_k
    z_k = z_all{i};

    % Build measurement struct for composite-likelihood PF families
    if ismember(filter_name, {'PDA_PF','MC_PF'})
        meas_struct.det = z_k;
        meas_struct.mag = signal{i};
        filt.timestep(meas_struct, GT(:,i));
    elseif ismember(filter_name, HMM_families) || strcmp(filter_name, 'HMM_RBPF')
        filt.timestep(z_k, GT(1:2, i));
    else
        filt.timestep(z_k, GT(:,i));
    end

    [x_est_all(:,i), P_est_all{i}] = filt.getGaussianEstimate();
    if has_map
        x_map_all(:,i) = filt.getMAPEstimate();
    end
    if has_grid
        hmm_grids(:,:,i) = filt.getMixtureGrid();
    end
    if has_standalone_hmm
        hmm_standalone_grid(:,:,i) = reshape(full(filt.ptarget_prob), npx, npx);
    end
    if has_any_particles
        particle_pos_all(:,:,i) = re_extract_pos(filt, filter_name);
        particle_w_all(:,i)     = re_extract_w(filt);
        if is_kf_rbpf_flag
            particle_cov_all(:,:,i) = re_extract_cov_kfrbpf(filt);
        end
    end

    if isprop(filt, 'current_ESS')
        ESS_all(i) = filt.current_ESS;
    end

    if mod(i,10) == 0
        pos_err = norm(x_est_all(1:2,i) - GT(1:2,i));
        fprintf('  Step %3d/%d | pos error=%.4f m | ESS=%s\n', ...
            i, n_k, pos_err, ...
            iif(~isnan(ESS_all(i)), sprintf('%.0f', ESS_all(i)), 'N/A'));
    end
end

fprintf('Tracking complete.\n');

%% ---- Package results ----------------------------------------------------
result.filter_name  = filter_name;
result.GT           = GT;
result.x_est        = x_est_all;   % MMSE estimate
result.x_map        = x_map_all;   % MAP estimate (2xN, NaN if unsupported)
result.P_est        = P_est_all;
result.measurements = z_all;
result.ESS          = ESS_all;
result.cfg          = cfg;
result.params       = Data.params;
result.hmm_mixture          = hmm_grids;            % [128x128xN_k], [] if unsupported
result.hmm_standalone_grid  = hmm_standalone_grid;  % [npx x npx x N_k], HMM families only

% Per-particle data for animation
result.particle_pos_all  = particle_pos_all;   % [2 x N_p x N_k], [] if no particles
result.particle_w_all    = particle_w_all;     % [N_p x N_k]
result.particle_cov_all  = particle_cov_all;   % [4 x N_p x N_k], KF_RBPF only

% Identify winner particle after full run
if has_cell_particles
    % RBPF: winner = particle with most non-zero associations (fewest missed detections)
    det_counts = zeros(1, filt.N_p);
    for ii = 1:filt.N_p
        if isfield(filt.particles{ii}, 'association_history')
            det_counts(ii) = sum(filt.particles{ii}.association_history > 0);
        end
    end
    [~, winner_idx] = max(det_counts);
    result.winner_assoc_history = filt.particles{winner_idx}.association_history;
elseif has_matrix_particles
    % PF: winner = particle whose final position is closest to GT
    final_pos = particle_pos_all(:, :, end);
    dists = vecnorm(final_pos - GT(1:2, end), 2, 1);
    [~, winner_idx] = min(dists);
    result.winner_assoc_history = [];
else
    winner_idx = [];
    result.winner_assoc_history = [];
end
result.winner_idx = winner_idx;

% Save full particle ancestral trajectories for statetraj animation
particle_final_traj    = {};
particle_all_assoc_hist = {};
if has_cell_particles
    particle_final_traj    = cell(1, filt.N_p);
    particle_all_assoc_hist = cell(1, filt.N_p);
    for ii = 1:filt.N_p
        if isfield(filt.particles{ii}, 'state_trajectory') && ~isempty(filt.particles{ii}.state_trajectory)
            traj = filt.particles{ii}.state_trajectory;
            particle_final_traj{ii} = traj(1:min(2,size(traj,1)), :);
        end
        if isfield(filt.particles{ii}, 'association_history')
            particle_all_assoc_hist{ii} = filt.particles{ii}.association_history;
        end
    end
end
result.particle_final_traj    = particle_final_traj;
result.particle_all_assoc_hist = particle_all_assoc_hist;

% Save as 'R' so plot_experiment_results can load with load(...,'R')
R = result; %#ok<NASGU>

%% ---- Save ---------------------------------------------------------------
if ismember(filter_name, PF_families)
    fname = sprintf('%s_N%d.mat', filter_name, opt.N_particles);
else
    fname = sprintf('%s.mat', filter_name);
end
out_file = fullfile(script_dir, dataset_dir, fname);
save(out_file, 'R', '-mat');
fprintf('Results saved: %s\n', out_file);

%% ---- Quick error summary ------------------------------------------------
valid_steps = ~isnan(x_est_all(1,:));
pos_errors  = vecnorm(x_est_all(1:2, valid_steps) - GT(1:2, valid_steps));
fprintf('\nPosition error summary:\n');
fprintf('  Mean  = %.4f m\n', mean(pos_errors));
fprintf('  RMSE  = %.4f m\n', sqrt(mean(pos_errors.^2)));
fprintf('  Max   = %.4f m\n', max(pos_errors));

end % function

%% ---- Helper: config printout --------------------------------------------
function print_filter_config(filter_name, cfg, Q, R, F, dt, filt)
    sep = repmat('-', 1, 54);
    fprintf('\n%s\n', sep);
    fprintf('  Config: %s\n', filter_name);
    fprintf('%s\n', sep);

    % dt
    fprintf('  dt              = %.3f s\n', dt);

    % N_particles
    if isprop(filt, 'N_p')
        fprintf('  N_particles     = %d\n', filt.N_p);
    end

    % Q diagonal
    if ~isempty(Q)
        qd = diag(Q)';
        fprintf('  Q (diag)        = [%s]\n', ...
            strjoin(arrayfun(@(x) sprintf('%.2e', x), qd, 'UniformOutput', false), ', '));
    else
        fprintf('  Q               = N/A (grid-based filter)\n');
    end

    % R diagonal
    if ~isempty(R) && isfield(cfg, 'R')
        rd = diag(R)';
        fprintf('  R (diag)        = [%s]\n', ...
            strjoin(arrayfun(@(x) sprintf('%.2e', x), rd, 'UniformOutput', false), ', '));
    else
        fprintf('  R               = N/A\n');
    end

    % F model type
    if ~isempty(F)
        % Detect ZOH vs discrete by checking F(1,5): ZOH expm gives a slightly
        % different value than the analytic dt^2/2 when dt is small.
        if isfield(cfg, 'F') && norm(cfg.F - F, 'fro') < 1e-10
            F_src = 'expm (ZOH, const-accel)';
        else
            F_src = 'discrete const-accel';
        end
        fprintf('  F model         = %s  [F(1,3)=%.4f, F(1,5)=%.4f]\n', ...
            F_src, F(1,3), F(1,5));
    end

    % Validation gate
    if isfield(cfg, 'ValidationSigma')
        fprintf('  ValidationSigma = %.1f sigma\n', cfg.ValidationSigma);
    end

    % ESS threshold
    if isprop(filt, 'ESS_threshold_percentage')
        fprintf('  ESS_threshold   = %.0f%% of N_p\n', filt.ESS_threshold_percentage * 100);
    end

    % PD / PFA
    if isprop(filt, 'PD')
        fprintf('  PD              = %.3f\n', filt.PD);
    end
    if isprop(filt, 'PFA')
        fprintf('  PFA             = %.3f\n', filt.PFA);
    end

    % lambda_clutter (KF family only)
    if isprop(filt, 'lambda_clutter')
        fprintf('  lambda_clutter  = %.3f\n', filt.lambda_clutter);
    end

    % PF-specific
    if isprop(filt, 'hybrid_resample_fraction')
        fprintf('  hybrid_resample = %.2f\n', filt.hybrid_resample_fraction);
    end
    if isprop(filt, 'composite_likelihood')
        fprintf('  composite_like  = %d\n', filt.composite_likelihood);
    end
    if isprop(filt, 'association_strategy')
        fprintf('  assoc_strategy  = %s\n', filt.association_strategy);
    end

    fprintf('%s\n\n', sep);
end

%% ---- Helper: inline if --------------------------------------------------
function v = iif(cond, a, b)
    if cond, v = a; else, v = b; end
end

%% ---- Helpers: per-particle extraction -----------------------------------
function pos = re_extract_pos(filt, filter_name)
    if isnumeric(filt.particles)
        pos = filt.particles(1:2, :);
    else
        N_p = filt.N_p;
        pos = zeros(2, N_p);
        for ii = 1:N_p
            if strcmp(filter_name, 'KF_RBPF')
                pos(:, ii) = filt.particles{ii}.kf.x(1:2);
            elseif strcmp(filter_name, 'HMM_RBPF')
                [xi, ~] = filt.particles{ii}.hmm.getGaussianEstimate();
                pos(:, ii) = xi;
            end
        end
    end
end

function w = re_extract_w(filt)
    if isnumeric(filt.particles)
        w = filt.weights;
    else
        w = cellfun(@(p) p.weight, filt.particles)';
    end
end

function covs = re_extract_cov_kfrbpf(filt)
    N_p = filt.N_p;
    covs = zeros(4, N_p);
    for ii = 1:N_p
        P_i = filt.particles{ii}.kf.P(1:2, 1:2);
        covs(:, ii) = P_i(:);
    end
end
