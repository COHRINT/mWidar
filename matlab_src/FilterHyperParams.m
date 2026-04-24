function params = FilterHyperParams(filter_name)
% FILTERHYPERPARAMS  THE ONE PLACE TO TUNE ALL FILTERS.
%
% DESCRIPTION:
%   Returns a struct of empirically tuned hyperparameters for the requested
%   filter.  FilterConfig calls this function to build Q, R, and F — so
%   editing this file is all that is needed when tuning any filter.
%
%   FilterConfig / FilterFactory provide the plumbing; this file has the
%   numbers.
%
% USAGE:
%   params = FilterHyperParams('KF_RBPF')
%   params = FilterHyperParams('PDA_PF')
%
% HOW TO TUNE:
%   1. Find your filter's case below.
%   2. Edit the parameter values in that section.
%   3. Run run_experiment (or test_3target for multi-target) to verify.
%
% -----------------------------------------------------------------------
% PARAMETER DESCRIPTIONS
% -----------------------------------------------------------------------
%
%   Q_diag (6x1 or 4x1)
%       Process noise diagonal [x, y, vx, vy, ax, ay].
%       Increase entry i to let the filter track faster changes in state i.
%       Decrease entry i to smooth out noise in state i.
%       Rule of thumb: ax/ay entries ≈ (max expected accel step)^2.
%       HMM-family filters have Q_diag = [] (grid-based, no state vector).
%
%   R_sigma (scalar, metres)
%       Measurement noise std dev. R = R_sigma^2 * eye(2).
%       Decrease to trust measurements more; increase to suppress noise.
%       HMM-family filters have R_sigma = [] (grid-based).
%
%   N_particles (integer)
%       Particle count for PF / RBPF families. More particles = more
%       accurate, but linearly slower. Start at 500; cut to 100 for tuning
%       sweeps, increase to 1000+ for final runs.
%       HMM memory is proportional to N_particles * N_grid, so keep small
%       (default 50) unless memory allows more.
%
%   PD (scalar in [0,1])
%       Probability that the target produces a detection.
%       Should match your sensor's actual empirical detection rate.
%
%   PFA (scalar in [0,1])
%       Probability of a false alarm per resolution cell (for PF/RBPF).
%       For KF families: clutter density lambda_clutter is used instead.
%
%   lambda_clutter (scalar)
%       Expected number of clutter returns per unit area per timestep.
%       Lower → filter assumes fewer false alarms, trusts nearby measurements more.
%       Tune: if filter ignores real measurements, lambda_clutter is too high.
%
%   ValidationSigma (scalar, sigma units)
%       Measurement gate size. A measurement is valid if its normalised
%       innovation score < ValidationSigma^2.
%       Too tight (2-sigma) → gate rejects measurements when particles drift,
%       causing total track loss with no recovery.
%       Default 5-sigma is safe; tighten only if clutter is very dense.
%
%   ESS_threshold (scalar in [0,1])
%       Resampling trigger: resample when ESS < ESS_threshold * N_p.
%       Lower → resample less often (more particle impoverishment risk).
%       Higher → resample more often (good diversity, but destroys history).
%
%   F_model ('zoh' | 'disc')
%       'zoh'  — F = expm(A*dt), zero-order hold (more accurate for KF families).
%       'disc' — F = analytic constant-accel discrete matrix (faster, fine for PF).
%
% -----------------------------------------------------------------------
% POST-CONSTRUCTION SETTINGS (set in run_experiment.m, not here)
% -----------------------------------------------------------------------
%   PDA_KF:   filt.lambda_clutter = 0.5  (override after construction)
%   PDA_PF:   filt.setDetectionModel(0.99, 0.25)
%             filt.hybrid_resample_fraction = 0.9
%             filt.composite_likelihood = false
%   KF_RBPF:  filt.association_strategy = 'likelihood'
%
% See also FilterConfig, FilterFactory, run_experiment

    switch upper(strtrim(filter_name))

        % ==============================================================
        % KALMAN FILTER FAMILY
        % ==============================================================

        case 'GNN_KF'
            % Larger position/velocity Q lets filter track maneuvers.
            % Too small → Kalman gain K ≈ 0, filter drifts on curves.
            params.Q_diag = [2e-4, 2e-4, ...   % x,  y    position
                             2e-3, 2e-3, ...   % vx, vy   velocity
                             2e-2, 2e-2];      % ax, ay   acceleration
            params.R_sigma        = 0.05;      % m — trust pixel meas tightly
            params.ValidationSigma = 2;
            params.F_model        = 'zoh';

        case 'PDA_KF'
            % Moderate Q so Kalman gain stays open.
            % Root cause of prior divergence: Q_pos=1e-4 + R=0.1 → K≈0.03,
            % filter nearly ignored measurements.
            % Keep R=0.1 — smaller R + larger Q → non-PD innovation covariance S.
            params.Q_diag = [5e-4, 5e-4, ...
                             2e-3, 2e-3, ...
                             5e-3, 5e-3];
            params.R_sigma        = 0.10;
            params.PD             = 0.95;
            params.PG             = 0.95;
            params.lambda_clutter = 0.5;       % TUNE: [0.1–5.0] lower = fewer clutter hits
            params.ValidationSigma = 2;
            params.F_model        = 'zoh';

        % ==============================================================
        % PARTICLE FILTER FAMILY
        % ==============================================================

        case {'GNN_PF', 'PDA_PF', 'MC_PF'}
            % Reference baseline: 0.11 m RMSE on T4_simulated_clutter.
            % Tightening Q to 5e-4 + ValidationSigma=2 caused total track
            % loss — gate rejected all measurements once particles drifted,
            % and there is no recovery mechanism without uniform exploration.
            params.Q_diag = [1e-3, 1e-3, ...
                             1e-2, 1e-2, ...
                             1e-1, 1e-1];
            params.R_sigma        = 0.10;
            params.N_particles    = 500;
            params.PD             = 0.95;
            params.PFA            = 0.05;
            params.lambda_clutter = 2.5;
            params.ValidationSigma = 5;        % TUNE: 5-sigma gate safe; 2-sigma risky
            params.ESS_threshold  = 0.2;
            params.F_model        = 'disc';

        % ==============================================================
        % RBPF FAMILY
        % ==============================================================

        case 'KF_RBPF'
            % Slightly open Q so embedded KFs track S-curves without
            % particle collapse (rapid ESS drop from uniform associations).
            params.Q_diag = [5e-4, 5e-4, ...
                             2e-3, 2e-3, ...
                             2e-3, 2e-3];
            params.R_sigma        = 0.05;
            params.N_particles    = 500;
            params.PD             = 0.95;
            params.PFA            = 0.05;
            params.lambda_clutter = 2.5;
            params.ESS_threshold  = 0.5;
            params.F_model        = 'zoh';

        case {'HMM_RBPF', 'HMM_RBPF_MULTI'}
            % Grid-based inner filters — no Q/R.  Keep N_particles small:
            % memory ~ N_p * N_grid (16384 for 128x128).
            params.Q_diag         = [];
            params.R_sigma        = [];
            params.N_particles    = 50;        % TUNE: memory-limited; 100 is comfortable
            params.PD             = 0.95;
            params.PFA            = 0.05;
            params.lambda_clutter = 2.5;
            params.ESS_threshold  = 0.5;
            params.F_model        = [];

        % ==============================================================
        % MULTI-TARGET EXTENSIONS
        % ==============================================================

        case 'KF_RBPF_MULTI'
            params.Q_diag = [5e-4, 5e-4, ...
                             2e-3, 2e-3, ...
                             2e-3, 2e-3];
            params.R_sigma        = 0.05;
            params.N_particles    = 500;
            params.PD             = 0.95;
            params.PFA            = 0.05;
            params.lambda_clutter = 2.5;
            params.ESS_threshold  = 0.5;
            params.F_model        = 'zoh';

        case 'PDA_PF_MULTI'
            params.Q_diag = [1e-3, 1e-3, ...
                             1e-2, 1e-2, ...
                             1e-1, 1e-1];
            params.R_sigma        = 0.10;
            params.N_particles    = 500;
            params.PD             = 0.95;
            params.PFA            = 0.05;
            params.lambda_clutter = 2.5;
            params.ValidationSigma = 5;
            params.ESS_threshold  = 0.2;
            params.F_model        = 'disc';

        % ==============================================================
        % HMM FAMILY (grid-based — no Q/R/F)
        % ==============================================================

        case {'GNN_HMM', 'PDA_HMM', 'HMM'}
            params.Q_diag         = [];
            params.R_sigma        = [];
            params.PD             = 0.95;
            params.PFA            = 0.05;
            params.ValidationSigma = 2;
            params.F_model        = [];

        otherwise
            params.Q_diag = [1e-3, 1e-3, 1e-2, 1e-2, 1e-1, 1e-1];
            params.R_sigma        = 0.10;
            params.N_particles    = 500;
            params.PD             = 0.95;
            params.PFA            = 0.05;
            params.lambda_clutter = 2.5;
            params.ValidationSigma = 2;
            params.ESS_threshold  = 0.5;
            params.F_model        = 'disc';
            warning('FilterHyperParams:Unknown', ...
                'No tuned parameters for ''%s'' — using generic defaults.', filter_name);
    end

    % Apply universal defaults for fields not set in a specific case
    if ~isfield(params, 'N_particles'),     params.N_particles    = 500;  end
    if ~isfield(params, 'PD'),              params.PD             = 0.95; end
    if ~isfield(params, 'PFA'),             params.PFA            = 0.05; end
    if ~isfield(params, 'lambda_clutter'),  params.lambda_clutter = 2.5;  end
    if ~isfield(params, 'ValidationSigma'), params.ValidationSigma = 2;   end
    if ~isfield(params, 'ESS_threshold'),   params.ESS_threshold  = 0.5;  end
    if ~isfield(params, 'F_model'),         params.F_model        = 'disc';end
    if ~isfield(params, 'PG'),              params.PG             = 0.95; end

end
