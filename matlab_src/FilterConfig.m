function cfg = FilterConfig(filter_name, varargin)
% FILTERCONFIG  Return a default configuration struct for any filter.
%
% *** TO TUNE FILTER PARAMETERS, EDIT FilterHyperParams.m ***
%
% DESCRIPTION:
%   Builds a configuration struct for the requested filter using per-filter
%   hyperparameters from FilterHyperParams.m.  FilterFactory uses the
%   returned struct to construct and initialise the actual filter object.
%
%   Parameter flow:
%     FilterHyperParams.m  ──► FilterConfig.m  ──► FilterFactory.m  ──► filter object
%        (TUNE HERE)              (plumbing)          (construction)
%
%   This file should rarely need to be edited.  Add a new filter by:
%     1. Adding its hyperparameters to FilterHyperParams.m
%     2. Adding a case block in the switch below that populates cfg fields
%     3. Adding a case in FilterFactory.m that calls the constructor
%
% SYNTAX:
%   cfg = FilterConfig(filter_name)
%   cfg = FilterConfig(filter_name, 'Name', value, ...)
%
% INPUTS:
%   filter_name - String identifier: 'GNN_KF', 'PDA_KF', 'GNN_HMM',
%                 'PDA_HMM', 'GNN_PF', 'PDA_PF', 'MC_PF',
%                 'KF_RBPF', 'HMM_RBPF',
%                 'KF_RBPF_multi', 'PDA_PF_multi', 'HMM_RBPF_multi'
%
% OPTIONAL NAME-VALUE PAIRS (override FilterHyperParams defaults):
%   'dt'              - Timestep in seconds            (default: 1)
%   'N_particles'     - Number of particles            (from FilterHyperParams)
%   'PD'              - Probability of detection       (from FilterHyperParams)
%   'PFA'             - Probability of false alarm     (from FilterHyperParams)
%   'lambda_clutter'  - Clutter density                (from FilterHyperParams)
%   'ValidationSigma' - Gate size in sigma units       (from FilterHyperParams)
%   'ESS_threshold'   - Resampling threshold           (from FilterHyperParams)
%   'store_full_history' - Store complete state history (default: true)
%   'Debug'           - Enable verbose debug output    (default: false)
%   'DynamicPlot'     - Enable real-time plotting      (default: false)
%
% OUTPUTS:
%   cfg - Struct with all parameters needed by FilterFactory.
%
% See also FilterHyperParams, FilterFactory, DA_Filter

    % ------------------------------------------------------------------
    % Parse optional overrides
    % ------------------------------------------------------------------
    p = inputParser;
    p.KeepUnmatched = true;

    addRequired(p, 'filter_name', @(x) ischar(x) || isstring(x));

    % Common overrides (values default to [] so we can tell if user set them)
    addParameter(p, 'dt',                 1,     @isnumeric);
    addParameter(p, 'N_particles',        [],    @(x) isempty(x) || (isnumeric(x) && x > 0));
    addParameter(p, 'PD',                 [],    @(x) isempty(x) || (isnumeric(x) && x > 0 && x <= 1));
    addParameter(p, 'PFA',                [],    @(x) isempty(x) || (isnumeric(x) && x >= 0));
    addParameter(p, 'lambda_clutter',     [],    @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'ValidationSigma',    [],    @(x) isempty(x) || (isnumeric(x) && x > 0));
    addParameter(p, 'ESS_threshold',      [],    @(x) isempty(x) || (isnumeric(x) && x > 0 && x <= 1));
    addParameter(p, 'store_full_history', true,  @islogical);
    addParameter(p, 'Debug',              false, @islogical);
    addParameter(p, 'DynamicPlot',        false, @islogical);

    parse(p, filter_name, varargin{:});
    opt = p.Results;

    % ------------------------------------------------------------------
    % Load per-filter tuned hyperparameters from FilterHyperParams.m
    % ------------------------------------------------------------------
    hp = FilterHyperParams(char(filter_name));

    % Apply any varargin overrides on top of FilterHyperParams defaults
    N_particles    = pick(opt.N_particles,    hp.N_particles);
    PD             = pick(opt.PD,             hp.PD);
    PFA            = pick(opt.PFA,            hp.PFA);
    lambda_clutter = pick(opt.lambda_clutter, hp.lambda_clutter);
    ValidationSigma = pick(opt.ValidationSigma, hp.ValidationSigma);
    ESS_threshold  = pick(opt.ESS_threshold,  hp.ESS_threshold);

    % ------------------------------------------------------------------
    % Build system matrices (constant-acceleration, 6-DOF state)
    % x = [x, y, vx, vy, ax, ay]', z = [x, y]'
    % ------------------------------------------------------------------
    dt = opt.dt;

    % Analytic discrete constant-acceleration matrix
    F_disc = [1 0 dt 0  dt^2/2 0;
              0 1 0  dt 0      dt^2/2;
              0 0 1  0  dt     0;
              0 0 0  1  0      dt;
              0 0 0  0  1      0;
              0 0 0  0  0      1];

    % ZOH (matrix exponential) constant-acceleration matrix
    Fc_cont = [0 0 1 0 0 0;
               0 0 0 1 0 0;
               0 0 0 0 1 0;
               0 0 0 0 0 1;
               0 0 0 0 0 0;
               0 0 0 0 0 0];
    F_zoh = expm(Fc_cont * dt);

    % Select F based on per-filter recommendation from FilterHyperParams
    if strcmpi(hp.F_model, 'zoh')
        F = F_zoh;
    elseif strcmpi(hp.F_model, 'disc')
        F = F_disc;
    else
        F = [];  % HMM-family: no F
    end

    % Build Q from per-filter diagonal in FilterHyperParams
    if ~isempty(hp.Q_diag)
        Q = diag(hp.Q_diag);
    else
        Q = [];  % HMM-family
    end

    % Build R from per-filter sigma in FilterHyperParams
    if ~isempty(hp.R_sigma)
        R = hp.R_sigma^2 * eye(2);
    else
        R = [];  % HMM-family
    end

    % Measurement matrix: observe [x, y] directly
    H = [1 0 0 0 0 0;
         0 1 0 0 0 0];

    % ------------------------------------------------------------------
    % Build base config
    % ------------------------------------------------------------------
    cfg = struct();
    cfg.filter_name        = char(filter_name);
    cfg.Debug              = opt.Debug;
    cfg.DynamicPlot        = opt.DynamicPlot;
    cfg.store_full_history = opt.store_full_history;

    % ------------------------------------------------------------------
    % Filter-family specific fields
    % ------------------------------------------------------------------
    switch upper(cfg.filter_name)

        % ==============================================================
        case {'GNN_KF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.R               = R;
            cfg.H               = H;
            cfg.ValidationSigma = ValidationSigma;

        % ==============================================================
        case {'PDA_KF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.R               = R;
            cfg.H               = H;
            cfg.PD              = PD;
            cfg.PG              = hp.PG;
            cfg.lambda_clutter  = lambda_clutter;
            cfg.ValidationSigma = ValidationSigma;

        % ==============================================================
        case {'GNN_HMM', 'PDA_HMM'}
        % ==============================================================
            % HMM filters use precomputed grid matrices loaded externally.
            % FilterFactory loads A_transition and pointlikelihood_image
            % from supplemental .mat files and passes them to the constructor.
            cfg.PD               = PD;
            cfg.PFA              = PFA;
            cfg.ValidationSigma  = ValidationSigma;

        % ==============================================================
        case {'GNN_PF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.H               = H;
            cfg.N_particles     = N_particles;
            cfg.PD              = PD;
            cfg.PFA             = PFA;
            cfg.ESS_threshold   = ESS_threshold;
            cfg.ValidationSigma = ValidationSigma;

        % ==============================================================
        case {'PDA_PF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.H               = H;
            cfg.N_particles     = N_particles;
            cfg.PD              = PD;
            cfg.PFA             = PFA;
            cfg.lambda_clutter  = lambda_clutter;
            cfg.ESS_threshold   = ESS_threshold;
            cfg.ValidationSigma = ValidationSigma;

        % ==============================================================
        case {'MC_PF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.H               = H;
            cfg.N_particles     = N_particles;
            cfg.PD              = PD;
            cfg.PFA             = PFA;
            cfg.lambda_clutter  = lambda_clutter;
            cfg.ESS_threshold   = ESS_threshold;
            cfg.ValidationSigma = ValidationSigma;

        % ==============================================================
        case {'KF_RBPF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.R               = R;
            cfg.H               = H;
            cfg.N_particles     = N_particles;
            cfg.PD              = PD;
            cfg.PFA             = PFA;
            cfg.lambda_clutter  = lambda_clutter;

        % ==============================================================
        case {'HMM_RBPF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.N_particles     = N_particles;
            cfg.PD              = PD;
            cfg.PFA             = PFA;
            cfg.lambda_clutter  = lambda_clutter;
            % HMM grid matrices (A_transition, pointlikelihood_image) loaded
            % externally by FilterFactory from supplemental .mat files.

        % ==============================================================
        % MULTI-TARGET FILTERS
        % ==============================================================

        case {'KF_RBPF_MULTI'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.R               = R;
            cfg.H               = H;
            cfg.N_particles     = N_particles;
            cfg.PD              = PD;
            cfg.PFA             = PFA;
            cfg.lambda_clutter  = lambda_clutter;
            cfg.ESS_threshold   = ESS_threshold;

        case {'PDA_PF_MULTI'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.R               = R;
            cfg.H               = H;
            cfg.N_particles     = N_particles;
            cfg.PD              = PD;
            cfg.PFA             = PFA;
            cfg.lambda_clutter  = lambda_clutter;
            cfg.ESS_threshold   = ESS_threshold;
            cfg.ValidationSigma = ValidationSigma;

        case {'HMM_RBPF_MULTI'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.N_particles     = N_particles;
            cfg.PD              = PD;
            cfg.PFA             = PFA;
            cfg.lambda_clutter  = lambda_clutter;
            cfg.ESS_threshold   = ESS_threshold;

        otherwise
            error('FilterConfig:UnknownFilter', ...
                ['Unknown filter: ''%s''.\n', ...
                 'Valid options: GNN_KF, PDA_KF, GNN_HMM, PDA_HMM, GNN_PF,\n', ...
                 '               PDA_PF, MC_PF, KF_RBPF, HMM_RBPF,\n', ...
                 '               KF_RBPF_multi, PDA_PF_multi, HMM_RBPF_multi'], ...
                cfg.filter_name);
    end

end

% ------------------------------------------------------------------
% Local helper: return val if non-empty, else default
% ------------------------------------------------------------------
function v = pick(val, default)
    if isempty(val)
        v = default;
    else
        v = val;
    end
end
