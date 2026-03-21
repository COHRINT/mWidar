function cfg = FilterConfig(filter_name, varargin)
% FILTERCONFIG  Return a default configuration struct for any filter.
%
% DESCRIPTION:
%   Central factory for filter parameter structs. Organises defaults around
%   the 3x3 DA x Tracking matrix:
%
%       DA \ Tracking |   KF    |   HMM   |   PF
%       --------------|---------|---------|--------
%       GNN           | GNN_KF  | GNN_HMM | GNN_PF
%       PDA           | PDA_KF  | PDA_HMM | PDA_PF
%       MC (sampling) |   —     |    —    | MC_PF
%       RBPF hybrid   | KF_RBPF |HMM_RBPF |   —
%
% SYNTAX:
%   cfg = FilterConfig(filter_name)
%   cfg = FilterConfig(filter_name, 'Name', value, ...)
%
% INPUTS:
%   filter_name - String identifier: 'GNN_KF', 'PDA_KF', 'GNN_HMM',
%                 'PDA_HMM', 'GNN_PF', 'PDA_PF', 'MC_PF',
%                 'KF_RBPF', 'HMM_RBPF'
%
% OPTIONAL NAME-VALUE PAIRS (override any default):
%   'dt'              - Timestep in seconds            (default: 1)
%   'N_particles'     - Number of particles            (default: 500)
%   'PD'              - Probability of detection       (default: 0.95)
%   'PFA'             - Probability of false alarm     (default: 0.05)
%   'lambda_clutter'  - Clutter density                (default: 2.5)
%   'store_full_history' - Store complete state history (default: true)
%   'Debug'           - Enable verbose debug output    (default: false)
%   'DynamicPlot'     - Enable real-time plotting      (default: false)
%
% OUTPUTS:
%   cfg - Struct with all parameters needed by FilterFactory to build
%         the requested filter. Fields vary by filter family (see below).
%
% COMMON CFG FIELDS (all filters):
%   cfg.filter_name        - Echo of input filter_name string
%   cfg.Debug              - Debug flag
%   cfg.DynamicPlot        - Real-time plot flag
%   cfg.store_full_history - History verbosity flag
%
% KF FAMILY (GNN_KF, PDA_KF) ADDITIONAL FIELDS:
%   cfg.dt, cfg.F, cfg.Q, cfg.R, cfg.H
%   cfg.PD, cfg.PFA, cfg.lambda_clutter (PDA_KF only)
%
% HMM FAMILY (GNN_HMM, PDA_HMM) ADDITIONAL FIELDS:
%   cfg.PD, cfg.PFA, cfg.ValidationSigma
%   (A_transition and pointlikelihood_image are loaded separately — see FilterFactory)
%
% PF FAMILY (GNN_PF, PDA_PF, MC_PF) ADDITIONAL FIELDS:
%   cfg.dt, cfg.F, cfg.Q, cfg.H
%   cfg.N_particles, cfg.PD, cfg.PFA
%   cfg.ESS_threshold, cfg.ValidationSigma
%
% RBPF FAMILY (KF_RBPF, HMM_RBPF) ADDITIONAL FIELDS:
%   cfg.dt, cfg.F, cfg.Q, cfg.R, cfg.H
%   cfg.N_particles, cfg.PD, cfg.PFA, cfg.lambda_clutter
%
% EXAMPLE:
%   cfg = FilterConfig('KF_RBPF', 'N_particles', 300, 'Debug', true);
%   filt = FilterFactory(cfg, x0, Data);
%
% See also FilterFactory, DA_Filter

    % ------------------------------------------------------------------
    % Parse optional overrides
    % ------------------------------------------------------------------
    p = inputParser;
    p.KeepUnmatched = true; % Forward filter-specific options

    addRequired(p, 'filter_name', @(x) ischar(x) || isstring(x));

    % Common overrides
    addParameter(p, 'dt',                 1,     @isnumeric);
    addParameter(p, 'N_particles',        500,   @(x) isnumeric(x) && x > 0);
    addParameter(p, 'PD',                 0.95,  @(x) isnumeric(x) && x > 0 && x <= 1);
    addParameter(p, 'PFA',                0.05,  @(x) isnumeric(x) && x >= 0);
    addParameter(p, 'lambda_clutter',     2.5,   @isnumeric);
    addParameter(p, 'ValidationSigma',    2,     @(x) isnumeric(x) && x > 0);
    addParameter(p, 'ESS_threshold',      0.5,   @(x) isnumeric(x) && x > 0 && x <= 1);
    addParameter(p, 'store_full_history', true,  @islogical);
    addParameter(p, 'Debug',              false, @islogical);
    addParameter(p, 'DynamicPlot',        false, @islogical);

    parse(p, filter_name, varargin{:});
    opt = p.Results;

    % ------------------------------------------------------------------
    % Shared system model (constant-acceleration, 6-DOF state)
    % x = [x, y, vx, vy, ax, ay]', z = [x, y]'
    % ------------------------------------------------------------------
    dt = opt.dt;

    F = [1 0 dt 0  dt^2/2 0;
         0 1 0  dt 0      dt^2/2;
         0 0 1  0  dt     0;
         0 0 0  1  0      dt;
         0 0 0  0  1      0;
         0 0 0  0  0      1];

    % Process noise — state is [x,y,vx,vy,ax,ay], noise drives ax and ay independently
    sigma_ax = 0.25; % m/s^2 std
    sigma_ay = 0.25;
    % Separate [6x1] noise-input vectors for each axis
    Gx = [dt^2/2; 0; dt; 0; 1; 0];   % affects x, vx, ax
    Gy = [0; dt^2/2; 0; dt; 0; 1];   % affects y, vy, ay
    Q  = sigma_ax^2 * (Gx * Gx') + sigma_ay^2 * (Gy * Gy');

    % Measurement noise — pixel-to-meter uncertainty for mWidar 128x128
    sigma_meas = 0.05; % metres
    R = sigma_meas^2 * eye(2);

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
            cfg.dt = dt;
            cfg.F  = F;
            cfg.Q  = Q;
            cfg.R  = R;
            cfg.H  = H;

        % ==============================================================
        case {'PDA_KF'}
        % ==============================================================
            cfg.dt             = dt;
            cfg.F              = F;
            cfg.Q              = Q;
            cfg.R              = R;
            cfg.H              = H;
            cfg.PD             = opt.PD;
            cfg.PG             = 0.95;  % Gate probability
            cfg.lambda_clutter = opt.lambda_clutter;

        % ==============================================================
        case {'GNN_HMM', 'PDA_HMM'}
        % ==============================================================
            % HMM filters use precomputed grid matrices loaded externally.
            % FilterFactory loads A_transition and pointlikelihood_image
            % from supplemental .mat files and passes them to the constructor.
            cfg.PD               = opt.PD;
            cfg.PFA              = opt.PFA;
            cfg.ValidationSigma  = opt.ValidationSigma;

        % ==============================================================
        case {'GNN_PF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.H               = H;
            cfg.N_particles     = opt.N_particles;
            cfg.PD              = opt.PD;
            cfg.PFA             = opt.PFA;
            cfg.ESS_threshold   = opt.ESS_threshold;
            cfg.ValidationSigma = opt.ValidationSigma;

        % ==============================================================
        case {'PDA_PF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.H               = H;
            cfg.N_particles     = opt.N_particles;
            cfg.PD              = opt.PD;
            cfg.PFA             = opt.PFA;
            cfg.lambda_clutter  = opt.lambda_clutter;
            cfg.ESS_threshold   = opt.ESS_threshold;
            cfg.ValidationSigma = opt.ValidationSigma;

        % ==============================================================
        case {'MC_PF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.H               = H;
            cfg.N_particles     = opt.N_particles;
            cfg.PD              = opt.PD;
            cfg.PFA             = opt.PFA;
            cfg.lambda_clutter  = opt.lambda_clutter;
            cfg.ESS_threshold   = opt.ESS_threshold;
            cfg.ValidationSigma = opt.ValidationSigma;

        % ==============================================================
        case {'KF_RBPF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.R               = R;
            cfg.H               = H;
            cfg.N_particles     = opt.N_particles;
            cfg.PD              = opt.PD;
            cfg.PFA             = opt.PFA;
            cfg.lambda_clutter  = opt.lambda_clutter;

        % ==============================================================
        case {'HMM_RBPF'}
        % ==============================================================
            cfg.dt              = dt;
            cfg.F               = F;
            cfg.Q               = Q;
            cfg.N_particles     = opt.N_particles;
            cfg.PD              = opt.PD;
            cfg.PFA             = opt.PFA;
            cfg.lambda_clutter  = opt.lambda_clutter;
            % HMM grid matrices (A_transition, pointlikelihood_image) loaded
            % externally by FilterFactory from supplemental .mat files.

        otherwise
            error('FilterConfig:UnknownFilter', ...
                'Unknown filter: ''%s''. Valid options: GNN_KF, PDA_KF, GNN_HMM, PDA_HMM, GNN_PF, PDA_PF, MC_PF, KF_RBPF, HMM_RBPF', ...
                cfg.filter_name);
    end

end
