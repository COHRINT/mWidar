function filt = FilterFactory(cfg, x0, varargin)
% FILTERFACTORY  Construct and return an initialised DA_Filter subclass.
%
% DESCRIPTION:
%   Takes a FilterConfig struct and an initial state estimate and returns
%   a fully initialised filter object ready to accept timestep() calls.
%   Handles the loading of HMM precomputed matrices when needed.
%
% SYNTAX:
%   filt = FilterFactory(cfg, x0)
%   filt = FilterFactory(cfg, x0, P0)
%   filt = FilterFactory(cfg, x0, P0, A_transition)
%   filt = FilterFactory(cfg, x0, P0, A_transition, pointlikelihood_image)
%   filt = FilterFactory(cfg, x0, P0, A_transition, pointlikelihood_image, pointlikelihood_mag)
%
% INPUTS:
%   cfg  - FilterConfig struct (from FilterConfig())
%   x0   - Initial state estimate [N_x x 1]  (typically [x, y, vx, vy, ax, ay]')
%   P0   - (optional) Initial covariance [N_x x N_x].
%          Default: diag([0.05, 0.05, 0.5, 0.5, 2, 2])
%   A_transition          - (optional) HMM state transition matrix [npx2 x npx2].
%                           Required for GNN_HMM, PDA_HMM, HMM_RBPF.
%   pointlikelihood_image - (optional) Precomputed spatial likelihood lookup
%                           table [npx2 x npx2].  Required for HMM-family
%                           and hybrid-likelihood PF filters.
%   pointlikelihood_mag   - (optional) Precomputed magnitude likelihood table
%                           [npx2 x 2].  Required for PDA_PF/MC_PF composite
%                           likelihood mode.
%
% OUTPUTS:
%   filt - Initialised DA_Filter subclass object. All properties set from
%          cfg; store_full_history and Debug flags applied.
%
% EXAMPLE:
%   cfg  = FilterConfig('KF_RBPF', 'N_particles', 300);
%   x0   = [0.5; 1.0; 0; 0; 0; 0];
%   filt = FilterFactory(cfg, x0);
%   filt.timestep(z);
%
% See also FilterConfig, DA_Filter, KF_RBPF, HMM_RBPF, PDA_KF, GNN_KF

    % ------------------------------------------------------------------
    % Parse optional arguments
    % ------------------------------------------------------------------
    P0                    = [];
    A_transition          = [];
    pointlikelihood_image = [];
    pointlikelihood_mag   = [];

    if nargin >= 3, P0                    = varargin{1}; end
    if nargin >= 4, A_transition          = varargin{2}; end
    if nargin >= 5, pointlikelihood_image = varargin{3}; end
    if nargin >= 6, pointlikelihood_mag   = varargin{4}; end

    % Default initial covariance (moderate uncertainty in all 6 states)
    if isempty(P0)
        P0 = diag([0.05, 0.05, 0.5, 0.5, 2, 2]);
    end

    % Common constructor option pairs — only 'Debug' is universal.
    % 'DynamicPlot' is added per-filter below for constructors that accept it.
    common_opts      = {'Debug', cfg.Debug, 'DynamicPlot', cfg.DynamicPlot};
    common_opts_nodp = {'Debug', cfg.Debug};

    % ------------------------------------------------------------------
    % Construct filter
    % ------------------------------------------------------------------
    switch upper(cfg.filter_name)

        % ==============================================================
        case 'GNN_KF'
        % ==============================================================
            filt = GNN_KF(x0, P0, cfg.F, cfg.Q, cfg.R, cfg.H, common_opts{:}, ...
                          'ValidationSigma', cfg.ValidationSigma);

        % ==============================================================
        case 'PDA_KF'
        % ==============================================================
            % PDA_KF requires a pointlikelihood_mag table; pass empty if
            % magnitude likelihood is not being used.
            filt = PDA_KF(x0, P0, cfg.F, cfg.Q, cfg.R, cfg.H, ...
                          pointlikelihood_mag, common_opts{:}, ...
                          'ValidationSigma', cfg.ValidationSigma);
            filt.PD             = cfg.PD;
            filt.PG             = cfg.PG;
            filt.lambda_clutter = cfg.lambda_clutter;

        % ==============================================================
        case 'GNN_HMM'
        % ==============================================================
            if isempty(A_transition) || isempty(pointlikelihood_image)
                error('FilterFactory:MissingLikelihood', ...
                    'GNN_HMM requires A_transition (4th arg) and pointlikelihood_image (5th arg).');
            end
            filt = GNN_HMM(x0(1:2), A_transition, pointlikelihood_image, ...
                           common_opts{:}, 'ValidationSigma', cfg.ValidationSigma);
            filt.PD  = cfg.PD;
            filt.PFA = cfg.PFA;

        % ==============================================================
        case 'PDA_HMM'
        % ==============================================================
            if isempty(A_transition) || isempty(pointlikelihood_image)
                error('FilterFactory:MissingLikelihood', ...
                    'PDA_HMM requires A_transition (4th arg) and pointlikelihood_image (5th arg).');
            end
            filt = PDA_HMM(x0(1:2), A_transition, pointlikelihood_image, ...
                           common_opts{:}, 'ValidationSigma', cfg.ValidationSigma);
            filt.PD  = cfg.PD;
            filt.PFA = cfg.PFA;

        % ==============================================================
        case 'GNN_PF'
        % ==============================================================
            filt = GNN_PF(x0, cfg.N_particles, cfg.F, cfg.Q, cfg.H, ...
                          pointlikelihood_image, ...
                          common_opts{:}, ...
                          'ValidationSigma', cfg.ValidationSigma, ...
                          'ESSThreshold',    cfg.ESS_threshold);
            filt.PD  = cfg.PD;
            filt.PFA = cfg.PFA;

        % ==============================================================
        case 'PDA_PF'
        % ==============================================================
            filt = PDA_PF(x0, cfg.N_particles, cfg.F, cfg.Q, cfg.H, ...
                          pointlikelihood_image, pointlikelihood_mag, ...
                          common_opts{:}, ...
                          'ValidationSigma', cfg.ValidationSigma, ...
                          'ESSThreshold',    cfg.ESS_threshold);
            filt.PD  = cfg.PD;
            filt.PFA = cfg.PFA;

        % ==============================================================
        case 'MC_PF'
        % ==============================================================
            filt = MC_PF(x0, cfg.N_particles, cfg.F, cfg.Q, cfg.H, ...
                         pointlikelihood_image, pointlikelihood_mag, ...
                         common_opts{:}, ...
                         'ValidationSigma', cfg.ValidationSigma, ...
                         'ESSThreshold',    cfg.ESS_threshold);
            filt.PD  = cfg.PD;
            filt.PFA = cfg.PFA;

        % ==============================================================
        case 'KF_RBPF'
        % KF_RBPF(x0, N_particles, F, Q, H, R, ...) — no DynamicPlot support
        % ==============================================================
            filt = KF_RBPF(x0, cfg.N_particles, cfg.F, cfg.Q, cfg.H, cfg.R, ...
                           common_opts_nodp{:});
            filt.PD  = cfg.PD;
            filt.PFA = cfg.PFA;

        % ==============================================================
        case 'HMM_RBPF'
        % HMM_RBPF(x0, N_particles, A_transition, pointlikelihood_image, ...)
        % ==============================================================
            if isempty(A_transition) || isempty(pointlikelihood_image)
                error('FilterFactory:MissingLikelihood', ...
                    'HMM_RBPF requires A_transition (4th arg) and pointlikelihood_image (5th arg).');
            end
            filt = HMM_RBPF(x0, cfg.N_particles, A_transition, ...
                            pointlikelihood_image, common_opts_nodp{:}, ...
                            'UniformInit', false);
            filt.PD  = cfg.PD;
            filt.PFA = cfg.PFA;

        otherwise
            error('FilterFactory:UnknownFilter', ...
                'Unknown filter: ''%s''. Run FilterConfig() for valid names.', ...
                cfg.filter_name);
    end

    % ------------------------------------------------------------------
    % Apply shared DA_Filter properties from config
    % ------------------------------------------------------------------
    filt.store_full_history = cfg.store_full_history;

end
