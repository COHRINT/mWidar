classdef (Abstract) TargetCountEstimator < handle
% TARGETCOUNTESTIMATOR  Abstract interface for online target-count estimation.
%
% Subclasses provide either a heuristic (signal-domain) or probabilistic
% (Bayesian) estimate of how many targets are currently present, plus
% candidate seed positions for newly-spawned tracks.
%
% Common signature:
%   [N_est, seed_xy, info] = update(signal_k, z_k)
%
% INPUTS:
%   signal_k - [npx x npx] mWidar frame at time k (or [] if not available)
%   z_k      - [N_z x N_meas] detections at time k (or [] if not available)
%
% OUTPUTS:
%   N_est    - integer estimated target count, or NaN during warm-up
%   seed_xy  - [2 x K_new] world-space candidate positions for new targets
%              (empty if N_est <= previous count, or estimator has none)
%   info     - struct with estimator-specific diagnostics (posterior,
%              cluster IDs, window state, ...)

    properties
        N_est       = NaN   % Last estimate
        seed_xy     = []    % Last candidate seed positions [2 x K]
        ready       = false % True once estimator has produced its first estimate
    end

    methods (Abstract)
        [N_est, seed_xy, info] = update(obj, signal_k, z_k)
    end

end
