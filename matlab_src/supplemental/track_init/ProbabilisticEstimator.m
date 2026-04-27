classdef ProbabilisticEstimator < TargetCountEstimator
% PROBABILISTICESTIMATOR  Recursive Bayesian object-count filter.
%
% Per-frame Bayesian recursion over a Poisson birth/death process:
%   prediction:  P(N_k | m_{1:k-1}) = T * P(N_{k-1} | m_{1:k-1})
%   update:      P(N_k | m_{1:k})  proportional to P(m_k | N_k) * pred
%
% The likelihood table P(m | N) is loaded from probObjCt_results.mat
% (built once by probObjCt.m). The transition matrix is built via
% buildPoissonCountTransition.
%
% Seed positions: this estimator only knows the count, not where targets
% are. When asked for seeds (i.e. when a signal frame is available and
% the count just went up), it delegates to a HeuristicEstimator (single-
% frame mode, T=1) to get cluster centroids from the current image.
%
% USAGE:
%   est = ProbabilisticEstimator(P_m_given_N, ...
%               'NVals', 0:5, 'MAxis', 0:7, ...
%               'LambdaArrival', 0.06, 'LambdaDepart', 0.05, ...
%               'SeedCfg', cfg);
%   for k = 1:K
%       [N_k, seeds, info] = est.update(signal{k}, z{k});
%   end

    properties
        P_m_given_N           % [n_states x n_obs] likelihood table
        N_vals                % Row vector of state values, e.g. 0:5
        m_axis                % Row vector of obs values,  e.g. 0:7
        T_transition          % [n_states x n_states] state transition
        P_prev                % [n_states x 1] last posterior
        post_hist             % [n_states x K] full posterior log
        N_map_seq             % [1 x K] MAP sequence
        alpha_hist            % [1 x K] log-evidence
        k_internal = 0        % frames processed
        seed_helper           % HeuristicEstimator with T=1 (single-frame clusterer)
    end

    methods
        function obj = ProbabilisticEstimator(P_m_given_N, varargin)
            p = inputParser;
            n_states = size(P_m_given_N, 1);
            n_obs    = size(P_m_given_N, 2);
            addParameter(p, 'NVals', 0:(n_states-1), @isnumeric);
            addParameter(p, 'MAxis', 0:(n_obs-1),    @isnumeric);
            addParameter(p, 'LambdaArrival', 0.06,   @isnumeric);
            addParameter(p, 'LambdaDepart',  0.05,   @isnumeric);
            addParameter(p, 'Prior', [],             @isnumeric);
            addParameter(p, 'SeedCfg', [],           @(x) isempty(x) || isstruct(x));
            parse(p, varargin{:});

            obj.P_m_given_N = P_m_given_N;
            obj.N_vals      = p.Results.NVals(:).';
            obj.m_axis      = p.Results.MAxis(:).';
            obj.T_transition = buildPoissonCountTransition( ...
                obj.N_vals, p.Results.LambdaArrival, p.Results.LambdaDepart);

            if isempty(p.Results.Prior)
                obj.P_prev = ones(n_states, 1) / n_states;
            else
                prior = p.Results.Prior(:);
                obj.P_prev = prior / sum(prior);
            end

            obj.post_hist  = zeros(n_states, 0);
            obj.N_map_seq  = zeros(1, 0);
            obj.alpha_hist = zeros(1, 0);

            % Single-frame heuristic clusterer used only to produce seeds
            % when count goes up. SeedCfg should match the cfg used by
            % HeuristicEstimator if both are in play.
            if ~isempty(p.Results.SeedCfg)
                seed_cfg = p.Results.SeedCfg;
                seed_cfg.T = 1;   % no averaging, react immediately
                obj.seed_helper = HeuristicEstimator(seed_cfg);
            else
                obj.seed_helper = [];
            end
        end

        function [N_est, seed_xy, info] = update(obj, signal_k, z_k)
            info = struct();

            % --- Bayes step on detection count m_k = size(z_k, 2) ---
            if isempty(z_k)
                m_k = 0;
            else
                m_k = size(z_k, 2);
            end
            mk_col = find(obj.m_axis == m_k, 1, 'first');
            if isempty(mk_col)
                mk_col = max(1, min(numel(obj.m_axis), ...
                    m_k - obj.m_axis(1) + 1));
            end

            likelihood_k  = obj.P_m_given_N(:, mk_col);
            pred_k        = obj.T_transition * obj.P_prev;
            unnorm_post_k = likelihood_k .* pred_k;
            s = sum(unnorm_post_k);

            if s > 0
                P_post = unnorm_post_k / s;
                alpha  = 1 / s;
            else
                P_post = obj.P_prev;
                alpha  = NaN;
            end

            obj.k_internal = obj.k_internal + 1;
            obj.post_hist(:, end+1)  = P_post;
            obj.alpha_hist(end+1)    = alpha;
            obj.P_prev = P_post;

            [~, map_idx] = max(P_post);
            N_est = obj.N_vals(map_idx);
            obj.N_map_seq(end+1) = N_est;

            N_prev_est = obj.N_est;
            obj.N_est = N_est;
            obj.ready = true;

            % --- Produce seed positions if count just increased and we
            %     have an image + a seed-helper available ---
            seed_xy = zeros(2, 0);
            if ~isempty(obj.seed_helper) && ~isempty(signal_k) && ...
                    ~isnan(N_prev_est) && N_est > N_prev_est
                [~, candidate_xy, ~] = obj.seed_helper.update(signal_k, z_k);
                seed_xy = candidate_xy;
            elseif ~isempty(obj.seed_helper) && ~isempty(signal_k)
                % Keep helper's window warm even when count is steady
                obj.seed_helper.update(signal_k, z_k);
            end
            obj.seed_xy = seed_xy;

            info.posterior = P_post;
            info.alpha     = alpha;
            info.m_k       = m_k;
        end
    end
end
