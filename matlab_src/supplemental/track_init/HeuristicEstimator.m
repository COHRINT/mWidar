classdef HeuristicEstimator < TargetCountEstimator
% HEURISTICESTIMATOR  Sliding-window CA-CFAR + clustering target counter.
%
% Maintains a T-frame sliding window of mWidar signal frames. Once full,
% averages them, runs the detector+clusterer (detectClusters.m), and
% reports both the cluster count and centroids as candidate seed
% positions for new tracks.
%
% This is the heuristic estimator described in supplemental/probObjCt.m
% and demonstrated by signalAvgTest.m, exposed here as a stateful object
% with the standard TargetCountEstimator interface.
%
% USAGE:
%   est = HeuristicEstimator(cfg);    % cfg as built in probObjCt.m
%   for k = 1:K
%       [N_k, seeds, info] = est.update(signal{k}, []);
%   end

    properties
        cfg              % Detection / clustering config struct
        T                % Sliding window length (default cfg.T = 25)
        window           % Cell array {1 x T} of frames; [] entries unfilled
    end

    methods
        function obj = HeuristicEstimator(cfg)
            if ~isfield(cfg, 'T'), cfg.T = 25; end
            obj.cfg = cfg;
            obj.T   = cfg.T;
            obj.window = cell(1, obj.T);
            obj.ready = false;
        end

        function [N_est, seed_xy, info] = update(obj, signal_k, ~)
            info = struct();

            if isempty(signal_k)
                N_est = obj.N_est;
                seed_xy = obj.seed_xy;
                info.window_filled = false;
                return
            end

            % Shift window: newest in slot 1, push older entries right.
            filled_idx = find(~cellfun(@isempty, obj.window), 1, 'last');
            if isempty(filled_idx), filled_idx = 0; end
            if filled_idx > 0
                obj.window(2:min(obj.T, filled_idx + 1)) = ...
                    obj.window(1:min(obj.T - 1, filled_idx));
            end
            obj.window{1} = signal_k;
            window_filled = (filled_idx >= obj.T - 1);

            if ~window_filled
                N_est = NaN;
                seed_xy = zeros(2, 0);
                info.window_filled = false;
                obj.N_est = N_est;
                obj.seed_xy = seed_xy;
                return
            end

            avgSignal = mean(cat(3, obj.window{:}), 3);
            [count, cluster_xy] = estimateDetectionCount(avgSignal, obj.cfg);

            obj.N_est = count;
            obj.seed_xy = cluster_xy;
            obj.ready = true;

            N_est = count;
            seed_xy = cluster_xy;
            info.window_filled = true;
            info.avgSignal = avgSignal;
        end
    end
end
