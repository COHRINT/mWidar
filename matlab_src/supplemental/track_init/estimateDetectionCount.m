function [det_count, clustered_meas_xy] = estimateDetectionCount(avgSignal, cfg)
% ESTIMATEDETECTIONCOUNT  Heuristic count of detected target clusters.
%
%   [det_count, clustered_meas_xy] = estimateDetectionCount(avgSignal, cfg)
%
% Lifted from probObjCt.m (was a script-local function); now a top-level
% function so it can be called from estimator classes and other scripts.
% Behaviour matches the original (commit 04bf7d4) with one additive
% change: the cluster centroids are now returned as a second output.
% Callers that need only the count can ignore the second return value.

    signal_normalized = normalizeSignalFrame(avgSignal, cfg);

    [~, peak_x, peak_y] = CA_CFAR(signal_normalized(21:128,:), ...
        cfg.Pfa, cfg.Ng, cfg.Nr);
    peak_x = peak_x + cfg.crop_rows;

    if isempty(peak_x)
        det_count = 0;
        clustered_meas_xy = zeros(2, 0);
        return
    end

    pvinds = sub2ind([cfg.npx, cfg.npx], peak_x, peak_y);
    meas_xy = [cfg.pxgrid(pvinds)'; cfg.pygrid(pvinds)'];
    valid = meas_xy(2,:) >= 0.5 & signal_normalized(pvinds)' > cfg.intensity_thr;
    valid_meas_xy = meas_xy(:, valid);

    clustered_meas_xy = clusterNearbyDetections(valid_meas_xy, cfg.cluster_radius);
    det_count = size(clustered_meas_xy, 2);
end
