% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mWidar Simulator implementation in MATLAB
%
% Anthony La Barca
%
% Function to calculate true positives (TP), false positives (FP), and
% false negatives (FN) for detector evaluation.
%
% This function compares detected peaks with ground truth points and
% determines match quality based on a distance threshold.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [TP, FP, FN] = calcTPFPFN(gt_points, detected_peaks, d_thresh_value)
    % calcTPFPFN - Calculate True Positives, False Positives, and False Negatives
    %
    % Inputs:
    %   gt_points - M×2 matrix of ground truth points [x, y]
    %   detected_peaks - N×2 (or N×K) matrix with first two columns as [x, y] coordinates
    %   d_thresh_value - Distance threshold for matching ground truth to detections
    %
    % Outputs:
    %   TP - Number of true positives (correctly detected points)
    %   FP - Number of false positives (incorrectly detected points)
    %   FN - Number of false negatives (missed ground truth points)

    % Handle empty cases
    if isempty(gt_points) && isempty(detected_peaks)
        TP = 0; FP = 0; FN = 0;
        return;
    elseif isempty(gt_points)
        TP = 0; FP = size(detected_peaks, 1); FN = 0;
        return;
    elseif isempty(detected_peaks)
        TP = 0; FP = 0; FN = size(gt_points, 1);
        return;
    end

    % Initialize match status arrays
    matched_gt = false(size(gt_points, 1), 1);
    matched_peaks = false(size(detected_peaks, 1), 1);

    % Match each detected peak to the nearest ground truth point
    for j = 1:size(detected_peaks, 1)
        % Find the closest GT point (using only x,y coordinates)
        dists = vecnorm(gt_points - detected_peaks(j, 1:2), 2, 2);
        [min_dist, idx] = min(dists);
        % disp(min_dist);

        if min_dist < d_thresh_value && ~matched_gt(idx)
            % If less than threshold pixels away and GT not matched, mark both as matched
            matched_gt(idx) = true;
            matched_peaks(j) = true;
        elseif min_dist < d_thresh_value && matched_gt(idx)
            % If less than threshold pixels away and GT already matched, just mark peak
            matched_peaks(j) = true;
        end
    end

    % Calculate metrics
    TP = sum(matched_gt);      % True positives: ground truth points that were detected
    FP = sum(~matched_peaks);  % False positives: detected peaks that don't match ground truth
    FN = sum(~matched_gt);     % False negatives: ground truth points that weren't detected
end
