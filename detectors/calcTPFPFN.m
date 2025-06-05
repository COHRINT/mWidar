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
    %   detected_peaks - N×K matrix with first two columns as [x, y] coordinates
    %                   For TDPF, the third column contains persistence score
    %   d_thresh_value - Distance threshold for matching ground truth to detections
    %
    % Outputs:
    %   TP - Number of true positives (correctly detected points)
    %   FP - Number of false positives (incorrectly detected points)
    %   FN - Number of false negatives (missed ground truth points)

    % Initialize match status arrays
    matched_gt = false(size(gt_points, 1), 1);
    matched_peaks = false(size(detected_peaks, 1), 1);

    % For TDPF detector, skip detections with persistence score <= 2
    if size(detected_peaks, 2) >= 3
        for j = 1:size(detected_peaks, 1)
            if detected_peaks(j, 3) <= 2
                continue; % Skip if not persistent
            end

            % Find the closest GT point
            dists = vecnorm(gt_points - detected_peaks(j, 1:2), 2, 2);
            [min_dist, idx] = min(dists);

            if min_dist < d_thresh_value && ~matched_gt(idx)
                % If less than threshold pixels away and GT not matched, mark both as matched
                matched_gt(idx) = true;
                matched_peaks(j) = true;
            elseif min_dist < d_thresh_value && matched_gt(idx)
                % If less than threshold pixels away and GT already matched, just mark peak
                matched_peaks(j) = true;
            end
        end
    else
        % For non-TDPF detectors (standard peak detection)
        for j = 1:size(detected_peaks, 1)
            % Find the closest GT point
            dists = vecnorm(gt_points - detected_peaks(j, 1:2), 2, 2);
            [min_dist, idx] = min(dists);

            if min_dist < d_thresh_value && ~matched_gt(idx)
                % If less than threshold pixels away and GT not matched, mark both as matched
                matched_gt(idx) = true;
                matched_peaks(j) = true;
            elseif min_dist < d_thresh_value && matched_gt(idx)
                % If less than threshold pixels away and GT already matched, just mark peak
                matched_peaks(j) = true;
            end
        end
    end

    % Calculate metrics
    TP = sum(matched_gt);      % True positives: ground truth points that were detected
    FP = sum(~matched_peaks);  % False positives: detected peaks that don't match ground truth
    FN = sum(~matched_gt);     % False negatives: ground truth points that weren't detected
end
