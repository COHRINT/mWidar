% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mWidar Simulator implementation in MATLAB
%
% Anthony La Barca
%
% Time Dependent Detector
%
% This detector uses the time-dependent signal model to detect objects in the
% radar signal. Using the prior 3 frames of data, the detector classifies detections
% as either new targets (new target), existing targets (1 frame in past 3),
% or persistent targets (2 or more frames in past 3). The detector uses a threshold to
% determine if a detection is a new target, existing target, or persistent target.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [detected_peaks] = TDPF(curr_peaks, prev_peak_score, threshold, min_peak_distance)
    % TDPF  Time Dependent Peak Finder
    %   detected_peaks = TDPF(curr_peaks, prev_peak_score, threshold, min_peak_distance) 
    %   compares current peaks to previous peak scores to classify detections as 
    %   new targets, existing targets, or persistent targets.
    %
    %   Inputs:
    %       curr_peaks - N×2 array of current peak locations [x, y]
    %       prev_peak_score - M×3 array of previous peaks with scores [x, y, score]
    %       threshold - Distance threshold for matching peaks between frames
    %       min_peak_distance - Minimum distance between peaks (optional, default=1)
    %
    %   Outputs:
    %       detected_peaks - K×3 array of detected peaks [x, y, score]
    
    % Set default min_peak_distance if not provided
    if nargin < 4
        min_peak_distance = 1;
    end

    % Initialize the output as an empty array (will build it as we process)
    detected_peaks = [];

    % Handle the case when there are no previous peaks
    if isempty(prev_peak_score)
        % If there are current peaks, initialize them all with score 1
        if ~isempty(curr_peaks)
            detected_peaks = [curr_peaks ones(size(curr_peaks, 1), 1)];
        end

        return;
    end

    % Handle the case when there are no current peaks
    if isempty(curr_peaks)
        % Decrement all previous peaks and keep only those with score >= 1
        prev_peak_score(:, 3) = prev_peak_score(:, 3) - 1;
        detected_peaks = prev_peak_score(prev_peak_score(:, 3) >= 1, :);
        return;
    end

    % Create a copy of the current peaks that we'll modify
    remaining_curr_peaks = curr_peaks;

    % Process each peak in prev_peak_score
    for i = 1:size(prev_peak_score, 1)
        % Skip processing if no current peaks remain
        if isempty(remaining_curr_peaks)
            % Decrement the score for the remaining previous peaks
            prev_peak_score(i:end, 3) = prev_peak_score(i:end, 3) - 1;

            % Add only those with scores >= 1 to detected_peaks
            valid_indices = prev_peak_score(i:end, 3) >= 1;

            if any(valid_indices)
                detected_peaks = [detected_peaks; prev_peak_score(i:end, :)];
            end

            break;
        end

        % Find the closest peak in remaining_curr_peaks
        distances = vecnorm(remaining_curr_peaks - prev_peak_score(i, 1:2), 2, 2);
        [min_dist, idx] = min(distances);

        if min_dist < threshold
            % Update the peak and increase its score (capped at 3)
            updated_peak = [remaining_curr_peaks(idx, :) min(prev_peak_score(i, 3) + 1, 4)]; % CAP SCORE AT 4
            detected_peaks = [detected_peaks; updated_peak];

            % Remove the matched peak from current peaks
            remaining_curr_peaks(idx, :) = [];
        else
            % Decrease the score if no match is found
            new_score = prev_peak_score(i, 3) - 1;

            if new_score >= 1
                detected_peaks = [detected_peaks; prev_peak_score(i, 1:2) new_score];
            end

        end

    end

    % Add remaining peaks from curr_peaks as new detections with score 1
    if ~isempty(remaining_curr_peaks)

        for i = 1:size(remaining_curr_peaks, 1)
            detected_peaks = [detected_peaks; remaining_curr_peaks(i, :) 1]; % START NEW DETECTIONS AT SCORE 1
        end

    end

    % Cleanup -- remove peaks within min_peak_distance, keeping higher score
    % This prevents noise around true targets from building up scores
    if size(detected_peaks, 1) > 1
        to_remove = false(size(detected_peaks, 1), 1);
        for i = 1:size(detected_peaks, 1)-1
            if to_remove(i), continue; end
            for j = i+1:size(detected_peaks, 1)
                if to_remove(j), continue; end
                dist = norm(detected_peaks(i, 1:2) - detected_peaks(j, 1:2));
                if dist < min_peak_distance
                    % Keep the one with higher score (or earlier one if tied)
                    if detected_peaks(j, 3) > detected_peaks(i, 3)
                        to_remove(i) = true;
                    else
                        to_remove(j) = true;
                    end
                end
            end
        end
        detected_peaks(to_remove, :) = [];
    end

end
