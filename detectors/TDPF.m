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
function [detected_peaks] = TDPF(curr_peaks, prev_peak_score, threshold)
    % TDPF  Time Dependent Peak Finder
    %   detected_peaks = TDPF(curr_peaks, prev_peak_score, threshold) compares current peaks to
    %   previous peak scores to classify detections as new targets, existing targets,
    %   or persistent targets.

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
            updated_peak = [remaining_curr_peaks(idx, :) min(prev_peak_score(i, 3) + 1, 3)];
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
            detected_peaks = [detected_peaks; remaining_curr_peaks(i, :) 1];
        end

    end

end
