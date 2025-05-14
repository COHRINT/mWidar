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
%
% TODO: Implement the time-dependent detector
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [detected_peaks] = TDPF(curr_peaks, prev_peak_score, threshold)
    % TDPF  Time Dependent Peak Finder
    %   detected_peaks = TDPF(curr_peaks, prev_peak_score, threshold) compares current peaks to
    %   previous peak scores to classify detections as new targets, existing targets,
    %   or persistent targets.

    % Initialize the output
    detected_peaks = prev_peak_score;

    % Process each peak in prev_peak_score
    for i = 1:size(prev_peak_score, 1)

        if prev_peak_score(i, 3) <= 0
            continue; % Skip if the score is already zero
        end

        % Find the closest peak in curr_peaks
        distances = vecnorm(curr_peaks - prev_peak_score(i, 1:2), 2, 2);
        [min_dist, idx] = min(distances);

        if min_dist < threshold
            % Update the peak and increase its score
            detected_peaks(i, 1:2) = curr_peaks(idx, :);
            detected_peaks(i, 3) = detected_peaks(i, 3) + 1;
            curr_peaks(idx, :) = []; % Remove the matched peak
        else
            % Decrease the score if no match is found
            detected_peaks(i, 3) = detected_peaks(i, 3) - 1;
        end

    end

    % Add remaining peaks from curr_peaks as new detections
    for i = 1:size(curr_peaks, 1)
        detected_peaks = [detected_peaks; curr_peaks(i, :) 1];
    end

    % Adjust scores based on conditions
    if ~isempty(detected_peaks)
        detected_peaks(detected_peaks(:, 3) > 3, 3) = 3; % Cap scores at 3
        detected_peaks(detected_peaks(:, 3) <= 0, :) = []; % Remove peaks with score <= 0
    end

end
