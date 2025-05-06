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
function [detected_peaks] = TDD(signal, threshold)
    % Set default threshold to 0.5
    if nargin < 2
        threshold = 0.5;
    end

    % Get the size of the signal
    [rows, cols] = size(signal);

    % Initialize the detected peaks matrix
    detected_peaks = zeros(rows, cols);

    % Loop through each column of the signal
    for col = 1:cols
        % Get the current column of the signal
        current_col = signal(:, col);

        % Find the maximum peak in the current column
        [~, locs] = findpeaks(current_col, 'MinPeakHeight', threshold);

        % Set the detected peaks in the detected peaks matrix
        detected_peaks(locs, col) = 1;
    end

end
