% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mWidar Simulator implementation in MATLAB
%
% Anthony La Barca
%
% h_maximas is a function implementing the H-maxima transformation for peak detection
%
% This function detects the regional maxima in the radar signal using the H-maxima
% transform. It suppresses maxima whose height is less than a specified threshold h.
% The function takes in the radar signal and returns a 2D binary matrix (the same
% size as the radar signal), where each "1" in the matrix indicates a detected peak.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [detected_peaks] = h_maximas(signal, h)
    % H-maxima transformation for peak detection -- set default h to 0.2
    if nargin < 2
        h = 0.4 * max(signal(:)); % dynamic based on max value
    end

    % Convert to grayscale if the signal is RGB (3D)
    if ndims(signal) == 3
        disp("Converting to grayscale");
        gray_signal = -rgb2gray(signal);
    else
        gray_signal = -signal;
    end

    % Remove noise with Gaussian smoothing and morphological opening
    gray_signal = imgaussfilt(gray_signal, 1.0);  % extra smoothing
    gray_signal = imopen(gray_signal, strel('disk', 2));  % slightly larger structuring element

    % Apply h-maxima transform
    h_transformed = imhmax(gray_signal, h);

    % Find regional maxima (peaks) efficiently
    peaks_mask = imregionalmax(h_transformed);
    peaks_mask = bwareaopen(peaks_mask, 20);  % suppress small peak regions
    [peak_xind, peak_yind] = find(peaks_mask);
    detected_peaks = [peak_yind, peak_xind];
    % detected_peaks_mask = peaks_mask; % Uncomment if binary map needed

    disp("Detected " + numel(peak_yind) + " peaks");
    disp(detected_peaks);

    %    figure(2);
    %    tiledlayout(1,2);
    %    nexttile;
    %     imshow(signal, []);
    %     title("Original Signal");
    %     nexttile;
    %     hold on;
    %     imshow(h_transformed, []);
    %     plot(peak_yind, peak_xind, 'r*', 'MarkerSize', 5);
    %     title("H-maxima Transformed Signal");
    %     hold off;
end
