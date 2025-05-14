% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mWidar Simulator implementation in MATLAB
%
% Anthony La Barca
%
% h_maxima_peaks is a function implementing the H-maxima transformation for peak detection
%
% This function detects the regional maxima in the radar signal using the H-maxima
% transform. It suppresses maxima whose height is less than a specified threshold h.
% The function takes in the radar signal and returns a 2D binary matrix (the same
% size as the radar git signal), where each "1" in the matrix indicates a detected peak.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x, y] = h_maxima_peaks(signal, h_val)
% Detect peaks using H-maxima transform after Gaussian filtering

signal = mat2gray(signal);                  % Normalize
% signal_filt = imgaussfilt(signal, 2);       % Smooth signal

% Set default h_val if not provided 
if nargin < 2 || isempty(h_val)
    disp('h_val not provided, using default value of 0.05');
    h_val = 0.05;                               
end
% Check if h_val is a scalar
if numel(h_val) > 1
    error('h_val must be a scalar value.');
end
peaks = imhmax(signal, h_val);         % Suppress shallow peaks
peak_mask = imregionalmax(peaks);           % Local maxima
[y, x] = find(peak_mask);                   % Return coordinates
end
