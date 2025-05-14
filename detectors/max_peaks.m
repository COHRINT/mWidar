% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mWidar Simulator implementation in MATLAB
%
% Anthony La Barca
%
% max_peaks is a wrapper for the maxpeaks2 function in MATLAB
%
% This function detects the maximum peaks in the radar signal
% using the peaks2.m function. The function takes in the radar signal and returns
% a 2D binary matrix (the same size as the radar signal), where each "1" in the matrix 
% indicates a detected peak. Assume single (signal is sizex x sizey x 1) signal, 
% where sizex and sizey are the dimensions of the signal. Return both integer values
% of indicies of the detected peaks and the binary matrix.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [detected_peaks] = max_peaks(signal, min_val)

% Find local maxima using the peaks2 function

% Min value is +3sigma above mean 
% min_val = mean(signal(:)) + std(signal(:));
[pvs,peak_xind,peak_yind] = peaks2(signal,'MinPeakHeight',min_val,'MinPeakDistance',4);

detected_peaks = [peak_yind, peak_xind];

end
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%