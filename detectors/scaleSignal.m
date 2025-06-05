% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mWidar Simulator implementation in MATLAB
%
% Anthony La Barca
%
% Function to scale radar signals using different methods
%
% This function scales the input signal using one of several methods:
% - tanh: Hyperbolic tangent scaling
% - gaussian: Gaussian filter with normalization
% - none: Min-max normalization
% - linear: Min-max normalization (same as none)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function scaled_signal = scaleSignal(signal_orig, scaling_string, blur_sigma)
    % scaleSignal - Scale the input signal using the specified method
    %
    % Inputs:
    %   signal_orig - Original input signal
    %   scaling_string - String specifying the scaling method:
    %                    "tanh", "gaussian", "none", or "linear"
    %   blur_sigma - Sigma value for Gaussian blur (default = 2)
    %                (only used for "gaussian" scaling)
    %
    % Output:
    %   scaled_signal - The scaled signal
    
    % Set default blur_sigma if not provided
    if nargin < 3
        blur_sigma = 2;
    end
    
    % Normalize signal to zero mean and unit variance for tanh scaling
    signal_std = (signal_orig - mean(signal_orig(:))) / std(signal_orig(:));
    
    % Apply the specified scaling method
    if strcmp(scaling_string, "tanh")
        % Hyperbolic tangent scaling, normalized to [0, 1]
        scaled_signal = (tanh(signal_std) + 1) / 2;
        
    elseif strcmp(scaling_string, "gaussian")
        % Apply Gaussian blur then normalize to [0, 1]
        blurred = imgaussfilt(signal_orig, blur_sigma);
        scaled_signal = (blurred - min(blurred(:))) / ...
                       (max(blurred(:)) - min(blurred(:)));
        
    elseif strcmp(scaling_string, "none") || strcmp(scaling_string, "linear")
        % Simple min-max normalization to [0, 1]
        scaled_signal = (signal_orig - min(signal_orig(:))) / ...
                       (max(signal_orig(:)) - min(signal_orig(:)));
        
    else
        % Default case: warn and use min-max normalization
        warning('Unknown scaling method "%s", using min-max normalization.', scaling_string);
        scaled_signal = (signal_orig - min(signal_orig(:))) / ...
                       (max(signal_orig(:)) - min(signal_orig(:)));
    end
end
