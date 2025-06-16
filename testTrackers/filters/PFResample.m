% function [sampsOut, wtsOut] = PFResample(sampsIn, wtsIn)
%     nsampsIn = size(sampsIn, 2);

%     % Compute cumulative sum of weights
%     sampcdf = cumsum(wtsIn)';
%     sampcdf = sampcdf / sampcdf(end);  % Normalize to ensure sum is 1

%     % Draw uniform random numbers
%     urands = rand(1, nsampsIn);

%     % Find resampled indices
%     [~, indsampsout] = histc(urands, [0; sampcdf]);

%     % Select resampled particles
%     sampsOut = sampsIn(:, indsampsout);

%     % Set uniform weights
%     wtsOut = (1/nsampsIn) * ones(size(wtsIn));
% end

function [sampsOut, wtsOut] = PFResample(sampsIn, wtsIn)
    nsampsIn = size(sampsIn, 2);
    
    % First, check if weights are valid
    if any(isnan(wtsIn)) || any(isinf(wtsIn))
        warning('Invalid weights detected before processing');
        % Replace invalid weights with small values
        wtsIn(isnan(wtsIn) | isinf(wtsIn)) = eps;
    end
    
    % Add small epsilon to zero weights to prevent issues
    epsilon = eps;  % Using MATLAB's eps instead of 1e-100
    wtsIn = wtsIn + epsilon;
    wtsIn = wtsIn / sum(wtsIn);  % Renormalize after adding epsilon
    
    % Compute cumulative sum of weights
    sampcdf = cumsum(wtsIn)';

    % Draw uniform random numbers
    urands = rand(1, nsampsIn);
    
    % Use simple linear search instead of interp1
    indsampsout = zeros(1, nsampsIn);
    for i = 1:nsampsIn
        indsampsout(i) = find(sampcdf >= urands(i), 1, 'first');
        if isempty(indsampsout(i))
            indsampsout(i) = nsampsIn;
        end
    end
    
    % Select resampled particles
    sampsOut = sampsIn(:, indsampsout);
    
    % Set uniform weights
    wtsOut = (1/nsampsIn) * ones(size(wtsIn));
end