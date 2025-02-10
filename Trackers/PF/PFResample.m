function [sampsOut, wtsOut] = PFResample(sampsIn, wtsIn)
    nsampsIn = size(sampsIn, 2);
    
    % Compute cumulative sum of weights
    sampcdf = cumsum(wtsIn)';
    sampcdf = sampcdf / sampcdf(end);  % Normalize to ensure sum is 1
    
    % Draw uniform random numbers
    urands = rand(1, nsampsIn);
    
    % Find resampled indices
    [~, indsampsout] = histc(urands, [0; sampcdf]);
    
    % Select resampled particles
    sampsOut = sampsIn(:, indsampsout);
    
    % Set uniform weights
    wtsOut = (1/nsampsIn) * ones(size(wtsIn));
end