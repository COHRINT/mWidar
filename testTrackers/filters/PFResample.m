function [sampsOut, wtsOut] = PFResample(sampsIn, wtsIn)
% PFRESAMPLE Bootstrap resampling for particle filter
%   Implements systematic resampling to reduce particle degeneracy by
%   selecting particles with replacement according to their weights.
%
%   Inputs:
%     sampsIn - Input particles (state_dim x N_particles matrix)
%     wtsIn   - Input particle weights (1 x N_particles vector)
%
%   Outputs:
%     sampsOut - Resampled particles (same size as sampsIn)
%     wtsOut   - Uniform weights after resampling (1 x N_particles vector)
%
%   Algorithm:
%     1. Normalizes input weights to sum to 1
%     2. Computes cumulative distribution function
%     3. Draws uniform random samples
%     4. Selects particles based on inverse CDF sampling
%     5. Returns resampled particles with uniform weights
%
%   Example:
%     [particles_new, weights_new] = PFResample(particles, weights)
%
%   See also TEST_HYBRID_PF

% Author: Anthony La Barca
% Date: 2025-06-17
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