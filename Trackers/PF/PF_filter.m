% PF_filter: Wrapper for post-simulation filtering of measurements
% Inputs:
%   - model: the system model (function handle that takes in particles and Q)
%   - particles: the particles at the current time step (dim: n_states x n_particles x time)
%   - weights : the weights of the particles at the current time step (dim: n_particles x time)
%   - Q: the process noise covariance matrix
%   - meas_model: the measurement model (function handle that takes in particles)
%   - measurements: the measurements at the current time step (dim: n_measurements x time)
%       - If there are not measurements for every timestep, non-measurements will be full of NaNs
%   - R: the measurement noise covariance matrix
%
% Outputs:
%   - particles: the particles at the next time step (dim: n_states x n_particles x time)
% EXAMPLE FROM UKF_FILTER
% function [xest, Pest, Pyy, innovation, K] = UKF_filter(f, q, h, r, y, x0, P0, h_params)
function particles = PF_filter(model, particles, weights, Q, meas_model, measurements, R)
    % Initialize variables
    n_states = size(particles, 1);
    n_particles = size(particles, 2);
    n_time = size(particles, 3);
    n_measurements = size(measurements, 1);

    weights_history = zeros(n_particles, n_time);

    % Loop through each time step
    for t = 1:n_time
        % Use the PF_iteration.m function to update the particles and the weights
        [particles(: ,:, ), weights(:, t)] = PF_iteration(model, particles(:, :, t), weights(:, t), Q, meas_model, measurements(:, t), R);
    end
end