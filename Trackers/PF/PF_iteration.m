% PF Iteration Step: Propagate particles through the system model, and
% if there are measurements, update the weights of the particles.
% Inputs:
%   - model: the system model (function handle that takes in particles and Q)
%   - particles: the particles at the current time step (dim: n_states x n_particles)
%   - weights : the weights of the particles at the current time step (dim: n_particles x 1)
%   - Q: the process noise covariance matrix
%   - meas_model: the measurement model (function handle that takes in particles)
%   - measurements: the measurements at the current time step (dim: n_measurements x 1)
%       - if there are no measurements, measurements = [] or measurements = [NaN]
%   - R: the measurement noise covariance matrix
%
% Outputs:
%   - particles: the particles at the next time step (dim: n_states x n_particles)
%   - weights: the weights of the particles at the next time step (dim: n_particles x 1)
% SAMPLE FROM UKF_UPDATE.M
% function [x_est, P_est, Pyy, innovation, K] = UKF_update(f, Q, h, R, y, x_prev, P_prev, h_params)
function [particles, weights] = PF_iteration(model, particles, weights, Q, meas_model, measurements, R)

% Propagate particles through the system model
particles = model(particles, Q);

% If there are measurements, update the weights of the particles
if ~isempty(measurements)
    % Calculate predicted measurements
    predicted_measurements = meas_model(particles);

    % Compute likelihood for all particles and landmarks simultaneously
    landmark_likelihoods = 1 ./ (sqrt(2*pi*R)) .* ...
        exp(-0.5 * ((measurements - predicted_measurements).^2 / R));

    % Multiply likelihoods across landmarks
    likelihood = prod(landmark_likelihoods, 1);

    % Update weights
    weights = weights .* likelihood;
    weights = weights / sum(weights);

    global RESAMPLE
    if RESAMPLE % Resample every 10 steps
        [particles, weights] = PFResample(particles, weights);
        RESAMPLE = 0;
    end
end
end
