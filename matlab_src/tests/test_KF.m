% Testing Kalman Filter implementation in MATLAB
clear; clc; close all;
% Add the DA_Track directory to the path
addpath('../DA_Track');

% Define initial state and covariance
x0 = [0; 0];
P0 = eye(2);

% Define system matrices
F = [1, 1; 0, 1]; % State transition matrix
Q = 0.1 * eye(2); % Process noise covariance
R = 0.5;          % Measurement noise covariance
H = [1, 0];       % Observation matrix

% Create Kalman Filter instance
kf = KF(x0, P0, F, Q, H, R);

% Example trajectory given these dynamics 
sim_times = 1:200;
state_simulation = zeros(2, size(sim_times,2));
state_simulation(:,1) = x0;
for t = 2:size(sim_times,2)
    state_simulation(:,t) = F * state_simulation(:,t-1) + mvnrnd([0;0], Q)';
end
measurements = H * state_simulation + mvnrnd(0, R, size(sim_times,2))';

% Run Kalman Filter through the measurements

x_estimates = zeros(2, size(sim_times,2));
P_estimates = zeros(2, 2, size(sim_times,2));
for t = 1:size(sim_times,2)
    [x_est, P_est] = kf.timestep(measurements(:,t));
    x_estimates(:,t) = x_est;
    P_estimates(:,:,t) = P_est;
end

% Plot results
figure;
subplot(2,1,1);
plot(sim_times, state_simulation(1,:), 'g-', 'DisplayName', 'True Position'); hold on;
plot(sim_times, measurements, 'rx', 'DisplayName', 'Measurements');
plot(sim_times, x_estimates(1,:), 'b-', 'DisplayName', 'KF Estimate');
xlabel('Time Step');
ylabel('Position');
legend;
title('Kalman Filter Position Estimation');

subplot(2,1,2);
plot(sim_times, state_simulation(2,:), 'g-', 'DisplayName', 'True Velocity'); hold on;
plot(sim_times, x_estimates(2,:), 'b-', 'DisplayName', 'KF Estimate');
xlabel('Time Step');
ylabel('Velocity');
legend;
title('Kalman Filter Velocity Estimation');

% Plot Errors
state_errors = state_simulation - x_estimates;
figure;
subplot(2,1,1);
plot(sim_times, state_errors(1,:), 'm-', 'DisplayName', 'Position Error');
hold on;
plot(sim_times, sqrt(squeeze(P_estimates(1,1,:)))', 'k--', 'DisplayName', 'Position Std Dev');
plot(sim_times, -sqrt(squeeze(P_estimates(1,1,:)))', 'k--', 'HandleVisibility','off');
xlabel('Time Step');
ylabel('Position Error');
legend;
title('Kalman Filter Position Estimation Error');

subplot(2,1,2);
plot(sim_times, state_errors(2,:), 'm-', 'DisplayName', 'Velocity Error');
hold on;
plot(sim_times, sqrt(squeeze(P_estimates(2,2,:)))', 'k--', 'DisplayName', 'Velocity Std Dev');
plot(sim_times, -sqrt(squeeze(P_estimates(2,2,:)))', 'k--', 'HandleVisibility','off');
xlabel('Time Step');
ylabel('Velocity Error');
legend;
title('Kalman Filter Velocity Estimation Error');


