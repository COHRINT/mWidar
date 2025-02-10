% Main script for Robot Localization Particle Filter

% Set random seed for reproducibility
rng(1928374655)

% Simulation parameters
dt = 0.02;
tvec = dt:dt:100;
global RESAMPLE;
RESAMPLE = 0;

% Velocity profiles
whist = 0.2 * sin(0.1 * tvec);
qhist = 0.5 * ones(size(tvec));

% Initial state
x0true = [0.1; 0.1; 0];

% Process and measurement noise
Q = diag([0.1; 0.1; 0.05]);
R = 1.5;

% Landmarks
landmarks = [ ...
                 -1.2856 5.8519
            %  1.5420 2.1169
            %  -0.1104 1.7926
            %  4.2603 9.7480
             2.6365 12.9204
            %  -3.5036 7.7518
            %  -1.6228 10.2106
            %  -9.8876 1.2568
            %  2.1522 0.5491
             -7.3594 11.9139];

% Simulation parameters
nLandmarks = size(landmarks, 1);
nSamples = 5000;
nskip = 1;

% Initialize particles
particles = -0.25 + 0.5 * rand(3, nSamples);
weights = (1 / nSamples) * ones(1, nSamples);
particles_hist = zeros(3, nSamples, length(tvec) + 1);
weights_hist = zeros(1, nSamples, length(tvec) + 1);
particles_hist(:, :, 1) = particles;
weights_hist(1, :, 1) = weights;

% True trajectory simulation
xtruehist = zeros(3, length(tvec) + 1);
xtruehist(:, 1) = x0true;
zhist = nan(nLandmarks, length(tvec) + 1);

% Simulation loop
for kk = 1:length(tvec)
    % Simulate true trajectory
    v = mvnrnd(zeros(3, 1), Q, 1);
    xtruehist(1, kk + 1) = xtruehist(1, kk) + dt * qhist(kk) * cos(xtruehist(3, kk)) + dt * v(1);
    xtruehist(2, kk + 1) = xtruehist(2, kk) + dt * qhist(kk) * sin(xtruehist(3, kk)) + dt * v(2);
    xtruehist(3, kk + 1) = xtruehist(3, kk) + dt * whist(kk) + dt * v(3);

    % Simulate measurements
    for ll = 1:nLandmarks
        r = mvnrnd(0, R, 1);
        zhist(ll, kk + 1) = sqrt((landmarks(ll, 1) - xtruehist(1, kk + 1)) ^ 2 + ...
            (landmarks(ll, 2) - xtruehist(2, kk + 1)) ^ 2) + r;
        zhist(ll, kk + 1) = max(zhist(ll, kk + 1), 0);
    end

    %%compute effective sample size for IS
    % coeffvar = var(wsamplehist)/(mean(wsamplehist))^2;
    % ess = nSamples/(1+coeffvar);
    % Particle Filter Iteration: if mod(kk,10)==0, resample
    if mod(kk, 100) == 0
        % if mod(kk, 20) == 0
        RESAMPLE = 1;
    end

    [particles, weights] = PF_iteration( ...
        @(x, Q) RobotLocalizationmodel(x, Q, whist(kk)), ...
        particles, weights, Q, ...
        @(x) RobotLocalizationMeasurementModel(x, landmarks), ...
        zhist(:, kk + 1), R);

    particles_hist(:, :, kk + 1) = particles;
    weights_hist(:, :, kk + 1) = weights;
end

%
figure;
tl = tiledlayout(1, 2, "TileSpacing", "compact");

% First plot (Particle Trajectories)
nexttile(1);
hold on;
plot(xtruehist(1, :), xtruehist(2, :), 'b-', 'LineWidth', 2);
plot(landmarks(:, 1), landmarks(:, 2), 'm+', 'MarkerSize', 15);

% Preallocate plot handles
hParticles = scatter(particles_hist(1, :, 1), particles_hist(2, :, 1), 5, 'green', 'filled');
hTrue = plot(xtruehist(1, 1), xtruehist(2, 1), 'ro', 'MarkerSize', 10);
title('Particle Trajectories');
xlabel('X position');
ylabel('Y position');
xlim([-9, 9]);
ylim([-1, 14]);

% Second plot (Particle Weights - Histogram)
nexttile(2);
hold on;
hWeights = histogram(weights_hist(1, :, 1), 'Normalization', 'probability'); % Normalize to probability distribution
title('Particle Weights');
xlabel('Weight Value');
ylabel('Frequency');

gifFilename = "particle_filter_full_evolution.gif";
% Loop and update plots
for kk = 2:length(tvec)
    % Update trajectory plot
    nexttile(1);
    xlim([-9, 9]);
    ylim([-1, 14]);
    set(hParticles, 'XData', particles_hist(1, :, kk), 'YData', particles_hist(2, :, kk));
    set(hTrue, 'XData', xtruehist(1, kk), 'YData', xtruehist(2, kk));

    % Update weights histogram
    nexttile(2);
    delete(hWeights); % Remove the old histogram
    hWeights = histogram(weights_hist(1, :, kk), 'Normalization', 'probability', 'FaceColor', 'b'); % Recreate with new data
    xlim([-1e-6, 1e-3])


    sgtitle(['Particle Filter Evolution kk = ', num2str(kk)]);

    drawnow;
end
