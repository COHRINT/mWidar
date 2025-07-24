clear; clc; close all

% Add paths for MATLAB functions
addpath(fullfile('DA_Track'))
addpath(fullfile('supplemental'))
addpath(fullfile('supplemental', 'Final_Test_Tracks'))
addpath(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObj'))

load(fullfile('supplemental', 'recovery.mat'))
load(fullfile('supplemental', 'sampling.mat'))

load(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObj', 'T5_parab_noise.mat'))

INTERACTIVE = false; % Set to true for interactive plotting
DISTRIBUTION = ~INTERACTIVE; % Set to true for distribution plotting
DEBUG = false; % Set to true to enable debug output

% Filter configuration
filter_type = "HybridPF"; % Options: "KF", "HybridPF"
DA = "PDA"; % Data Association: "PDA", "GNN"

GT = Data.GT;
GT_meas = GT(1:2, :);
z = Data.y;
signal = Data.signal;

n_k = size(GT, 2);
performance = cell(1, n_k);

dt = 0.1; % sec
% Define KF Matrices state vector - {x,y,vx,vy,ax,ay}

% A = [0 0 1 0 0 0;
%     0 0 0 1 0 0;
%     0 0 0 0 1 0;
%     0 0 0 0 0 1;
%     0 0 0 0 0 0;
%     0 0 0 0 0 0];
%
% F = expm(A*dt);

% Use the direct discrete-time formulation (matches test_hybrid_PF)
F = [1, 0, dt, 0, dt ^ 2/2, 0;
     0, 1, 0, dt, 0, dt ^ 2/2;
     0, 0, 1, 0, dt, 0;
     0, 0, 0, 1, 0, dt;
     0, 0, 0, 0, 1, 0;
     0, 0, 0, 0, 0, 1];

Q = 1e-2 * eye(6);

R = 0.1 * eye(2);

H = [1 0 0 0 0 0;
     0 1 0 0 0 0];

P0 = diag([0.1 0.1 0.25 0.25 0.5 0.5]);
% performance{1}.x = GT(:,1); % Initial State
% performance{1}.P = P0; % Initial State Covaraince

% Load likelihood lookup table for PDA_PF

%current_class = GNN_KF(performance{1}.x, performance{1}.P, F, Q, R, H);
% current_class = PDAF(performance{1}.x, performance{1}.P, F, Q, R, H);

switch filter_type

    case "KF"
        fprintf("Using Kalman Filter ")

        if DA == "PDA"
            current_class = PDAF(GT(:, 1), P0, F, 100*Q, R, H);

        elseif DA == "GNN"
            current_class = GNN_KF(GT(:, 1), P0, F, Q, R, H);
        else
            error('Unknown data association method: %s', DA);
        end

        performance{1}.x = GT(:, 1); % Initial State
        performance{1}.P = P0; % Initial State Covariance

    case 'HybridPF'
        fprintf("Using Hybrid Particle Filter ")
        load(fullfile('supplemental', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');

        if DA == "PDA"
            fprintf("with PDA data association\n");
            current_class = PDA_PF(GT(:, 1), 1000, F, Q, H, pointlikelihood_image);
            current_class.debug = DEBUG;

            % Harder test for the HybridPF... set all particles to uniform over the state space
            % x in -2,2... y in 0,4... vx,vy = [-1, 1] m/s, ax,ay = [-0.5, 0.5] m/s^2
            % This is a more challenging test for the HybridPF, as it requires the filter to
            % effectively sample from a uniform distribution over the state space.
            % n_particles = size(current_class.particles, 2);
            % current_class.particles(1, :) = -2 + 4 * rand(1, n_particles); % x: uniform in [-2, 2]
            % current_class.particles(2, :) = 0 + 4 * rand(1, n_particles); % y: uniform in [0, 4]
            % current_class.particles(3, :) = -1 + 2 * rand(1, n_particles); % vx: uniform in [-1, 1] m/s
            % current_class.particles(4, :) = -1 + 2 * rand(1, n_particles); % vy: uniform in [-1, 1] m/s
            % current_class.particles(5, :) = -0.5 + 1 * rand(1, n_particles); % ax: uniform in [-0.5, 0.5] m/s^2
            % current_class.particles(6, :) = -0.5 + 1 * rand(1, n_particles); % ay: uniform in [-0.5, 0.5] m/s^2

        elseif DA == "GNN"
            fprintf("with GNN data association\n");
            current_class = GNN_PF(GT(:, 1), 1000, F, Q, H, pointlikelihood_image);
            current_class.debug = DEBUG;

        else
            error('Unknown data association method: %s', DA);
        end

        performance{1}.particles = current_class.particles; % Store initial particles
        performance{1}.weights = current_class.weights; % Store initial weights
        [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate(); % Initial Gaussian estimate
    otherwise
        error('Unknown filter type: %s', filter_type);
        exit(1);

end

%% State Estimation
for i = 2:n_k
    fprintf('Processing time step %d/%d\n', i, n_k);

    switch filter_type
        case "KF"
            current_meas = z{i}; % Use original measurements

            [X, P] = current_class.timestep(performance{i - 1}.x, performance{i - 1}.P, current_meas);

            % update performance
            performance{i}.x = X;
            performance{i}.P = P;
        case 'HybridPF'
            current_meas = z{i}; % Use original measurements

            % [X,P] = current_class.timestep(performance{i-1}.x, performance{i-1}.P,current_meas);
            current_class.timestep(current_meas);

            % Update performance
            performance{i}.particles = current_class.particles; % Store particles
            performance{i}.weights = current_class.weights; % Store weights

            % update performance
            [performance{i}.x, performance{i}.P] = current_class.getGaussianEstimate();

        otherwise
            error('Unknown filter type in timestep loop: %s', filter_type);
    end

end

%% Plotting

% initial_state.x0 = GT(:,1);
% initial_state.P0 = P0;
if INTERACTIVE
    mWidar_FilterPlot_Interactive(performance, Data, 0:dt:10, filter_type); % Interactive plotting function with slider
    fprintf('Interactive plot is ready! Use the slider and controls to navigate through timesteps.\n');
    fprintf('Close the figure window when you are done.\n');
elseif DISTRIBUTION
    mWidar_FilterPlot_Distribution(performance, Data, 0:dt:10, filter_type, fullfile("..", "figures", "DA_Track", "PDA_PF_animation_nogating.gif")); % Original plotting function for distribution
else
    fprintf('No plotting mode selected. Set INTERACTIVE or DISTRIBUTION to true.\n');
end

% NEES(current_class,initial_state,A,10,M,G)

% Keep the figure open for interaction
