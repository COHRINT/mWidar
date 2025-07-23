clear; clc; close all

% Add paths for MATLAB functions
addpath(fullfile('DA_Track'))
addpath(fullfile('supplemental'))
addpath(fullfile('supplemental', 'Final_Test_Tracks'))
addpath(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObj'))

load(fullfile('supplemental', 'recovery.mat'))
load(fullfile('supplemental', 'sampling.mat'))

load(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObj', 'T5_parab_noise.mat'))

%% This can all be temporary, change/clean up as needed

PLOT = 1;
GT = Data.GT;
GT_meas = GT(1:2, :);
z = Data.y;
signal = Data.signal;

n_k = size(GT,2);
performance = cell(1,n_k);

dt = 0.1; % sec

% Define KF Matrices state vector - {x,y,vx,vy,ax,ay}

A = [0 0 1 0 0 0;
    0 0 0 1 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0;
    0 0 0 0 0 0];

F = expm(A*dt);

Q = 1e-2*eye(6);

R = 0.1*eye(2);

H = [1 0 0 0 0 0;
    0 1 0 0 0 0];

P0 = diag([0.1 0.1 0.25 0.25 0.5 0.5]);
performance{1}.x = GT(:,1); % Initial State
performance{1}.P = P0; % Initial State Covaraince

%current_class = GNN_KF(performance{1}.x, performance{1}.P, F, Q, R, H);
current_class = PDAF(performance{1}.x, performance{1}.P, F, Q, R, H);



%% State Estimation

for i = 2:n_k
    % Get current measurements
    % current_meas = z{i};
    current_meas = GT_meas(:, i);

    [X,P] = current_class.timestep(performance{i-1}.x, performance{i-1}.P,current_meas);

    % update performance
    performance{i}.x = X;
    performance{i}.P = P;

end

%% Plotting

initial_state.x0 = GT(:,1);
initial_state.P0 = P0;


if PLOT
    mWidar_FilterPlot_Distribution(performance, Data, 0:dt:10, 'KF'); % Plotting function for distribution
    % mWidar_FilterPlot(performance,Data,0:dt:10) % Basic plotting function for now, should be able to work for all types of filters for basic trajectory/error plots
    % NEES(current_class,initial_state,A,10,M,G)

    pause(.1)
end