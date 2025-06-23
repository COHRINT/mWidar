clear; clc; close all

% Do the main we discussed
addpath('matlab_src\DA_Track')
addpath('matlab_src\supplemental')
addpath('Kalman_Filters\Final_Test_Tracks\SingleObj')

load T4_parab.mat

%% This can all be temporary, change/clean up as needed

PLOT = 1;
GT = Data.GT;
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

Q = diag([0.001 0.001 0.01 0.01 0.1 0.1]);

R = 0.1*eye(2);

H = [1 0 0 0 0 0;
    0 1 0 0 0 0];

P0 = diag([0.1 0.1 0.25 0.25 0.5 0.5]);
performance{1}.x = GT(:,1); % Initial State
performance{1}.P = P0; % Initial State Covaraince

%current_class = GNN_KF(performance{1}.x, performance{1}.P, F, Q, R, H);
current_class = PDAF(performance{1}.x, performance{1}.P, F, Q, R, H);
% current_class = GNN_HMM(X, P0, Q, R, H, F);

%% State Estimation

for i = 2:n_k
    
    [X,P] = current_class.timestep(performance{i-1}.x, performance{i-1}.P,z{i});

    % update performance
    performance{i}.x = X;
    performance{i}.P = P;

end

%% Plotting

if PLOT
    mWidar_FilterPlot(performance,Data,0:dt:10) % Basic plotting function for now, should be able to work for all types of filters for basic trajectory/error plots
end