%%%%%%%%%%%% Test script for validating JPDAF %%%%%%%%%%%%%%%%%%%%%
clear; clc; close all

addpath(fullfile('DA_Track'))
load(fullfile('supplemental', 'Final_Test_Tracks', 'MultObj', 'JPDAF_test_traj_3.mat'))



%% Initialize KF Matrices

P0 = diag([0.25 0.25 0.5 0.5 0.5 0.5]); % State Cov, constant for each obj

% State Transition Matrix
A = [0 1 0 0 0 0;    % dx/dt = vx
    0 0 1 0 0 0;     % dvx/dt = ax  
    0 0 0 0 0 0;     % dax/dt = 0 (constant acceleration)
    0 0 0 0 1 0;     % dy/dt = vy
    0 0 0 0 0 1;     % dvy/dt = ay
    0 0 0 0 0 0];    % day/dt = 0 (constant acceleration)

dt = 0.1;

F = expm(A*dt);

% Process Noise
Q = 0.001*eye(6);
% Meas Noise
R = 0.1*eye(2);
% Meas Model
H = [1 0 0 0 0 0;
    0 1 0 0 0 0];
% PD -> Probability of detecting an object
PD = 0.9;
% t -> Number of tracks, assumed known
t = size(Data.GT,2);

%n_t -> Number of timesteps
n_t = size(Data.GT{1},2);
% Unpack 
GT = Data.GT;
y = Data.y;

% Initial State
x0 = {};
P0_cell = {};
performance = cell(t,n_t);

for i = 1:t
    x0 = [x0, {GT{i}(:,1)}];
    
    P0_cell = [P0_cell,P0];
    
    
    
    performance{i,1}.x = GT{i}(:,1);
    performance{i,1}.P = P0;

end

curr_class = JPDAF(x0,P0_cell,F,Q,R,H,PD,i);

for k = 2:n_t
    
    z = y{k};

    x_prior = {};
    P_prior = {};

    for i = 1:t
        x_prior = [x_prior,performance{i,k-1}.x];
        P_prior = [P_prior,performance{i,k-1}.P];
    end
    [X,P] = curr_class.timestep(x_prior,P_prior,z);
    

    for i = 1:t
        performance{i,k}.x = X{i};
        performance{i,k}.P = P{i};
    end
    
end

mWidar_FilterPlot_multiObj_Distribution(performance, Data, 0:dt:10,'JPDAF_traj3.gif')
