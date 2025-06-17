clear;close all; clc

load tracks/linear_const_v_mWidarsim.mat xplus
load tracks/linear_const_v_mWidarsim2.mat xplus2
load tracks/cv_mWidarSim_2obj_15.mat Data -mat

load ../recovery.mat
load ../sampling.mat

%% For the following PDAF implementation, save all KF matrices as a struct

y = Data.y;
sim_signal = Data.signal;

dt = 0.1;
t_vec = 0:dt:20;

to_plot = 0;

GT1  = xplus;
x0 = xplus(:,1);

n_k = size(xplus,2); % # of timesteps

% Initialize KF Matrices
P0 = diag([0.75 1.25 1.5 0.75 1.25 1.5]);

Q = 0.01*diag([0.01 0.01 0.1 0.01 0.01 0.1]);

R = 0.25*eye(2);

H = [1 0 0 0 0 0;
    0 0 0 1 0 0];
A = [0 1 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0];
F = expm(A*dt);

% Save matrices to KF Struct
KF = cell(1,n_k);

% PDAF 1
for i = 1:n_k
    KF{i}.Q = Q; %Process Noise
    KF{i}.R = R; % Measurment Noise
    KF{i}.H = H; % Measurment function
    KF{i}.F = F; % Dynamics function
    KF{i}.x = x0; % State
    KF{i}.P = P0; % Covaraince
    KF{i}.z = y{i}; % Measurments
    KF{i}.valid_z = []; % Validated Measurments
    KF{i}.S = zeros(2,2); % Innovation Covaraince
    KF{i}.innov = [0;0];
end

% PDAF 2
x0 = xplus2(:,1);
GT2  = xplus2;
KF2 = cell(1,n_k);

for i = 1:n_k
    KF2{i}.Q = Q; %Process Noise
    KF2{i}.R = R; % Measurment Noise
    KF2{i}.H = H; % Measurment function
    KF2{i}.F = F; % Dynamics function
    KF2{i}.x = x0; % State
    KF2{i}.P = P0; % Covaraince
    KF2{i}.z = y{i}; % Measurments
    KF2{i}.valid_z = []; % Validated Measurments
    KF2{i}.S = zeros(2,2); % Innovation Covaraince
    KF2{i}.innov = [0;0];
end
%% Call PDAF

[KF] = mWidar_PDAF(KF);
[KF2] = mWidar_PDAF(KF2);

%% Plot Results

% Which results to plots/save
plot_traj = 1;
plot_mWidarimg = 1;
plot_error = 1;
plot_innov = 0;
save_traj = 1;
save_error = 1;
save_innov = 0;
states = ["X Position [m]","X Velocity [m/s]","X Acceleration [m/s^2]","Y Position [m]","Y Velocity [m/s]","Y Acceleration [m/s^2]"];
color = ['k','b'];

% Call plotting function
KF_plot = {KF, KF2};
GT = {GT1, GT2};
mWidar_FilterPlot(KF_plot,GT,t_vec,sim_signal,plot_traj,plot_mWidarimg,plot_error,plot_innov,save_traj,save_error,save_innov,1,4,states,color)