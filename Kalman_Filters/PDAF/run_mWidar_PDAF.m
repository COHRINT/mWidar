clear;close all; clc

load ../Final_Test_Tracks/SingleObj/T4_parab.mat

%load ../recovery.mat
%load ../sampling.mat

%% For the following PDAF implementation, save all KF matrices as a struct

GT = Data.GT;
xplus = GT;
y = Data.y;
sim_signal = Data.signal;

dt = 0.1;
t_vec = 0:dt:10;

to_plot = 0;


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

%% Call PDAF

[KF] = mWidar_PDAF(KF);

%% Plot Results

mWidar_FilterPlot(KF,Data,t_vec)

