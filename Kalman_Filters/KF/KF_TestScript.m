clear; clc; close all

%% Initialize

% Test and validate 6 state KF
rng(100)

% Choose Trajectory
load tracks/linear_const_a.mat xplus

GT = xplus;

% Initialize Constants
dt = 0.1;
t_vec = 0:dt:20;

P0 = diag([0.1 0.25 0.5 0.1 0.25 0.5]);

% Length of time vector
n_k = length(GT);

% Preallocate space for filter estimates
X = zeros(6,n_k);
P = cell(1,n_k);
X(:,1) = GT(:,1);
P{1} = P0;

% KF Matrices
Q = diag([0.01 0.01 0.1 0.01 0.01 0.1]);
Q = 0.001*Q;
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

%% Save relevent KF matrices/outputs to a struct
KF = cell(1,n_k);

for i = 1:n_k
    KF{i}.Q = Q; %Process Noise
    KF{i}.R = R; % Measurment Noise
    KF{i}.H = H; % Measurment function
    KF{i}.F = F; % Dynamics function
    KF{i}.x = X(:,1); % State
    KF{i}.P = P0; % State Covaraince
    KF{i}.S = zeros(2,2); % Innovation Covaraince
    KF{i}.y = [GT(1,i);GT(4,i)] + mvnrnd([0;0],0.01*R)'; % Measurments
    KF{i}.innov = [0;0];
end

%% Call KF

for k = 2:n_k
    k

    [x_minus,y_minus,P_minus,S] = KF_DS(KF{k-1});

    KF{k-1}.S = S;

    [X,P,innov] = KF_MS(KF{k-1},x_minus,y_minus,P_minus);

    KF{k}.x = X;
    KF{k}.P = P;
    KF{k-1}.innov = innov;
end

%% Plot Results

mWidar_LinearKF_plot(KF,GT,t_vec)

NEES_KF(KF,GT,100)

