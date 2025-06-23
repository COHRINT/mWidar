clear; close all; clear

load tracks / linear_const_v_mWidarsim.mat xplus
load tracks / cv_mWidarSim_25.mat Data -mat

y = Data.y;
sim_signal = Data.signal;

dt = 0.1;
t_vec = 0:dt:20;

to_plot = 0;

GT = xplus;
x0 = xplus(:, 1);

n_k = size(xplus, 2); % # of timesteps

% Initialize KF Matrices
P0 = diag([0.75 1.25 1.5 0.75 1.25 1.5]);

Q = 0.01 * diag([0.01 0.01 0.1 0.01 0.01 0.1]);

R = 0.25 * eye(2);

H = [1 0 0 0 0 0;
     0 0 0 1 0 0];
A = [0 1 0 0 0 0;
     0 0 1 0 0 0;
     0 0 0 0 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 1;
     0 0 0 0 0 0];
F = expm(A * dt);

% Save matrices to KF Struct
KF = cell(1, n_k);

% GNN-KF
for i = 1:n_k
    KF{i}.Q = Q; %Process Noise
    KF{i}.R = R; % Measurment Noise
    KF{i}.H = H; % Measurment function
    KF{i}.F = F; % Dynamics function
    KF{i}.x = x0; % State
    KF{i}.P = P0; % Covaraince
    KF{i}.z = y{i}; % Measurments
    KF{i}.valid_z = []; % Validated Measurments
    KF{i}.S = zeros(2, 2); % Innovation Covaraince
    KF{i}.innov = [0; 0];
end

[KF] = GNN_KF(KF);

% Which results to plots/save
plot_traj = 1;
plot_mWidarimg = 1;
plot_error = 1;
plot_innov = 0;
save_traj = 0;
save_error = 0;
save_innov = 0;
states = ["X Position [m]", "X Velocity [m/s]", "X Acceleration [m/s^2]", "Y Position [m]", "Y Velocity [m/s]", "Y Acceleration [m/s^2]"];
color = 'k';

KF_plot = {KF};
GT_plot = {GT};

mWidar_FilterPlot(KF_plot, GT_plot, t_vec, sim_signal, plot_traj, plot_mWidarimg, plot_error, plot_innov, save_traj, save_error, save_innov, 1, 4, states, color)
