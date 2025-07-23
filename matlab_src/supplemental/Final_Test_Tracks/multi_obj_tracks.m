clear; clc; close all

load ../recovery.mat
load ../sampling.mat


dt = 0.1; % [sec]
tvec = 0:dt:10; % 10 seconds for each trajectory, 100 timesteps
n_t = size(tvec,2); % # of tsteps
toplot = 1; % Plot signal at end of script
% Select detector
detector = "peaks2";
%detector = "CFAR";

%% Traj 1: JPDA Test (fairly easy scenario)

% X will be our ground truth state time history

A = [0 0 1 0 0 0;
    0 0 0 1 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0;
    0 0 0 0 0 0];

x0 = cell(1,2);
X = cell(1,2);

% IC1 -> Near array, y ~ 0
x0{1} = [-1.75 0.5 0 0 0.065 0]'; % No y acceleration

% IC2 -> Far from array, opposite direction
x0{2} = [1.75 3.5 0 0 -0.065 0]'; % No y acceleration

X{1} = generate_track(x0{1},A,tvec);
X{2} = generate_track(x0{2},A,tvec);

% Generate an mWidar image for each timestep, save the signal, each
% detection, and ground truth into one .mat file
[y, Signal] = sim_mWidar_image(n_t,X,M,G,detector,false);

Data.GT = X;
Data.y = y;
Data.signal = Signal;

save MultObj/JPDAF_test_traj.mat Data -mat

