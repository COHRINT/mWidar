% Simple test script to debug PDA_PF particle propagation
clear; clc; close all

% Add paths
addpath(fullfile('DA_Track'))
addpath(fullfile('supplemental'))

% Load data
load(fullfile('supplemental', 'recovery.mat'))
load(fullfile('supplemental', 'sampling.mat'))
load(fullfile('supplemental', 'Final_Test_Tracks', 'SingleObj', 'T5_parab_noise.mat'))
load(fullfile('supplemental', 'precalc_imagegridHMMEmLike.mat'), 'pointlikelihood_image');

dt = 0.1;
GT = Data.GT;

% Use the direct discrete-time formulation (matches test_hybrid_PF)
F = [1, 0, dt, 0, dt^2/2, 0;
     0, 1, 0, dt, 0, dt^2/2;
     0, 0, 1, 0, dt, 0;
     0, 0, 0, 1, 0, dt;
     0, 0, 0, 0, 1, 0;
     0, 0, 0, 0, 0, 1];

Q = 1e-2*eye(6);
H = [1 0 0 0 0 0; 0 1 0 0 0 0];

% Initialize PDA_PF
pf = PDA_PF(GT(:, 1), 1000, F, Q, H, pointlikelihood_image);
pf.debug = true;

fprintf('Initial state: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', GT(:, 1));
fprintf('Initial particle mean: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', mean(pf.particles, 2));

% Test one prediction step
fprintf('\n--- Testing prediction step ---\n');
pf.prediction();
fprintf('After prediction particle mean: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', mean(pf.particles, 2));

% Test one measurement update
fprintf('\n--- Testing measurement update ---\n');
meas = GT(1:2, 2);
fprintf('Measurement: [%.4f, %.4f]\n', meas);
pf.state_Estimation([], meas);
[x_est, P_est] = pf.getGaussianEstimate();
fprintf('After update estimate: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n', x_est);
