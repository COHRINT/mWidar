clear; clc; close all


M = load("sampling.mat").M;
G = load("recovery.mat").G;

X = [-1.4;0;0;3.25;0;0];

[Meas,Height] = SimulateImages(X,M,G);
