% Script to generate 3 datasets for detector testing.
clear;
clc;
close all;

% Set default MATLAB plotting parameters
% Set default font size to be larger
set(0, 'DefaultAxesFontSize', 16);      % Axis tick label size
set(0, 'DefaultTextFontSize', 16);      % Text (including legend) size
set(0, 'DefaultLegendFontSize', 16);    % Legend text size
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesTitleFontSizeMultiplier', 20/16); % Title font size = 20

% Set default text interpreter to LaTeX
set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex');

% Make titles bold
set(0, 'DefaultAxesTitleFontWeight', 'bold');

load(fullfile("matlab_src","supplemental","recovery.mat"))
load(fullfile("matlab_src","supplemental","sampling.mat"))

%% Generate Trackset
 PLOT_FLAG = true;
    
% System Dynamics
A = [0 0 1 0 0 0;
    0 0 0 1 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0;
    0 0 0 0 0 0];

% Maximum timespan, 100 time steps, 5 seconds. 
dt = 0.1;

%% Track 1: Horizontal line
GT_Track1 = zeros(6,100);
X_1 = zeros(1,2,100);
X_1(1,1:2,1) = [1 60];
X0 = [1 60 10 0 0 0]';
GT_Track1(:,1) = X0;

for k = 2:100
    GT_Track1(:,k) = expm(A*dt) * GT_Track1(:,k-1);
    X_1(1,:,k) = [GT_Track1(1,k) GT_Track1(2,k)];
    if GT_Track1(1,k) > 128 || GT_Track1(2,k) > 128
        fprintf("TRACK 1 LEAVES ARRAY BOUNDS")
    end
end

%% Track 2: Straight line
GT_Track2 = zeros(6,100);
X_2 = zeros(1,2,100);
X_2(1,1:2,1) = [1 25];
X0 = [1 25 10 9 0 0]';
GT_Track2(:,1) = X0;

for k = 2:100
    GT_Track2(:,k) = expm(A*dt) * GT_Track2(:,k-1);
    X_2(1,:,k) = [GT_Track2(1,k) GT_Track2(2,k)];
    if GT_Track2(1,k) > 128 || GT_Track2(2,k) > 128
        fprintf("TRACK 2 LEAVES ARRAY BOUNDS")
    end
end

%% Track 3: Parabola
GT_Track3 = zeros(6,100);
X_3 = zeros(1,2,100);
X_3(1,1:2,1) = [1 120];
X0 = [1 120 10 -25 0 5]';
GT_Track3(:,1) = X0;

for k = 2:100
    GT_Track3(:,k) = expm(A*dt) * GT_Track3(:,k-1);
    X_3(1,:,k) = [GT_Track3(1,k) GT_Track3(2,k)];
    if GT_Track3(1,k) > 128 || GT_Track3(2,k) > 128
        fprintf("TRACK 3 LEAVES ARRAY BOUNDS")
    end
end

%% Validate Trajectories
if PLOT_FLAG

    figure(1); hold on; grid on
    plot(GT_Track1(1,:),GT_Track1(2,:),'r')
    plot(GT_Track2(1,:),GT_Track2(2,:),'b')
    plot(GT_Track3(1,:),GT_Track3(2,:),'g')
    title("Object Tracks")
    xlabel("X [px]")
    ylabel("Y [px]")
end

%% Save Tracjectors

% Single -> Single object case, only use track 1
Data = X_1;
save(fullfile("detectors","Detector_Tracks", "Single_Object_Track.mat"),'Data','-mat')
% Double -> Double object case, use track 1 and 2
Data = [X_1; X_2];
save(fullfile("detectors","Detector_Tracks", "Double_Object_Track.mat"),'Data','-mat')
% Triple -> Use all 3 tracks
Data = [X_1; X_2;X_3];
save(fullfile("detectors","Detector_Tracks", "Triple_Object_Track.mat"),'Data','-mat')

%% Simulate and Save mWidar signal for each case

Signal_Single = zeros(128,128,100);
Signal_Double = zeros(128,128,100);
Signal_Triple = zeros(128,128,100);

for k = 1:100

    

    % Get Object Positions
    px1 = floor(X_1(1,1,k));
    py1 = floor(X_1(1,2,k));
    px2 = floor(X_2(1,1,k));
    py2 = floor(X_2(1,2,k));
    px3 = floor(X_3(1,1,k));
    py3 = floor(X_3(1,2,k));

    % Single Signal
    S = zeros(128,128);
    S(py1,px1) = 1;

    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';
    Signal_Single(:,:,k) = imgaussfilt(sim_signal, 2);
    

    % Double Signal
    S = zeros(128,128);
    S(py1,px1) = 1;
    S(py2,px2) = 1;

    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';
    Signal_Double(:,:,k) = imgaussfilt(sim_signal, 2);

    % Triple Signal
    S = zeros(128,128);
    S(py1,px1) = 1;
    S(py2,px2) = 1;
    S(py3,px3) = 1;

    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';
    Signal_Triple(:,:,k) = imgaussfilt(sim_signal, 2);
end

%% Save Signals

% Single -> Single object case, only use track 1
Signal = Signal_Single;
save(fullfile("detectors","Detector_Tracks", "Single_Signal.mat"),'Signal','-mat')
% Double -> Double object case, use track 1 and 2
Signal = Signal_Double;
save(fullfile("detectors","Detector_Tracks", "Double_Signal.mat"),'Signal','-mat')
% Triple -> Use all 3 tracks
Signal = Signal_Triple;
save(fullfile("detectors","Detector_Tracks", "Triple_Signal.mat"),'Signal','-mat')