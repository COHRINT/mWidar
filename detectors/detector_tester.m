% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mWidar Simulator implementation in MATLAB
%
% Anthony La Barca
%
% Script to test the performance of detection algorithms
% against simulated data
%
% Detectors to be tested:
%   - Max Peak Detector
%   - H-maxima Detector
%   - Time-Dependent Detector
%
% Metrics to be tested:
%   - Statistical Methods (True Positive Rate, False Positive Rate, etc.)
%   - Computational Methods (Time Complexity, Space Complexity, etc.)
%   - Performance Metrics (ROC Curve, Precision-Recall Curve, etc.)
%   - Robustness Metrics (Noise, Clutter, etc.)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize Script Environment
clear;
clc;
close all;

% Set default MATLAB plotting parameters
% Set default font size to be larger
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultTextFontSize', 14);
set(0, 'DefaultLineLineWidth', 2);

% Set default text interpreter to LaTeX
set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex');

% Make titles bold
set(0, 'DefaultAxesTitleFontWeight', 'bold');

% Set PLOT_FLAG to 1 to plot the detection simulation
PLOT_FLAG = 0;

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the simulation data -- Single, Double, Triple
dataset = "Triple";
GT = load("../data_tracks/" + dataset + "_GT.mat").GT;
sim_signal = load("../data_tracks/" + dataset + "_simulated_signal.mat").simulated_signal;
GT_traj = load("../data_tracks/" + dataset + "_objects_traj.mat").objects_traj;

fprintf("Loaded %s dataset\n", dataset);
fprintf("Size of GT: %d x %d x %d\n", size(GT));
fprintf("Size of signal: %d x %d x %d\n", size(sim_signal));

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load detectors into Dictionary
color_palette = [
                 0.3, 0.3, 0.7; % Darker blue/purple
                 0.7, 0.3, 0.3; % Darker red/maroon
                 0.3, 0.6, 0.3  % Darker green
                 ];
detectors = containers.Map('KeyType', 'char', 'ValueType', 'any');
detectors("Max Peak") = @max_peaks;
detectors("H-maxima") = @h_maximas;
% detectors("Time Dependent") = (@TDD, color_palette(3, :));

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(Position = [100, 100, 1800, 600])
tile_sim = tiledlayout(1, 2, 'TileSpacing', 'Compact');
ax1 = nexttile(tile_sim);
ax2 = nexttile(tile_sim);
hold(ax1, 'on')
hold(ax2, 'on')
xlim(ax2, [-1.1, 1.1])
ylim(ax2, [0, 7])
xlim(ax1, [0, 128])
ylim(ax1, [0, 128])

for t = 1:size(sim_signal, 3)
    % Iterate through detectors and find peaks
    % blur current signal
    signal = imgaussfilt(sim_signal(:, :, t));
    % Scale signal using hyperbolic tangent with scaling factor
    % Optionally scale to zero mean, unit variance
    signal_std = (signal - mean(signal(:))) / std(signal(:));
    signal = tanh(signal_std);

    % Plot distribution of signal intensity
    cla(ax2) % Clear previous histogram
    histogram(ax2, signal(:), 'Normalization', 'pdf', 'BinWidth', 0.1, 'FaceColor', 'blue', 'EdgeColor', 'none');
    xlim(ax2, [-1.1, 1.1])
    ylabel(ax2, 'Probability Density Function')
    xlabel(ax2, 'Signal Intensity')

    ylim(ax2, [0, 7])

    % Plot surface
    cla(ax1)
    s = surface(ax1, signal, 'FaceAlpha', 0.5);
    s.EdgeColor = 'none';
    % colormap(ax1, 'parula');
    colormap(ax1, 'gray');
    colorbar(ax1)
    axis equal
    xlim(ax1, [1, 128])
    ylim(ax1, [1, 128])
    title(ax1, "Radar Signal at Time " + t, 'Interpreter', 'latex', 'FontSize', 14);

    % Process each detector with its corresponding color
    detector_names = keys(detectors);
    plots = [];
    names = detector_names(:); % Ensure it's a column vector

    for i = 1:length(detector_names)
        detector_name = detector_names{i};
        detector = detectors(detector_name);
        detected_peaks = detector(signal);

        % Use color from palette for each detector
        color = color_palette(i, :);

        % Plot detected peaks
        if ~isempty(detected_peaks)
            detector_plot = plot(ax1, detected_peaks(:, 1), detected_peaks(:, 2), '+', 'MarkerSize', 20, 'LineWidth', 5, 'MarkerFaceColor', color, 'MarkerEdgeColor', color);
            plots = [plots; detector_plot]; % Store plot handles for legend
        end

    end

    % Plot ground truth
    true_traj = plot(ax1, GT_traj(1, :, t), GT_traj(2, :, t), 'x', 'MarkerSize', 20, 'LineWidth', 4, 'Color', 'blue');
    plots = [plots; true_traj]; % Store plot handles for legend
    names = [names; "Ground Truth"];

    legend(ax1, plots, names, 'Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 14);

    pause(.2)

end
