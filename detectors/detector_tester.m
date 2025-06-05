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

% Set PLOT_FLAG to 1 to plot the detection simulation
PLOT_FLAG = 0;

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the simulation data -- Single, Double, Triple
scaling_string = "gaussian"; % Options: "tanh", "gaussian", "none", "linear"
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
                 0.3, 0.6, 0.3 % Darker green
                 ];

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(Position = [100, 100, 1800, 600])
% tile_sim = tiledlayout(1, 2, 'TileSpacing', 'Compact');
% ax1 = nexttile(tile_sim);
% ax2 = nexttile(tile_sim);
% hold(ax1, 'on')
% hold(ax2, 'on')
% xlim(ax2, [-1.1, 1.1])
% ylim(ax2, [0, 7])
% xlim(ax1, [0, 128])
% ylim(ax1, [0, 128])
method_strings = ["Max-Peak", "CA-CFAR", "Time Dependent"];

num_methods = 2;
num_thresholds = 100;
global_TP_MP = zeros(num_methods, num_thresholds + 1);
global_FP_MP = zeros(num_methods, num_thresholds + 1);
global_FN_MP = zeros(num_methods, num_thresholds + 1);

global_TPR_MP = zeros(num_methods, num_thresholds + 1); ;
global_FPR_MP = zeros(num_methods, num_thresholds + 1); ;

% Thresholds for Max Peak Detector and CA_CFAR Detector
thresholds = 1 - exp(-linspace(0, 5, num_thresholds)); % Exponential spacing to increase density near 1
thresholds = (thresholds - min(thresholds)) / (max(thresholds) - min(thresholds)); % Scale to [0, 1]
thresholds = sort(thresholds); % Ensure thresholds are sorted in ascending order
thresholds = [thresholds, 1];
% disp("Max Peak Thresholds: " + num2str(thresholds));
disp("Number of Max Peak Thresholds: " + num2str(length(thresholds)));

% Threshold for TDPF (range of "dist_thresh" values)
dist_thresh = linspace(0, 15, num_thresholds + 1); % Distance thresholds
% dist_thresh = [dist_thresh, 50]; % Add a fixed threshold of 5
dist_thresh = sort(dist_thresh); % Ensure thresholds are sorted in ascending order
disp("Number of TDPF Thresholds: " + num2str(length(dist_thresh)));

for t = 1:size(sim_signal, 3)
    disp(t)
    % Iterate through detectors and find peaks
    % blur current signal
    signal_orig = sim_signal(20:end, :, t);

    % Scale signal using the scaleSignal function
    signal = scaleSignal(signal_orig, scaling_string);

    TPR_MP = [];
    FPR_MP = [];
    ROC_DETECT_DISTANCE = 5;

    gt_points = squeeze(GT_traj(:, :, t)).'; % M×2
    % Subtract 20 from all x-coordinates to account for the 20 pixel offset
    gt_points(:, 2) = gt_points(:, 2) - 20;

    %%%%%%%%%%%%
    %% Detectors
    %%%%%%%%%%%%
    % MAX PEAKS DETECTOR
    for i = 1:length(thresholds)

        % Max Peak Detector
        th = thresholds(i);
        [~, px, py] = peaks2(signal, 'MinPeakHeight', th, 'MinPeakDistance', 4);
        peaks = [py, px];

        % Use the calcTPFPFN function to calculate TP, FP, FN
        [TP, FP, FN] = calcTPFPFN(gt_points, peaks, ROC_DETECT_DISTANCE);

        global_TP_MP(1, i) = global_TP_MP(1, i) + TP;
        global_FP_MP(1, i) = global_FP_MP(1, i) + FP;
        global_FN_MP(1, i) = global_FN_MP(1, i) + FN;
    end

    % CA_CFAR DETECTOR
    for i = 1:length(thresholds)

        % CA_CFAR Detector
        th = thresholds(i);
        [~, px, py] = CA_CFAR(signal, th, 3, 10);
        peaks = [py, px];

        % Use the calcTPFPFN function to calculate TP, FP, FN
        [TP, FP, FN] = calcTPFPFN(gt_points, peaks, ROC_DETECT_DISTANCE);

        global_TP_MP(2, i) = global_TP_MP(2, i) + TP;
        global_FP_MP(2, i) = global_FP_MP(2, i) + FP;
        global_FN_MP(2, i) = global_FN_MP(2, i) + FN;
    end

    % TDPF DETECTOR
    % if t == 1
    %     prev_peak_score_loop = struct();

    %     for i = 1:length(dist_thresh)
    %         % Use the threshold as the field name
    %         field_name = sprintf('thresh_%d', i);
    %         prev_peak_score_loop.(field_name) = []; % Initialize with an empty array
    %     end

    % end

    % for i = 1:length(dist_thresh)
    %     % CA_CFAR detector for current frame
    %     % [~, px, py] = CA_CFAR(signal, .8, 2, 15);

    %     % MaxPeaks Detector
    %     [~, px, py] = peaks2(signal, 'MinPeakHeight', .6, 'MinPeakDistance', 10);
    %     peaks = [py, px];

    %     % TDPF Detector
    %     d_thresh_value = dist_thresh(i);
    %     field_name = sprintf('thresh_%d', i);
    %     prev_peak_score = prev_peak_score_loop.(field_name);

    %     % Update prev_peak_score using TDPF
    %     prev_peak_score = TDPF(peaks, prev_peak_score, d_thresh_value);
    %     % Predited peaks MUST be persistent
    %     predicted_peaks = prev_peak_score(prev_peak_score(:, 3) > 2, 1:2);

    %     % Replace the previous peak score with the new one
    %     prev_peak_score_loop.(field_name) = prev_peak_score;

    %     % Use the calcTPFPFN function to calculate TP, FP, FN
    %     [TP, FP, FN] = calcTPFPFN(gt_points, predicted_peaks, ROC_DETECT_DISTANCE);

    %     global_TP_MP(3, i) = global_TP_MP(3, i) + TP;
    %     global_FP_MP(3, i) = global_FP_MP(3, i) + FP;
    %     global_FN_MP(3, i) = global_FN_MP(3, i) + FN;
    % end

    %%%%%%%%%%%%
    %% Plotting
    %%%%%%%%%%%%

    % Create a single figure for animation at the beginning of the loop
    if t == 1
        % fig_anim = figure('Position', [100, 100, 1800, 600], "Name", "Radar Signal Animation");
        fig_anim = figure("Visible", "off");
    end

    % Clear the figure but keep the window
    clf(fig_anim);

    % Create the tiled layout
    tile_sim = tiledlayout(fig_anim, 1, 2, 'TileSpacing', 'Compact');

    % First subplot - Surface plot
    ax1 = nexttile(tile_sim);
    hold(ax1, 'on')

    % Plot the GT points on first axis
    plot(ax1, GT_traj(1, :, t), GT_traj(2, :, t) - 20, 'd', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'magenta', 'Color', 'magenta', 'DisplayName', 'Ground Truth');

    % Plot text next to each GT point
    for i = 1:size(gt_points, 1)
        text(ax1, gt_points(i, 1), gt_points(i, 2), ['GT ' num2str(i)], 'Color', 'magenta', 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    end

    % Plot the signal surface
    [X, Y] = meshgrid(1:size(signal, 2), 1:size(signal, 1));
    s = surf(ax1, X, Y, signal, 'FaceAlpha', .5, 'EdgeAlpha', 0.5);
    s.EdgeColor = 'none';
    colormap(ax1, 'gray');
    c = colorbar(ax1);
    c.Label.String = 'Signal Intensity';
    c.Label.Interpreter = 'latex';

    % Plot the detected peaks
    [~, px, py] = peaks2(signal, 'MinPeakHeight', 0.75, 'MinPeakDistance', 4);
    peaks = [py, px];
    plot(ax1, peaks(:, 1), peaks(:, 2), '+', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', color_palette(1, :), 'Color', color_palette(1, :), 'DisplayName', 'Max-Peaks');

    % Plot the CA_CFAR peaks
    [~, px, py] = CA_CFAR(signal, .2, 3, 10);
    peaks = [py, px];

    if size(peaks, 1) > 0
        % Plot the detected peaks
        plot(ax1, peaks(:, 1), peaks(:, 2), 'x', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', color_palette(2, :), 'Color', color_palette(2, :), 'DisplayName', 'CA-CFAR');
    end

    % Plot TDPF detected peaks
    if t == 1
        % Initialize previous peak score
        prev_peak_score = [];
    end

    % TDPF_String = ["New Target", "Existing Target", "Persistent Target"];
    % [~, px, py] = peaks2(signal, 'MinPeakHeight', .6, 'MinPeakDistance', 15);
    % peaks = [py, px];
    % prev_peak_score = TDPF(peaks, prev_peak_score, 2);

    % if size(prev_peak_score, 1) > 0
    %     color_small = ["#FF0000", "#00FF00", "#0000FF"];

    %     for i = 1:3
    %         plot_peaks = prev_peak_score(prev_peak_score(:, 3) == i, 1:2);

    %         if size(plot_peaks, 1) > 0
    %             plot(ax1, plot_peaks(:, 1), plot_peaks(:, 2), 'o', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', color_small(i), 'Color', color_small(i), 'DisplayName', TDPF_String(i));
    %         end

    %     end

    % end

    % Configure first subplot
    axis(ax1, 'equal');
    xlim(ax1, [1, size(signal, 2)]);
    ylim(ax1, [1, size(signal, 1)]);
    zlim(ax1, [min(signal(:)), max(signal(:))]);
    view(ax1, 2); % Set the view to 2D
    title(ax1, "Radar Signal at Time " + t, 'Interpreter', 'latex', 'FontSize', 14);
    legend(ax1, 'Location', 'northeast', "Interpreter", "latex", 'FontSize', 12);

    % Second subplot - Histogram
    ax2 = nexttile(tile_sim);
    hold(ax2, 'on')

    % Plot histogram of signal values
    histogram(ax2, signal(:), 'Normalization', 'pdf', 'BinWidth', 0.05, 'FaceColor', 'blue', 'EdgeColor', 'none');

    % Add vertical lines for each GT point's signal intensity
    for i = 1:size(gt_points, 1)
        % Get coordinates of GT point (rounded to nearest integer)
        gt_x = round(gt_points(i, 1));
        gt_y = round(gt_points(i, 2));

        % Make sure coordinates are within bounds
        gt_x = max(1, min(gt_x, size(signal, 2)));
        gt_y = max(1, min(gt_y, size(signal, 1)));

        % Get signal value at GT position
        gt_signal_val = signal(gt_y, gt_x);

        % Get y-limits of current plot for line height
        y_lims = ylim(ax2);

        % Plot vertical line at the signal intensity with GT color
        line(ax2, [gt_signal_val gt_signal_val], [0 y_lims(2) * 0.9], 'Color', 'magenta', 'LineWidth', 2, 'LineStyle', '--');

        % Add text label with GT index
        text(ax2, gt_signal_val, y_lims(2) * 0.95, ['GT ' num2str(i)], 'Color', 'magenta', 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end

    xlim(ax2, [0, 1]);
    ylabel(ax2, 'Probability Density', 'Interpreter', 'latex');
    xlabel(ax2, 'Signal Intensity', 'Interpreter', 'latex');
    title(ax2, "Signal Intensity Distribution", 'Interpreter', 'latex', 'FontSize', 14);
    grid(ax2, 'on');
    sgtitle("Radar Signal Analysis - Frame " + t, 'Interpreter', 'latex', 'FontSize', 16);

    pause(0.1);

end

%% ROC Curves 

figure('Name', 'ROC Curves', 'Position', [100, 100, 300, 300]);
hold('on')

% Plot ROC curves
for meth = 1:num_methods
    global_TPR_MP(meth, :) = global_TP_MP(meth, :) ./ (global_TP_MP(meth, :) + global_FN_MP(meth, :) + eps);
    global_FPR_MP(meth, :) = global_FP_MP(meth, :) ./ (global_FP_MP(meth, :) + global_TP_MP(meth, :) + eps);

    % [~, sort_idx] = sort(global_FPR_MP(meth, :));
    plot(global_FPR_MP(meth, :), global_TPR_MP(meth, :), '-o', 'Color', color_palette(meth, :), "DisplayName", method_strings(meth), 'LineWidth', 2, 'MarkerSize', 6);

    fprintf("Method: %s\n", method_strings(meth));
    fprintf("Number of datapoints: %d\n", length(global_TPR_MP(meth, :)));

    % Every 10th point label with threshold
    % if meth == 3

    %     for i = 1:10:length(thresholds)
    %         text(global_FPR_MP(meth, i), global_TPR_MP(meth, i) - .05, sprintf('%.5f', dist_thresh(i)), 'FontSize', 8, 'Color', color_palette(meth, :));
    %     end

    % else

        % for i = 1:10:length(thresholds)
        %     text(global_FPR_MP(meth, i), global_TPR_MP(meth, i) - .05, sprintf('%.2f', thresholds(i)), 'FontSize', 8, 'Color', color_palette(meth, :));
        % end

    % end

end

% Plot "Random" line
plot([0 1], [0 1], '--k', 'LineWidth', 1, "HandleVisibility", "off");

% Axis
axis('square')
xlim([0 1])
ylim([0 1])

% Labels
xlabel('False Positive Rate', 'Interpreter', 'latex')
ylabel('True Positive Rate', 'Interpreter', 'latex')
% title(sprintf('ROC Curve - Dataset: %s - %s scaling', dataset, scaling_string), 'Interpreter', 'latex')
title(sprintf('ROC Curve - Dataset: %s', dataset), 'Interpreter', 'latex')
% Legend + Grid
grid('on')
legend('Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 14);

% Save figure
exportgraphics(gcf, sprintf('../data_tracks/ROC_%s.png', dataset), 'Resolution', 600);
