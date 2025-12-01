clear; clc; close all

%% plot_mWidar_image_MC
% Generate sample mWidar images with detector outputs for visualization
% This script creates the same signals used in run_MC_Detector without 
% running the full Monte Carlo simulation

%% Parameters
rng(1) % Fixed seed for reproducibility
d_thresh = 15; % Distance threshold for matching (same as in MC simulation)

NUM_OBJECTS = 8; % Number of objects to simulate (1-10)
FRAMES_TO_SAVE = 5; % Number of frames to save (will save the last N frames)

% Detector thresholds (using middle values from MC simulation)
THRESH_MP = 0.2; % MaxPeaks MinPeakHeight
THRESH_CFAR = 0.3; % CA-CFAR Pfa
THRESH_TDPF = 15; % TDPF Distance threshold

% Output directory
OUTPUT_DIR = fullfile("..", "figures", "Detectors", "MCRunSamples");
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

%% Load mWidar matrices
load(fullfile("..", "matlab_src", "supplemental", "recovery.mat"))
load(fullfile("..", "matlab_src", "supplemental", "sampling.mat"))
xgrid = 1:128;
ygrid = 1:128;
[X, Y] = meshgrid(xgrid, ygrid);

%% Generate signal
while true
    [scaled_signal, POS] = random_tracks2(NUM_OBJECTS, M, G);
    if all(~isnan(scaled_signal(128, 128, :, :)), 'all')
        break
    end
    fprintf("Bad Signal Generation. Trying again.\n");
end
nk = size(POS, 3);

fprintf("Generated signal with %d objects over %d timesteps\n", NUM_OBJECTS, nk)

% Determine which frames to save (last FRAMES_TO_SAVE frames)
frames_to_plot = max(1, nk - FRAMES_TO_SAVE + 1):nk;
fprintf("Saving frames: %s\n", mat2str(frames_to_plot))

%% Setup for TDPF (need to run through all frames to build history)
TDPF_Dict = {};

%% Process all frames (TDPF needs history)
fprintf("\nProcessing frames for TDPF...\n")

for k = 1:nk
    % Extract valid region
    signal_full = scaled_signal(:, :, 2, k);
    signal_valid = signal_full(21:128, :);
    y_offset = 20;
    
    % TDPF processing
    [pks_height, py, px] = peaks2(signal_valid);
    py = py + y_offset;

    % Sort peaks by height (descending)
    [~, sort_idx] = sort(pks_height, 'descend');
    px = px(sort_idx);
    py = py(sort_idx);
    fprintf("\n\nTDPF Processing New Frame\n")
    disp([pks_height(sort_idx), px, py]);
    current_peaks = [px, py];
    
    if k == 1
        first_score = [current_peaks, ones(size(current_peaks, 1), 1)];
        TDPF_Dict{1} = first_score;
    else
        TDPF_Dict{k} = TDPF(current_peaks, TDPF_Dict{k - 1}, THRESH_TDPF, THRESH_TDPF / 2);
    end
end

fprintf("TDPF history built.\n\n")

%% Plot and save the last FRAMES_TO_SAVE frames
fprintf("Generating and saving images...\n\n")

% Create figure with detectors in rows, frames in columns
if FRAMES_TO_SAVE == 10 
    fig = figure('Position', [100, 100, 2400, 1200], 'Color', 'w');
else
    fig = figure('Position', [100, 100, 1600, 1200], 'Color', 'w');
end
tiledlayout(4, FRAMES_TO_SAVE, 'TileSpacing', 'compact', 'Padding', 'compact');

% Pre-process all frames to get detector results
all_results = cell(length(frames_to_plot), 1);

for idx = 1:length(frames_to_plot)
    k = frames_to_plot(idx);
    
    fprintf("Processing frame %d/%d (timestep %d)...\n", idx, FRAMES_TO_SAVE, k)
    
    % Get ground truth points for this frame
    gt_points = POS(:, :, k);
    
    % Extract valid region
    signal_full = scaled_signal(:, :, 2, k);
    signal_valid = signal_full(21:128, :);
    y_offset = 20;
    
    % Filter ground truth to valid region
    if ~isempty(gt_points)
        valid = gt_points(:, 1) >= 1 & gt_points(:, 1) <= 128 & ...
            gt_points(:, 2) >= 20 & gt_points(:, 2) <= 128 & ...
            ~isnan(gt_points(:, 1)) & ~isnan(gt_points(:, 2));
        gt_points = gt_points(valid, :);
    end
    
    %% Run detectors
    
    % MaxPeaks (peaks2)
    [~, py, px] = peaks2(signal_valid, 'MinPeakHeight', THRESH_MP, 'MinPeakDistance', 20);
    py = py + y_offset;
    peaks_MP = [px, py];
    
    % CA-CFAR
    Ng = 5; Ns = 20;
    [~, py, px] = CA_CFAR(signal_valid, THRESH_CFAR, Ng, Ns);
    py = py + y_offset;
    peaks_CFAR = [px, py];
    
    % TDPF (use pre-computed dictionary)
    peaks_tdpf = TDPF_Dict{k}(TDPF_Dict{k}(:, 3) > 2, 1:2);
    
    %% Calculate TP, FP, FN for each detector
    [TP_MP, FP_MP, FN_MP] = calcTPFPFN(gt_points, peaks_MP, d_thresh);
    [TP_CFAR, FP_CFAR, FN_CFAR] = calcTPFPFN(gt_points, peaks_CFAR, d_thresh);
    [TP_TDPF, FP_TDPF, FN_TDPF] = calcTPFPFN(gt_points, peaks_tdpf, d_thresh);
    
    % Store results
    all_results{idx} = struct('k', k, 'gt_points', gt_points, ...
        'peaks_MP', peaks_MP, 'TP_MP', TP_MP, 'FP_MP', FP_MP, 'FN_MP', FN_MP, ...
        'peaks_CFAR', peaks_CFAR, 'TP_CFAR', TP_CFAR, 'FP_CFAR', FP_CFAR, 'FN_CFAR', FN_CFAR, ...
        'peaks_tdpf', peaks_tdpf, 'TP_TDPF', TP_TDPF, 'FP_TDPF', FP_TDPF, 'FN_TDPF', FN_TDPF);
end

% Define colors
color_gt = [1, 0, 0];        % Bright red for ground truth
color_mp = [1, 0.5, 0];      % Orange for MaxPeaks
color_cfar = [0, 1, 0];      % Bright green for CA-CFAR
color_tdpf = [0, 0.8, 1];    % Bright cyan/light blue for TDPF

% Calculate overall metrics across all frames
total_TP_MP = 0; total_FP_MP = 0; total_FN_MP = 0;
total_TP_CFAR = 0; total_FP_CFAR = 0; total_FN_CFAR = 0;
total_TP_TDPF = 0; total_FP_TDPF = 0; total_FN_TDPF = 0;

for idx = 1:length(frames_to_plot)
    res = all_results{idx};
    total_TP_MP = total_TP_MP + res.TP_MP;
    total_FP_MP = total_FP_MP + res.FP_MP;
    total_FN_MP = total_FN_MP + res.FN_MP;
    
    total_TP_CFAR = total_TP_CFAR + res.TP_CFAR;
    total_FP_CFAR = total_FP_CFAR + res.FP_CFAR;
    total_FN_CFAR = total_FN_CFAR + res.FN_CFAR;
    
    total_TP_TDPF = total_TP_TDPF + res.TP_TDPF;
    total_FP_TDPF = total_FP_TDPF + res.FP_TDPF;
    total_FN_TDPF = total_FN_TDPF + res.FN_TDPF;
end

% Calculate Precision and Recall
precision_MP = total_TP_MP / (total_TP_MP + total_FP_MP);
recall_MP = total_TP_MP / (total_TP_MP + total_FN_MP);

precision_CFAR = total_TP_CFAR / (total_TP_CFAR + total_FP_CFAR);
recall_CFAR = total_TP_CFAR / (total_TP_CFAR + total_FN_CFAR);

precision_TDPF = total_TP_TDPF / (total_TP_TDPF + total_FP_TDPF);
recall_TDPF = total_TP_TDPF / (total_TP_TDPF + total_FN_TDPF);

fprintf("\n=== Overall Performance Metrics ===\n")
fprintf("MaxPeaks:  Precision = %.3f, Recall = %.3f\n", precision_MP, recall_MP)
fprintf("CA-CFAR:   Precision = %.3f, Recall = %.3f\n", precision_CFAR, recall_CFAR)
fprintf("TDPF:      Precision = %.3f, Recall = %.3f\n\n", precision_TDPF, recall_TDPF)

%% Plot Row 1: Ground Truth
for idx = 1:length(frames_to_plot)
    k = frames_to_plot(idx);
    res = all_results{idx};
    
    nexttile;
    surf(X, Y, scaled_signal(:, :, 2, k), 'EdgeColor', 'none');
    shading interp
    colormap(gca, 'gray')
    hold on
    
    % Plot only ground truth
    if ~isempty(res.gt_points)
        plot3(res.gt_points(:, 1), res.gt_points(:, 2), ones(size(res.gt_points(:, 1))), ...
            'x', 'Color', color_gt, 'LineWidth', 3, 'MarkerSize', 30);
    end
    
    axis square; view(2); xlim([1 129]); ylim([21 129]);
    set(gca, 'XTick', [], 'YTick', []);
    title(sprintf('Frame %d', k), 'FontSize', 9);
    if idx == 1
        ylabel('Ground Truth', 'FontSize', 11, 'FontWeight', 'bold');
    end
    hold off
end

%% Plot Row 2: MaxPeaks
for idx = 1:length(frames_to_plot)
    k = frames_to_plot(idx);
    res = all_results{idx};
    
    nexttile;
    surf(X, Y, scaled_signal(:, :, 2, k), 'EdgeColor', 'none');
    shading interp
    colormap(gca, 'gray')
    hold on
    
    % Plot ground truth and MaxPeaks
    if ~isempty(res.gt_points)
        plot3(res.gt_points(:, 1), res.gt_points(:, 2), ones(size(res.gt_points(:, 1))), ...
            'x', 'Color', color_gt, 'LineWidth', 3, 'MarkerSize', 30);
    end
    if ~isempty(res.peaks_MP)
        plot3(res.peaks_MP(:, 1), res.peaks_MP(:, 2), ones(size(res.peaks_MP(:, 1))), ...
            'o', 'Color', color_mp, 'LineWidth', 2, 'MarkerSize', 12);
    end
    
    axis square; view(2); xlim([1 129]); ylim([21 129]);
    set(gca, 'XTick', [], 'YTick', []);
    title(sprintf('TP:%d FP:%d FN:%d', res.TP_MP, res.FP_MP, res.FN_MP), 'FontSize', 9);
    if idx == 1
        ylabel('MaxPeaks', 'FontSize', 11, 'FontWeight', 'bold');
    end
    hold off
end

%% Plot Row 3: CA-CFAR
for idx = 1:length(frames_to_plot)
    k = frames_to_plot(idx);
    res = all_results{idx};
    
    nexttile;
    surf(X, Y, scaled_signal(:, :, 2, k), 'EdgeColor', 'none');
    shading interp
    colormap(gca, 'gray')
    hold on
    
    % Plot ground truth and CA-CFAR
    if ~isempty(res.gt_points)
        plot3(res.gt_points(:, 1), res.gt_points(:, 2), ones(size(res.gt_points(:, 1))), ...
            'x', 'Color', color_gt, 'LineWidth', 3, 'MarkerSize', 30);
    end
    if ~isempty(res.peaks_CFAR)
        plot3(res.peaks_CFAR(:, 1), res.peaks_CFAR(:, 2), ones(size(res.peaks_CFAR(:, 1))), ...
            'o', 'Color', color_cfar, 'LineWidth', 2, 'MarkerSize', 8);
    end
    
    axis square; view(2); xlim([1 129]); ylim([21 129]);
    set(gca, 'XTick', [], 'YTick', []);
    title(sprintf('TP:%d FP:%d FN:%d', res.TP_CFAR, res.FP_CFAR, res.FN_CFAR), 'FontSize', 9);
    if idx == 1
        ylabel('CA-CFAR', 'FontSize', 11, 'FontWeight', 'bold');
    end
    hold off
end

%% Plot Row 4: TDPF
for idx = 1:length(frames_to_plot)
    k = frames_to_plot(idx);
    res = all_results{idx};
    
    nexttile;
    surf(X, Y, scaled_signal(:, :, 2, k), 'EdgeColor', 'none');
    shading interp
    colormap(gca, 'gray')
    hold on
    
    % Plot ground truth and TDPF
    if ~isempty(res.gt_points)
        plot3(res.gt_points(:, 1), res.gt_points(:, 2), ones(size(res.gt_points(:, 1))), ...
            'x', 'Color', color_gt, 'LineWidth', 3, 'MarkerSize', 30);
    end
    if ~isempty(res.peaks_tdpf)
        plot3(res.peaks_tdpf(:, 1), res.peaks_tdpf(:, 2), ones(size(res.peaks_tdpf(:, 1))), ...
            'o', 'Color', color_tdpf, 'LineWidth', 2, 'MarkerSize', 4);
    end
    
    axis square; view(2); xlim([1 129]); ylim([21 129]);
    set(gca, 'XTick', [], 'YTick', []);
    title(sprintf('TP:%d FP:%d FN:%d', res.TP_TDPF, res.FP_TDPF, res.FN_TDPF), 'FontSize', 9);
    if idx == 1
        ylabel('TDPF', 'FontSize', 11, 'FontWeight', 'bold');
    end
    hold off
end

% Add overall title and legend explanation (commented out due to cutoff issues)
% sgtitle_str = sprintf('%d Objects (Thresholds: MP=%.2f, CFAR=%.2f, TDPF=%.0f)\nMetrics format: TP/FP/FN', ...
    % NUM_OBJECTS, THRESH_MP, THRESH_CFAR, THRESH_TDPF);
% sgtitle(sgtitle_str, 'FontSize', 13, 'FontWeight', 'bold')

% Save figure
filename = sprintf('frames_obj%d.png', NUM_OBJECTS);
filepath = fullfile(OUTPUT_DIR, filename);
saveas(fig, filepath);
fprintf("\n✓ Saved: %s\n", filepath)

% close(fig)

fprintf("\n✓ All frames saved to: %s\n", OUTPUT_DIR)
