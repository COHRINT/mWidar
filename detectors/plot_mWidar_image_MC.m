clear; clc; close all

%% plot_mWidar_image_MC
% Generate sample mWidar images with detector outputs for visualization
% This script creates the same signals used in run_MC_Detector without 
% running the full Monte Carlo simulation

%% Parameters
NUM_OBJECTS = 3; % Number of objects to simulate (1-10)
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
rng(1) % Fixed seed for reproducibility
[scaled_signal, POS] = random_tracks2(NUM_OBJECTS, M, G);
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
    [pks_height, px, py] = peaks2(signal_valid);
    px = px + y_offset;

    % Sort peaks by height (descending)
    [~, sort_idx] = sort(pks_height, 'descend');
    px = px(sort_idx);
    py = py(sort_idx);
    current_peaks = [px, py];
    
    if k == 1
        first_score = [current_peaks, ones(size(current_peaks, 1), 1)];
        TDPF_Dict{1} = first_score;
    else
        TDPF_Dict{k} = TDPF(current_peaks, TDPF_Dict{k - 1}, THRESH_TDPF, 15);
    end
end

fprintf("TDPF history built.\n\n")

%% Plot and save the last FRAMES_TO_SAVE frames
fprintf("Generating and saving images...\n\n")

% Create single figure with all frames
fig = figure('Position', [100, 100, 1600, 400], 'Color', 'w');
tiledlayout(1, FRAMES_TO_SAVE, 'TileSpacing', 'compact', 'Padding', 'compact');

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
    [~, px, py] = peaks2(signal_valid, 'MinPeakHeight', THRESH_MP, 'MinPeakDistance', 20);
    px = px + y_offset;
    peaks_MP = [px, py];
    
    % CA-CFAR
    Ng = 5; Ns = 20;
    [~, px, py] = CA_CFAR(signal_valid, THRESH_CFAR, Ng, Ns);
    px = px + y_offset;
    peaks_CFAR = [px, py];
    
    % TDPF (use pre-computed dictionary)
    peaks_tdpf = TDPF_Dict{k}(TDPF_Dict{k}(:, 3) > 2, 1:2);
    
    %% Calculate TP, FP, FN for each detector
    d_thresh = 20; % Distance threshold for matching (same as in MC simulation)
    
    % MaxPeaks metrics
    [TP_MP, FP_MP, FN_MP] = calcTPFPFN(gt_points, peaks_MP, d_thresh);
    
    % CA-CFAR metrics
    [TP_CFAR, FP_CFAR, FN_CFAR] = calcTPFPFN(gt_points, peaks_CFAR, d_thresh);
    
    % TDPF metrics
    [TP_TDPF, FP_TDPF, FN_TDPF] = calcTPFPFN(gt_points, peaks_tdpf, d_thresh);
    
    % Count ground truth objects
    num_gt = size(gt_points, 1);
    
    %% Create subplot
    nexttile;
    
    % Plot signal in grayscale
    surf(X, Y, scaled_signal(:, :, 2, k), 'EdgeColor', 'none');
    shading interp
    colormap(gca, 'gray')  % Use grayscale colormap for this subplot
    hold on
    
    % Define more visible colors
    color_gt = [1, 0, 0];        % Bright red for ground truth
    color_mp = [1, 0.5, 0];      % Orange for MaxPeaks
    color_cfar = [0, 1, 0];      % Bright green for CA-CFAR
    color_tdpf = [0, 0.8, 1];    % Bright cyan/light blue for TDPF
    
    % Plot ground truth (bright red 'x', largest)
    if ~isempty(gt_points)
        h_gt = plot3(gt_points(:, 1), gt_points(:, 2), ones(size(gt_points(:, 1))), ...
            'x', 'Color', color_gt, 'LineWidth', 3, 'MarkerSize', 30);
    else
        h_gt = plot3(NaN, NaN, NaN, 'x', 'Color', color_gt, 'LineWidth', 3, 'MarkerSize', 16);
    end
    
    % Plot MaxPeaks detections (orange 'o', medium-large)
    if ~isempty(peaks_MP)
        h_mp = plot3(peaks_MP(:, 2), peaks_MP(:, 1), ones(size(peaks_MP(:, 1))), ...
            'o', 'Color', color_mp, 'LineWidth', 2, 'MarkerSize', 12);
    else
        h_mp = plot3(NaN, NaN, NaN, 'o', 'Color', color_mp, 'LineWidth', 2, 'MarkerSize', 12);
    end
    
    % Plot CA-CFAR detections (bright green 'o', medium)
    if ~isempty(peaks_CFAR)
        h_cfar = plot3(peaks_CFAR(:, 2), peaks_CFAR(:, 1), ones(size(peaks_CFAR(:, 1))), ...
            'o', 'Color', color_cfar, 'LineWidth', 2, 'MarkerSize', 8);
    else
        h_cfar = plot3(NaN, NaN, NaN, 'o', 'Color', color_cfar, 'LineWidth', 2, 'MarkerSize', 8);
    end
    
    % Plot TDPF detections (bright cyan 'o', small)
    if ~isempty(peaks_tdpf)
        h_tdpf = plot3(peaks_tdpf(:, 2), peaks_tdpf(:, 1), ones(size(peaks_tdpf(:, 1))), ...
            'o', 'Color', color_tdpf, 'LineWidth', 2, 'MarkerSize', 4);
    else
        h_tdpf = plot3(NaN, NaN, NaN, 'o', 'Color', color_tdpf, 'LineWidth', 2, 'MarkerSize', 4);
    end
    
    % Formatting
    axis square
    view(2)
    xlim([0 128])
    ylim([20 128])
    xlabel('X', 'FontSize', 10)
    ylabel('Y', 'FontSize', 10)
    
    % Create title with metrics included
    title_str = sprintf('Frame %d\nGT: %d | MP: %d/%d/%d | CFAR: %d/%d/%d | TDPF: %d/%d/%d', ...
        k, num_gt, TP_MP, FP_MP, FN_MP, TP_CFAR, FP_CFAR, FN_CFAR, TP_TDPF, FP_TDPF, FN_TDPF);
    t = title(title_str, 'FontSize', 9, 'Interpreter', 'none');
    t.Position(2) = t.Position(2) + 5; % Move title up by 5 units
    hold off
end

% Add overall title and legend explanation (commented out due to cutoff issues)
% sgtitle_str = sprintf('%d Objects (Thresholds: MP=%.2f, CFAR=%.2f, TDPF=%.0f)\nMetrics format: TP/FP/FN', ...
%     NUM_OBJECTS, THRESH_MP, THRESH_CFAR, THRESH_TDPF);
% sgtitle(sgtitle_str, 'FontSize', 13, 'FontWeight', 'bold')

% Save figure
filename = sprintf('frames_obj%d.png', NUM_OBJECTS);
filepath = fullfile(OUTPUT_DIR, filename);
saveas(fig, filepath);
fprintf("\n✓ Saved: %s\n", filepath)

% close(fig)

fprintf("\n✓ All frames saved to: %s\n", OUTPUT_DIR)
