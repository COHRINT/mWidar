clear; clc; close all

%% plot_MC_results_multi
% Plot ROC curves from saved MC detector results
% Supports loading multiple object counts from a single MC run directory
% Each object count gets a different marker shape
% Bijan Jourabchi / Anthony La Barca

%% Configuration
% Set the directory containing all obj*.mat files from a single MC run
MC_RUN_DIR = fullfile('..', 'Results', 'detectors', '20251119_063843_MCRuns100_NThresh20');

fprintf("Loading results from: %s\n", MC_RUN_DIR)

% Find all obj*.mat files in the directory
obj_files = dir(fullfile(MC_RUN_DIR, 'obj*.mat'));

if isempty(obj_files)
    error('No obj*.mat files found in directory: %s', MC_RUN_DIR);
end

fprintf("Found %d object count files:\n", length(obj_files))

%% Load all data files
all_data = cell(length(obj_files), 1);
object_counts = zeros(length(obj_files), 1);

for i = 1:length(obj_files)
    file_path = fullfile(MC_RUN_DIR, obj_files(i).name);
    all_data{i} = load(file_path);
    object_counts(i) = all_data{i}.o;
    fprintf("  - %s (%d objects)\n", obj_files(i).name, object_counts(i));
end

% Sort by object count for consistent ordering
[object_counts, sort_idx] = sort(object_counts);
all_data = all_data(sort_idx);

% Get common parameters from first file
data = all_data{1};
detectors_list = data.detectors_list;
detectors_count = data.detectors_count;
thresh_ranges = data.thresh_ranges;
thresh_labels = data.thresh_labels;
MC_RUNS = data.MC_RUNS;
NUM_THRESHOLDS = data.NUM_THRESHOLDS;

fprintf("\nLoaded %d MC runs with %d thresholds per detector\n\n", MC_RUNS, NUM_THRESHOLDS)

% Define marker shapes for different object counts
marker_shapes = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h', '*'};

%% Plot 1: Individual scatter plots for each object count
for obj_idx = 1:length(all_data)
    data = all_data{obj_idx};
    o = object_counts(obj_idx);
    marker = marker_shapes{mod(obj_idx-1, length(marker_shapes)) + 1};
    
    figure_name = sprintf("ROC_OBJ_COUNT%d.png", o);
    fig = figure('Name', figure_name, 'Color', 'w', 'Position', [100, 100, 1400, 400]);

    for detector = 1:detectors_count
        ax = subplot(1, detectors_count, detector);
        
        % Call plotting method for scatter plot
        plot_scatter_all_runs(ax, data, detector, o, marker, ...
            thresh_ranges, thresh_labels, MC_RUNS, NUM_THRESHOLDS, detectors_list);
    end
end

fprintf("Completed Individual Scatter Plots");

%% Plot 2: Combined plot - All object counts on same figure
fig2 = figure('Name', 'ROC Combined - All Object Counts', 'NumberTitle', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1400, 400]);

for detector = 1:detectors_count
    ax = subplot(1, detectors_count, detector);
    
    % Call plotting method for combined mean curves
    plot_combined_mean_curves(ax, all_data, object_counts, marker_shapes, detector, ...
        thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list);
end

fprintf("Completed Combined Plots ");
%% Plot 3: Individual error bar plots for each object count
for obj_idx = 1:length(all_data)
    data = all_data{obj_idx};
    o = object_counts(obj_idx);
    marker = marker_shapes{mod(obj_idx-1, length(marker_shapes)) + 1};
    
    fig3 = figure('Name', sprintf('ROC Error Bars - %d Objects', o), 'NumberTitle', 'off', ...
        'Color', 'w', 'Position', [100, 100, 1400, 400]);

    for detector = 1:detectors_count
        ax = subplot(1, detectors_count, detector);
        
        % Call plotting method for error bars with fitted curve
        plot_error_bars_with_fit(ax, data, detector, o, marker, ...
            thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list);
    end
end

fprintf("Completed Individual Error Bar Plots plots");
fprintf("Plotting complete!\n")

%% ======================== PLOTTING METHODS ========================

function plot_scatter_all_runs(ax, data, detector, obj_count, marker, ...
    thresh_ranges, thresh_labels, MC_RUNS, NUM_THRESHOLDS, detectors_list)
    % Plot scatter of all MC runs and thresholds for a single detector/object count
    %
    % Inputs:
    %   ax - axes handle to plot on
    %   data - loaded data structure containing FPR, TPR arrays
    %   detector - detector index (1, 2, or 3)
    %   obj_count - number of objects in this dataset
    %   marker - marker shape string (e.g., 'o', 's', '^')
    %   thresh_ranges - cell array of threshold ranges for each detector
    %   thresh_labels - cell array of threshold labels
    %   MC_RUNS - number of Monte Carlo runs
    %   NUM_THRESHOLDS - number of threshold values tested
    %   detectors_list - cell array of detector names
    
    hold(ax, 'on'); 
    grid(ax, 'on');

    % Get threshold range for this detector
    thresh_vals = thresh_ranges{detector};

    % Plot all MC_run/threshold combinations
    for m = 1:MC_RUNS
        for t = 1:NUM_THRESHOLDS
            % Use threshold value for color
            scatter(ax, data.FPR(detector, m, t), data.TPR(detector, m, t), 50, ...
                thresh_vals(t), 'filled', marker, 'MarkerEdgeColor', 'k');
        end
    end

    % Add colorbar
    c = colorbar(ax);
    c.Label.String = thresh_labels{detector};
    c.Label.Interpreter = 'tex';
    colormap(ax, parula);
    caxis(ax, [min(thresh_vals), max(thresh_vals)]);

    % Plot "Random" line
    plot(ax, [0 1], [0 1], '--k', 'LineWidth', 1, "HandleVisibility", "off");

    % Axis formatting
    axis(ax, 'square')
    xlim(ax, [0 1])
    ylim(ax, [0 1])

    % Labels
    xlabel(ax, 'False Positive Rate', 'Interpreter', 'latex')
    ylabel(ax, 'True Positive Rate', 'Interpreter', 'latex')
    title(ax, sprintf('%s - Obj Count: %d', detectors_list(detector), obj_count), 'Interpreter', 'latex')
end

function plot_combined_mean_curves(ax, all_data, object_counts, marker_shapes, detector, ...
    thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list)
    % Plot mean ROC curves for all object counts on same axes
    % Colors points by threshold value, differentiates object counts by marker shape
    %
    % Inputs:
    %   ax - axes handle to plot on
    %   all_data - cell array of loaded data structures
    %   object_counts - array of object counts corresponding to all_data
    %   marker_shapes - cell array of marker shapes
    %   detector - detector index
    %   thresh_ranges - cell array of threshold ranges for each detector
    %   thresh_labels - cell array of threshold labels
    %   NUM_THRESHOLDS - number of threshold values tested
    %   detectors_list - cell array of detector names
    
    hold(ax, 'on'); 
    grid(ax, 'on');
    
    % Get threshold range and colormap for this detector
    thresh_vals = thresh_ranges{detector};
    
    % Plot each object count with different marker, colored by threshold
    for obj_idx = 1:length(all_data)
        data = all_data{obj_idx};
        o = object_counts(obj_idx);
        marker = marker_shapes{mod(obj_idx-1, length(marker_shapes)) + 1};

        % Compute mean for each threshold
        mean_FPR = zeros(NUM_THRESHOLDS, 1);
        mean_TPR = zeros(NUM_THRESHOLDS, 1);

        for t = 1:NUM_THRESHOLDS
            mean_FPR(t) = mean(data.FPR(detector, :, t), 'omitnan');
            mean_TPR(t) = mean(data.TPR(detector, :, t), 'omitnan');
        end

        % Plot each threshold point individually to color by threshold value
        for t = 1:NUM_THRESHOLDS
            scatter(ax, mean_FPR(t), mean_TPR(t), 100, thresh_vals(t), marker, ...
                'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        end
        
        % Add connecting line for this object count
        plot(ax, mean_FPR, mean_TPR, 'LineStyle', '-', 'LineWidth', 1.5, ...
            'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    end
    
    % Create custom legend for marker shapes
    legend_handles = [];
    legend_labels = {};
    for obj_idx = 1:length(all_data)
        o = object_counts(obj_idx);
        marker = marker_shapes{mod(obj_idx-1, length(marker_shapes)) + 1};
        h = plot(ax, NaN, NaN, marker, 'MarkerSize', 10, 'LineWidth', 1.5, ...
            'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerEdgeColor', 'k');
        legend_handles = [legend_handles; h];
        legend_labels{end+1} = sprintf('%d objects', o);
    end

    % Add colorbar for threshold values
    c = colorbar(ax);
    c.Label.String = thresh_labels{detector};
    c.Label.Interpreter = 'tex';
    colormap(ax, parula);
    caxis(ax, [min(thresh_vals), max(thresh_vals)]);

    % Plot "Random" line
    plot(ax, [0 1], [0 1], '--k', 'LineWidth', 1, "HandleVisibility", "off");

    % Axis formatting
    axis(ax, 'square')
    
    ylim(ax, [8.5e-1 1])
    xlim(ax, [8.5e-1 1])

    % Labels
    xlabel(ax, 'False Positive Rate', 'Interpreter', 'latex')
    ylabel(ax, 'True Positive Rate', 'Interpreter', 'latex')
    title(ax, sprintf('%s - All Object Counts', detectors_list(detector)), 'Interpreter', 'latex')

    % Legend for marker shapes
    legend(legend_handles, legend_labels, 'Location', 'southeast')
end

function plot_error_bars_with_fit(ax, data, detector, obj_count, marker, ...
    thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list)
    % Plot error bars (mean ± std) with fitted curve for a single detector/object count
    %
    % Inputs:
    %   ax - axes handle to plot on
    %   data - loaded data structure containing FPR, TPR arrays
    %   detector - detector index
    %   obj_count - number of objects in this dataset
    %   marker - marker shape string
    %   thresh_ranges - cell array of threshold ranges
    %   thresh_labels - cell array of threshold labels
    %   NUM_THRESHOLDS - number of threshold values tested
    %   detectors_list - cell array of detector names
    
    hold(ax, 'on'); 
    grid(ax, 'on');

    % Get threshold range for this detector
    thresh_vals = thresh_ranges{detector};

    % Compute mean and std for each threshold
    mean_FPR = zeros(NUM_THRESHOLDS, 1);
    std_FPR = zeros(NUM_THRESHOLDS, 1);
    mean_TPR = zeros(NUM_THRESHOLDS, 1);
    std_TPR = zeros(NUM_THRESHOLDS, 1);

    for t = 1:NUM_THRESHOLDS
        mean_FPR(t) = mean(data.FPR(detector, :, t), 'omitnan');
        std_FPR(t) = std(data.FPR(detector, :, t), 'omitnan');
        mean_TPR(t) = mean(data.TPR(detector, :, t), 'omitnan');
        std_TPR(t) = std(data.TPR(detector, :, t), 'omitnan');
    end

    % Get colormap for threshold coloring
    cmap = parula(NUM_THRESHOLDS);

    % Plot error bars colored by threshold
    for t = 1:NUM_THRESHOLDS
        errorbar(ax, mean_FPR(t), mean_TPR(t), std_TPR(t), std_TPR(t), ...
            std_FPR(t), std_FPR(t), marker, 'MarkerSize', 8, 'LineWidth', 2, ...
            'CapSize', 10, 'Color', cmap(t, :), 'MarkerFaceColor', cmap(t, :));
    end

    % Recolor points by threshold using scatter (for proper colorbar)
    for t = 1:NUM_THRESHOLDS
        scatter(ax, mean_FPR(t), mean_TPR(t), 100, thresh_vals(t), marker, ...
            'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    end

    % Fit a curve to the mean values (sort by FPR first)
    [sorted_FPR, sort_idx] = sort(mean_FPR);
    sorted_TPR = mean_TPR(sort_idx);

    % Remove NaN values and duplicate FPR values for fitting
    valid_idx = ~isnan(sorted_FPR) & ~isnan(sorted_TPR);
    sorted_FPR = sorted_FPR(valid_idx);
    sorted_TPR = sorted_TPR(valid_idx);

    % Remove duplicate FPR values (keep first occurrence)
    [sorted_FPR, unique_idx] = unique(sorted_FPR, 'stable');
    sorted_TPR = sorted_TPR(unique_idx);

    if length(sorted_FPR) >= 3
        % Fit a smooth curve using spline interpolation
        FPR_fit = linspace(min(sorted_FPR), max(sorted_FPR), 100);
        TPR_fit = interp1(sorted_FPR, sorted_TPR, FPR_fit, 'pchip');
        plot(ax, FPR_fit, TPR_fit, '-', 'LineWidth', 2.5, 'Color', [0.2, 0.2, 0.2], ...
            'DisplayName', 'Fitted Curve');
    end

    % Add colorbar
    c = colorbar(ax);
    c.Label.String = strrep(thresh_labels{detector}, '_', '\_');
    c.Label.Interpreter = 'tex';
    colormap(ax, parula);
    caxis(ax, [min(thresh_vals), max(thresh_vals)]);

    % Plot "Random" line
    plot(ax, [0 1], [0 1], '--k', 'LineWidth', 1, "HandleVisibility", "off");

    % Axis formatting
    axis(ax, 'square')
    xlim(ax, [8.5e-1 1])
    ylim(ax, [8.5e-1 1])

    % Labels
    xlabel(ax, 'False Positive Rate', 'Interpreter', 'latex')
    ylabel(ax, 'True Positive Rate', 'Interpreter', 'latex')
    title(ax, sprintf('%s - Obj Count: %d (Mean \\pm 1SD)', ...
        strrep(detectors_list(detector), '_', '\_'), obj_count), 'Interpreter', 'tex')
end
