clear; clc; close all

%% plot_MC_results
% Plot ROC curves from saved MC detector results
% Bijan Jourabchi / Anthony La Barca

%% Configuration
% Set the data file to load (hardcoded for ease of use)
MC_RUN_DATAFILE = fullfile('..', 'Results', 'detectors', '20251119_060525_MCRuns100_NThresh20','obj5.mat');
fprintf("Loading results from: %s\n", MC_RUN_DATAFILE)

% Load the results
load(MC_RUN_DATAFILE);

fprintf("Loaded results for %d objects, %d MC runs, %d thresholds\n", o, MC_RUNS, NUM_THRESHOLDS)

%% Plot 1: Scatter plot with all MC runs and thresholds
figure_name = sprintf("ROC_OBJ_COUNT%d.png", o);
fig = figure('Name', figure_name, 'Color', 'w', 'Position', [100, 100, 1400, 400]);

for detector = 1:detectors_count
    subplot(1, detectors_count, detector); hold on; grid on

    % Get threshold range for this detector
    thresh_vals = thresh_ranges{detector};

    % Plot all MC_run/threshold combinations
    for m = 1:MC_RUNS

        for t = 1:NUM_THRESHOLDS
            % Use threshold value for color
            scatter(FPR(detector, m, t), TPR(detector, m, t), 50, thresh_vals(t), 'filled', 'MarkerEdgeColor', 'k');
        end

    end

    % Add colorbar
    c = colorbar;
    c.Label.String = thresh_labels{detector};
    c.Label.Interpreter = 'tex';
    colormap(gca, parula);
    caxis([min(thresh_vals), max(thresh_vals)]);

    % Plot "Random" line
    plot([0 1], [0 1], '--k', 'LineWidth', 1, "HandleVisibility", "off");


    % Axis with proper limits for log scale
    axis('square')
    xlim([0 1]) % Log scale needs lower bound > 0
    ylim([0 1])

    % Labels
    xlabel('False Positive Rate (log scale)', 'Interpreter', 'latex')
    ylabel('True Positive Rate', 'Interpreter', 'latex')
    title(sprintf('%s - Obj Count: %d', detectors_list(detector), o), 'Interpreter', 'latex')

    % Grid
    grid('on')
end

%% Plot 2: Aggregated ROC plot with error bars and fitted curves
fig2 = figure('Name', 'ROC with Error Bars and Fit', 'NumberTitle', 'off', 'Color', 'w', 'Position', [100, 100, 1400, 400]);

for detector = 1:detectors_count
    subplot(1, detectors_count, detector); hold on; grid on

    % Get threshold range for this detector
    thresh_vals = thresh_ranges{detector};

    % Compute mean and std for each threshold
    mean_FPR = zeros(NUM_THRESHOLDS, 1);
    std_FPR = zeros(NUM_THRESHOLDS, 1);
    mean_TPR = zeros(NUM_THRESHOLDS, 1);
    std_TPR = zeros(NUM_THRESHOLDS, 1);

    for t = 1:NUM_THRESHOLDS
        mean_FPR(t) = mean(FPR(detector, :, t), 'omitnan');
        std_FPR(t) = std(FPR(detector, :, t), 'omitnan');
        mean_TPR(t) = mean(TPR(detector, :, t), 'omitnan');
        std_TPR(t) = std(TPR(detector, :, t), 'omitnan');
    end

    % Get colormap for threshold coloring
    cmap = parula(NUM_THRESHOLDS);

    % Plot error bars colored by threshold
    for t = 1:NUM_THRESHOLDS
        % Plot error bars with specific color for this threshold
        errorbar(mean_FPR(t), mean_TPR(t), std_TPR(t), std_TPR(t), std_FPR(t), std_FPR(t), ...
            'o', 'MarkerSize', 8, 'LineWidth', 2, 'CapSize', 10, ...
            'Color', cmap(t, :), 'MarkerFaceColor', cmap(t, :));
    end

    % Recolor points by threshold using scatter (to get proper colorbar)
    for t = 1:NUM_THRESHOLDS
        scatter(mean_FPR(t), mean_TPR(t), 100, thresh_vals(t), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
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
        plot(FPR_fit, TPR_fit, '-', 'LineWidth', 2.5, 'Color', [0.2, 0.2, 0.2], 'DisplayName', 'Fitted Curve');
    end

    % Add colorbar
    c = colorbar;
    c.Label.String = strrep(thresh_labels{detector}, '_', '\_'); % Escape underscores
    c.Label.Interpreter = 'tex';
    colormap(gca, parula);
    caxis([min(thresh_vals), max(thresh_vals)]);

    % Plot "Random" line
    plot([0 1], [0 1], '--k', 'LineWidth', 1, "HandleVisibility", "off");

    % Use semi-log scale (log on FPR, linear on TPR)
    % set(gca, 'XScale', 'log');

    % Axis with proper limits for log scale
    axis('square')
    xlim([8.5e-1 1]) % Log scale needs lower bound > 0
    ylim([8.5e-1 1])

    % Labels
    xlabel('False Positive Rate (log scale)', 'Interpreter', 'latex')
    ylabel('True Positive Rate', 'Interpreter', 'latex')
    title(sprintf('%s - Obj Count: %d (Mean \\pm 1SD)', strrep(detectors_list(detector), '_', '\_'), o), 'Interpreter', 'tex')

    % Grid
    grid('on')
end

fprintf("Plotting complete!\n")
