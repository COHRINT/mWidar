clear; clc; close all

%% plot_MC_results_multi
% Plot ROC curves from saved MC detector results
% Supports loading multiple object counts from a single MC run directory
% Each object count gets a different marker shape
% Bijan Jourabchi / Anthony La Barca

%% Configuration
% Set the directory containing all obj*.mat files from a series of MC Runs
MC_RUN_NAME = '20251120_115621_MCRuns100_NThresh20';
MC_RUN_DIR = fullfile('..', 'Results', 'detectors', MC_RUN_NAME);

MC_FIG_SAVEDIR = fullfile('..', 'figures', 'Detectors', 'MCRunResults', MC_RUN_NAME);

if ~exist(MC_FIG_SAVEDIR, 'dir')
    mkdir(MC_FIG_SAVEDIR);
end

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

%% ======================== CALCULATE ADVANCED STATISTICS ========================

fprintf("\n\n############# CALCULATING ADVANCED STATISTICS #############\n\n")

% Initialize statistics structure for each object count
all_stats = cell(length(all_data), 1);

for obj_idx = 1:length(all_data)
    data = all_data{obj_idx};
    o = object_counts(obj_idx);
    
    fprintf("Calculating statistics for %d object(s)...\n", o)
    
    stats = struct();
    
    for detector = 1:detectors_count
        detector_name = char(detectors_list(detector));
        detector_field = strrep(detector_name, '-', '_');
        
        % Extract data for this detector across all MC runs and thresholds
        precision_data = squeeze(data.Precision(detector, :, :)); % (MC_runs x thresholds)
        recall_data = squeeze(data.Recall(detector, :, :));       % (MC_runs x thresholds)
        TPR_data = squeeze(data.TPR(detector, :, :));             % (MC_runs x thresholds)
        FPR_data = squeeze(data.FPR(detector, :, :));             % (MC_runs x thresholds)
        
        %% 1. Calculate F1 Score for each MC run and threshold
        % F1 Score: Harmonic mean of Precision and Recall
        % GOOD VALUES: F1 > 0.8 (excellent), 0.6-0.8 (good), < 0.6 (poor)
        % Higher is better; balances precision and recall
        F1_data = 2 * (precision_data .* recall_data) ./ (precision_data + recall_data);
        F1_data(isnan(F1_data)) = 0; % Handle division by zero
        
        % Mean and std of F1 across MC runs for each threshold
        F1_mean = mean(F1_data, 1, 'omitnan')'; % (thresholds x 1)
        F1_std = std(F1_data, 0, 1, 'omitnan')';
        
        %% 2. Find best operating point (highest mean F1)
        % Best threshold maximizes F1 score - optimal balance of precision/recall
        [best_F1, best_idx] = max(F1_mean);
        
        stats.(detector_field).best_threshold_idx = best_idx;
        stats.(detector_field).best_threshold_value = thresh_ranges{detector}(best_idx);
        stats.(detector_field).best_F1_mean = best_F1;
        stats.(detector_field).best_F1_std = F1_std(best_idx);
        stats.(detector_field).best_Precision_mean = mean(precision_data(:, best_idx), 'omitnan');
        stats.(detector_field).best_Precision_std = std(precision_data(:, best_idx), 'omitnan');
        stats.(detector_field).best_Recall_mean = mean(recall_data(:, best_idx), 'omitnan');
        stats.(detector_field).best_Recall_std = std(recall_data(:, best_idx), 'omitnan');
        
        %% 3. Calculate AUC-PR (Area Under Precision-Recall Curve)
        % AUC-PR: Summary of precision-recall tradeoff across all thresholds
        % GOOD VALUES: For detection tasks with low precision/high recall preference:
        %   - High recall (>0.9) is critical - we want to catch all objects
        %   - Precision can be lower (>0.1) - false alarms are acceptable
        %   - AUC-PR depends on application: higher is better but interpret with recall priority
        % Higher is better; less sensitive to class imbalance than AUC-ROC
        % Use mean values across MC runs for each threshold
        mean_precision = mean(precision_data, 1, 'omitnan')';
        mean_recall = mean(recall_data, 1, 'omitnan')';
        
        % Sort by recall for proper integration
        [sorted_recall, sort_idx] = sort(mean_recall);
        sorted_precision = mean_precision(sort_idx);
        
        % Remove NaN values
        valid = ~isnan(sorted_recall) & ~isnan(sorted_precision);
        
        if sum(valid) > 1
            % Use trapezoidal integration
            stats.(detector_field).AUC_PR = trapz(sorted_recall(valid), sorted_precision(valid));
        else
            stats.(detector_field).AUC_PR = NaN;
        end
        
        %% 4. Calculate 95% Confidence Intervals using bootstrapping
        % Confidence Intervals: Range where true mean likely falls
        % INTERPRETATION: Narrower CI = more reliable estimate, wider CI = more uncertainty
        % For best operating point only (to save computation)
        if MC_RUNS >= 10 % Only do bootstrap if we have enough samples
            n_bootstrap = 1000;
            alpha = 0.05; % 95% CI
            
            % Bootstrap F1 at best threshold
            F1_boot = bootstrp(n_bootstrap, @mean, F1_data(:, best_idx));
            stats.(detector_field).best_F1_CI = prctile(F1_boot, [alpha/2, 1-alpha/2]*100);
            
            % Bootstrap Precision at best threshold
            Precision_boot = bootstrp(n_bootstrap, @mean, precision_data(:, best_idx));
            stats.(detector_field).best_Precision_CI = prctile(Precision_boot, [alpha/2, 1-alpha/2]*100);
            
            % Bootstrap Recall at best threshold
            Recall_boot = bootstrp(n_bootstrap, @mean, recall_data(:, best_idx));
            stats.(detector_field).best_Recall_CI = prctile(Recall_boot, [alpha/2, 1-alpha/2]*100);
        else
            % Use t-distribution for small samples
            stats.(detector_field).best_F1_CI = [NaN, NaN];
            stats.(detector_field).best_Precision_CI = [NaN, NaN];
            stats.(detector_field).best_Recall_CI = [NaN, NaN];
        end
        
        %% 5. Store full distributions for later analysis
        stats.(detector_field).F1_mean_all = F1_mean;
        stats.(detector_field).F1_std_all = F1_std;
        stats.(detector_field).Precision_mean_all = mean_precision;
        stats.(detector_field).Precision_std_all = std(precision_data, 0, 1, 'omitnan')';
        stats.(detector_field).Recall_mean_all = mean_recall;
        stats.(detector_field).Recall_std_all = std(recall_data, 0, 1, 'omitnan')';
        
    end
    
    % Store statistics for this object count
    all_stats{obj_idx} = stats;
    
    %% Print Summary Table for this object count
    fprintf("\n========== SUMMARY STATISTICS FOR %d OBJECT(S) ==========\n", o)
    fprintf("%-10s | %-10s | %-10s | %-10s | %-10s\n", ...
        'Detector', 'Recall±σ', 'Prec±σ', 'F1±σ', 'AUC-PR')
    fprintf(repmat('-', 1, 75))
    fprintf("\n")
    
    for detector = 1:detectors_count
        detector_name = char(detectors_list(detector));
        detector_field = strrep(detector_name, '-', '_');
        s = stats.(detector_field);
        
        fprintf("%-10s | %.3f±%.3f | %.3f±%.3f | %.3f±%.3f | %.4f\n", ...
            detector_name, ...
            s.best_Recall_mean, s.best_Recall_std, ...
            s.best_Precision_mean, s.best_Precision_std, ...
            s.best_F1_mean, s.best_F1_std, ...
            s.AUC_PR)
    end
    
    fprintf(repmat('=', 1, 75))
    fprintf("\n")
    
    if MC_RUNS >= 10
        fprintf("\n95%% Confidence Intervals (Bootstrap) - Prioritized by Recall:\n")
        for detector = 1:detectors_count
            detector_name = char(detectors_list(detector));
            detector_field = strrep(detector_name, '-', '_');
            s = stats.(detector_field);
            
            fprintf("%-10s | Recall: [%.3f, %.3f] | Precision: [%.3f, %.3f]\n", ...
                detector_name, ...
                s.best_Recall_CI(1), s.best_Recall_CI(2), ...
                s.best_Precision_CI(1), s.best_Precision_CI(2))
        end
        fprintf("\n")
    end
    
    %% Generate LaTeX table for this object count
    % Create custom LaTeX table with "mean ± std" formatting
    
    % Create caption
    caption_str = sprintf('Performance Metrics for %d Object(s) at Best Operating Point (%d MC Runs)', o, MC_RUNS);
    
    % Build LaTeX string manually for custom formatting
    latex_str = sprintf('\\begin{table}[htbp]\n');
    latex_str = [latex_str sprintf('\\centering\n')];
    latex_str = [latex_str sprintf('\\caption{%s}\n', caption_str)];
    latex_str = [latex_str sprintf('%% Requires \\usepackage{booktabs}\n')];
    latex_str = [latex_str sprintf('\\begin{tabular}{c|cc}\n')];
    latex_str = [latex_str sprintf('\\toprule\n')];
    latex_str = [latex_str sprintf(' & Recall (\\%%) & Precision (\\%%) \\\\\n')];
    latex_str = [latex_str sprintf('\\midrule\n')];
    
    for detector = 1:detectors_count
        detector_name = char(detectors_list(detector));
        detector_field = strrep(detector_name, '-', '_');
        s = stats.(detector_field);
        
        % Format as "mean ± std" in percentages
        recall_str = sprintf('%.1f $\\pm$ %.1f', s.best_Recall_mean * 100, s.best_Recall_std * 100);
        precision_str = sprintf('%.1f $\\pm$ %.1f', s.best_Precision_mean * 100, s.best_Precision_std * 100);
        
        latex_str = [latex_str sprintf('%s & %s & %s \\\\\n', detector_name, recall_str, precision_str)];
    end
    
    latex_str = [latex_str sprintf('\\bottomrule\n')];
    latex_str = [latex_str sprintf('\\end{tabular}\n')];
    latex_str = [latex_str sprintf('\\end{table}\n')];
    
    % Save to file
    latex_filename = sprintf('stats_table_obj%d.tex', o);
    latex_filepath = fullfile(MC_FIG_SAVEDIR, latex_filename);
    fid = fopen(latex_filepath, 'w');
    fprintf(fid, '%s', latex_str);
    fclose(fid);
    
    fprintf("LaTeX table saved to: %s\n\n", latex_filepath);
    
end

% Save all statistics to file
stats_save_path = fullfile(MC_FIG_SAVEDIR, 'advanced_statistics.mat');
save(stats_save_path, 'all_stats', 'object_counts', 'detectors_list');
fprintf("Advanced statistics saved to: %s\n\n", stats_save_path)

fprintf("############# STATISTICS CALCULATION COMPLETE #############\n\n")

%% ======================== AGGREGATE STATISTICS ACROSS ALL OBJECT COUNTS ========================

fprintf("\n\n############# CALCULATING AGGREGATE STATISTICS ACROSS ALL SCENARIOS #############\n\n")

% Initialize aggregate statistics structure
aggregate_stats = struct();

for detector = 1:detectors_count
    detector_name = char(detectors_list(detector));
    detector_field = strrep(detector_name, '-', '_');
    
    % Collect all F1, Precision, Recall values across ALL object counts, MC runs, and thresholds
    all_F1 = [];
    all_Precision = [];
    all_Recall = [];
    all_TPR = [];
    all_FPR = [];
    
    % Also collect data at best operating point for each object count
    best_F1_by_obj = [];
    best_Precision_by_obj = [];
    best_Recall_by_obj = [];
    
    % For weighted AUC (weight by object count complexity)
    AUC_PR_by_obj = [];
    
    for obj_idx = 1:length(all_data)
        data = all_data{obj_idx};
        stats = all_stats{obj_idx};
        
        % Extract all data for this object count
        precision_data = squeeze(data.Precision(detector, :, :)); % (MC_runs x thresholds)
        recall_data = squeeze(data.Recall(detector, :, :));
        TPR_data = squeeze(data.TPR(detector, :, :));
        FPR_data = squeeze(data.FPR(detector, :, :));
        
        % Calculate F1 for all points
        F1_data = 2 * (precision_data .* recall_data) ./ (precision_data + recall_data);
        F1_data(isnan(F1_data)) = 0;
        
        % Concatenate all values
        all_F1 = [all_F1; F1_data(:)];
        all_Precision = [all_Precision; precision_data(:)];
        all_Recall = [all_Recall; recall_data(:)];
        all_TPR = [all_TPR; TPR_data(:)];
        all_FPR = [all_FPR; FPR_data(:)];
        
        % Extract best operating point for this object count
        best_idx = stats.(detector_field).best_threshold_idx;
        best_F1_by_obj = [best_F1_by_obj; F1_data(:, best_idx)]; % All MC runs at best threshold
        best_Precision_by_obj = [best_Precision_by_obj; precision_data(:, best_idx)];
        best_Recall_by_obj = [best_Recall_by_obj; recall_data(:, best_idx)];
        
        % Collect AUC values
        AUC_PR_by_obj = [AUC_PR_by_obj; stats.(detector_field).AUC_PR];
    end
    
    %% 1. Overall statistics across ALL scenarios (all object counts, MC runs, thresholds)
    % These metrics show average performance across entire parameter space
    % INTERPRETATION: Shows general capability but may hide optimal performance
    % NOTE: For detection, RECALL is the priority metric (catch all objects, tolerate false alarms)
    aggregate_stats.(detector_field).overall_Recall_mean = mean(all_Recall, 'omitnan');
    aggregate_stats.(detector_field).overall_Recall_std = std(all_Recall, 'omitnan');
    aggregate_stats.(detector_field).overall_Recall_median = median(all_Recall, 'omitnan');
    
    aggregate_stats.(detector_field).overall_Precision_mean = mean(all_Precision, 'omitnan');
    aggregate_stats.(detector_field).overall_Precision_std = std(all_Precision, 'omitnan');
    
    aggregate_stats.(detector_field).overall_F1_mean = mean(all_F1, 'omitnan');
    aggregate_stats.(detector_field).overall_F1_std = std(all_F1, 'omitnan');
    aggregate_stats.(detector_field).overall_F1_median = median(all_F1, 'omitnan');
    
    %% 2. Best operating point statistics (across all object counts)
    % *** MOST IMPORTANT METRICS FOR PUBLICATION ***
    % These show real-world performance at optimal threshold for each scenario
    % PRIORITY: Recall > 0.9 (critical - must catch all objects)
    % ACCEPTABLE: Precision > 0.1 (false alarms are tolerable in detection)
    % Lower std = more consistent across scenarios
    aggregate_stats.(detector_field).best_Recall_mean = mean(best_Recall_by_obj, 'omitnan');
    aggregate_stats.(detector_field).best_Recall_std = std(best_Recall_by_obj, 'omitnan');
    aggregate_stats.(detector_field).best_Recall_median = median(best_Recall_by_obj, 'omitnan');
    
    aggregate_stats.(detector_field).best_Precision_mean = mean(best_Precision_by_obj, 'omitnan');
    aggregate_stats.(detector_field).best_Precision_std = std(best_Precision_by_obj, 'omitnan');
    
    aggregate_stats.(detector_field).best_F1_mean = mean(best_F1_by_obj, 'omitnan');
    aggregate_stats.(detector_field).best_F1_std = std(best_F1_by_obj, 'omitnan');
    aggregate_stats.(detector_field).best_F1_median = median(best_F1_by_obj, 'omitnan');
    
    %% 3. Mean AUC across all object counts
    % Average of AUC-PR computed for each scenario - summary of precision-recall tradeoff
    % INTERPRETATION: Higher is better, but prioritize high recall over high AUC-PR
    % Lower std = consistent performance across different complexities
    aggregate_stats.(detector_field).mean_AUC_PR = mean(AUC_PR_by_obj, 'omitnan');
    aggregate_stats.(detector_field).std_AUC_PR = std(AUC_PR_by_obj, 'omitnan');
    
    %% 4. Robustness metrics (how consistent is performance across scenarios?)
    % Coefficient of Variation (CV = std/mean): Normalized variability measure for RECALL
    % GOOD VALUES: CV < 0.1 (excellent consistency), 0.1-0.2 (good), > 0.2 (variable)
    % Lower is better - indicates stable recall across scenarios
    aggregate_stats.(detector_field).Recall_CV = std(best_Recall_by_obj, 'omitnan') / mean(best_Recall_by_obj, 'omitnan');
    
    % Inter-quartile range (IQR): Spread of middle 50% of recall performance
    % GOOD VALUES: Smaller IQR = more consistent, compare relative to mean
    aggregate_stats.(detector_field).Recall_IQR = iqr(best_Recall_by_obj);
    
    %% 5. 95% Confidence intervals for aggregate performance
    % Statistical uncertainty in estimated mean performance
    % INTERPRETATION: We're 95% confident the true mean lies in this range
    % Narrower intervals = more reliable estimate (need more MC runs for narrow CI)
    if length(best_F1_by_obj) >= 10
        n_bootstrap = 1000;
        alpha = 0.05;
        
        F1_boot = bootstrp(n_bootstrap, @mean, best_F1_by_obj);
        aggregate_stats.(detector_field).best_F1_CI = prctile(F1_boot, [alpha/2, 1-alpha/2]*100);
        
        Precision_boot = bootstrp(n_bootstrap, @mean, best_Precision_by_obj);
        aggregate_stats.(detector_field).best_Precision_CI = prctile(Precision_boot, [alpha/2, 1-alpha/2]*100);
        
        Recall_boot = bootstrp(n_bootstrap, @mean, best_Recall_by_obj);
        aggregate_stats.(detector_field).best_Recall_CI = prctile(Recall_boot, [alpha/2, 1-alpha/2]*100);
    else
        aggregate_stats.(detector_field).best_F1_CI = [NaN, NaN];
        aggregate_stats.(detector_field).best_Precision_CI = [NaN, NaN];
        aggregate_stats.(detector_field).best_Recall_CI = [NaN, NaN];
    end
end

%% Print Aggregate Statistics Table
fprintf("\n========== AGGREGATE STATISTICS ACROSS ALL OBJECT COUNTS ==========\n")
fprintf("%-10s | %-12s | %-12s | %-12s | %-10s | %-8s\n", ...
    'Detector', 'Recall*', 'Precision', 'F1', 'AUC-PR', 'CV(Rec)')
fprintf(repmat('-', 1, 85))
fprintf("\n")

for detector = 1:detectors_count
    detector_name = char(detectors_list(detector));
    detector_field = strrep(detector_name, '-', '_');
    s = aggregate_stats.(detector_field);
    
    fprintf("%-10s | %.3f±%.3f | %.3f±%.3f | %.3f±%.3f | %.4f±%.3f | %.3f\n", ...
        detector_name, ...
        s.best_Recall_mean, s.best_Recall_std, ...
        s.best_Precision_mean, s.best_Precision_std, ...
        s.best_F1_mean, s.best_F1_std, ...
        s.mean_AUC_PR, s.std_AUC_PR, ...
        s.Recall_CV)
end

fprintf(repmat('=', 1, 85))
fprintf("\n")

% Print confidence intervals if available
if length(all_data) * MC_RUNS >= 10
    fprintf("\n95%% Confidence Intervals (Bootstrap) for Best Operating Point:\n")
    for detector = 1:detectors_count
        detector_name = char(detectors_list(detector));
        detector_field = strrep(detector_name, '-', '_');
        s = aggregate_stats.(detector_field);
        
        fprintf("%-10s | Recall: [%.3f, %.3f] | Precision: [%.3f, %.3f]\n", ...
            detector_name, ...
            s.best_Recall_CI(1), s.best_Recall_CI(2), ...
            s.best_Precision_CI(1), s.best_Precision_CI(2))
    end
end

fprintf("\n")
fprintf("Legend (* indicates PRIMARY metric for detection tasks):\n")
fprintf("  Recall*     : TP/(TP+FN) - fraction of true objects detected [PRIMARY METRIC]\n")
fprintf("                GOOD: >0.90 (catch nearly all objects) | CRITICAL: >0.85 minimum | HIGHER IS BETTER\n")
fprintf("  Precision   : TP/(TP+FP) - fraction of detections that are correct\n")
fprintf("                ACCEPTABLE: >0.10 for detection (false alarms tolerable) | HIGHER IS BETTER\n")
fprintf("  F1          : Harmonic mean of Precision and Recall\n")
fprintf("                NOTE: Low F1 is OK if Recall is high (detection prioritizes recall)\n")
fprintf("  AUC-PR      : Area Under Precision-Recall curve - tradeoff summary\n")
fprintf("                Context-dependent; prioritize high recall region of curve\n")
fprintf("  CV(Rec)     : Coefficient of Variation for Recall (std/mean) - consistency measure\n")
fprintf("                GOOD: <0.05 (very consistent), 0.05-0.10 (acceptable), >0.10 (variable) | LOWER IS BETTER\n")
fprintf("\n")
fprintf("  ±σ notation indicates standard deviation across scenarios\n")
fprintf("  Smaller σ values indicate more consistent performance across different scenarios\n")
fprintf("\n")

%% Generate LaTeX table for aggregate statistics
% Create custom LaTeX table with "mean ± std" formatting

% Create caption
caption_str = sprintf('Aggregate Performance Metrics Across All Object Counts (%d Total Scenarios, %d MC Runs Each)', ...
    length(object_counts), MC_RUNS);

% Build LaTeX string manually for custom formatting
latex_str = sprintf('\\begin{table}[htbp]\n');
latex_str = [latex_str sprintf('\\centering\n')];
latex_str = [latex_str sprintf('\\caption{%s}\n', caption_str)];
latex_str = [latex_str sprintf('%% Requires \\usepackage{booktabs}\n')];
latex_str = [latex_str sprintf('\\begin{tabular}{c|cc}\n')];
latex_str = [latex_str sprintf('\\toprule\n')];
latex_str = [latex_str sprintf(' & Recall (\\%%) & Precision (\\%%) \\\\\n')];
latex_str = [latex_str sprintf('\\midrule\n')];

for detector = 1:detectors_count
    detector_name = char(detectors_list(detector));
    detector_field = strrep(detector_name, '-', '_');
    s = aggregate_stats.(detector_field);
    
    % Format as "mean ± std" in percentages
    recall_str = sprintf('%.1f $\\pm$ %.1f', s.best_Recall_mean * 100, s.best_Recall_std * 100);
    precision_str = sprintf('%.1f $\\pm$ %.1f', s.best_Precision_mean * 100, s.best_Precision_std * 100);
    
    latex_str = [latex_str sprintf('%s & %s & %s \\\\\n', detector_name, recall_str, precision_str)];
end

latex_str = [latex_str sprintf('\\bottomrule\n')];
latex_str = [latex_str sprintf('\\end{tabular}\n')];
latex_str = [latex_str sprintf('\\end{table}\n')];

% Save to file
latex_filename = 'aggregate_stats_table.tex';
latex_filepath = fullfile(MC_FIG_SAVEDIR, latex_filename);
fid = fopen(latex_filepath, 'w');
fprintf(fid, '%s', latex_str);
fclose(fid);

fprintf("Aggregate LaTeX table saved to: %s\n\n", latex_filepath);


% Save aggregate statistics
aggregate_save_path = fullfile(MC_FIG_SAVEDIR, 'aggregate_statistics.mat');
save(aggregate_save_path, 'aggregate_stats', 'detectors_list');
fprintf("Aggregate statistics saved to: %s\n\n", aggregate_save_path)

fprintf("############# AGGREGATE STATISTICS COMPLETE #############\n\n")

%{
%% ======================== ROC Curves CURVES ========================

fprintf("\n\n############# GENERATING ROC Curves CURVES #############\n\n")

%% Plot 1: Individual scatter plots for each object count
for obj_idx = 1:length(all_data)
    data = all_data{obj_idx};
    o = object_counts(obj_idx);
    marker = marker_shapes{mod(obj_idx - 1, length(marker_shapes)) + 1};

    figure_name = sprintf("ROC_OBJ_COUNT%d.png", o);
    fig = figure('Name', figure_name, 'Color', 'w', 'Position', [100, 100, 1400, 400]);

    for detector = 1:detectors_count
        ax = subplot(1, detectors_count, detector);

        % Call plotting method for scatter plot
        plot_scatter_all_runs(ax, data, detector, o, marker, ...
            thresh_ranges, thresh_labels, MC_RUNS, NUM_THRESHOLDS, detectors_list);
    end

    % Save figure
    saveas(fig, fullfile(MC_FIG_SAVEDIR, figure_name));
    fprintf("Saved figure: %s\n", fullfile(MC_FIG_SAVEDIR, figure_name));

end

fprintf("Completed Individual Scatter Plots\n");

%% Plot 2: Individual error bar plots for each object count
for obj_idx = 1:length(all_data)
    data = all_data{obj_idx};
    o = object_counts(obj_idx);
    marker = marker_shapes{mod(obj_idx - 1, length(marker_shapes)) + 1};

    fig2 = figure('Name', sprintf('ROC Error Bars - %d Objects', o), 'NumberTitle', 'off', ...
        'Color', 'w', 'Position', [100, 100, 1400, 400]);

    for detector = 1:detectors_count
        ax = subplot(1, detectors_count, detector);

        % Call plotting method for error bars with fitted curve
        plot_error_bars_with_fit(ax, data, detector, o, marker, ...
            thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list);
    end

    % Save figure
    saveas(fig2, fullfile(MC_FIG_SAVEDIR, sprintf('ROC_ErrorBars_OBJ_COUNT%d.png', o)));
    fprintf("Saved figure: %s\n", fullfile(MC_FIG_SAVEDIR, sprintf('ROC_ErrorBars_OBJ_COUNT%d.png', o)));

end

fprintf("Completed Individual Error Bar Plots plots\n");

%% Plot 3: Combined plot zoomed out- All object counts on same figure
fig3 = figure('Name', 'ROC Combined - All Object Counts', 'NumberTitle', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1400, 400]);

xlimits = [0.5, 1];
ylimits = [0.75, 1];

for detector = 1:detectors_count
    ax = subplot(1, detectors_count, detector);

    % Call plotting method for combined mean curves
    plot_combined_mean_curves(ax, all_data, object_counts, marker_shapes, detector, ...
        thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list, xlimits, ylimits);

end

% Save figure
saveas(fig3, fullfile(MC_FIG_SAVEDIR, 'ROC_Combined_All_Object_Counts.png'));
fprintf("Saved figure: %s\n", fullfile(MC_FIG_SAVEDIR, 'ROC_Combined_All_Object_Counts.png'));

fprintf("Completed Combined Plots ");

%% Plot 3: Combined plot zoomed in- All object counts on same figure
fig3 = figure('Name', 'ROC Combined - All Object Counts Zoom', 'NumberTitle', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1400, 400]);

xlimits = [0.9, 1];
ylimits = [0.95, 1];

for detector = 1:detectors_count
    ax = subplot(1, detectors_count, detector);

    % Call plotting method for combined mean curves
    plot_combined_mean_curves(ax, all_data, object_counts, marker_shapes, detector, ...
        thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list, xlimits, ylimits);

end

% Save figure
saveas(fig3, fullfile(MC_FIG_SAVEDIR, 'ROC_Combined_All_Object_Counts_Zoom.png'));
fprintf("Saved figure: %s\n", fullfile(MC_FIG_SAVEDIR, 'ROC_Combined_All_Object_Counts.png'));

fprintf("Completed Combined Plots ");

fprintf("Plotting complete!\n")
%}
%% ======================== PRECISION-RECALL CURVES ========================

fprintf("\n\n############# GENERATING PRECISION-RECALL CURVES #############\n\n")

%% Plot 4: Individual scatter plots for each object count (Precision-Recall)
for obj_idx = 1:length(all_data)
    data = all_data{obj_idx};
    o = object_counts(obj_idx);
    marker = marker_shapes{mod(obj_idx - 1, length(marker_shapes)) + 1};

    figure_name = sprintf("PR_OBJ_COUNT%d.png", o);
    fig = figure('Name', figure_name, 'Color', 'w', 'Position', [100, 100, 1400, 400]);

    for detector = 1:detectors_count
        ax = subplot(1, detectors_count, detector);

        % Call plotting method for scatter plot (Precision-Recall)
        plot_scatter_all_runs_PR(ax, data, detector, o, marker, ...
            thresh_ranges, thresh_labels, MC_RUNS, NUM_THRESHOLDS, detectors_list);
    end

    % Save figure
    saveas(fig, fullfile(MC_FIG_SAVEDIR, figure_name));
    fprintf("Saved figure: %s\n", fullfile(MC_FIG_SAVEDIR, figure_name));

end

fprintf("Completed Individual Scatter Plots (Precision-Recall)\n");

%% Plot 5: Individual error bar plots for each object count (Precision-Recall)
for obj_idx = 1:length(all_data)
    data = all_data{obj_idx};
    o = object_counts(obj_idx);
    marker = marker_shapes{mod(obj_idx - 1, length(marker_shapes)) + 1};

    fig2 = figure('Name', sprintf('PR Error Bars - %d Objects', o), 'NumberTitle', 'off', ...
        'Color', 'w', 'Position', [100, 100, 1400, 400]);

    for detector = 1:detectors_count
        ax = subplot(1, detectors_count, detector);

        % Call plotting method for error bars with fitted curve (Precision-Recall)
        plot_error_bars_with_fit_PR(ax, data, detector, o, marker, ...
            thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list);
    end

    % Save figure
    saveas(fig2, fullfile(MC_FIG_SAVEDIR, sprintf('PR_ErrorBars_OBJ_COUNT%d.png', o)));
    fprintf("Saved figure: %s\n", fullfile(MC_FIG_SAVEDIR, sprintf('PR_ErrorBars_OBJ_COUNT%d.png', o)));

end

fprintf("Completed Individual Error Bar Plots (Precision-Recall)\n");

%% Plot 6: Combined plot zoomed out - All object counts (Precision-Recall)
fig3 = figure('Name', 'PR Combined - All Object Counts', 'NumberTitle', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1400, 400]);

xlimits = [0.8, 1];
ylimits = [0, 0.2];

for detector = 1:detectors_count
    ax = subplot(1, detectors_count, detector);

    % Call plotting method for combined mean curves (Precision-Recall)
    plot_combined_mean_curves_PR(ax, all_data, object_counts, marker_shapes, detector, ...
        thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list, xlimits, ylimits);

end

% Save figure
saveas(fig3, fullfile(MC_FIG_SAVEDIR, 'PR_Combined_All_Object_Counts.png'));
fprintf("Saved figure: %s\n", fullfile(MC_FIG_SAVEDIR, 'PR_Combined_All_Object_Counts.png'));

fprintf("Completed Combined Plots (Precision-Recall)\n");

%% Plot 7: Combined plot zoomed in - All object counts (Precision-Recall)
fig3 = figure('Name', 'PR Combined - All Object Counts Zoom', 'NumberTitle', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1400, 400]);

xlimits = [0.99, 1];
ylimits = [0, 0.1];

for detector = 1:detectors_count
    ax = subplot(1, detectors_count, detector);

    % Call plotting method for combined mean curves (Precision-Recall)
    plot_combined_mean_curves_PR(ax, all_data, object_counts, marker_shapes, detector, ...
        thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list, xlimits, ylimits);

end

% Save figure
saveas(fig3, fullfile(MC_FIG_SAVEDIR, 'PR_Combined_All_Object_Counts_Zoom.png'));
fprintf("Saved figure: %s\n", fullfile(MC_FIG_SAVEDIR, 'PR_Combined_All_Object_Counts_Zoom.png'));

fprintf("Completed Combined Plots (Precision-Recall)\n");

fprintf("\n\nAll plotting complete!\n")

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
        thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list, xlimits, ylimits)
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
    %   xlimits - array of xlimits forplot
    %   ylimits - array of ylimits for plot

    hold(ax, 'on');
    grid(ax, 'on');

    % Get threshold range and colormap for this detector
    thresh_vals = thresh_ranges{detector};

    % Plot each object count with different marker, colored by threshold
    for obj_idx = 1:length(all_data)
        data = all_data{obj_idx};
        o = object_counts(obj_idx);
        marker = marker_shapes{mod(obj_idx - 1, length(marker_shapes)) + 1};

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
        marker = marker_shapes{mod(obj_idx - 1, length(marker_shapes)) + 1};
        h = plot(ax, NaN, NaN, marker, 'MarkerSize', 10, 'LineWidth', 1.5, ...
            'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerEdgeColor', 'k');
        legend_handles = [legend_handles; h];
        legend_labels{end + 1} = sprintf('%d objects', o);
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

    ylim(ax, ylimits)
    xlim(ax, xlimits)

    % Labels
    xlabel(ax, 'False Positive Rate', 'Interpreter', 'latex')
    ylabel(ax, 'True Positive Rate', 'Interpreter', 'latex')
    title(ax, sprintf('%s - All Object Counts', detectors_list(detector)), 'Interpreter', 'latex')

    % Legend for marker shapes
    legend(legend_handles, legend_labels, 'Location', 'southwest')
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
    xlim(ax, [0 1])
    ylim(ax, [8.5e-1 1])

    % Labels
    xlabel(ax, 'False Positive Rate', 'Interpreter', 'latex')
    ylabel(ax, 'True Positive Rate', 'Interpreter', 'latex')
    title(ax, sprintf('%s - Obj Count: %d (Mean \\pm 1SD)', ...
        strrep(detectors_list(detector), '_', '\_'), obj_count), 'Interpreter', 'tex')
end

%% ======================== PRECISION-RECALL PLOTTING METHODS ========================

function plot_scatter_all_runs_PR(ax, data, detector, obj_count, marker, ...
        thresh_ranges, thresh_labels, MC_RUNS, NUM_THRESHOLDS, detectors_list)
    % Plot scatter of all MC runs and thresholds for Precision-Recall
    %
    % Inputs:
    %   ax - axes handle to plot on
    %   data - loaded data structure containing Precision, Recall arrays
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
            scatter(ax, data.Recall(detector, m, t), data.Precision(detector, m, t), 50, ...
                thresh_vals(t), 'filled', marker, 'MarkerEdgeColor', 'k');
        end

    end

    % Add colorbar
    c = colorbar(ax);
    c.Label.String = thresh_labels{detector};
    c.Label.Interpreter = 'tex';
    colormap(ax, parula);
    caxis(ax, [min(thresh_vals), max(thresh_vals)]);

    % Axis formatting
    axis(ax, 'square')
    xlim(ax, [0 1])
    ylim(ax, [0 1])

    % Labels
    xlabel(ax, 'Recall', 'Interpreter', 'latex')
    ylabel(ax, 'Precision', 'Interpreter', 'latex')
    title(ax, sprintf('%s - Obj Count: %d', detectors_list(detector), obj_count), 'Interpreter', 'latex')
end

function plot_combined_mean_curves_PR(ax, all_data, object_counts, marker_shapes, detector, ...
        thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list, xlimits, ylimits)
    % Plot mean Precision-Recall curves for all object counts on same axes
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
    %   xlimits - array of xlimits for plot
    %   ylimits - array of ylimits for plot

    hold(ax, 'on');
    grid(ax, 'on');

    % Get threshold range and colormap for this detector
    thresh_vals = thresh_ranges{detector};

    % Plot each object count with different marker, colored by threshold
    for obj_idx = 1:length(all_data)
        data = all_data{obj_idx};
        o = object_counts(obj_idx);
        marker = marker_shapes{mod(obj_idx - 1, length(marker_shapes)) + 1};

        % Compute mean for each threshold
        mean_Recall = zeros(NUM_THRESHOLDS, 1);
        mean_Precision = zeros(NUM_THRESHOLDS, 1);

        for t = 1:NUM_THRESHOLDS
            mean_Recall(t) = mean(data.Recall(detector, :, t), 'omitnan');
            mean_Precision(t) = mean(data.Precision(detector, :, t), 'omitnan');
        end

        % Plot each threshold point individually to color by threshold value
        for t = 1:NUM_THRESHOLDS
            scatter(ax, mean_Recall(t), mean_Precision(t), 100, thresh_vals(t), marker, ...
                'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        end

        % Add connecting line for this object count
        plot(ax, mean_Recall, mean_Precision, 'LineStyle', '-', 'LineWidth', 1.5, ...
            'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    end

    % Create custom legend for marker shapes
    legend_handles = [];
    legend_labels = {};

    for obj_idx = 1:length(all_data)
        o = object_counts(obj_idx);
        marker = marker_shapes{mod(obj_idx - 1, length(marker_shapes)) + 1};
        h = plot(ax, NaN, NaN, marker, 'MarkerSize', 10, 'LineWidth', 1.5, ...
            'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerEdgeColor', 'k');
        legend_handles = [legend_handles; h];
        legend_labels{end + 1} = sprintf('%d objects', o);
    end

    % Add colorbar for threshold values
    c = colorbar(ax);
    c.Label.String = thresh_labels{detector};
    c.Label.Interpreter = 'tex';
    colormap(ax, parula);
    caxis(ax, [min(thresh_vals), max(thresh_vals)]);

    % Axis formatting
    axis(ax, 'square')

    ylim(ax, ylimits)
    xlim(ax, xlimits)

    % Labels
    xlabel(ax, 'Recall', 'Interpreter', 'latex')
    ylabel(ax, 'Precision', 'Interpreter', 'latex')
    title(ax, sprintf('%s - All Object Counts', detectors_list(detector)), 'Interpreter', 'latex')

    % Legend for marker shapes
    legend(legend_handles, legend_labels, 'Location', 'northeast')
end

function plot_error_bars_with_fit_PR(ax, data, detector, obj_count, marker, ...
        thresh_ranges, thresh_labels, NUM_THRESHOLDS, detectors_list)
    % Plot error bars (mean ± std) with fitted curve for Precision-Recall
    %
    % Inputs:
    %   ax - axes handle to plot on
    %   data - loaded data structure containing Precision, Recall arrays
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
    mean_Recall = zeros(NUM_THRESHOLDS, 1);
    std_Recall = zeros(NUM_THRESHOLDS, 1);
    mean_Precision = zeros(NUM_THRESHOLDS, 1);
    std_Precision = zeros(NUM_THRESHOLDS, 1);

    for t = 1:NUM_THRESHOLDS
        mean_Recall(t) = mean(data.Recall(detector, :, t), 'omitnan');
        std_Recall(t) = std(data.Recall(detector, :, t), 'omitnan');
        mean_Precision(t) = mean(data.Precision(detector, :, t), 'omitnan');
        std_Precision(t) = std(data.Precision(detector, :, t), 'omitnan');
    end

    % Get colormap for threshold coloring
    cmap = parula(NUM_THRESHOLDS);

    % Plot error bars colored by threshold
    for t = 1:NUM_THRESHOLDS
        errorbar(ax, mean_Recall(t), mean_Precision(t), std_Precision(t), std_Precision(t), ...
            std_Recall(t), std_Recall(t), marker, 'MarkerSize', 8, 'LineWidth', 2, ...
            'CapSize', 10, 'Color', cmap(t, :), 'MarkerFaceColor', cmap(t, :));
    end

    % Recolor points by threshold using scatter (for proper colorbar)
    for t = 1:NUM_THRESHOLDS
        scatter(ax, mean_Recall(t), mean_Precision(t), 100, thresh_vals(t), marker, ...
            'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    end

    % Fit a curve to the mean values (sort by Recall first)
    [sorted_Recall, sort_idx] = sort(mean_Recall);
    sorted_Precision = mean_Precision(sort_idx);

    % Remove NaN values and duplicate Recall values for fitting
    valid_idx = ~isnan(sorted_Recall) & ~isnan(sorted_Precision);
    sorted_Recall = sorted_Recall(valid_idx);
    sorted_Precision = sorted_Precision(valid_idx);

    % Remove duplicate Recall values (keep first occurrence)
    [sorted_Recall, unique_idx] = unique(sorted_Recall, 'stable');
    sorted_Precision = sorted_Precision(unique_idx);

    if length(sorted_Recall) >= 3
        % Fit a smooth curve using spline interpolation
        Recall_fit = linspace(min(sorted_Recall), max(sorted_Recall), 100);
        Precision_fit = interp1(sorted_Recall, sorted_Precision, Recall_fit, 'pchip');
        plot(ax, Recall_fit, Precision_fit, '-', 'LineWidth', 2.5, 'Color', [0.2, 0.2, 0.2], ...
            'DisplayName', 'Fitted Curve');
    end

    % Add colorbar
    c = colorbar(ax);
    c.Label.String = strrep(thresh_labels{detector}, '_', '\_');
    c.Label.Interpreter = 'tex';
    colormap(ax, parula);
    caxis(ax, [min(thresh_vals), max(thresh_vals)]);

    % Axis formatting
    axis(ax, 'square')
    xlim(ax, [5e-1 1])
    ylim(ax, [0 2e-1])

    % Labels
    xlabel(ax, 'Recall', 'Interpreter', 'latex')
    ylabel(ax, 'Precision', 'Interpreter', 'latex')
    title(ax, sprintf('%s - Obj Count: %d (Mean \\pm 1SD)', ...
        strrep(detectors_list(detector), '_', '\_'), obj_count), 'Interpreter', 'tex')
end
