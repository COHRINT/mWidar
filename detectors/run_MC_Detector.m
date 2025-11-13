clear; clc; close all

%% run_MC_Detector

% MC simulation of different detectors for mWidar Signal
% Bijan Jourabchi


%% Set up simulation

PLOT_DEBUG = true;
pks_plt = [];
signal_plt = {};

load(fullfile("matlab_src","supplemental","recovery.mat"))
load(fullfile("matlab_src","supplemental","sampling.mat"))

rng(42352)

detectors_list = ["peaks2", "CA_CFAR"];
detectors_count = 2;
MC_RUNS = 50;
d_thresh_value = 5;

TP_ovr = zeros(detectors_count, MC_RUNS);
FP_ovr = zeros(detectors_count, MC_RUNS);
FN_ovr = zeros(detectors_count, MC_RUNS);
TPR = zeros(detectors_count, MC_RUNS);
FPR = zeros(detectors_count, MC_RUNS);

fprintf("############# BEGINNING MC SIMULATION ################### \n\n")

for o = 1:5
    fprintf("############# MC SIMULATION WITH %d OBJECT(S)################# \n\n",o)
    for m = 1:MC_RUNS

    fprintf("MC Run #%d \n",m)

    % Generate new signal and GT
    [scaled_signal, POS] = random_tracks(o,M,G);

    nk = size(POS,3);
    %% Run Signal Through Detectors

    for k = 1:nk
        
        gt_points = POS(:,:,k);

        % Keep rows where col1 in [1,128] and col2 in [20,128]
        if ~isempty(gt_points)
            valid = gt_points(:,1) >= 1  & gt_points(:,1) <= 128 & ...
                    gt_points(:,2) >= 20 & gt_points(:,2) <= 128;
            gt_points = gt_points(valid, :);
        end

        % If no GT points remain, skip this time step
        if isempty(gt_points)
            fprintf("No valid points, skipping... \n")
            continue;
        end

        % Peaks2
        % Another loop
        [~, px, py] = peaks2(scaled_signal(:,:,k), 'MinPeakHeight', 0.85, 'MinPeakDistance', 10); 
        peaks = [px, py];

        % Use the calcTPFPFN function to calculate TP, FP, FN
        [TP, FP, FN] = calcTPFPFN(gt_points, peaks, d_thresh_value);

        TP_ovr(1, m) = TP_ovr(1, m) + TP;
        FP_ovr(1, m) = FP_ovr(1, m) + FP;
        FN_ovr(1, m) = FN_ovr(1, m) + FN;


        % CA_CFAR
        [~, px, py] = CA_CFAR(scaled_signal(:,:,k), 0.36, 5, 20); 
        peaks = [px, py];

        % Use the calcTPFPFN function to calculate TP, FP, FN
        [TP, FP, FN] = calcTPFPFN(gt_points, peaks, d_thresh_value);

        TP_ovr(2, m) = TP_ovr(2, m) + TP;
        FP_ovr(2, m) = FP_ovr(2, m) + FP;
        FN_ovr(2, m) = FN_ovr(2, m) + FN;
    end

    end

    fprintf("MC Simulation Completed for object count %d, Plotting \n\n",o)


    color_palette = [
                    0.3, 0.3, 0.7; % Darker blue/purple
                    0.7, 0.3, 0.3; % Darker red/maroon
                    0.3, 0.6, 0.3 % Darker green
                    ];

    fig = figure('Visible','off','Name','ROC Figure','NumberTitle','off','Color','w');
    for detector = 1:detectors_count
    TPR(detector,:) = TP_ovr(detector,:)./(TP_ovr(detector,:) + FN_ovr(detector,:));
    FPR(detector,:) = FP_ovr(detector,:)./(FP_ovr(detector,:) + TP_ovr(detector,:));
    subplot(1,detectors_count,detector);hold on; grid on
    plot(FPR(detector, :), TPR(detector, :), 'o', 'Color', color_palette(detector, :), "DisplayName", detectors_list(detector), 'LineWidth', 2, 'MarkerSize', 6);
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
    title(sprintf('ROC Curve - Object Count: %d', o), 'Interpreter', 'latex')
    % Legend + Grid
    grid('on')
    legend('Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 14);

    end

    figName = sprintf("detectors/Figures/ROC_OBJ_COUNT%d.png",o);
    print(fig, figName, '-dpng', '-r300');
    close(fig);
end