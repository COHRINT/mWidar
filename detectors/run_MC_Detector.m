clear; clc; close all

%% Load in relevent dataset
Dataset = "Single";
GT = load("detectors/Detector_Tracks/" + Dataset + "_Object_Track.mat").Data;
Signal = load("detectors/Detector_Tracks/" + Dataset + "_Signal.mat").Signal;

rng(42352)

detectors_list = ["peaks2", "CA_CFAR"];
detectors_count = 2;
MC_RUNS = 50;

% Acquire all new threshold values 

% Peaks2 -> changing "MinPeakDistance" with values from 1 - 15

p2_iter = linspace(1,15,MC_RUNS);

% CA-CFAR -> changing "Pfa" with values from 0.1 - 0.4

ca_iten = linspace(0.1,0.4,MC_RUNS);

d_thresh_value = 5;

TP_ovr = zeros(detectors_count, MC_RUNS);
FP_ovr = zeros(detectors_count, MC_RUNS);
FN_ovr = zeros(detectors_count, MC_RUNS);
TPR = zeros(detectors_count, MC_RUNS);
FPR = zeros(detectors_count, MC_RUNS);

for k = 1:size(Signal,3)

    fprintf("Processing Timestep %d....\n",k)
    signal_original = Signal(:,:,k);
    % Normalize signal [0 1]
    scaled_signal = (signal_original - min(signal_original(:)))/ (max(signal_original(:)) - min(signal_original(:)));
    gt_points = GT(:,:,k);
    %% Run Signal Through Detectors

    % Peaks2
    for i = 1:MC_RUNS
        th = thresholds(i);
        [~, px, py] = peaks2(scaled_signal, 'MinPeakHeight', th, 'MinPeakDistance', 4); % Not peak height?
        peaks = [py, px];

        % Use the calcTPFPFN function to calculate TP, FP, FN
        [TP, FP, FN] = calcTPFPFN(gt_points, peaks, d_thresh_value);

        TP_ovr(1, i) = TP_ovr(1, i) + TP;
        FP_ovr(1, i) = FP_ovr(1, i) + FP;
        FN_ovr(1, i) = FN_ovr(1, i) + FN;
    end

    % CA_CFAR
    for i = 1:MC_RUNS
        th = thresholds(i);
        [~, px, py] = CA_CFAR(scaled_signal, th, 3, 10); % Keep th between 0.01 and 0.4

        % Use the calcTPFPFN function to calculate TP, FP, FN
        [TP, FP, FN] = calcTPFPFN(gt_points, peaks, d_thresh_value);

        TP_ovr(2, i) = TP_ovr(2, i) + TP;
        FP_ovr(2, i) = FP_ovr(2, i) + FP;
        FN_ovr(2, i) = FN_ovr(2, i) + FN;
    end
end



color_palette = [
                 0.3, 0.3, 0.7; % Darker blue/purple
                 0.7, 0.3, 0.3; % Darker red/maroon
                 0.3, 0.6, 0.3 % Darker green
                 ];

figure(1); hold on; grid on


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
    title(sprintf('ROC Curve - Dataset: %s', Dataset), 'Interpreter', 'latex')
    % Legend + Grid
    grid('on')
    legend('Location', 'southeast', 'Interpreter', 'latex', 'FontSize', 14);

end