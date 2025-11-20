clear; clc; close all

%% run_MC_Detector

% MC simulation of different detectors for mWidar Signal
% Bijan Jourabchi / Anthony La Barca

%% Set up simulation

% Timestamp for saving results
script_start_time = datetime('now', 'Format', 'yyyyMMdd_HHmmss');

PLOT_DEBUG = true;
pks_plt = [];
signal_plt = {};

load(fullfile("..", "matlab_src", "supplemental", "recovery.mat"))
load(fullfile("..", "matlab_src", "supplemental", "sampling.mat"))
xgrid = 1:128;
ygrid = 1:128;
[X, Y] = meshgrid(xgrid, ygrid);

rng(42352)

detectors_list = ["peaks2", "CA-CFAR", "TDPF"];
detectors_count = 3;
MC_RUNS = 100;
d_thresh_value = 20;

% Threshold sweep parameters
NUM_THRESHOLDS = 20; % Number of threshold values to test

% Define threshold ranges for each detector
thresh_MP_range = linspace(0.01, 0.75, NUM_THRESHOLDS); % MaxPeaks: MinPeakHeight
thresh_CFAR_range = linspace(0.3, 0.75, NUM_THRESHOLDS); % CA-CFAR: Pfa
thresh_TDPF_range = linspace(5, 15, NUM_THRESHOLDS); % TDPF: Distance threshold

% Setup for TDPF
TDPF_Dict = {};
TDPF_String = ["New", "Existing", "Confirmed", "Persistent"];

TP_ovr = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);
FP_ovr = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);
FN_ovr = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);
TPR = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);
FPR = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);
Precision = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);
Recall = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);

fprintf("############# BEGINNING MC SIMULATION ################### \n\n")
% objects_count = [1, 5, 10];
objects_count = 1:10;

% Create results directory structure with timestamp
results_base_dir = fullfile("..", "Results", "detectors");
run_dir_name = sprintf('%s_MCRuns%d_NThresh%d', char(script_start_time), MC_RUNS, NUM_THRESHOLDS);
results_dir = fullfile(results_base_dir, run_dir_name);

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

fprintf("Results will be saved to: %s\n\n", results_dir)

for o = objects_count
    fprintf("############# MC SIMULATION WITH %d OBJECT(S)################# \n\n", o)

    for m = 1:MC_RUNS

        fprintf("MC Run #%d \n", m)

        % Generate new signal and GT (same for all thresholds)
        [scaled_signal, POS] = random_tracks2(o, M, G);
        nk = size(POS, 3);

        for t = 1:NUM_THRESHOLDS

            % fprintf("  Threshold %d/%d \n", t, NUM_THRESHOLDS)

            % Reset TDPF dictionary for each threshold test
            TDPF_Dict = {};

            % Get current threshold values for each detector
            current_thresh_MP = thresh_MP_range(t);
            current_thresh_CFAR = thresh_CFAR_range(t);
            current_thresh_TDPF = thresh_TDPF_range(t);

            %% Run Signal Through Detectors
            for k = 1:nk

                gt_points = POS(:, :, k);

                % Extract valid region without NaN values (rows 21-128, y-axis from 21 to 128)
                signal_full = scaled_signal(:, :, 2, k);
                signal_valid = signal_full(21:128, :); % Remove bottom 20 rows with NaN
                y_offset = 20; % Offset to transform back to original frame

                % Keep rows where col1 in [1,128] and col2 in [20,128]
                if ~isempty(gt_points)
                    valid = gt_points(:, 1) >= 1 & gt_points(:, 1) <= 128 & ...
                        gt_points(:, 2) >= 20 & gt_points(:, 2) <= 128;
                    gt_points = gt_points(valid, :);
                end

                % If no GT points remain, skip this time step
                if isempty(gt_points)
                    continue;
                end

                % Max_Peaks (Peaks2) - use valid signal region with varying MinPeakHeight
                [~, px, py] = peaks2(signal_valid, 'MinPeakHeight', current_thresh_MP, 'MinPeakDistance', 20);
                % Transform back to original frame
                px = px + y_offset;
                peaks_MP = [px, py];

                % Use the calcTPFPFN function to calculate TP, FP, FN
                if k >= 5 % Make statistics consistent by only evaluating after 5th frame for TDPF sake
                    [TP, FP, FN] = calcTPFPFN(gt_points, peaks_MP, d_thresh_value);

                    TP_ovr(1, m, t) = TP_ovr(1, m, t) + TP;
                    FP_ovr(1, m, t) = FP_ovr(1, m, t) + FP;
                    FN_ovr(1, m, t) = FN_ovr(1, m, t) + FN;
                end

                % CA-CFAR - use valid signal region with varying Pfa
                Ng = 5; Ns = 20;
                [~, px, py] = CA_CFAR(signal_valid, current_thresh_CFAR, Ng, Ns);
                % Transform back to original frame
                px = px + y_offset;
                peaks_CFAR = [px, py];

                % Use the calcTPFPFN function to calculate TP, FP, FN
                if k >= 5 % Make statistics consistent by only evaluating after 5th frame for TDPF sake
                    [TP, FP, FN] = calcTPFPFN(gt_points, peaks_CFAR, d_thresh_value);

                    TP_ovr(2, m, t) = TP_ovr(2, m, t) + TP;
                    FP_ovr(2, m, t) = FP_ovr(2, m, t) + FP;
                    FN_ovr(2, m, t) = FN_ovr(2, m, t) + FN;
                end

                % TDPF - use valid signal region with varying distance threshold
                [~, px, py] = peaks2(signal_valid, 'MinPeakHeight', 0.2); % Find ALL the peaks
                % Transform back to original frame
                px = px + y_offset;
                current_peaks = [px, py];

                if k == 1
                    first_score = [current_peaks, ones(size(current_peaks, 1), 1)];
                    TDPF_Dict{1} = first_score;
                else
                    TDPF_Dict{k} = TDPF( ...
                        current_peaks, ...
                        TDPF_Dict{k - 1}, ...
                        current_thresh_TDPF, ... % Distance threshold (varies)
                        current_thresh_TDPF / 2 ... % MinPeakDistance to prevent noise clustering
                    );
                end

                peaks_tdpf = TDPF_Dict{k}(TDPF_Dict{k}(:, 3) > 2, 1:2); % Reported peaks with score > 2, only x,y

                % Use the calcTPFPFN function to calculate TP, FP, FN
                % Since TDPF takes time to warm up (optimally 3 frames), we only start evaluating after 5th frame
                if k >= 5 % Make statistics consistent by only evaluating after 5th frame for TDPF sake
                    [TP, FP, FN] = calcTPFPFN(gt_points, peaks_tdpf, d_thresh_value);

                    TP_ovr(3, m, t) = TP_ovr(3, m, t) + TP;
                    FP_ovr(3, m, t) = FP_ovr(3, m, t) + FP;
                    FN_ovr(3, m, t) = FN_ovr(3, m, t) + FN;
                end

            end

        end % end threshold loop

    end % end MC runs loop

    fprintf("MC Simulation Completed for object count %d\n", o)
    fprintf("Saving results...\n")

    % Calculate TPR and FPR for all detector/MC_run/threshold combinations
    % for detector = 1:detectors_count
    %
    %     for m = 1:MC_RUNS
    %
    %         for t = 1:NUM_THRESHOLDS
    %             TPR(detector, m, t) = TP_ovr(detector, m, t) ./ (TP_ovr(detector, m, t) + FN_ovr(detector, m, t));
    %             FPR(detector, m, t) = FP_ovr(detector, m, t) ./ (FP_ovr(detector, m, t) + TP_ovr(detector, m, t));
    %
    %             % Calculate Precision and Recall
    %             Precision(detector, m, t) = TP_ovr(detector, m, t) ./ (TP_ovr(detector, m, t) + FP_ovr(detector, m, t));
    %             Recall(detector, m, t) = TP_ovr(detector, m, t) ./ (TP_ovr(detector, m, t) + FN_ovr(detector, m, t));
    %         end
    %
    %     end
    %
    % end
    TPR = TP_ovr ./ (TP_ovr + FN_ovr);
    FPR = FP_ovr ./ (FP_ovr + TP_ovr); % NOTE: this is unusual, but matches your loop
    Precision = TP_ovr ./ (TP_ovr + FP_ovr);
    Recall = TP_ovr ./ (TP_ovr + FN_ovr);

    % Define threshold ranges and labels for each detector
    thresh_ranges = {thresh_MP_range, thresh_CFAR_range, thresh_TDPF_range};
    thresh_labels = {'MinPeakHeight', 'P_{fa}', 'Distance Threshold'};

    % Create filename for this object count
    results_filename = sprintf('obj%d.mat', o);
    results_filepath = fullfile(results_dir, results_filename);

    % Save all results to .mat file
    save(results_filepath, ...
        'TP_ovr', 'FP_ovr', 'FN_ovr', ...
        'TPR', 'FPR', ...
        'Precision', 'Recall', ...
        'thresh_MP_range', 'thresh_CFAR_range', 'thresh_TDPF_range', ...
        'thresh_ranges', 'thresh_labels', ...
        'detectors_list', 'detectors_count', ...
        'MC_RUNS', 'NUM_THRESHOLDS', ...
        'd_thresh_value', 'o');

    fprintf("Results saved to: %s\n", results_filepath);

end

fprintf("############# MC SIMULATION COMPLETE ################### \n")
