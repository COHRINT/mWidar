clear; clc; close all

%% run_MC_Detector

% MC simulation of different detectors for mWidar Signal
% Bijan Jourabchi

%% Set up simulation

PLOT_DEBUG = true;
pks_plt = [];
signal_plt = {};

load(fullfile("..", "matlab_src", "supplemental", "recovery.mat"))
load(fullfile("..", "matlab_src", "supplemental", "sampling.mat"))
xgrid = 1:128;
ygrid = 1:128;
[X, Y] = meshgrid(xgrid, ygrid);

rng(42352)

detectors_list = ["peaks2", "CA_CFAR", "TDPF"];
detectors_count = 3;
MC_RUNS = 20;
d_thresh_value = 20;

% Threshold sweep parameters
NUM_THRESHOLDS = 10; % Number of threshold values to test

% Define threshold ranges for each detector
thresh_MP_range = linspace(1, 20, NUM_THRESHOLDS);     % MaxPeaks: MinPeakDistance
thresh_CFAR_range = linspace(0.1, 0.5, NUM_THRESHOLDS); % CA_CFAR: Pfa
thresh_TDPF_range = linspace(1, 10, NUM_THRESHOLDS);    % TDPF: Distance threshold

% Setup for TDPF
TDPF_Dict = {};
TDPF_String = ["New", "Existing", "Confirmed", "Persistent"];

TP_ovr = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);
FP_ovr = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);
FN_ovr = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);
TPR = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);
FPR = zeros(detectors_count, MC_RUNS, NUM_THRESHOLDS);

fprintf("############# BEGINNING MC SIMULATION ################### \n\n")
% objects_count = 1:5;
objects_count = [1,2,3];

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

            % DIAGNOSTIC PLOTTING (only for first threshold to avoid clutter)
            % if t == 1
            %     figure(1)
            %     tiledlayout(2, ceil(nk / 2), 'TileSpacing', 'Compact', 'Padding', 'Compact');
            % 
            %     for k = 1:nk
            %         nexttile(k);
            %         surf(X, Y, scaled_signal(:, :, 2, k), 'EdgeColor', 'none');
            %         shading interp
            %         hold on
            %         plot3(POS(:, 1, k), POS(:, 2, k), ones(size(POS(:, 1, k))), 'rx', 'LineWidth', 2, 'MarkerSize', 10)
            %         title(sprintf("Time Step %d", k))
            %         axis square
            %         view(2)
            %         xlim([0 128])
            %         ylim([20 128])
            %     end
            % 
            %     sgtitle(sprintf("MC Run #%d - Object Count: %d", m, o), 'FontSize', 14, 'FontWeight', 'bold');
            % end

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

                % Max_Peaks (Peaks2) - use valid signal region with varying MinPeakDistance
                [~, px, py] = peaks2(signal_valid, 'MinPeakHeight', 0.5, 'MinPeakDistance', current_thresh_MP);
                % Transform back to original frame
                px = px + y_offset;
                peaks_MP = [px, py];
                
                % Plot peaks (only for first threshold)
                % if t == 1
                %     nexttile(k);
                %     hold on;
                %     plot3(peaks_MP(:, 2), peaks_MP(:, 1), ones(size(peaks_MP(:, 1))), 'mo', 'LineWidth', 2, 'MarkerSize', 4);
                %     hold off;
                % end

                % Use the calcTPFPFN function to calculate TP, FP, FN
                [TP, FP, FN] = calcTPFPFN(gt_points, peaks_MP, d_thresh_value);

                TP_ovr(1, m, t) = TP_ovr(1, m, t) + TP;
                FP_ovr(1, m, t) = FP_ovr(1, m, t) + FP;
                FN_ovr(1, m, t) = FN_ovr(1, m, t) + FN;

                % CA_CFAR - use valid signal region with varying Pfa
                Ng = 5; Ns = 20;
                [~, px, py] = CA_CFAR(signal_valid, current_thresh_CFAR, Ng, Ns);
                % Transform back to original frame
                px = px + y_offset;
                peaks_CFAR = [px, py];
                
                % Plot peaks (only for first threshold)
                % if t == 1
                %     nexttile(k);
                %     hold on;
                %     plot3(peaks_CFAR(:, 2), peaks_CFAR(:, 1), ones(size(peaks_CFAR(:, 1))), 'go', 'LineWidth', 2, 'MarkerSize', 12);
                %     hold off;
                % end

                % Use the calcTPFPFN function to calculate TP, FP, FN
                [TP, FP, FN] = calcTPFPFN(gt_points, peaks_CFAR, d_thresh_value);

                TP_ovr(2, m, t) = TP_ovr(2, m, t) + TP;
                FP_ovr(2, m, t) = FP_ovr(2, m, t) + FP;
                FN_ovr(2, m, t) = FN_ovr(2, m, t) + FN;

                % TDPF - use valid signal region with varying distance threshold
                [~, px, py] = peaks2(signal_valid, 'MinPeakHeight', 0.3); % Find ALL the peaks
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
                        current_thresh_TDPF ... % Distance threshold (varies)
                    );
                end

                peaks_tdpf = TDPF_Dict{k}(TDPF_Dict{k}(:, 3) > 2, 1:2); % Reported peaks with score > 2, only x,y
                
                % Plot peaks (only for first threshold)
                % if t == 1
                %     nexttile(k);
                %     hold on;
                %     plot3(peaks_tdpf(:, 2), peaks_tdpf(:, 1), ones(size(peaks_tdpf(:, 1))), 'co', 'LineWidth', 2, 'MarkerSize', 8);
                %     hold off;
                % end

                % Use the calcTPFPFN function to calculate TP, FP, FN
                % Since TDPF takes time to warm up (optimally 3 frames), we only start evaluating after 5th frame
                if k >= 5
                    [TP, FP, FN] = calcTPFPFN(gt_points, peaks_tdpf, d_thresh_value);

                    TP_ovr(3, m, t) = TP_ovr(3, m, t) + TP;
                    FP_ovr(3, m, t) = FP_ovr(3, m, t) + FP;
                    FN_ovr(3, m, t) = FN_ovr(3, m, t) + FN;
                end

            end
            
        end % end threshold loop

        % if m == 1
        %     pause(1);
        % end

    end % end MC runs loop

    fprintf("MC Simulation Completed for object count %d, Plotting \n\n", o)

    % Calculate TPR and FPR for all detector/MC_run/threshold combinations
    for detector = 1:detectors_count
        for m = 1:MC_RUNS
            for t = 1:NUM_THRESHOLDS
                TPR(detector, m, t) = TP_ovr(detector, m, t) ./ (TP_ovr(detector, m, t) + FN_ovr(detector, m, t));
                FPR(detector, m, t) = FP_ovr(detector, m, t) ./ (FP_ovr(detector, m, t) + TP_ovr(detector, m, t));
            end
        end
    end

    % Define threshold ranges and labels for each detector
    thresh_ranges = {thresh_MP_range, thresh_CFAR_range, thresh_TDPF_range};
    thresh_labels = {'MinPeakDistance', 'P_{fa}', 'Distance Threshold'};

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

        % Scale x axis and y axis to log (try to make look better)
        xscale('log')
        yscale('log')

        % Axis
        axis('square')
        xlim([0 1])
        ylim([0 1])

        % Labels
        xlabel('False Positive Rate', 'Interpreter', 'latex')
        ylabel('True Positive Rate', 'Interpreter', 'latex')
        title(sprintf('%s - Obj Count: %d', detectors_list(detector), o), 'Interpreter', 'latex')
        
        % Grid
        grid('on')
    end

    % figName = sprintf("Figures/ROC_OBJ_COUNT%d.png", o);
    % print(fig, figName, '-dpng', '-r300');
    % close(fig);

end
