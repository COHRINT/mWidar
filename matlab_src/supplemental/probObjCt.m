%% Monte Carlo estimate of object-count detection probabilities
%
% This script estimates:
%   1) P(N | m): the probability of observing N detections given m objects
%   2) P(m | N): the posterior probability of m objects given N detections
%
% The posterior is derived from the Monte Carlo likelihood estimate using a
% user-defined prior over m.

clc; close all; clear
rng(400)

%% --- Environment configuration -----------------------------------------
addpath(fullfile('DA_Track'))
addpath(fullfile('DA_Track', 'multi'))
addpath(fullfile('supplemental'))
addpath(fullfile('supplemental', 'Final_Test_Tracks'))
addpath(fullfile('supplemental', 'Final_Test_Tracks', 'MultiObj'))

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

load(fullfile(script_dir, 'recovery.mat'), 'G');
load(fullfile(script_dir, 'sampling.mat'), 'M');

%% --- Monte Carlo settings ----------------------------------------------
cfg = struct();
cfg.max_obj = 5;
cfg.MC = 100;
cfg.verbose = true;

cfg.Pfa = 0.285;
cfg.Ng = 15;
cfg.Nr = 20;

cfg.T = 25;                  % Sliding-window length
cfg.intensity_thr = 0.25;     % Detection intensity threshold
cfg.cluster_radius = 0.35;   % Merge nearby detections [m]
cfg.blur_sigma = 1.3;
cfg.crop_rows = 20;          % Remove near-range clutter rows

cfg.npx = 128;
cfg.Lscene = 4;
cfg.xgrid = linspace(-2, 2, cfg.npx);
cfg.ygrid = linspace(0, cfg.Lscene, cfg.npx);
[cfg.pxgrid, cfg.pygrid] = meshgrid(cfg.xgrid, cfg.ygrid);

cfg.min_track_y = 1.0;
cfg.min_start_separation = 0.75;
cfg.dt = 0.01;
cfg.tvec = 0:cfg.dt:10;

cfg.traj_types = ["LINE", "PARABOLA"];
cfg.prior_m = ones(cfg.max_obj + 1, 1) ./ (cfg.max_obj + 1);

n_k = numel(cfg.tvec);

%% --- Run Monte Carlo ----------------------------------------------------
det_counts_by_obj = cell(cfg.max_obj+1, 1);

for n_obj = 0:cfg.max_obj
    if cfg.verbose
        fprintf('Simulating object count m = %d\n', n_obj);
    end

    det_counts_by_obj{n_obj+1} = runObjectCountTrials(n_obj, cfg, G, M);
end

%% --- Convert counts to probability tables -------------------------------
max_det_observed = max(cellfun(@maxZeroSafe, det_counts_by_obj));
det_axis = 0:max_det_observed;

% N_k is indexed by row i (state), and m_k is the detection count.
N_vals = 0:cfg.max_obj;
m_axis = det_axis(:).'; % detection-count support (columns in lookup table)

P_m_given_N = zeros(numel(N_vals), numel(m_axis)); % row i -> N_k = i

for i = 1:numel(N_vals)
    counts_i = det_counts_by_obj{N_vals(i) + 1}(:).';
    for j = 1:length(counts_i)
        col = counts_i(j) - m_axis(1) + 1;   % usually counts_i(j)+1 when m_axis starts at 0
        if col >= 1 && col <= numel(m_axis)
            P_m_given_N(i, col) = P_m_given_N(i, col) + 1;
        end
    end

    row_sum = sum(P_m_given_N(i, :));
    if row_sum > 0
        P_m_given_N(i, :) = P_m_given_N(i, :) ./ row_sum;
    end
end

%% --- Visualization ------------------------------------------------------
figure;
bar3(P_m_given_N);
xlabel('Detections N');
ylabel( 'Objects m');
zlabel('Probability');
title('P(m | N)');
xticks( 1:numel(det_axis));
xticklabels( string(det_axis));
yticks(1:numel(N_vals));
yticklabels(string(N_vals));
zlim( [0, max(P_m_given_N(:), [], 'omitnan') * 1.05 + eps]);

%% --- Export results -----------------------------------------------------
results = struct();
results.P_m_given_N = P_m_given_N;
results.det_counts_by_obj = det_counts_by_obj;
results.cfg = cfg;

out_file = fullfile(script_dir, 'probObjCt_results.mat');
%save(out_file, 'results', '-mat');
fprintf('Saved dataset to: %s\n', out_file);


%% FUNCTIONS

function det_counts = runObjectCountTrials(n_obj, cfg, G, M)
    
    det_counts = zeros(cfg.MC, 1);
    

    for trial = 1:cfg.MC
        
        % Pick n_obj random pixels to place objects in
        X_GT = cell(n_obj, 1);
        for n = 1:n_obj
            x = randi(128);
            y = randi([30 128]);
            X_GT{n} = [x; y];
        end
        
        sim_signal = simulateSignalFrame(X_GT, cfg, G, M);
        

        if cfg.verbose
            fprintf('  Trial %d / %d\n', trial, cfg.MC);
        end

        det_counts(trial) = estimateDetectionCount(sim_signal, cfg);
        
    end

end


function sim_signal = simulateSignalFrame(X_GT, cfg, G, M)
    S = zeros(cfg.npx, cfg.npx);

    for j = 1:numel(X_GT)
        true_pos = X_GT{j}(1:2);
        Gx = true_pos(1);
        Gy = true_pos(2);


        if ~isempty(Gx) && ~isempty(Gy) && Gx >= 1 && Gx <= cfg.npx && Gy >= 1 && Gy <= cfg.npx
            S(Gy, Gx) = S(Gy, Gx) + 1;
        end
    end

    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, cfg.npx, cfg.npx)';
end

function [window, avgSignal, window_ready] = updateSlidingWindow(window, new_signal)
    T = numel(window);
    filled_idx = find(~cellfun(@isempty, window), 1, 'last');
    if isempty(filled_idx)
        filled_idx = 0;
    end

    if filled_idx > 0
        window(2:min(T, filled_idx + 1)) = window(1:min(T - 1, filled_idx));
    end
    window{1} = new_signal;

    window_ready = filled_idx >= T - 1;
    if window_ready
        avgSignal = mean(cat(ndims(window{1}) + 1, window{:}), ndims(window{1}) + 1);
    else
        avgSignal = [];
    end
end

function det_count = estimateDetectionCount(avgSignal, cfg)
    signal_normalized = normalizeSignalFrame(avgSignal, cfg);

    [~, peak_x, peak_y] = CA_CFAR(signal_normalized(21:128,:), ...
        cfg.Pfa, cfg.Ng, cfg.Nr);
    peak_x = peak_x + cfg.crop_rows;

    if isempty(peak_x)
        det_count = 0;
        return
    end

    pvinds = sub2ind([cfg.npx, cfg.npx], peak_x, peak_y);
    meas_xy = [cfg.pxgrid(pvinds)'; cfg.pygrid(pvinds)'];
    valid = meas_xy(2,:) >= 0.5 & signal_normalized(pvinds)' > cfg.intensity_thr;
    valid_meas_xy = meas_xy(:, valid);

    clustered_meas_xy = clusterNearbyDetections(valid_meas_xy, cfg.cluster_radius);
    det_count = size(clustered_meas_xy, 2);
end

function signal_normalized = normalizeSignalFrame(avgSignal, cfg)
    blurred = imgaussfilt(avgSignal, cfg.blur_sigma);
    blurred(1:cfg.crop_rows, :) = NaN;
    signal_scaled = asinh(blurred);
    if (max(signal_scaled(:)) - min(signal_scaled(:))) ~= 0
        signal_normalized = (signal_scaled - min(signal_scaled(:))) ./ ...
            (max(signal_scaled(:)) - min(signal_scaled(:)));
    else
        signal_normalized = signal_scaled;
    end
end

function clustered_xy = clusterNearbyDetections(meas_xy, cluster_radius)
    if isempty(meas_xy)
        clustered_xy = zeros(2, 0);
        return
    end

    n_det = size(meas_xy, 2);
    visited = false(1, n_det);
    clustered_xy = zeros(2, 0);

    for i = 1:n_det
        if visited(i)
            continue
        end

        component = i;
        queue = i;
        visited(i) = true;

        while ~isempty(queue)
            current = queue(1);
            queue(1) = [];

            for j = 1:n_det
                if visited(j)
                    continue
                end

                if norm(meas_xy(:, current) - meas_xy(:, j)) <= cluster_radius
                    visited(j) = true;
                    queue(end + 1) = j; %#ok<AGROW>
                    component(end + 1) = j; %#ok<AGROW>
                end
            end
        end

        clustered_xy(:, end + 1) = mean(meas_xy(:, component), 2); %#ok<AGROW>
    end
end

function value = maxZeroSafe(x)
    if isempty(x)
        value = 0;
    else
        value = max(x);
    end
end

