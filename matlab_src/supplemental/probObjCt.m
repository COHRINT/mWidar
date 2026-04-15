%% Monte Carlo estimate of object-count detection probabilities
%
% This script estimates:
%   1) P(N | m): the probability of observing N detections given m objects
%   2) P(m | N): the posterior probability of m objects given N detections
%
% The posterior is derived from the Monte Carlo likelihood estimate using a
% user-defined prior over m.

clc; close all; clear
rng(467)

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

cfg.Pfa = 0.295;
cfg.Ng = 15;
cfg.Nr = 20;

cfg.T = 25;                  % Sliding-window length
cfg.intensity_thr = 0.5;     % Detection intensity threshold
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
cfg.prior_m = ones(cfg.max_obj, 1) ./ cfg.max_obj;

n_k = numel(cfg.tvec);

%% --- Run Monte Carlo ----------------------------------------------------
det_counts_by_obj = cell(cfg.max_obj, 1);

for n_obj = 1:cfg.max_obj
    if cfg.verbose
        fprintf('Simulating object count m = %d\n', n_obj);
    end

    det_counts_by_obj{n_obj} = runObjectCountTrials(n_obj, cfg, G, M, n_k);
end

%% --- Convert counts to probability tables -------------------------------
max_det_observed = max(cellfun(@maxZeroSafe, det_counts_by_obj));
det_axis = 0:max_det_observed;

P_N_given_m = zeros(cfg.max_obj, numel(det_axis));
for n_obj = 1:cfg.max_obj
    counts = det_counts_by_obj{n_obj};
    if isempty(counts)
        continue
    end

    hist_counts = accumarray(counts + 1, 1, [numel(det_axis), 1]);
    P_N_given_m(n_obj, :) = (hist_counts ./ sum(hist_counts)).';
end

P_m_given_N = computePosteriorFromLikelihood(P_N_given_m, cfg.prior_m);

%% --- Visualization ------------------------------------------------------
fig = figure('Color', 'w', 'Position', [100 100 1100 430]);
tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
bar3(ax1, P_N_given_m);
xlabel(ax1, 'Detections N');
ylabel(ax1, 'Objects m');
zlabel(ax1, 'Probability');
title(ax1, 'P(N | m)');
xticks(ax1, 1:numel(det_axis));
xticklabels(ax1, string(det_axis));
yticks(ax1, 1:cfg.max_obj);
zlim(ax1, [0, max(P_N_given_m(:), [], 'omitnan') * 1.05 + eps]);

ax2 = nexttile;
bar3(ax2, P_m_given_N);
xlabel(ax2, 'Detections N');
ylabel(ax2, 'Objects m');
zlabel(ax2, 'Probability');
title(ax2, 'P(m | N)');
xticks(ax2, 1:numel(det_axis));
xticklabels(ax2, string(det_axis));
yticks(ax2, 1:cfg.max_obj);
zlim(ax2, [0, max(P_m_given_N(:), [], 'omitnan') * 1.05 + eps]);

%% --- Export results -----------------------------------------------------
results = struct();
results.cfg = cfg;
results.det_axis = det_axis;
results.P_N_given_m = P_N_given_m;
results.P_m_given_N = P_m_given_N;
results.det_counts_by_obj = det_counts_by_obj;

save(fullfile(script_dir, 'probObjCt_results.mat'), 'results');


%% FUNCTIONS

function det_counts = runObjectCountTrials(n_obj, cfg, G, M, n_k)
    n_valid_frames = n_k - cfg.T + 1;
    det_counts = zeros(cfg.MC * n_valid_frames, 1);
    write_idx = 1;

    for trial = 1:cfg.MC
        X_GT = generateTracks(n_obj, cfg);
        window = cell(1, cfg.T);

        if cfg.verbose
            fprintf('  Trial %d / %d\n', trial, cfg.MC);
        end

        for k = 1:n_k
            sim_signal = simulateSignalFrame(X_GT, k, cfg, G, M);
            [window, avgSignal, window_ready] = updateSlidingWindow(window, sim_signal);
            if ~window_ready
                continue
            end

            det_counts(write_idx) = estimateDetectionCount(avgSignal, cfg);
            write_idx = write_idx + 1;
        end
    end

    det_counts = det_counts(1:write_idx-1);
end

function X_GT = generateTracks(n_obj, cfg)
    X_GT = cell(1, n_obj);
    starts = sample_start_points(n_obj, cfg.min_track_y, cfg.min_start_separation);
    stops = sample_stop_points(n_obj, cfg.min_track_y);

    for idx = 1:n_obj
        traj_type = cfg.traj_types(mod(idx - 1, numel(cfg.traj_types)) + 1);
        switch traj_type
            case "LINE"
                X_GT{idx} = generate_line_track(cfg.tvec, cfg.dt, starts(:,idx), stops(:,idx));
            case "PARABOLA"
                X_GT{idx} = generate_parabola_track(cfg.tvec, cfg.dt, starts(:,idx), stops(:,idx));
            otherwise
                error('Unsupported trajectory type: %s', traj_type);
        end
    end
end

function sim_signal = simulateSignalFrame(X_GT, k, cfg, G, M)
    S = zeros(cfg.npx, cfg.npx);

    for j = 1:numel(X_GT)
        true_pos = X_GT{j}(1:2, k);
        px = true_pos(1);
        py = true_pos(2);

        if px <= -2 || px >= 2 || py <= 0 || py >= 4
            continue
        end

        Gx = find(px <= cfg.xgrid, 1, 'first');
        Gy = find(py <= cfg.ygrid, 1, 'first');
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

    [~, peak_x, peak_y] = CA_CFAR(signal_normalized(cfg.crop_rows + 1:end, :), ...
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

    finite_mask = isfinite(signal_scaled);
    finite_vals = signal_scaled(finite_mask);
    if isempty(finite_vals)
        signal_normalized = zeros(size(signal_scaled));
        return
    end

    min_val = min(finite_vals);
    max_val = max(finite_vals);
    denom = max(max_val - min_val, eps);

    signal_normalized = zeros(size(signal_scaled));
    signal_normalized(finite_mask) = (signal_scaled(finite_mask) - min_val) ./ denom;
end

function P_m_given_N = computePosteriorFromLikelihood(P_N_given_m, prior_m)
    weighted = P_N_given_m .* prior_m;
    evidence = sum(weighted, 1);

    P_m_given_N = zeros(size(P_N_given_m));
    valid_cols = evidence > 0;
    P_m_given_N(:, valid_cols) = weighted(:, valid_cols) ./ evidence(valid_cols);
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

function starts = sample_start_points(n_obj, min_track_y, min_separation)
    starts = sample_points_with_min_separation(n_obj, min_track_y, 3.4, min_separation);
end

function stops = sample_stop_points(n_obj, min_track_y)
    stops = zeros(2, n_obj);
    stops(1,:) = -1.8 + 3.6 .* rand(1, n_obj);
    stops(2,:) = min_track_y + (3.4 - min_track_y) .* rand(1, n_obj);
end

function pts = sample_points_with_min_separation(n_obj, y_min, y_max, min_separation)
    pts = zeros(2, n_obj);
    max_attempts = 500;

    for k = 1:n_obj
        placed = false;
        for attempt = 1:max_attempts
            candidate = [-1.8 + 3.6 .* rand(); y_min + (y_max - y_min) .* rand()];
            if k == 1
                pts(:,k) = candidate;
                placed = true;
                break
            end

            deltas = pts(:,1:k-1) - candidate;
            distances = sqrt(sum(deltas.^2, 1));
            if all(distances >= min_separation)
                pts(:,k) = candidate;
                placed = true;
                break
            end
        end

        if ~placed
            error(['Could not place %d start points with %.2f m separation ' ...
                'after %d attempts.'], n_obj, min_separation, max_attempts);
        end
    end
end

function X = generate_line_track(tvec, dt, start_xy, stop_xy)
    T = tvec(end);
    delta_pos = stop_xy - start_xy;
    a_progress = -0.01 + 0.02 .* rand();
    v0_progress = (1 - 0.5 * a_progress * T^2) / T;

    progress = v0_progress .* tvec + 0.5 .* a_progress .* tvec.^2;
    x = start_xy(1) + delta_pos(1) .* progress;
    y = start_xy(2) + delta_pos(2) .* progress;

    x = clamp(x, -2, 2);
    y = clamp(y, 1, 4);

    vx = gradient(x, dt);
    vy = gradient(y, dt);
    ax = gradient(vx, dt);
    ay = gradient(vy, dt);
    X = [x; y; vx; vy; ax; ay];
end

function X = generate_parabola_track(tvec, dt, start_xy, stop_xy)
    n_t = numel(tvec);
    T = tvec(end);
    tau = tvec ./ T;

    x = start_xy(1) + (stop_xy(1) - start_xy(1)) .* tau;

    mid = 0.5 * (start_xy + stop_xy);
    y_peak = min(3.8, max(1.1, mid(2) + 0.5 + 0.4 .* rand()));

    c = start_xy(2);
    a = 2 * (start_xy(2) + stop_xy(2) - 2 * y_peak);
    b = stop_xy(2) - c - a;
    y = a .* tau.^2 + b .* tau + c;

    x = clamp(x, -2, 2);
    y = clamp(y, 1, 4);

    vx = gradient(x, dt);
    vy = gradient(y, dt);
    ax = gradient(vx, dt);
    ay = gradient(vy, dt);

    X = [x; y; vx; vy; ax; ay];
    if size(X, 2) ~= n_t
        error('Parabola track length mismatch.');
    end
end

function out = clamp(in, lo, hi)
    out = min(hi, max(lo, in));
end
