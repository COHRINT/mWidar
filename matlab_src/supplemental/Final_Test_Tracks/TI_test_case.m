clear; clc; close all
rng(400)

script_dir = fileparts(mfilename('fullpath'));
repo_dir   = fileparts(script_dir);
root_dir   = fileparts(repo_dir);

addpath(fullfile(root_dir, 'DA_Track'));
addpath(fullfile(root_dir, 'DA_Track', 'multi'));
addpath(repo_dir);
addpath(script_dir);

% Track-Init dataset generator for mWidar.
% Produces Data in the same core format as test_track.m:
%   Data.y      -> 1 x n_t cell, each [2 x N_det]
%   Data.signal -> 1 x n_t cell, each [128 x 128], raw signal
%   Data.obj_ct -> 1 x n_t vector, containing true object count

%% --- User config --------------------------------------------------------
max_obj = 4; % only matter for rw mode
min_obj = 1;
min_track_y = 1.0;
min_start_separation = 0.75;
count_mode = 'constant';   % 'random_walk' or 'constant'
constant_obj_count = 1;       % used only when count_mode='constant'

dt   = 0.01;
tvec = 0:dt:10;
n_t  = numel(tvec);

Pfa = 0.285;
Ng  = 15;
Nr  = 20;

T = 1;                % per-frame processing (no window averaging)
thr = 0.5;           % intensity threshold (matches probObjCt.m)
cluster_radius = 0.35;
blur_sigma = 1.3;
crop_rows = 20;

min_dwell = 80;       % frames
max_dwell = 160;      % frames

%% --- Scene / grid setup ------------------------------------------------
npx   = 128;
xgrid = linspace(-2, 2, npx);
ygrid = linspace(0, 4, npx);
[pxgrid, pygrid] = meshgrid(xgrid, ygrid);

%% --- Load simulation matrices ------------------------------------------
fprintf('Loading mWidar simulation matrices...\n');
load(fullfile(repo_dir, 'recovery.mat'), 'G');
load(fullfile(repo_dir, 'sampling.mat'), 'M');
fprintf('  M: %dx%d, G: %dx%d\n', size(M,1), size(M,2), size(G,1), size(G,2));

%% --- Generate random-walk object count ---------------------------------
switch lower(count_mode)
    case 'random_walk'
        obj_ct = generate_obj_count_walk(n_t, min_obj, max_obj, min_dwell, max_dwell);
    case 'constant'
        if constant_obj_count < min_obj || constant_obj_count > max_obj
            error('constant_obj_count must be within [%d, %d].', min_obj, max_obj);
        end
        obj_ct = constant_obj_count * ones(1, n_t);
    otherwise
        error('Unsupported count_mode: %s. Use ''random_walk'' or ''constant''.', count_mode);
end

%% --- Build up-to-4 GT tracks (same style as multi_obj_tracks.m) --------
traj_types = ["LINE", "PARABOLA"];
X_GT = cell(1, max_obj);
starts = sample_start_points(max_obj, min_track_y, min_start_separation);
stops  = sample_stop_points(max_obj, min_track_y);

for k = 1:max_obj
    traj_type = traj_types(mod(k-1, numel(traj_types)) + 1);
    switch traj_type
        case "LINE"
            X_GT{k} = generate_line_track(tvec, dt, starts(:,k), stops(:,k));
        case "PARABOLA"
            X_GT{k} = generate_parabola_track(tvec, dt, starts(:,k), stops(:,k));
        otherwise
            error('Unsupported trajectory type: %s', traj_type);
    end
end

active_mask = assign_active_tracks(obj_ct, max_obj);

%% --- Simulate raw signal sequence --------------------------------------
Signal = cell(1, n_t);
for i = 1:n_t
    S = zeros(npx, npx);

    active_ids = find(active_mask(:, i));
    for idx = 1:numel(active_ids)
        j = active_ids(idx);
        true_pos = X_GT{j}(1:2, i);
        px = true_pos(1);
        py = true_pos(2);

        if px > -2 && px < 2 && py > 0 && py < 4
            Gx = find(px <= xgrid, 1, 'first');
            Gy = find(py <= ygrid, 1, 'first');
            if ~isempty(Gx) && ~isempty(Gy) && Gx >= 1 && Gx <= npx && Gy >= 1 && Gy <= npx
                S(Gy, Gx) = S(Gy, Gx) + 1;
            end
        end
    end

    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    Signal{i}   = reshape(signal_flat, npx, npx)';
end

%% --- Process signal + CA-CFAR (per-frame, no window averaging) ---------
y = cell(1, n_t);

for k = 1:n_t
    frameSignal = Signal{k};
    blurred = imgaussfilt(frameSignal, blur_sigma);
    blurred(1:crop_rows,:) = NaN;
    signal_scaled = asinh(blurred);
    signal_normalized = (signal_scaled - min(signal_scaled(:))) / ...
        (max(signal_scaled(:)) - min(signal_scaled(:)));

    [~, peak_x, peak_y] = CA_CFAR(signal_normalized(crop_rows+1:128,:), Pfa, Ng, Nr);
    peak_x = peak_x + crop_rows;

    if isempty(peak_x)
        y{k} = zeros(2, 0);
        continue
    end

    pvinds = sub2ind([npx, npx], peak_x, peak_y);
    meas_xy = [pxgrid(pvinds)'; pygrid(pvinds)'];
    valid = meas_xy(2,:) >= 0.5 & signal_normalized(pvinds)' > thr;
    valid_meas_xy = meas_xy(:, valid);
    clustered_meas_xy = clusterNearbyDetections(valid_meas_xy, cluster_radius);
    y{k} = clustered_meas_xy;
end

%% --- Save dataset -------------------------------------------------------
Data = struct();
Data.GT = X_GT;
Data.y = y;
Data.signal = Signal;
Data.obj_ct = obj_ct;
Data.active_mask = active_mask;
Data.params = struct( ...
    'rng_seed', 400, ...
    'dt', dt, ...
    'tvec', tvec, ...
    'Pfa', Pfa, ...
    'Ng', Ng, ...
    'Nr', Nr, ...
    'T', T, ...
    'thr', thr, ...
    'blur_sigma', blur_sigma, ...
    'crop_rows', crop_rows, ...
    'cluster_radius', cluster_radius, ...
    'count_mode', count_mode, ...
    'constant_obj_count', constant_obj_count, ...
    'min_dwell', min_dwell, ...
    'max_dwell', max_dwell);

sim_dur_s = tvec(end) - tvec(1);
if strcmpi(count_mode, 'constant')
    mode_tag = sprintf('const%d', constant_obj_count);
else
    mode_tag = sprintf('rw%dto%d', min_obj, max_obj);
end
dur_tag = strrep(sprintf('%0.0f', sim_dur_s), '.', 'p');
dt_tag = strrep(sprintf('%0.3f', dt), '.', 'p');
out_name = sprintf('TI_test_case_%s_%ss_dt%s.mat', mode_tag, dur_tag, dt_tag);
out_file = fullfile(script_dir, out_name);
save(out_file, 'Data', '-mat');
fprintf('Saved dataset to: %s\n', out_file);


function obj_ct = generate_obj_count_walk(n_t, min_obj, max_obj, min_dwell, max_dwell)
    max_attempts = 200;

    for attempt = 1:max_attempts
        obj_ct = zeros(1, n_t);
        idx = 1;
        cur = min_obj;
        transitions = zeros(1, n_t);
        trans_idx = 0;

        while idx <= n_t
            dwell = randi([min_dwell, max_dwell]);
            j_end = min(n_t, idx + dwell - 1);
            obj_ct(idx:j_end) = cur;
            idx = j_end + 1;

            if idx <= n_t
                if cur <= min_obj
                    step = 1;
                elseif cur >= max_obj
                    step = -1;
                else
                    if rand() < 0.2
                        step = -1;
                    else
                        step = 1;
                    end
                end
                cur = cur + step;
                trans_idx = trans_idx + 1;
                transitions(trans_idx) = step;
            end
        end

        used_steps = transitions(1:trans_idx);
        has_up = any(used_steps > 0);
        has_down = any(used_steps < 0);
        if has_up && has_down
            return
        end
    end

    warning('Could not realize both up/down transitions; using last generated walk.');
end

function active_mask = assign_active_tracks(obj_ct, max_obj)
    n_t = numel(obj_ct);
    active_mask = false(max_obj, n_t);

    active = false(max_obj, 1);
    prev_count = max(0, min(max_obj, round(obj_ct(1))));
    if prev_count > 0
        perm0 = randperm(max_obj);
        active(perm0(1:prev_count)) = true;
    end
    active_mask(:, 1) = active;

    for k = 2:n_t
        curr_count = obj_ct(k);

        if curr_count > prev_count
            n_add = curr_count - prev_count;
            inactive_ids = find(~active);
            perm = randperm(numel(inactive_ids));
            add_ids = inactive_ids(perm(1:n_add));
            active(add_ids) = true;
        elseif curr_count < prev_count
            n_remove = prev_count - curr_count;
            active_ids = find(active);
            perm = randperm(numel(active_ids));
            remove_ids = active_ids(perm(1:n_remove));
            active(remove_ids) = false;
        end

        active_mask(:, k) = active;
        prev_count = curr_count;
    end
end

function clustered_xy = clusterNearbyDetections(meas_xy, cluster_radius)
    if isempty(meas_xy)
        clustered_xy = zeros(2,0);
        return
    end

    n_det = size(meas_xy, 2);
    dist_mat = pdist2(meas_xy', meas_xy');
    adjacency = dist_mat <= cluster_radius;

    visited = false(1, n_det);
    clustered_xy = zeros(2,0);

    for i = 1:n_det
        if visited(i)
            continue
        end

        queue = i;
        component = i;
        visited(i) = true;

        while ~isempty(queue)
            current = queue(1);
            queue(1) = [];

            neighbors = find(adjacency(current,:) & ~visited);
            if ~isempty(neighbors)
                visited(neighbors) = true;
                queue = [queue, neighbors]; %#ok<AGROW>
                component = [component, neighbors]; %#ok<AGROW>
            end
        end

        clustered_xy(:, end+1) = mean(meas_xy(:,component), 2); %#ok<AGROW>
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
