clear; clc; close all
rng(400)

script_dir = fileparts(mfilename('fullpath'));
repo_dir   = fileparts(script_dir);
addpath(script_dir);

% Multi-object synthetic dataset generator for mWidar.
% Produces Data in the same core format as test_track.m:
%   Data.GT     -> 1 x N cell, each [6 x n_t]
%   Data.y      -> 1 x n_t cell, each [2 x N_det]
%   Data.signal -> 1 x n_t cell, each [128 x 128]

%% User config
object_count = 5;      % Any positive integer
track_name   = "multi_obj_"+num2str(object_count);
write_gif    = true;
noise_flag   = false;
min_track_y  = 1.0;    % Tracks must remain at or above this y-position [m]
min_start_separation = 0.75; % Minimum pairwise distance between start points [m]

dt   = 0.01;
tvec = 0:dt:10;
n_t  = numel(tvec);

if object_count < 1 || floor(object_count) ~= object_count
    error('object_count must be a positive integer.');
end

out_dir_abs = fullfile(repo_dir, 'Final_Test_Tracks', 'MultiObj');

if ~exist(out_dir_abs, 'dir')
    mkdir(out_dir_abs);
end

fprintf('Loading mWidar simulation matrices...\n');
load(fullfile(repo_dir, 'recovery.mat'), 'G');
load(fullfile(repo_dir, 'sampling.mat'), 'M');
fprintf('  M: %dx%d, G: %dx%d\n', size(M,1), size(M,2), size(G,1), size(G,2));

%% Build GT tracks
%traj_types = ["LINE", "PARABOLA", "SCURVE"];
traj_types = ["LINE", "PARABOLA"];
X_GT = cell(1, object_count);

starts = sample_start_points(object_count, min_track_y, min_start_separation);
stops  = sample_stop_points(object_count, min_track_y);
for k = 1:object_count
    traj_type = traj_types(mod(k-1, numel(traj_types)) + 1);

    switch traj_type
        case "LINE"
            X_GT{k} = generate_line_track(tvec, dt, starts(:,k), stops(:,k));
        case "PARABOLA"
            X_GT{k} = generate_parabola_track(tvec, dt, starts(:,k), stops(:,k));
        case "SCURVE"
            X_GT{k} = generate_scurve_track(tvec, dt, starts(:,k), stops(:,k));
    end
    
end

%% Simulate measurements and signals

%% ---- Scene / grid setup ------------------------------------------------
npx    = 128;
Lscene = 4;           % scene height [m]
xgrid  = linspace(-2, 2,      npx);  % [-2, 2] m
ygrid  = linspace( 0, Lscene, npx);  % [0,  4] m
[pxgrid, pygrid] = meshgrid(xgrid, ygrid);

%% ---- CA-CFAR parameters ------------------------------------------------
%%% Might want to include this in a config file at some point ??

Pfa = 0.295;   % false alarm probability (tuned for this scene)
Ng  = 5;       % guard cells
Nr  = 20;      % training (reference) cells

y   = cell(1, n_t);
Signal = cell(1, n_t);

for i = 1:n_t
    
    %% Populate S
    S = zeros(128, 128);
    for j = 1:object_count
        true_pos = X_GT{j}(1:2, i);
        px = true_pos(1);
        py = true_pos(2);
    
        % ---- mWidar forward model ----
        
        if px > -2 && px < 2 && py > 0 && py < 4
            Gx = find(px <= xgrid, 1, 'first');
            Gy = find(py <= ygrid, 1, 'first');
            if Gx >= 1 && Gx <= 128 && Gy >= 1 && Gy <= 128
                
                S(Gy, Gx) = 1;
            end
        else
            fprintf("Timestep %i not in scene", i)
        end

    end
    %% Construct Signal
    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal  = reshape(signal_flat, 128, 128)';

    blurred            = imgaussfilt(sim_signal, 1.3);
    Signal{i}        = sim_signal;
    signal_scaled = blurred;
    signal_scaled(1:20,:) = NaN;
    signal_scaled = asinh(signal_scaled);
    signal_normalized  = (signal_scaled - min(signal_scaled(:))) / (max(signal_scaled(:)) - min(signal_scaled(:)));

    % ---- CA-CFAR detection ----
    try
        
        [~, peak_x, peak_y] = CA_CFAR(signal_normalized(21:128,:), Pfa, Ng, Nr);
        peak_x = peak_x + 20;
        if ~isempty(peak_x)
            pvinds   = sub2ind([npx, npx], peak_x, peak_y);
            meas_xy  = [pxgrid(pvinds)'; pygrid(pvinds)'];
            % Remove detections below y=0.5 m (clutter floor)
            valid    = meas_xy(2,:) >= 0.5;
            y{i}   = meas_xy(:, valid);
        else
            fprintf('  [t=%d] No detections\n', i);
            y{i} = zeros(2, 0);
        end

    catch err
        fprintf('  [t=%d] CA_CFAR error: %s — using noisy truth\n', i, err.message);
        y{i} = true_pos + 0.1*randn(detector_stream, 2, 1);
    end
    
    

    if mod(i, 10) == 0
        n_det = size(y{i}, 2);
        fprintf('  Step %2d/%d: %d detections (truth=[%.3f, %.3f])\n', ...
            i, n_t, n_det, true_pos(1), true_pos(2));
    end

    Data.GT     = X_GT;
    Data.y      = y;
    Data.signal = Signal;
    Data.params = struct( ...
        'dt',     dt, ...
        'n_t',    n_t, ...
        'Pfa',    Pfa, ...
        'Ng',     Ng, ...
        'Nr',     Nr, ...
        'rng_seed', 400, ...
        'xgrid_range', [-2 2], ...
        'ygrid_range', [0  4], ...
        'npx',    npx);
end


file_path = fullfile(out_dir_abs, track_name + "_TI_test"+".mat");
save(file_path, 'Data', '-mat');
fprintf('Saved dataset to: %s\n', file_path);

%% Plot summary + optional GIF
%plot_multi_object_dataset(Data, tvec, object_count, out_dir_abs, track_name, write_gif);


function starts = sample_start_points(n_obj, min_track_y, min_separation)
    starts = sample_points_with_min_separation(n_obj, min_track_y, 3.4, min_separation);
end

function stops = sample_stop_points(n_obj, min_track_y)
    stops = zeros(2, n_obj);
    stops(1,:) = -1.8 + 3.6 .* rand(1, n_obj);
    stops(2,:) =  min_track_y + (3.4 - min_track_y) .* rand(1, n_obj);
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
                break;
            end

            deltas = pts(:,1:k-1) - candidate;
            distances = sqrt(sum(deltas.^2, 1));
            if all(distances >= min_separation)
                pts(:,k) = candidate;
                placed = true;
                break;
            end
        end

        if ~placed
            error('Could not place %d start points with %.2f m separation after %d attempts.', ...
                n_obj, min_separation, max_attempts);
        end
    end
end

function X = generate_line_track(tvec, dt, start_xy, stop_xy)
    n_t = numel(tvec);
    T = tvec(end);

    delta_pos = stop_xy - start_xy;
    a_progress = -0.01 + 0.02 .* rand();
    v0_progress = (1 - 0.5 * a_progress * T^2) / T;

    progress      = v0_progress .* tvec + 0.5 .* a_progress .* tvec.^2;
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
    if size(X,2) ~= n_t
        error('Parabola track length mismatch.');
    end
end

function X = generate_scurve_track(tvec, dt, start_xy, stop_xy)
    T = tvec(end);
    tau = tvec ./ T;

    x_base = start_xy(1) + (stop_xy(1) - start_xy(1)) .* tau;
    y_base = start_xy(2) + (stop_xy(2) - start_xy(2)) .* tau;

    x = x_base + 0.5 .* sin(2*pi*tau) .* (1 - tau).^2 .* tau.^2;
    y = y_base + 0.35 .* sin(pi*tau) .* sin(2*pi*tau);

    x = clamp(x, -2, 2);
    y = clamp(y, 1, 4);

    vx = gradient(x, dt);
    vy = gradient(y, dt);
    ax = gradient(vx, dt);
    ay = gradient(vy, dt);

    X = [x; y; vx; vy; ax; ay];
end

function out = clamp(in, lo, hi)
    out = min(hi, max(lo, in));
end

function plot_multi_object_dataset(Data, tvec, n_obj, out_dir, track_name, write_gif)
    n_t = numel(tvec);

    npx = 128;
    xgrid = linspace(-2, 2, npx);
    ygrid = linspace(0, 4, npx);
    [pxgrid, pygrid] = meshgrid(xgrid, ygrid);

    colors = lines(max(n_obj, 7));

    fig = figure('Color', 'w', 'Position', [100 100 980 700]);

    if write_gif
        gif_path = fullfile(out_dir, track_name + ".gif");
    end

    for i = 1:n_t
        clf(fig);
        tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

        nexttile(1);
        if i <= numel(Data.signal) && ~isempty(Data.signal{i})
            surf(pxgrid, pygrid, Data.signal{i}, 'EdgeColor', 'none');
            view(2);
            axis equal tight;
            xlim([-2, 2]);
            ylim([0, 4]);
            title('Signal');
            xlabel('x (m)');
            ylabel('y (m)');
            colormap(turbo);
            colorbar;
        else
            imagesc(xgrid, ygrid, zeros(npx));
            axis xy;
            axis equal tight;
            xlim([-2, 2]);
            ylim([0, 4]);
            title('Signal (empty)');
            xlabel('x (m)');
            ylabel('y (m)');
            colormap(turbo);
            colorbar;
        end

        ax2 = nexttile(2);
        hold(ax2, 'on');

        for k = 1:n_obj
            Xk = Data.GT{k};
            clr = colors(mod(k-1, size(colors,1)) + 1, :);
            plot(Xk(1,1:i), Xk(2,1:i), '-', 'LineWidth', 1.8, 'Color', clr);
            plot(Xk(1,i), Xk(2,i), 'o', 'MarkerSize', 6, 'MarkerFaceColor', clr, 'MarkerEdgeColor', clr);
        end

        if i <= numel(Data.y) && ~isempty(Data.y{i})
            scatter(Data.y{i}(1,:), Data.y{i}(2,:), 14, 'r', 'filled');
        end

        axis equal;
        xlim([-2, 2]);
        ylim([0, 4]);
        xlabel('x (m)');
        ylabel('y (m)');
        title(sprintf('GT + Detections (%d objects)', n_obj));
        title(tl, sprintf('Multi-Object Track Playback (t = %.1f s)', tvec(i)));
        drawnow;

        if write_gif
            frame = getframe(fig);
            [imind, cm] = rgb2ind(frame2im(frame), 256);
            if i == 1
                imwrite(imind, cm, gif_path, 'gif', 'Loopcount', inf, 'DelayTime', tvec(2)-tvec(1));
            else
                imwrite(imind, cm, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', tvec(2)-tvec(1));
            end
        end
    end

    if write_gif
        fprintf('GIF written: %s\n', gif_path);
    end
end
