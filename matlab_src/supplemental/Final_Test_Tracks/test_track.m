clear; clc; close all
rng(400)

%%% Very early iteration of multi-target track generator. Will need to be
%%% changed to make it work for n tracks / different trajectories etc.

%% Housekeeping
% addpath(fullfile("include/"))
out_dir = "supplemental/Final_Test_Tracks/MultiObj";

%% ---- Load mWidar forward-model matrices ---------------------------------
fprintf('Loading mWidar simulation matrices...\n');
load(fullfile('supplemental', 'sampling.mat'),  'M');
load(fullfile('supplemental', 'recovery.mat'),  'G');
fprintf('  M: %dx%d,  G: %dx%d\n', size(M,1), size(M,2), size(G,1), size(G,2));

%% ---- 2 Object Test Track ------------------------------------------------
n_t = 100;
dt = 0.1;
tvec = dt:dt:n_t*dt;
n_track = 2;

% T1
start = [-1.5, 1.5]; stop = [1.5,3.5];
[X_GT1] = generate_Scurve(tvec,n_t,dt,start,stop);

% T2
start = [1.5, 3.5]; stop = [-1.5,1.5];
[X_GT2] = generate_Scurve(tvec,n_t,dt,start,stop);

X_GT = {X_GT1, X_GT2};

%% ---- Scene / grid setup ------------------------------------------------
npx    = 128;
Lscene = 4;           % scene height [m]
xgrid  = linspace(-2, 2,      npx);  % [-2, 2] m
ygrid  = linspace( 0, Lscene, npx);  % [0,  4] m
[pxgrid, pygrid] = meshgrid(xgrid, ygrid);

%% ---- CA-CFAR parameters ------------------------------------------------
%%% Might want to include this in a config file at some point ??

Pfa = 0.367;   % false alarm probability (tuned for this scene)
Ng  = 5;       % guard cells
Nr  = 20;      % training (reference) cells

%% ---- Generate measurements via mWidar + CA-CFAR ------------------------
fprintf('Running mWidar forward model + CA-CFAR detector...\n');

detector_stream = RandStream('mt19937ar', 'Seed', 400);

y_4      = cell(1, n_t);
Signal_4 = cell(1, n_t);

for i = 1:n_t
    
    %% Populate S
    S = zeros(128, 128);
    for j = 1:n_track
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

        %% REMEMBER TO REMOVE - For Debugging
        %y_4{i} = [y_4{i} true_pos + 0.1*randn(detector_stream, 2, 1) ];
    end
    %% Construct Signal
    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal  = reshape(signal_flat, 128, 128)';

    blurred            = imgaussfilt(sim_signal, 2);
    Signal_4{i}        = blurred;
    signal_normalized  = blurred;

    % ---- CA-CFAR detection ----
    try
        [~, peak_x, peak_y] = CA_CFAR(signal_normalized, Pfa, Ng, Nr);

        if ~isempty(peak_x)
            pvinds   = sub2ind([npx, npx], peak_x, peak_y);
            meas_xy  = [pxgrid(pvinds)'; pygrid(pvinds)'];
            % Remove detections below y=0.5 m (clutter floor)
            valid    = meas_xy(2,:) >= 0.5;
            y_4{i}   = meas_xy(:, valid);
        else
            fprintf('  [t=%d] No detections\n', i);
            y_4{i} = zeros(2, 0);
        end

    catch err
        fprintf('  [t=%d] CA_CFAR error: %s — using noisy truth\n', i, err.message);
        y_4{i} = true_pos + 0.1*randn(detector_stream, 2, 1);
    end
    
    

    if mod(i, 10) == 0
        n_det = size(y_4{i}, 2);
        fprintf('  Step %2d/%d: %d detections (truth=[%.3f, %.3f])\n', ...
            i, n_t, n_det, true_pos(1), true_pos(2));
    end

    Data.GT     = X_GT;
    Data.y      = y_4;
    Data.signal = Signal_4;
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

%% ---- Save ---------------------------------------------------------------
file_path = fullfile(out_dir, "demo_track.mat");
save(file_path,'Data', '-mat')
fprintf('\nSaved dataset to: %s\n', file_path);


%% ---- Plot + GIF: signal surface (top-down), GT trajectories, measurements
gif_path = fullfile(out_dir, "two_target_trajectories.gif");
fprintf('Rendering GIF: %s\n', gif_path);

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fig = figure('Color', 'w', 'Position', [100 100 900 700]);

for i = 1:n_t
    clf(fig);
    tl = tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % Left subplot: signal only
    ax1 = nexttile(1); 
    Z = Signal_4{i};
    surf(pxgrid, pygrid, Z, 'EdgeColor', 'none');
    view(2);
    axis equal tight;
    xlim([min(xgrid), max(xgrid)]);
    ylim([min(ygrid), max(ygrid)]);
    xlabel('x (m)');
    ylabel('y (m)');
    title('Signal');
    colormap(turbo);
    colorbar;

    % Right subplot: trajectories + current timestep measurements only
    nexttile(2);
    hold on;
    plot(X_GT1(1,1:i), X_GT1(2,1:i), 'b-', 'LineWidth', 2.0);
    plot(X_GT2(1,1:i), X_GT2(2,1:i), 'k-', 'LineWidth', 2.0);
    plot(X_GT1(1,i), X_GT1(2,i), 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
    plot(X_GT2(1,i), X_GT2(2,i), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 7);
    if ~isempty(y_4{i})
        scatter(y_4{i}(1,:), y_4{i}(2,:), 16, 'r', 'filled');
    end
    axis equal;
    xlim([-2, 2]);
    ylim([0, 4]);
    set(gca, 'XLimMode', 'manual', 'YLimMode', 'manual');
    xlabel('x (m)');
    ylabel('y (m)');
    title('Trajectories + Current Measurements');
    lgd = legend({'Target 1 Trajectory', 'Target 2 Trajectory', ...
        'Target 1 Current', 'Target 2 Current', 'Measurements'}, ...
        'Location', 'eastoutside', 'Orientation', 'vertical');
    lgd.Box = 'off';
    title(tl, sprintf('Two-Target Tracking (t = %.1f s)', tvec(i)));
    drawnow;

    frame = getframe(fig);
    [imind, cm] = rgb2ind(frame2im(frame), 256);
    if i == 1
        imwrite(imind, cm, gif_path, 'gif', 'Loopcount', inf, 'DelayTime', dt);
    else
        imwrite(imind, cm, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', dt);
    end
end

fprintf('GIF written: %s\n', gif_path);


function [X_GT] = generate_Scurve(tvec,n_t,dt,start,stop)
    
    x_start = start(1); x_end = stop(1);
    y_start = start(2); y_end = stop(2);

    T   = tvec(end);
    tau = tvec / T;  % [0, 1]
    
    % Position
    x_traj = x_start + (x_end - x_start)*tau + 0.6*sin(2*pi*tau).*(1-tau).^2.*tau.^2;
    y_base  = y_start + (y_end - y_start)*tau;
    y_traj  = y_base  + 0.4*sin(pi*tau).*sin(2*pi*tau);
    
    % Velocity  (d/dt analytically)
    dtau_dt  = 1/T;
    dx_dtau  = (x_end-x_start) + 0.6.*(2*pi*cos(2*pi*tau).*(1-tau).^2.*tau.^2 + ...
                sin(2*pi*tau).*(2*(1-tau).*tau.^2.*(-1) + 2*(1-tau).^2.*tau));
    vx_traj  = dx_dtau .* dtau_dt;
    dy_dtau  = (y_end-y_start) + 0.4.*(pi*cos(pi*tau).*sin(2*pi*tau) + ...
                sin(pi*tau).*2*pi.*cos(2*pi*tau));
    vy_traj  = dy_dtau .* dtau_dt;
    
    % Acceleration (d²/dt² analytically) — use .* throughout to avoid [1xN]*[1xN] errors
    d2x_dtau2 = 0.6.*(-4.*pi^2.*sin(2.*pi.*tau).*(1-tau).^2.*tau.^2 + ...
        4.*pi.*cos(2.*pi.*tau).*(2.*(1-tau).*tau.^2.*(-1) + 2.*(1-tau).^2.*tau) + ...
        2.*pi.*cos(2.*pi.*tau).*(2.*tau.^2.*(-1) + 4.*(1-tau).*tau) + ...
        sin(2.*pi.*tau).*2.*(2.*tau.*(-1) + 2.*(1-tau)));
    ax_traj   = d2x_dtau2 .* dtau_dt^2;
    
    d2y_dtau2 = 0.4.*(-pi^2.*sin(pi.*tau).*sin(2.*pi.*tau) + ...
        2.*pi.*cos(pi.*tau).*2.*pi.*cos(2.*pi.*tau) + ...
        pi.*cos(pi.*tau).*2.*pi.*cos(2.*pi.*tau) - ...
        sin(pi.*tau).*4.*pi^2.*sin(2.*pi.*tau));
    ay_traj   = d2y_dtau2 .* dtau_dt^2;
    
    % Clamp within scene bounds
    x_traj = max(-2, min(2, x_traj));
    y_traj = max( 0, min(4, y_traj));
    
    X_GT = [x_traj; y_traj; vx_traj; vy_traj; ax_traj; ay_traj];  % 6 x n_t
    
    fprintf('Trajectory: S-curve, %d timesteps, dt=%.2f s\n', n_t, dt);
    fprintf('  Vx max=%.2f m/s, Vy max=%.2f m/s\n', max(abs(vx_traj)), max(abs(vy_traj)));

end

function [X_GT] = generate_Line(tvec,n_t,dt, start, stop)
    x_start = start(1); x_end = stop(1);
    y_start = start(2); y_end = stop(2);

    T = tvec(end);

    delta_pos = [x_end - x_start; y_end - y_start];

    % Use a scalar progress variable with constant acceleration so the
    % trajectory stays on the straight segment between start and end.
    a_progress = -0.01 + 0.02 .* rand();
    v0_progress = (1 - 0.5 * a_progress * T^2) / T;

    progress      = v0_progress .* tvec + 0.5 .* a_progress .* tvec.^2;
    progress_dot  = v0_progress + a_progress .* tvec;
    progress_ddot = a_progress .* ones(1, n_t);

    x_traj = x_start + delta_pos(1) .* progress;
    y_traj = y_start + delta_pos(2) .* progress;

    vx_traj = delta_pos(1) .* progress_dot;
    vy_traj = delta_pos(2) .* progress_dot;

    ax_traj = delta_pos(1) .* progress_ddot;
    ay_traj = delta_pos(2) .* progress_ddot;

    x_traj = max(-2, min(2, x_traj));
    y_traj = max( 0, min(4, y_traj));

    X_GT = [x_traj; y_traj; vx_traj; vy_traj; ax_traj; ay_traj];

    fprintf('Trajectory: Line, %d timesteps, dt=%.2f s\n', n_t, dt);
    fprintf('  Ax=%.3f m/s^2, Ay=%.3f m/s^2\n', ax_traj(1), ay_traj(1));
    fprintf('  Vx max=%.2f m/s, Vy max=%.2f m/s\n', max(abs(vx_traj)), max(abs(vy_traj)));
end

