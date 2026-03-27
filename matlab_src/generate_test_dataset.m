%% generate_test_dataset.m
%
% Extenstion of generate_tuning_dataset.m
%
% Generates a full mWidar simulation + CA-CFAR detector dataset
% from chosen track and saves it to:
%   matlab_src/data/TEST_DATASET/{track_name}/TRIAL{number}.mat
%
% Run from matlab_src/ after startup.m:
%   generate_tuning_dataset
%
% Output data.mat contains:
%   Data.GT     - [6 x N_k] ground truth [x; y; vx; vy; ax; ay]
%   Data.y      - {1 x N_k} cell, each entry [2 x N_det] detections (may be empty)
%   Data.signal - {1 x N_k} cell, each entry [128 x 128] processed mWidar image
%   Data.params - struct of generation parameters (for reference)

clear; clc; close all
rng(400)

% chosen_track = "RW";
% chosen_track = "LINE";
% chosen_track = "PARABOLA";
chosen_track = "SCURVE";


n_tracks = 10;

%% ---- Paths ---------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'supplemental'));
addpath(fullfile(script_dir, 'supplemental', 'Final_Test_Tracks'));

out_dir  = fullfile(script_dir, 'data/TEST_DATASET', chosen_track);

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
    fprintf('Created output directory: %s\n', out_dir);
end

%% ---- Load mWidar forward-model matrices ---------------------------------
fprintf('Loading mWidar simulation matrices...\n');
load(fullfile(script_dir, 'supplemental', 'sampling.mat'),  'M');
load(fullfile(script_dir, 'supplemental', 'recovery.mat'),  'G');
fprintf('  M: %dx%d,  G: %dx%d\n', size(M,1), size(M,2), size(G,1), size(G,2));

for i = 1:n_tracks

    out_file = fullfile(out_dir, chosen_track + num2str(i) + '.mat');

    [Data] = generate_dataset(chosen_track,M,G);

    %% ---- Save ---------------------------------------------------------------

    save(out_file, 'Data', '-mat');
    fprintf('\nSaved dataset to: %s\n', out_file);
    
    % Quick summary
    n_det_all = cellfun(@(c) size(c,2), Data.y);
    fprintf('Detection counts: min=%d, max=%d, mean=%.1f per timestep\n', ...
        min(n_det_all), max(n_det_all), mean(n_det_all));
    fprintf('Done.\n');
end

plot_test_dataset_gt(out_dir,chosen_track)

function [Data] = generate_dataset(chosen_track,M,G)
    %% ---- Scene / grid setup ------------------------------------------------
    npx    = 128;
    Lscene = 4;           % scene height [m]
    xgrid  = linspace(-2, 2,      npx);  % [-2, 2] m
    ygrid  = linspace( 0, Lscene, npx);  % [0,  4] m
    [pxgrid, pygrid] = meshgrid(xgrid, ygrid);
    
    dt        = 0.1;       % [s]
    num_steps = 99;
    tvec      = (0:num_steps) * dt;
    n_t       = length(tvec);  % 100 timesteps
    
    %% ---- Track Selection ---------------------------------------------------
    
    switch upper(chosen_track)
        case "LINE"
            [X_GT] = generate_Line(tvec,n_t,dt);
        case "PARABOLA"
            [X_GT] = generate_Parabola(tvec,n_t,dt);
        case "SCURVE"
            %% ---- Trajectory: analytical S-curve ------------------------------------
            [X_GT] = generate_Scurve(tvec,n_t,dt);
        case "RW"
            [X_GT] = generate_RWcurve(tvec,n_t,dt);
    end
    %% ---- CA-CFAR parameters ------------------------------------------------
    Pfa = 0.365;   % false alarm probability (tuned for this scene)
    Ng  = 5;       % guard cells
    Nr  = 20;      % training (reference) cells
    
    
    %% ---- Generate measurements via mWidar + CA-CFAR ------------------------
    fprintf('Running mWidar forward model + CA-CFAR detector...\n');
    
    detector_stream = RandStream('mt19937ar', 'Seed', 400);
    
    y_4      = cell(1, n_t);
    Signal_4 = cell(1, n_t);
    
    for i = 1:n_t
        true_pos = X_GT(1:2, i);
        px = true_pos(1);
        py = true_pos(2);
    
        % ---- mWidar forward model ----
        S = zeros(128, 128);
        if px > -2 && px < 2 && py > 0 && py < 4
            Gx = find(px <= xgrid, 1, 'first');
            Gy = find(py <= ygrid, 1, 'first');
            if Gx >= 1 && Gx <= 128 && Gy >= 1 && Gy <= 128
                S(Gy, Gx) = 1;
            end
        else
            fprintf("Timestep %i not in scene", i)
        end
    
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
function [X_GT] = generate_Scurve(tvec,n_t,dt)
    
    x_start = -1.9 + (-1-(-1.9)).*rand(); x_end = 1 + (1.9-(1)).*rand();
    y_start = 0.5 + (1.5-(0.5)).*rand(); y_end = 3 + (3.9-(3)).*rand();

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

function [X_GT] = generate_Line(tvec,n_t,dt)
    x_start = -1.9 + (-1-(-1.9)).*rand(); x_end = 1 + (1.9-(1)).*rand();
    y_start = 0.5 + (1.5-(0.5)).*rand(); y_end = 3 + (3.9-(3)).*rand();

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

function [X_GT] = generate_Parabola(tvec,n_t,dt)
    
    A = [0 0 1 0 0 0;
        0 0 0 1 0 0;
        0 0 0 0 1 0;
        0 0 0 0 0 1;
        0 0 0 0 0 0;
        0 0 0 0 0 0];

    xstart = unifrnd(-1.8,-1.3);
    ystart = unifrnd(0.5,1);
    vxstart = 0.25;
    vystart = 0.7;
    axstart = 0;
    aystart = -0.13;

    x0 = [xstart;ystart;vxstart;vystart;axstart;aystart];

    X_GT = zeros(6,n_t);
    X_GT(:,1) = x0;
    for i = 2:n_t
        X_GT(:,i) = (expm(A*dt)) * X_GT(:,i-1);
    end
    

    fprintf('Trajectory: Parabola, %d timesteps, dt=%.2f s\n', n_t, dt);
    fprintf('  Ax=%.3f m/s^2, Ay=%.3f m/s^2\n', X_GT(5,1),  X_GT(6,1));
    fprintf('  Vx max=%.2f m/s, Vy max=%.2f m/s\n', max(abs(X_GT(3,:))), max(abs(X_GT(4,:))));
end


function [X_GT] = generate_RWcurve(tvec,n_t,dt)
    x_start = -1.9 + (-1-(-1.9)).*rand();
    x_end   = 1 + (1.9-(1)).*rand();
    y_start = 0.5 + (1.5-(0.5)).*rand();
    y_end   = 3 + (3.9-(3)).*rand();

    T   = tvec(end);
    tau = tvec / T;

    % Start from the same nominal S-curve used in generate_Scurve.
    x_base = x_start + (x_end - x_start).*tau + 0.6.*sin(2.*pi.*tau).*(1-tau).^2.*tau.^2;
    y_line = y_start + (y_end - y_start).*tau;
    y_base = y_line + 0.4.*sin(pi.*tau).*sin(2.*pi.*tau);

    % Every ~5 timesteps, let the acceleration bias take a random walk.
    change_interval = 5;
    accel_step_sigma = 0.04;
    accel_bias_limit = 0.1;

    accel_bias = zeros(2, n_t);
    current_bias = [0; 0];
    for i = 2:n_t
        if mod(i-1, change_interval) == 0
            current_bias = current_bias + accel_step_sigma .* randn(2,1);
            current_bias = max(-accel_bias_limit, min(accel_bias_limit, current_bias));
        end
        accel_bias(:, i) = current_bias;
    end

    % Integrate the bias to get a smooth deviation from the nominal S-curve.
    vel_offset = zeros(2, n_t);
    pos_offset = zeros(2, n_t);
    for i = 2:n_t
        pos_offset(:, i) = pos_offset(:, i-1) + vel_offset(:, i-1).*dt + 0.5.*accel_bias(:, i-1).*dt.^2;
        vel_offset(:, i) = vel_offset(:, i-1) + accel_bias(:, i-1).*dt;
    end

    % Keep the perturbation small near the start/end so the result still
    % resembles the analytical S-curve and stays in-scene more reliably.
    envelope = sin(pi.*tau).^2;
    x_traj = x_base + envelope .* pos_offset(1, :);
    y_traj = y_base + envelope .* pos_offset(2, :);

    x_traj = max(-2, min(2, x_traj));
    y_traj = max( 0, min(4, y_traj));

    % Recompute derivatives from the final trajectory so the reported state
    % is consistent with the perturbed path.
    vx_traj = gradient(x_traj, dt);
    vy_traj = gradient(y_traj, dt);
    ax_traj = gradient(vx_traj, dt);
    ay_traj = gradient(vy_traj, dt);

    X_GT = [x_traj; y_traj; vx_traj; vy_traj; ax_traj; ay_traj];

    fprintf('Trajectory: RW S-curve, %d timesteps, dt=%.2f s\n', n_t, dt);
    fprintf('  Accel bias updates every %d steps\n', change_interval);
    fprintf('  Vx max=%.2f m/s, Vy max=%.2f m/s\n', max(abs(vx_traj)), max(abs(vy_traj)));

end
