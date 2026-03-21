%% generate_tuning_dataset.m
%
% Generates a T4-style (full mWidar simulation + CA-CFAR detector) dataset
% from an S-curve trajectory and saves it to:
%   matlab_src/data/TUNING_DATASET1/data.mat
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

%% ---- Paths ---------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'supplemental'));
addpath(fullfile(script_dir, 'supplemental', 'Final_Test_Tracks'));

out_dir  = fullfile(script_dir, 'data', 'TUNING_DATASET1');
out_file = fullfile(out_dir, 'data.mat');

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
    fprintf('Created output directory: %s\n', out_dir);
end

%% ---- Load mWidar forward-model matrices ---------------------------------
fprintf('Loading mWidar simulation matrices...\n');
load(fullfile(script_dir, 'supplemental', 'sampling.mat'),  'M');
load(fullfile(script_dir, 'supplemental', 'recovery.mat'),  'G');
fprintf('  M: %dx%d,  G: %dx%d\n', size(M,1), size(M,2), size(G,1), size(G,2));

%% ---- Scene / grid setup -------------------------------------------------
npx    = 128;
Lscene = 4;           % scene height [m]
xgrid  = linspace(-2, 2,      npx);  % [-2, 2] m
ygrid  = linspace( 0, Lscene, npx);  % [0,  4] m
[pxgrid, pygrid] = meshgrid(xgrid, ygrid);

%% ---- Trajectory: analytical S-curve ------------------------------------
dt        = 0.1;       % [s]
num_steps = 50;
tvec      = (0:num_steps) * dt;
n_t       = length(tvec);  % 51 timesteps

x_start = -1.5;  x_end = 1.5;
y_start =  0.5;  y_end = 3.5;
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

%% ---- CA-CFAR parameters ------------------------------------------------
Pfa = 0.365;   % false alarm probability (tuned for this scene)
Ng  = 5;       % guard cells
Nr  = 20;      % training (reference) cells

%% ---- Generate measurements via mWidar + CA-CFAR ------------------------
fprintf('Running mWidar forward model + CA-CFAR detector...\n');

rng(400);   % reproducible seed (matches T4 in generate_tuning_tracks.m)

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
        y_4{i} = true_pos + 0.1*randn(2,1);
    end

    if mod(i, 10) == 0
        n_det = size(y_4{i}, 2);
        fprintf('  Step %2d/%d: %d detections (truth=[%.3f, %.3f])\n', ...
            i, n_t, n_det, true_pos(1), true_pos(2));
    end
end

%% ---- Save ---------------------------------------------------------------
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

save(out_file, 'Data', '-mat');
fprintf('\nSaved dataset to: %s\n', out_file);

% Quick summary
n_det_all = cellfun(@(c) size(c,2), Data.y);
fprintf('Detection counts: min=%d, max=%d, mean=%.1f per timestep\n', ...
    min(n_det_all), max(n_det_all), mean(n_det_all));
fprintf('Done.\n');
