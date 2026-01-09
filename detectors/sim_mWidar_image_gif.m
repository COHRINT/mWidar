%% sim_mWidar_image_gif
% Generate a simple 3-target trajectory GIF using mWidar image simulation
% Shows only the grayscale mWidar signal without any detector outputs

clear; clc; close all

%% Parameters
NUM_OBJECTS = 3; % 3 targets
GIF_FILENAME = fullfile("..", "figures", "Detectors", "3target_trajectory.gif");
DELAY_TIME = 0.1; % Delay between frames in seconds

% Create output directory if it doesn't exist
output_dir = fileparts(GIF_FILENAME);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf("Generating 3-target trajectory GIF with kinematic trajectories...\n")

%% Load mWidar matrices
load(fullfile("..", "matlab_src", "supplemental", "recovery.mat"))
load(fullfile("..", "matlab_src", "supplemental", "sampling.mat"))

%% Define kinematic trajectories
% Time parameters
dt = 0.1; % Time step in seconds
T = 5.0; % Total duration in seconds
tvec = 0:dt:T;
nk = length(tvec);

% Scene bounds (matching mWidar grid)
% X: 1-128 (columns), Y: 1-128 (rows), but valid region is Y >= 21
x_min = 10; x_max = 118;
y_min = 30; y_max = 118;

% Initialize position storage
POS = zeros(NUM_OBJECTS, 2, nk);

% Target 1: Diagonal (bottom-left to top-right)
% Start: (20, 30), End: (110, 110)
x1_start = 20; y1_start = 30;
x1_end = 110; y1_end = 110;
vx1 = (x1_end - x1_start) / T;
vy1 = (y1_end - y1_start) / T;

for k = 1:nk
    t = tvec(k);
    POS(1, 1, k) = x1_start + vx1 * t; % x position
    POS(1, 2, k) = y1_start + vy1 * t; % y position
end

% Target 2: Horizontal (right to left)
% Start: (110, 70), End: (20, 70)
x2_start = 110; y2_start = 70;
x2_end = 20; y2_end = 70;
vx2 = (x2_end - x2_start) / T;
vy2 = 0; % No vertical movement

for k = 1:nk
    t = tvec(k);
    POS(2, 1, k) = x2_start + vx2 * t; % x position
    POS(2, 2, k) = y2_start + vy2 * t; % y position
end

% Target 3: Towards array then away (vertical oscillation)
% Starts at top, moves down (towards array), then back up
% Start: (65, 110), Bottom: (65, 35), End: (65, 110)
x3 = 65; % Constant x position
y3_start = 110;
y3_min = 35;

% Use sinusoidal motion for smooth back-and-forth
for k = 1:nk
    t = tvec(k);
    % Oscillate from y_start down to y_min and back
    y3_amplitude = (y3_start - y3_min) / 2;
    y3_center = (y3_start + y3_min) / 2;
    POS(3, 1, k) = x3; % x position (constant)
    POS(3, 2, k) = y3_center - y3_amplitude * cos(2 * pi * t / T); % y position
end

fprintf("Generated %d kinematic trajectories over %d timesteps (%.1f seconds)\n", NUM_OBJECTS, nk, T)
fprintf("  Target 1: Diagonal (bottom-left to top-right)\n")
fprintf("  Target 2: Horizontal (right to left)\n")
fprintf("  Target 3: Vertical oscillation (towards/away from array)\n")

%% Create meshgrid for plotting
xgrid = 1:128;
ygrid = 1:128;
[X, Y] = meshgrid(xgrid, ygrid);

%% Generate mWidar signal for each timestep
Signal = zeros(128, 128, 2, nk);

fprintf("\nGenerating mWidar signals...\n")
for k = 1:nk
    if mod(k, 10) == 0
        fprintf("  Processing timestep %d/%d...\n", k, nk)
    end
    
    % Create object binary scene
    S = zeros(128, 128);
    
    for obj = 1:NUM_OBJECTS
        px = round(POS(obj, 1, k)); % x position (column)
        py = round(POS(obj, 2, k)); % y position (row)
        
        % Place object in scene if within bounds
        if px >= 1 && px <= 128 && py >= 1 && py <= 128
            S(py, px) = 1; % Note: S(row, col) = S(y, x)
        end
    end
    
    % Generate mWidar signal
    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';
    signal_original = imgaussfilt(sim_signal, 1.3);
    
    Signal(:, :, 1, k) = signal_original; % Store unscaled signal
    
    % Normalize signal [0 1] and apply nonlinearity
    signal_scaled = signal_original;
    signal_scaled(1:20, :) = NaN; % Focus on valid region
    scaled_signal = asinh(signal_scaled);
    scaled_signal = (signal_scaled - min(signal_scaled(:))) / (max(signal_scaled(:)) - min(signal_scaled(:)));
    
    Signal(:, :, 2, k) = scaled_signal; % Store scaled signal
end

fprintf("Signal generation complete!\n\n")

%% Create figure
fig = figure('Position', [100, 100, 800, 800], 'Color', 'w');

%% Generate GIF
fprintf("Creating GIF animation...\n")
for k = 1:nk
    if mod(k, 10) == 0
        fprintf("  Frame %d/%d...\n", k, nk)
    end
    
    % Clear figure for next frame
    clf;
    
    % Plot mWidar signal with viridis colormap
    surf(X, Y, Signal(:, :, 2, k), 'EdgeColor', 'none');
    shading interp
    colormap('parula')
    hold on
    
    % Plot true target locations as magenta hollow squares
    for obj = 1:NUM_OBJECTS
        px = POS(obj, 1, k); % x position
        py = POS(obj, 2, k); % y position
        
        % Plot hollow square at target location
        plot3(px, py, 1, 's', 'MarkerEdgeColor', 'm', 'LineWidth', 2, 'MarkerSize', 15);
    end
    
    hold off
    
    % Set view and limits
    view(2)
    xlim([1 128])
    ylim([1 128])
    axis square
    
    % Add title with frame number and time
    title(sprintf('3-Target Trajectory - Frame %d/%d (t=%.1fs)', k, nk, tvec(k)), ...
        'FontSize', 14, 'FontWeight', 'bold')
    
    % Remove tick labels for cleaner look
    set(gca, 'XTick', [], 'YTick', [])
    
    % Capture frame and save to GIF
    drawnow
    frame = getframe(fig);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    if k == 1
        % Write first frame and create GIF file
        imwrite(imind, cm, GIF_FILENAME, 'gif', ...
            'Loopcount', inf, 'DelayTime', DELAY_TIME);
    else
        % Append subsequent frames
        imwrite(imind, cm, GIF_FILENAME, 'gif', ...
            'WriteMode', 'append', 'DelayTime', DELAY_TIME);
    end
end

fprintf("\n✓ GIF saved to: %s\n", GIF_FILENAME)
fprintf("  Total frames: %d\n", nk)
fprintf("  Animation duration: %.1f seconds\n", nk * DELAY_TIME)
fprintf("  Trajectory duration: %.1f seconds\n", T)

close(fig)
