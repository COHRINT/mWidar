% TDPF_TESTSCRIPT - Test script for Time Dependent Peak Finder (TDPF)
%
% This script tests the TDPF algorithm before further implementation.
% It generates synthetic peak data with three types of targets:
%   1. Stationary persistent target
%   2. Random-walk persistent target
%   3. Diagonal-moving persistent target
%   4-N. Random clutter peaks
%
% The script visualizes the TDPF classification results across multiple
% timesteps using a 2x5 tiledlayout, showing how the algorithm categorizes
% peaks as "New Target", "Existing Target", or "Persistent Target".
%
% Author: Anthony La Barca

%% Initialization
clear; clf; close all;
rng(0);

SAVEFIG = true;

%% Parameters
num_time = 10; % Number of timesteps
num_peaks = 30; % Total number of peaks per timestep
num_dim = 2; % Dimensionality (x, y)
grid_size = 128; % Size of detection grid

%% Generate Initial Peak Positions
% Randomly generate 3 constant peaks away from edges
const_peaks = randi([2, grid_size - 2], 3, 2);

% Initialize time_peaks array: [time x peaks x dimensions]
time_peaks = zeros(num_time, num_peaks, num_dim);

%% Setup TDPF
TDPF_Dict = {};
TDPF_String = ["New", "Existing", "Confirmed", "Persistent"];

% Initialize moving peaks
moving_peak_1 = const_peaks(2, :); % Random walk target
moving_peak_2 = const_peaks(3, :); % Diagonal moving target

%% Generate Peak Data and Run TDPF
for i = 1:num_time
    % Peak 1: Stationary
    time_peaks(i, 1, :) = const_peaks(1, :);

    % Peak 2: Random walk
    time_peaks(i, 2, :) = moving_peak_1;
    step = randi([-2, 2], 1, 2);
    moving_peak_1 = moving_peak_1 + step;
    moving_peak_1 = max(min(moving_peak_1, grid_size), 0);
    moving_peak_1 = round(moving_peak_1);

    % Peak 3: Diagonal movement (alternating right and up)
    time_peaks(i, 3, :) = moving_peak_2;

    if mod(i, 2) == 1
        moving_peak_2(1) = moving_peak_2(1) + 1;
    else
        moving_peak_2(2) = moving_peak_2(2) + 1;
    end

    % Peaks 4-N: Random clutter
    time_peaks(i, 4:end, :) = floor(grid_size * rand(num_peaks - 3, num_dim));

    % Run TDPF algorithm
    current_peaks = time_peaks(i, :, :);

    if i == 1
        first_score = [squeeze(current_peaks), ones(num_peaks, 1)];
        TDPF_Dict{1} = first_score;
    else
        TDPF_Dict{i} = TDPF( ...
            squeeze(current_peaks), ...
            TDPF_Dict{i - 1}, ...
            5 ... % Distance threshold
        );
    end

end

%% Setup Plotting Parameters
% Ground truth colors and sizes
detection_colors = [1 0 0; 1 0 0; 1 0 0]; % Red for true targets
detection_sizes = [120, 120, 120];

for i = 4:num_peaks
    detection_colors = [detection_colors; 0 0 0]; % Black for clutter
    detection_sizes = [detection_sizes, 80];
end

% TDPF classification colors
colors_map = [
              1 0.5 0; % New Target: orange
              1 0 1; % Existing Target: magenta
              0 0 1; % Confirmed Target: blue
              0 1 0 % Persistent Target: green
              ];

%% Create Visualization
figure(1);
set(gcf, 'Color', 'w', 'Position', [100 100 1600 700]);
tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:num_time
    nexttile;

    % Plot ground truth peaks
    scatter(time_peaks(i, :, 1), time_peaks(i, :, 2), detection_sizes, detection_colors, 'LineWidth', 2);
    hold on;

    % Overlay TDPF classifications
    curr_TDPF = TDPF_Dict{i};

    for j = 1:size(curr_TDPF, 1)
        score = curr_TDPF(j, 3);
        scatter(curr_TDPF(j, 1), curr_TDPF(j, 2), 100, colors_map(score, :), 'LineWidth', 2, "Marker", '+');
    end

    % "Reported Peaks" are those with score > 2 (i.e., Existing or Persistent Targets)
    num_reported_peaks = size(curr_TDPF(curr_TDPF(:, 3) > 2, :), 1);
    title_string = sprintf('Timestep %d - Reported Peaks: %d', i, num_reported_peaks);
    title(title_string, 'FontSize', 12);
    xlabel('X'); ylabel('Y');
    xlim([0 grid_size]);
    ylim([0 grid_size]);
    grid on;
    axis square;
    set(gca, 'FontSize', 10);
end

sgtitle('Time Dependent Peak Finder', 'FontSize', 14, 'FontWeight', 'bold');

% Add legend
h1 = scatter(NaN, NaN, 120, 'r', 'LineWidth', 2, 'DisplayName', 'True Targets');
hold on;
h2 = scatter(NaN, NaN, 80, 'k', 'LineWidth', 2, 'DisplayName', 'Clutter');
h3 = scatter(NaN, NaN, 100, [1 0.5 0], 'LineWidth', 2, "Marker", '+', 'DisplayName', "New Target");
h4 = scatter(NaN, NaN, 100, [1 0 1], 'LineWidth', 2, "Marker", '+', 'DisplayName', "Existing Target");
h5 = scatter(NaN, NaN, 100, [0 0 1], 'LineWidth', 2, "Marker", '+', 'DisplayName', "Confirmed Target");
h6 = scatter(NaN, NaN, 100, [0 1 0], 'LineWidth', 2, "Marker", '+', 'DisplayName', "Persistent Target");
lgd = legend([h1 h2 h3 h4 h5 h6], 'Orientation', 'horizontal', 'FontSize', 10, 'Position', [0.2 0.02 0.6 0.03]);

%% Save Figure
if SAVEFIG
    time = datetime('now');
    figure_path = fullfile("..", "figures", "Detectors", "TDPF");

    if ~exist(figure_path, 'dir')
        mkdir(figure_path);
    end

    filename = sprintf('TDPF_TestScript_Figure_%s.png', datestr(time, 'yyyymmdd_HHMMSS'));
    saveas(gcf, fullfile(figure_path, filename));
end
