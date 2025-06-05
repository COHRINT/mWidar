clear; clf; close all;
rng(0);

% Parameters
num_time = 20;
num_peaks = 10;
num_dim = 2;
grid_size = 10;

% Explicitly define 3 constant peaks
const_peaks = [2 3; 7 8; 5 1]; % [x y] for each constant peak

% Generate random peaks for the rest
init_peaks = [const_peaks; floor(grid_size * rand(num_peaks-3, num_dim))];

% Initialize time_peaks
time_peaks = zeros(num_time, num_peaks, num_dim);

% TDPF Setup
 
prev_peak_score = [];


TDPF_String = ["New Target", "Existing Target", "Persistent Target"];

prev_peak_score = TDPF(peaks, prev_peak_score, 2);


for i = 1:num_time
    % GENERATE NEW PEAKS
    % First 3 are constant
    time_peaks(i, 1:3, :) = const_peaks;
    % 4:end are random at each timestep
    time_peaks(i, 4:end, :) = floor(grid_size * rand(num_peaks-3, num_dim));

    % TDPF
    current_peaks = time_peaks(i, :, :);

end

figure(1);
set(gcf, 'Color', 'w');
for i = 1:num_time
    clf;
    scatter(time_peaks(i, 1:3, 1), time_peaks(i, 1:3, 2), 120, 'r',  'DisplayName', 'Constant');
    hold on;
    scatter(time_peaks(i, 4:end, 1), time_peaks(i, 4:end, 2), 80, 'b', 'DisplayName', 'New');
    title("Time Dependent Peak Finder", 'FontSize', 14);
    xlabel('X'); ylabel('Y');
    xlim([0 grid_size]);
    ylim([0 grid_size]);
    grid on;
    legend('Location', 'northeast');
    set(gca, 'FontSize', 12);
    pause(0.5);
end


% 