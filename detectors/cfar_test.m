clf; clear all; close all;

% Example usage:
data = rand(100, 100); % Example data
true_locs = [51, 51; 60, 60]; % True locations of peaks
for i = 1:size(true_locs, 1)
    data(true_locs(i, 1), true_locs(i, 2)) = 10; % Add a peak
end
threshold = 0.8; % Increased threshold to reduce false detections
guard_cells = 3; % Adjusted guard cells for better peak isolation
noise_cells = 7; % Increased noise cells for more robust noise estimation
[~, locs_y_CFAR, locs_x_CFAR] = CA_CFAR(data, threshold, guard_cells, noise_cells);
[~, locs_y_MP, locs_x_MP] = max_peaks(data);


% Improved visualization
figure;
[X, Y] = meshgrid(1:size(data, 2), 1:size(data, 1));
surf(X, Y, data, 'EdgeColor', 'none', "FaceAlpha", .5); % 3D surface plot for better depth perception
view(2); % Adjust view angle for better visualization
colormap('hot'); % Use a better colormap for contrast
colorbar; % Display color scale
hold on; 
% plot(locs_x, locs_y, 'b*', 'MarkerSize', 10, 'LineWidth', 1.5); % Larger, blue markers for peaks
plot(true_locs(:, 2), true_locs(:, 1), 'go', 'MarkerSize', 10, 'LineWidth', 1.5); % True locations in green
title('Detected Peaks', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X-axis', 'FontSize', 12); 
ylabel('Y-axis', 'FontSize', 12);
grid on; % Add grid for better readability
hold off;