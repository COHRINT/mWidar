%% ========== SINGLE POINT SIMULATION ==========
% Run MC simulation to generate magnitude image grid likelihoods for HMM filter
%
%
clc; clf; clear; close all;
% Load mWidar simulation matrices
load(fullfile('recovery.mat'))
load(fullfile('sampling.mat'))

%
FULL_GEN = false;

% Put into mWidar params struct
mWidarParams.sampling = M;
mWidarParams.recovery = G;

xgrid = 1:128;
ygrid = 1:128;
[grid_points_x, grid_points_y] = meshgrid(xgrid, ygrid);
grid_points = [grid_points_x(:), grid_points_y(:)];

% MC settings for mWidar sim
Nsamples = 100; % Reduced for full generation to save computation time
grid_size = 5;
half_size = floor(grid_size / 2); % 2 for 5x5 grid
sigma = 1.0; % Standard deviation for Gaussian

simpoint = [31, 31; 30, 30; 100, 100]

for sp = 1:size(simpoint, 1)

    sim_point = simpoint(sp, :); % (x,y) coordinates of point to simulate

    % Create impulse at the specified grid point
    sampleArray = zeros(1, 128, 128);
    sampleArray(1, sim_point(2), sim_point(1)) = 1; % (row, col) = (y, x)

    % Generate mWidar response
    simImage = genmWidarImage(sampleArray, mWidarParams);

    % Plot the result using PDA_PF style
    npx = 128;
    xvec = linspace(-2, 2, npx);
    yvec = linspace(0, 4, npx);

    figure(1); clf;
    imagesc(xvec, yvec, squeeze(simImage(1, :, :)));
    set(gca, 'YDir', 'normal');
    colormap('parula');
    colorbar;
    axis image;
    xlabel('X (m)');
    ylabel('Y (m)');

    if sp == 1
        title("mWidar Image Signal)")
    else
        title(sprintf('mWidar Image for Point Source at (%d, %d)', sim_point(1), sim_point(2)));
    end

    % Save the figure
    saveas(gcf, sprintf('mWidarSample%d_%d.png', sim_point(1), sim_point(2)));
end
