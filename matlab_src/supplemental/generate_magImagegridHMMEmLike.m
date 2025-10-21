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

% MC settings for mWidar sim
Nsamples = 100; % Reduced for full generation to save computation time
grid_size = 5;
half_size = floor(grid_size / 2); % 2 for 5x5 grid
sigma = 1.0; % Standard deviation for Gaussian

if ~FULL_GEN
    %% ========== SINGLE POINT SIMULATION ==========
    sim_point = [30, 30]; % Pixel location to test (center point where we measure brightness)

    % Create 5x5 grid centered at sim_point
    [X_offset, Y_offset] = meshgrid(-half_size:half_size, -half_size:half_size);
    grid_points = [sim_point(1) + X_offset(:), sim_point(2) + Y_offset(:)]; % 25x2 array

    % Create Gaussian weights for the 5x5 grid (centered at middle of grid)
    gaussian_weights = exp(- (X_offset .^ 2 + Y_offset .^ 2) / (2 * sigma ^ 2));
    gaussian_weights = gaussian_weights / sum(gaussian_weights(:)); % Normalize to sum to 1
    gaussian_weights = gaussian_weights(:); % Flatten to vector

    fprintf('5x5 Grid points around (%d,%d):\n', sim_point(1), sim_point(2));

    for i = 1:length(grid_points)
        fprintf('Point %d: (%d,%d) with weight %.4f\n', i, grid_points(i, 1), grid_points(i, 2), gaussian_weights(i));
    end

    % Pre-compute mWidar responses for all 25 grid points
    fprintf('\nPre-computing mWidar responses for all 25 grid points...\n');
    true_responses = zeros(25, 1);

    for i = 1:25
        % Create impulse at grid point i
        sampleArray = zeros(1, 128, 128);
        sampleArray(1, grid_points(i, 2), grid_points(i, 1)) = 1; % Note: MATLAB uses (row, col) = (y, x)

        % Generate mWidar response
        simImage = genmWidarImage(sampleArray, mWidarParams);

        % Record the brightness at our measurement point (sim_point)
        true_responses(i) = simImage(1, sim_point(2), sim_point(1)); % brightness at sim_point due to impulse at grid_points(i)

        fprintf('Grid point %d at (%d,%d): brightness at (%d,%d) = %.6f\n', ...
            i, grid_points(i, 1), grid_points(i, 2), sim_point(1), sim_point(2), true_responses(i));
    end

    % Run Monte Carlo simulation
    fprintf('\nRunning Monte Carlo simulation with %d samples...\n', Nsamples);
    pixel_counts = zeros(25, 1); % Count frequency of each pixel being selected

    for i = 1:Nsamples
        % Categorically sample from the 25 grid points using Gaussian weights
        sampled_idx = randsample(1:25, 1, true, gaussian_weights);
        pixel_counts(sampled_idx) = pixel_counts(sampled_idx) + 1;

        if mod(i, 100) == 0
            fprintf('MC run %d / %d complete\r', i, Nsamples);
        end

    end

    % Compute weighted average and variance using the sampling frequencies
    sample_frequencies = pixel_counts / Nsamples;
    weighted_center_value = sum(sample_frequencies .* true_responses);

    % Calculate variance: Var(X) = E[X^2] - (E[X])^2
    weighted_center_variance = sum(sample_frequencies .* (true_responses .^ 2)) - weighted_center_value ^ 2;
    weighted_center_std = sqrt(weighted_center_variance);

    fprintf('\nMonte Carlo Results:\n');
    fprintf('Pixel selection frequencies:\n');

    for i = 1:25
        fprintf('Point (%d,%d): selected %d times (%.3f%%), true response = %.6f\n', ...
            grid_points(i, 1), grid_points(i, 2), pixel_counts(i), 100 * sample_frequencies(i), true_responses(i));
    end

    fprintf('\nWeighted average center value: %.6f\n', weighted_center_value);
    fprintf('Weighted standard deviation: %.6f\n', weighted_center_std);
    fprintf('Direct center response: %.6f\n', true_responses(13)); % Middle point (should be index 13 for 5x5 grid)

    % Visualization
    figure('Position', [200, 200, 900, 600]);
    tiledlayout(2, 2);

    % Plot 1: Histogram of pixel selection frequencies
    nexttile;
    bar(1:25, pixel_counts);
    title('Pixel Selection Frequency over MC runs');
    xlabel('Grid Point Index');
    ylabel('Selection Count');
    grid on;

    % Plot 2: True responses for each grid point
    nexttile;
    bar(1:25, true_responses);
    title('True mWidar Response at Center for Each Grid Point');
    xlabel('Grid Point Index');
    ylabel('Brightness Value');
    grid on;

    % Plot 3: 5x5 Gaussian weight distribution
    nexttile;
    imagesc(reshape(gaussian_weights, 5, 5));
    colorbar;
    title('Gaussian Weight Distribution (5x5)');
    xlabel('X Offset');
    ylabel('Y Offset');
    axis equal tight;

    % Plot 4: 5x5 true response distribution
    nexttile;
    imagesc(reshape(true_responses, 5, 5));
    colorbar;
    title('True Response Distribution (5x5)');
    xlabel('X Offset');
    ylabel('Y Offset');
    axis equal tight;

    sgtitle(sprintf('Monte Carlo Analysis: Weighted Avg = %.2f$\\pm$%.2f, Direct Response = %.6f', ...
        weighted_center_value, weighted_center_std, true_responses(13)), 'Interpreter', 'latex');

    % Display comparison
    fprintf('\n=== COMPARISON ===\n');
    fprintf('Weighted average method: %.6f\n', weighted_center_value);
    fprintf('Direct center response: %.6f\n', true_responses(13));
    fprintf('Difference: %.6f (%.3f%%)\n', abs(weighted_center_value - true_responses(13)), ...
        100 * abs(weighted_center_value - true_responses(13)) / true_responses(13));

    % Save the figure
    saveas(gcf, sprintf('pointlikelihood_mc_singlepoint_%d_%d.png', sim_point(1), sim_point(2)));

else
    %% ========== FULL GENERATION MODE ==========
    fprintf('Running FULL GENERATION mode for all 128x128 pixels...\n');

    % Initialize output arrays
    likelihood_means = zeros(128, 128);
    likelihood_stds = zeros(128, 128);

    % Pre-compute Gaussian weights (same for all points)
    [X_offset, Y_offset] = meshgrid(-half_size:half_size, -half_size:half_size);
    gaussian_weights = exp(- (X_offset .^ 2 + Y_offset .^ 2) / (2 * sigma ^ 2));
    gaussian_weights = gaussian_weights / sum(gaussian_weights(:)); % Normalize to sum to 1
    gaussian_weights = gaussian_weights(:); % Flatten to vector

    %% ========== OPTIMIZATION: PRE-COMPUTE LOOKUP TABLE ==========
    fprintf('Pre-computing mWidar response lookup table...\n');

    % Find all unique impulse locations we'll need (all points within 5x5 of any measurement point)
    unique_impulse_locations = [];

    for x = 1:128

        for y = 1:128
            % Get 5x5 grid around this measurement point
            grid_points = [x + X_offset(:), y + Y_offset(:)];

            % Keep only valid grid points within bounds
            valid_indices = (grid_points(:, 1) >= 1) & (grid_points(:, 1) <= 128) & ...
                (grid_points(:, 2) >= 1) & (grid_points(:, 2) <= 128);
            valid_grid_points = grid_points(valid_indices, :);

            % Add to unique list
            unique_impulse_locations = [unique_impulse_locations; valid_grid_points];
        end

    end

    % Remove duplicates
    unique_impulse_locations = unique(unique_impulse_locations, 'rows');
    n_unique = size(unique_impulse_locations, 1);

    fprintf('Found %d unique impulse locations to pre-compute (out of %d total needed)\n', n_unique, 25 * 128 * 128);
    fprintf('Optimization factor: %.1fx speedup expected\n', 25 * 128 * 128 / n_unique);

    % Pre-compute all mWidar responses and store in lookup table
    fprintf('Computing mWidar responses for lookup table (PARALLELIZED)...\n');

    % Check if parallel pool exists, if not create one
    if isempty(gcp('nocreate'))
        fprintf('Starting parallel pool with %d workers...\n', feature('numcores'));
        parpool('local', feature('numcores')); % Use all available cores
    else
        fprintf('Using existing parallel pool with %d workers...\n', gcp().NumWorkers);
    end

    mwidar_lookup = containers.Map('KeyType', 'char', 'ValueType', 'any');

    % Pre-allocate cell arrays for parallel computation
    keys_cell = cell(n_unique, 1);
    responses_cell = cell(n_unique, 1);

    tic;

    % Parallel computation of mWidar responses
    parfor i = 1:n_unique
        impulse_loc = unique_impulse_locations(i, :);

        % Create impulse at this location
        sampleArray = zeros(1, 128, 128);
        sampleArray(1, impulse_loc(2), impulse_loc(1)) = 1; % Note: MATLAB uses (row, col) = (y, x)

        % Generate mWidar response
        simImage = genmWidarImage(sampleArray, mWidarParams);

        % Store key and response in cell arrays (can't modify containers.Map directly in parfor)
        keys_cell{i} = sprintf('%d_%d', impulse_loc(1), impulse_loc(2));
        responses_cell{i} = squeeze(simImage(1, :, :)); % Store the 128x128 response
    end

    % Populate the lookup table after parallel computation
    fprintf('Populating lookup table...\n');

    for i = 1:n_unique
        mwidar_lookup(keys_cell{i}) = responses_cell{i};

        if mod(i, 1000) == 0 || i == n_unique
            elapsed = toc;
            fprintf('Populated %d/%d responses (%.1f%%) - Elapsed: %.1fs\n', ...
                i, n_unique, 100 * i / n_unique, elapsed);
        end

    end

    fprintf('Lookup table pre-computation complete! Starting main computation...\n');
    fprintf('Using CPU-only acceleration (M1 GPU not supported by MATLAB)\n');

    % Create real-time visualization
    [xgrid, ygrid] = meshgrid(linspace(-2, 2, 128), linspace(0, 4, 128));

    fig = figure('Position', [100, 100, 1400, 600]);
    tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

    % Plot 1: Mean likelihood (real-time)
    ax1 = nexttile;
    h_mean = imagesc(ax1, xgrid(1, :), ygrid(:, 1), likelihood_means);
    set(ax1, 'YDir', 'normal');
    cb1 = colorbar(ax1);
    title(ax1, 'Mean Likelihood Values (Real-time)');
    xlabel(ax1, 'X Position (m)');
    ylabel(ax1, 'Y Position (m)');
    axis(ax1, 'equal', 'tight');
    clim(ax1, [0, 1]); % Initial color limits, will auto-adjust

    % Plot 2: Standard deviation (real-time)
    ax2 = nexttile;
    h_std = imagesc(ax2, xgrid(1, :), ygrid(:, 1), likelihood_stds);
    set(ax2, 'YDir', 'normal');
    cb2 = colorbar(ax2);
    title(ax2, 'Standard Deviation (Real-time)');
    xlabel(ax2, 'X Position (m)');
    ylabel(ax2, 'Y Position (m)');
    axis(ax2, 'equal', 'tight');
    clim(ax2, [0, 0.1]); % Initial color limits, will auto-adjust

    % Plot 3: Progress visualization
    ax3 = nexttile;
    progress_map = zeros(128, 128);
    h_progress = imagesc(ax3, xgrid(1, :), ygrid(:, 1), progress_map);
    set(ax3, 'YDir', 'normal');
    cb3 = colorbar(ax3);
    title(ax3, 'Computation Progress');
    xlabel(ax3, 'X Position (m)');
    ylabel(ax3, 'Y Position (m)');
    axis(ax3, 'equal', 'tight');
    clim(ax3, [0, 1]);
    colormap(ax3, 'hot');

    sgtitle('Monte Carlo Likelihood Statistics - Real-time Progress (Optimized + Parallel)');

    % Batch processing parameters for parallelization
    batch_size = 100; % Process this many measurement points per batch
    update_frequency = 10; % Update display every N batches processed

    % Convert 2D grid to linear indices for easier batching
    [X_coords, Y_coords] = meshgrid(1:128, 1:128);
    all_coords = [X_coords(:), Y_coords(:)]; % All 16384 measurement points
    total_pixels = size(all_coords, 1);
    num_batches = ceil(total_pixels / batch_size);

    fprintf('Processing %d measurement points in %d batches of size %d\n', ...
        total_pixels, num_batches, batch_size);

    tic; % Start timing

    % Process measurement points in parallel batches
    for batch_idx = 1:num_batches
        % Determine batch range
        start_idx = (batch_idx - 1) * batch_size + 1;
        end_idx = min(batch_idx * batch_size, total_pixels);
        batch_coords = all_coords(start_idx:end_idx, :);
        current_batch_size = size(batch_coords, 1);

        % Pre-allocate batch results
        batch_means = zeros(current_batch_size, 1);
        batch_stds = zeros(current_batch_size, 1);
        batch_x_coords = batch_coords(:, 1);
        batch_y_coords = batch_coords(:, 2);

        % Parallel processing of measurement points in this batch
        parfor point_idx = 1:current_batch_size
            x = batch_x_coords(point_idx);
            y = batch_y_coords(point_idx);
            sim_point = [x, y]; % Current measurement point

            % Create 5x5 grid centered at sim_point
            grid_points = [sim_point(1) + X_offset(:), sim_point(2) + Y_offset(:)]; % 25x2 array

            % Check bounds - only keep grid points within [1,128] x [1,128]
            valid_indices = (grid_points(:, 1) >= 1) & (grid_points(:, 1) <= 128) & ...
                (grid_points(:, 2) >= 1) & (grid_points(:, 2) <= 128);

            if sum(valid_indices) == 0
                % No valid grid points, skip this measurement point
                batch_means(point_idx) = 0;
                batch_stds(point_idx) = 0;
                continue;
            end

            % Keep only valid grid points and their weights
            valid_grid_points = grid_points(valid_indices, :);
            valid_weights = gaussian_weights(valid_indices);
            valid_weights = valid_weights / sum(valid_weights); % Renormalize

            % Get mWidar responses using LOOKUP TABLE (OPTIMIZED!)
            true_responses = zeros(length(valid_grid_points), 1);

            for i = 1:length(valid_grid_points)
                % Get response from lookup table instead of computing
                key = sprintf('%d_%d', valid_grid_points(i, 1), valid_grid_points(i, 2));
                full_response = mwidar_lookup(key);

                % Extract brightness at our measurement point (sim_point)
                true_responses(i) = full_response(sim_point(2), sim_point(1)); % brightness at sim_point
            end

            % Run Monte Carlo simulation for this measurement point (CPU-based in parfor)
            pixel_counts = zeros(length(valid_grid_points), 1);

            for mc = 1:Nsamples
                % Categorically sample from valid grid points using valid weights
                sampled_idx = randsample(1:length(valid_grid_points), 1, true, valid_weights);
                pixel_counts(sampled_idx) = pixel_counts(sampled_idx) + 1;
            end

            % Compute weighted statistics
            sample_frequencies = pixel_counts / Nsamples;
            weighted_mean = sum(sample_frequencies .* true_responses);
            weighted_variance = sum(sample_frequencies .* (true_responses .^ 2)) - weighted_mean ^ 2;
            weighted_std = sqrt(weighted_variance);

            % Store batch results
            batch_means(point_idx) = weighted_mean;
            batch_stds(point_idx) = weighted_std;
        end

        % Update global results with batch results
        for point_idx = 1:current_batch_size
            x = batch_coords(point_idx, 1);
            y = batch_coords(point_idx, 2);
            likelihood_means(y, x) = batch_means(point_idx);
            likelihood_stds(y, x) = batch_stds(point_idx);
            progress_map(y, x) = 1; % Mark as completed
        end

        % Real-time visualization update
        if mod(batch_idx, update_frequency) == 0 || batch_idx == num_batches
            % Update mean plot
            set(h_mean, 'CData', likelihood_means);

            % Update std plot
            set(h_std, 'CData', likelihood_stds);

            % Update progress plot
            set(h_progress, 'CData', progress_map);

            % Auto-adjust color limits based on current data
            if max(likelihood_means(:)) > 0
                clim(ax1, [0, max(likelihood_means(:))]);
            end

            if max(likelihood_stds(:)) > 0
                clim(ax2, [0, max(likelihood_stds(:))]);
            end

            % Update title with progress info
            elapsed_time = toc;
            pixels_processed = end_idx;
            est_total_time = elapsed_time * total_pixels / pixels_processed;
            remaining_time = est_total_time - elapsed_time;

            sgtitle(sprintf('MC Stats (Parallel) - Batch: %d/%d | Progress: %d/%d (%.1f%%) | Elapsed: %.1fs | ETA: %.1fs', ...
                batch_idx, num_batches, pixels_processed, total_pixels, 100 * pixels_processed / total_pixels, elapsed_time, remaining_time));

            % Force display update
            drawnow;

            % Console progress
            fprintf('Batch %d/%d complete - Processed %d/%d pixels (%.1f%%) - Elapsed: %.1fs - ETA: %.1fs\n', ...
                batch_idx, num_batches, pixels_processed, total_pixels, 100 * pixels_processed / total_pixels, elapsed_time, remaining_time);
        end

    end

    fprintf('Full generation complete!\n');

    % Final update with better color limits
    clim(ax1, [0, max(likelihood_means(:))]);
    clim(ax2, [0, max(likelihood_stds(:))]);
    sgtitle('Monte Carlo Likelihood Statistics - COMPLETE (Parallel + Optimized)');
    drawnow;

    % Save results in correct format: 128^2 x 2 (npx^2 x 2)
    % Convert 128x128 arrays to 16384x2 format for HMM compatibility
    pointlikelihood_image = [likelihood_means(:), likelihood_stds(:)];

    % Save in sparse format to match original HMM code expectations
    pointlikelihood_image = sparse(pointlikelihood_image);

    save('precalc_imagegridHMMEmLike.mat', 'pointlikelihood_image', '-mat');
    fprintf('Results saved to precalc_imagegridHMMEmLike.mat in correct npx^2 x 2 format\n');

    % Save the figure
    saveas(gcf, 'pointlikelihood_mc_realtime.png');
    fprintf('Real-time plot saved to pointlikelihood_mc_realtime.png\n');

end
