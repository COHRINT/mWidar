function visualize_RBPFHMM_history(rbpf, varargin)
    % VISUALIZE_RBPFHMM_HISTORY Standalone visualization of HMM-RBPF filter history
    %
    % USAGE:
    %   visualize_RBPFHMM_history(rbpf)
    %   visualize_RBPFHMM_history(rbpf, 'Animate', true)
    %   visualize_RBPFHMM_history(rbpf, 'SaveGIF', 'output.gif')
    %   visualize_RBPFHMM_history(rbpf, 'SaveFinalFigure', 'final.png')
    %
    % INPUTS:
    %   rbpf - HMM_RBPF object with populated history field
    %
    % OPTIONAL PARAMETERS:
    %   'Animate'            - Boolean: Show timestep-by-timestep animation (default: false)
    %   'AnimationSpeed'     - Pause duration between frames in seconds (default: 0.2)
    %   'SaveGIF'            - String: Filename to save animation as GIF (default: '')
    %   'SaveFinalFigure'    - String: Filename to save final timestep figure (default: '')
    %   'PlotMargin'         - Margin percentage for plot bounds (default: 0.2 = 20%)
    %   'SignalData'         - Cell array of mWidar signals [128x128] per timestep (default: {})
    %   'PlotTrajectories'   - Boolean: Plot all particle trajectory histories (default: false)
    %   'MaxParticlesToPlot' - Maximum number of particle trajectories to plot (default: inf)
    %   'SaveIndividualGIFs' - Boolean: Save each subplot as separate GIF (default: false)
    %   'GIFDirectory'       - String: Directory for individual GIFs (default: '')
    %
    % FIGURE LAYOUT (2x3 with signal data, 1x3 without):
    %   TOP ROW:    Position (weighted) | HMM Posterior Heatmap | HMM Entropy
    %   BOTTOM ROW: mWidar Signal       | Association histogram | Trajectory Tree

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'Animate', false, @islogical);
    addParameter(p, 'AnimationSpeed', 0.2, @(x) x > 0);
    addParameter(p, 'SaveGIF', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'SaveFinalFigure', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'PlotMargin', 0.2, @(x) x >= 0);
    addParameter(p, 'SignalData', {}, @iscell);
    addParameter(p, 'PlotTrajectories', false, @islogical);
    addParameter(p, 'MaxParticlesToPlot', inf, @isnumeric);
    addParameter(p, 'SaveIndividualGIFs', false, @islogical);
    addParameter(p, 'GIFDirectory', '', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});

    animate = p.Results.Animate;
    anim_speed = p.Results.AnimationSpeed;
    gif_filename = char(p.Results.SaveGIF);
    final_fig_filename = char(p.Results.SaveFinalFigure);
    margin = p.Results.PlotMargin;
    signal_data = p.Results.SignalData;
    plot_trajectories = p.Results.PlotTrajectories;
    max_particles_to_plot = p.Results.MaxParticlesToPlot;
    save_individual_gifs = p.Results.SaveIndividualGIFs;
    gif_directory = char(p.Results.GIFDirectory);

    % Check that history exists
    if isempty(rbpf.history)
        error('HMM_RBPF history is empty. Run filter first before visualizing.');
    end

    n_steps = length(rbpf.history);
    fprintf('Visualizing %d timesteps of HMM-RBPF...\n', n_steps);

    % Compute static plot bounds
    bounds = computePlotBounds(rbpf, margin);

    % Animation or just final frame?
    if animate
        frames_to_plot = 1:n_steps;
    else
        frames_to_plot = n_steps;
    end

    % Determine if we should generate individual GIFs
    if save_individual_gifs
        generateIndividualGIFs(rbpf, frames_to_plot, bounds, signal_data, ...
                              plot_trajectories, max_particles_to_plot, ...
                              gif_directory, anim_speed, animate);
    else
        % Combined figure approach
        if ~isempty(signal_data)
            fig = figure('Name', 'HMM-RBPF Tracking Results', 'Position', [100, 100, 1600, 800]);
        else
            fig = figure('Name', 'HMM-RBPF Tracking Results', 'Position', [100, 100, 1600, 400]);
        end

        % Setup for GIF export
        gif_first_frame = true;

        for k = frames_to_plot
            plotTimestep(rbpf, k, bounds, fig, signal_data, plot_trajectories, max_particles_to_plot);

            % Save GIF frame if requested
            if ~isempty(gif_filename)
                frame = getframe(fig);
                im = frame2im(frame);
                [imind, cm] = rgb2ind(im, 256);

                if gif_first_frame
                    imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', anim_speed);
                    gif_first_frame = false;
                else
                    imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', anim_speed);
                end
            end

            % Pause for animation
            if animate && k < n_steps
                pause(anim_speed);
            end
        end

        % Save final figure if requested
        if ~isempty(final_fig_filename)
            saveas(fig, final_fig_filename);
            fprintf('Saved final figure to: %s\n', final_fig_filename);
        end

        if ~isempty(gif_filename)
            fprintf('Saved animation GIF to: %s\n', gif_filename);
        end
    end

end

%% =====================================================================
%%  PLOT BOUNDS
%% =====================================================================

function bounds = computePlotBounds(~, ~)
    % Compute static plot bounds for mWidar simulations
    % (HMM is 2-state position only — no velocity/acceleration bounds needed)

    bounds.x = [-2, 2];
    bounds.y = [0.5, 4];   % Crop out detector at y=0
end

%% =====================================================================
%%  MAIN TIMESTEP PLOTTER
%% =====================================================================

function plotTimestep(rbpf, k, bounds, fig, signal_data, plot_trajectories, max_particles_to_plot)
    % Plot a single timestep of HMM-RBPF
    %
    % LAYOUT:
    %   2x3 (with signal): Position | HMM Posterior | Entropy
    %                       Signal  | Association   | Trajectory
    %   1x3 (no signal):   Position | HMM Posterior | Entropy

    figure(fig);

    % Extract data for this timestep
    hist = rbpf.history(k);
    particle_states = hist.particle_states;       % [2 x N_p]
    particle_assocs = hist.particle_associations;  % [1 x N_p]
    particle_weights = hist.particle_weights;      % [1 x N_p]
    measurements = hist.measurements;
    true_state = hist.true_state;
    true_meas_flag = hist.true_meas_flag;
    x_est = hist.estimate;      % [2 x 1]
    P_est = hist.covariance;    % [2 x 2]
    ESS = hist.ESS;

    % Determine layout
    has_signal = ~isempty(signal_data) && k <= length(signal_data) && ~isempty(signal_data{k});

    %% ========== TOP ROW: Position | HMM Posterior | Entropy ==========

    %% SUBPLOT 1: Position (weighted particle cloud)
    if has_signal
        subplot(2, 3, 1);
    else
        subplot(1, 3, 1);
    end

    cla; hold on;

    scatter(particle_states(1, :), particle_states(2, :), 20, particle_weights, ...
        'filled', 'MarkerFaceAlpha', 0.8);
    colormap(gca, hot(256));
    colorbar;

    % Plot mean estimate with covariance ellipse
    plot(x_est(1), x_est(2), 'go', 'MarkerSize', 12, 'LineWidth', 3);
    pos_cov = P_est(1:2, 1:2);
    ellipse_1sigma = computeCovarianceEllipse(x_est(1:2), pos_cov, 1);
    plot(ellipse_1sigma(1, :), ellipse_1sigma(2, :), 'g-', 'LineWidth', 2);

    % Plot true state
    if ~isempty(true_state)
        plot(true_state(1), true_state(2), 'md', 'MarkerSize', 10, ...
            'LineWidth', 2, 'MarkerFaceColor', 'm');
    end

    % Plot measurements
    if ~isempty(measurements)
        plot(measurements(1, :), measurements(2, :), 'cx', 'MarkerSize', 8, 'LineWidth', 1.5);
    end

    title(sprintf('Position (Weighted)\\newlineESS: %.1f', ESS), 'Interpreter', 'tex');
    xlabel('X (m)'); ylabel('Y (m)');
    xlim(bounds.x); ylim(bounds.y);
    axis square; grid on;

    %% SUBPLOT 2: HMM Posterior Heatmap (weighted average across particles)
    if has_signal
        subplot(2, 3, 2);
    else
        subplot(1, 3, 2);
    end

    cla; hold on;

    % Compute weighted-average posterior over the HMM grid
    % We reconstruct from the live particles if they still exist,
    % otherwise fall back to particle_states
    try
        % Access live HMM posteriors from the rbpf object
        N_p = rbpf.N_p;
        npx = rbpf.particles{1}.hmm.npx;
        npx2 = npx * npx;
        avg_posterior = zeros(npx2, 1);

        for i = 1:N_p
            w_i = rbpf.particles{i}.weight;
            posterior_i = rbpf.particles{i}.hmm.ptarget_prob;
            avg_posterior = avg_posterior + w_i * full(posterior_i(:));
        end

        % Reshape to 2D grid
        posterior_grid = reshape(avg_posterior, [npx, npx]);

        % Read grid bounds from HMM
        scene_x = rbpf.particles{1}.hmm.scenebounds_x;
        scene_y = rbpf.particles{1}.hmm.scenebounds_y;

        % Crop to match plot bounds (y >= 0.5)
        % The grid covers scene_y(1) to scene_y(2). We want y >= 0.5
        y_edges = linspace(scene_y(1), scene_y(2), npx + 1);
        row_start = find(y_edges(1:end-1) >= bounds.y(1), 1, 'first');
        if isempty(row_start), row_start = 1; end

        posterior_cropped = posterior_grid(row_start:end, :);
        y_plot = [max(scene_y(1), bounds.y(1)), scene_y(2)];

        imagesc(scene_x, y_plot, posterior_cropped);
        colormap(gca, hot(256));
        colorbar;
        set(gca, 'YDir', 'normal');

        % Overlay estimate and truth
        plot(x_est(1), x_est(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
        if ~isempty(true_state)
            plot(true_state(1), true_state(2), 'md', 'MarkerSize', 10, ...
                'LineWidth', 2, 'MarkerFaceColor', 'm');
        end

        title(sprintf('HMM Posterior (Weighted Avg)\\newlinek=%d', k), 'Interpreter', 'tex');
    catch
        % Fallback: if live particles are unavailable (e.g. final-only mode)
        text(0.5, 0.5, sprintf('HMM Posterior\n(Live particles unavailable)'), ...
            'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', 10);
        title(sprintf('HMM Posterior (k=%d)', k));
    end

    xlabel('X (m)'); ylabel('Y (m)');
    xlim(bounds.x); ylim(bounds.y);
    axis square; grid on;

    %% SUBPLOT 3: HMM Entropy
    if has_signal
        subplot(2, 3, 3);
    else
        subplot(1, 3, 3);
    end

    cla; hold on;

    % Plot entropy distribution across particles
    if isfield(hist, 'particle_entropies') && ~isempty(hist.particle_entropies)
        entropies = hist.particle_entropies;
        mean_entropy = sum(particle_weights .* entropies);
        max_entropy = log(rbpf.particles{1}.hmm.npx^2);  % Maximum entropy = log(npx2)

        histogram(entropies, 20, 'Normalization', 'probability', ...
            'FaceColor', [0.3 0.3 0.8], 'FaceAlpha', 0.7);

        % Mark weighted mean entropy
        xline(mean_entropy, 'g-', 'LineWidth', 2);
        xline(max_entropy, 'r--', 'LineWidth', 1.5);

        % Legend
        legend({'Particle H', sprintf('Weighted Mean: %.2f', mean_entropy), ...
                sprintf('Max H: %.2f', max_entropy)}, ...
            'Location', 'best', 'FontSize', 7);

        xlabel('Entropy (nats)');
        ylabel('Probability');
        title(sprintf('HMM Entropy (k=%d)\\newlineMean: %.2f / %.2f', ...
            k, mean_entropy, max_entropy), 'Interpreter', 'tex');
    else
        % Fallback: entropy data not stored in history
        text(0.5, 0.5, sprintf('Entropy data\nnot available'), ...
            'HorizontalAlignment', 'center', 'Units', 'normalized');
        title(sprintf('HMM Entropy (k=%d)', k));
    end

    grid on;

    %% ========== BOTTOM ROW: Signal | Association | Trajectory ==========

    %% SUBPLOT 4: mWidar Signal with measurements
    if has_signal
        subplot(2, 3, 4);
        cla; hold on;

        signal = signal_data{k};  % Full signal [128x128]

        % Crop signal to y: 0.5-4 (remove detector region at y=0-0.5)
        signal_y_range = [0, 4];
        crop_y_range = [0.5, 4];

        n_rows = size(signal, 1);
        row_indices = round((crop_y_range - signal_y_range(1)) / diff(signal_y_range) * n_rows);
        row_indices = max(1, min(n_rows, row_indices));
        row_start = row_indices(1);
        row_end = n_rows;

        signal_cropped = signal(row_start:row_end, :);

        % Normalize for better contrast
        signal_range = max(signal_cropped(:)) - min(signal_cropped(:));
        if signal_range > 0
            signal_norm = (signal_cropped - min(signal_cropped(:))) / signal_range;
        else
            signal_norm = zeros(size(signal_cropped));
        end

        imagesc(bounds.x, bounds.y, signal_norm);
        colormap(gca, hot(256));
        colorbar;
        set(gca, 'YDir', 'normal');

        % Plot measurements
        if ~isempty(measurements)
            plot(measurements(1, :), measurements(2, :), 'co', ...
                'MarkerSize', 8, 'LineWidth', 2);
        end

        % Plot true state
        if ~isempty(true_state) && true_meas_flag
            plot(true_state(1), true_state(2), 'mp', 'MarkerSize', 15, ...
                'LineWidth', 2, 'MarkerFaceColor', 'm');
        end

        % Plot estimate
        plot(x_est(1), x_est(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);

        xlim(bounds.x); ylim(bounds.y);
        xlabel('X (m)'); ylabel('Y (m)');
        title(sprintf('mWidar Signal (t=%d)', k));
        axis square; grid on;
    end

    %% SUBPLOT 5: Association histogram
    if has_signal
        subplot(2, 3, 5);
    else
        % Without signal data, skip signal panel — show association at bottom?
        % For now, skip association if no signal
        return;
    end

    cla; hold on;

    % Determine correct association (nearest measurement to truth)
    correct_assoc = [];
    if ~isempty(true_state) && true_meas_flag && ~isempty(measurements)
        true_pos = true_state(1:2);
        dists = vecnorm(measurements - true_pos, 2, 1);
        [~, closest_idx] = min(dists);
        correct_assoc = closest_idx;
    end

    N_meas = max(particle_assocs);
    N_meas_total = max([N_meas, size(measurements, 2)]);

    if N_meas_total > 0
        h = histogram(particle_assocs, 'BinEdges', -0.5:(N_meas_total + 0.5), ...
            'Normalization', 'probability', 'FaceColor', [0.3 0.3 0.8]);

        set(gca, 'YScale', 'log');
        ylim([1e-4, 2]);

        % Highlight correct association
        if ~isempty(correct_assoc)
            bin_idx = correct_assoc + 1;

            if bin_idx <= length(h.Values) && h.Values(bin_idx) > 0
                bar(correct_assoc, h.Values(bin_idx), 'FaceColor', 'none', ...
                    'EdgeColor', [0 0.8 0], 'LineWidth', 3);
                star_y = max(h.Values(bin_idx) * 1.5, 1.2);
                plot(correct_assoc, star_y, 'p', 'MarkerSize', 20, ...
                    'MarkerFaceColor', [1 0.8 0], 'MarkerEdgeColor', [0.8 0.6 0], 'LineWidth', 1.5);
            end
        end

        xlabel('Association ID');
        ylabel('Fraction (log scale)');
        title(sprintf('Associations (k=%d)', k), 'Interpreter', 'tex');
        xticks(0:N_meas_total);

        xlabels = cell(1, N_meas_total + 1);
        xlabels{1} = 'Miss';
        for i = 1:N_meas_total
            if ~isempty(correct_assoc) && i == correct_assoc
                xlabels{i + 1} = sprintf('%d *', i);
            else
                xlabels{i + 1} = num2str(i);
            end
        end
        xticklabels(xlabels);
        grid on;
    else
        bar(0, 1, 'FaceColor', [0.3 0.3 0.8]);
        xlabel('Association ID');
        ylabel('Fraction');
        title(sprintf('Associations (k=%d)', k), 'Interpreter', 'tex');
        xticks(0);
        xticklabels({'Miss'});
        ylim([0 1.1]);
        grid on;
    end

    %% SUBPLOT 6: Trajectory hypothesis tree
    if has_signal
        subplot(2, 3, 6);
        cla; hold on;

        if plot_trajectories && k > 1
            if isfield(hist, 'particle_trajectories') && ~isempty(hist.particle_trajectories)
                particle_trajs = hist.particle_trajectories;
                N_particles = length(particle_trajs);
                N_to_plot = min(N_particles, max_particles_to_plot);

                for pp = 1:N_to_plot
                    traj = particle_trajs{pp};
                    if ~isempty(traj) && size(traj, 2) >= 2
                        plot(traj(1, :), traj(2, :), 'Color', [0.7 0.7 0.7 0.3], 'LineWidth', 0.5);
                    end
                end
            else
                text(0.5, 0.5, sprintf('Trajectory Tree\n(Data not available)\nRun with updated HMM_RBPF'), ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', 10);
            end

            % Plot estimated trajectory from history
            est_traj = zeros(2, k);
            for t = 1:k
                est_traj(:, t) = rbpf.history(t).estimate(1:2);
            end
            plot(est_traj(1, :), est_traj(2, :), 'g-', 'LineWidth', 3, 'DisplayName', 'Estimate');

            % Plot true trajectory if available
            if ~isempty(true_state)
                true_traj = zeros(2, k);
                for t = 1:k
                    if ~isempty(rbpf.history(t).true_state)
                        true_traj(:, t) = rbpf.history(t).true_state(1:2);
                    end
                end
                plot(true_traj(1, :), true_traj(2, :), 'm--', 'LineWidth', 3, 'DisplayName', 'True');
            end

            % Plot current particle cloud
            scatter(particle_states(1, :), particle_states(2, :), 10, particle_weights, ...
                'filled', 'MarkerFaceAlpha', 0.6);

            xlabel('X (m)'); ylabel('Y (m)');
            title(sprintf('Trajectory Tree (k=%d)', k));
            xlim(bounds.x); ylim(bounds.y);
            grid on; axis square;
        else
            text(0.5, 0.5, sprintf('Trajectory Tree\n(k=%d)', k), ...
                'HorizontalAlignment', 'center', 'Units', 'normalized');
            xlim(bounds.x); ylim(bounds.y);
            axis square; grid on;
        end
    end

    drawnow;
end

%% =====================================================================
%%  INDIVIDUAL GIF GENERATION
%% =====================================================================

function generateIndividualGIFs(rbpf, frames_to_plot, bounds, signal_data, ...
                                plot_trajectories, max_particles_to_plot, ...
                                gif_directory, anim_speed, animate)
    % Generate separate GIF for each subplot type

    subplot_types = {'position', 'hmm_posterior', 'entropy', 'association'};
    if ~isempty(signal_data)
        subplot_types{end+1} = 'signal';
    end
    if plot_trajectories
        subplot_types{end+1} = 'trajectory';
    end

    fprintf('\nGenerating %d individual HMM-RBPF GIF files...\n', length(subplot_types));

    for i = 1:length(subplot_types)
        subplot_type = subplot_types{i};
        gif_filename = fullfile(gif_directory, [subplot_type '.gif']);

        fprintf('  Creating %s...\n', gif_filename);

        fig = figure('Name', sprintf('HMM-RBPF %s', upper(subplot_type)), ...
                    'Position', [100, 100, 800, 600]);

        gif_first_frame = true;

        for k = frames_to_plot
            clf(fig);

            plotIndividualSubplot(rbpf, k, bounds, subplot_type, signal_data, max_particles_to_plot);

            frame = getframe(fig);
            im = frame2im(frame);
            [imind, cm] = rgb2ind(im, 256);

            if gif_first_frame
                imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', anim_speed);
                gif_first_frame = false;
            else
                imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', anim_speed);
            end

            if animate && k < frames_to_plot(end)
                pause(anim_speed);
            end
        end

        close(fig);
        fprintf('    Saved: %s\n', gif_filename);
    end

    fprintf('\nAll individual GIFs saved successfully!\n');
end

%% =====================================================================
%%  INDIVIDUAL SUBPLOT PLOTTER (for standalone GIFs)
%% =====================================================================

function plotIndividualSubplot(rbpf, k, bounds, subplot_type, signal_data, max_particles_to_plot)
    % Plot a specific subplot type for timestep k

    hist = rbpf.history(k);
    particle_states = hist.particle_states;
    particle_weights = hist.particle_weights;
    true_state = hist.true_state;

    switch subplot_type
        case 'position'
            % Position subplot (weighted particle cloud)
            scatter(particle_states(1, :), particle_states(2, :), 50, particle_weights, ...
                'filled', 'MarkerFaceAlpha', 0.8);
            colormap(gca, hot(256)); colorbar;
            hold on;

            est = hist.estimate(1:2);
            est_cov = hist.covariance(1:2, 1:2);
            plot(est(1), est(2), 'gx', 'MarkerSize', 15, 'LineWidth', 3);
            ellipse_pts = computeCovarianceEllipse(est, est_cov, 1);
            plot(ellipse_pts(1, :), ellipse_pts(2, :), 'g-', 'LineWidth', 2);

            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'mo', 'MarkerSize', 12, 'LineWidth', 3);
            end

            measurements = hist.measurements;
            if ~isempty(measurements)
                plot(measurements(1, :), measurements(2, :), 'cx', 'MarkerSize', 8, 'LineWidth', 1.5);
            end

            xlabel('X (m)'); ylabel('Y (m)');
            title(sprintf('Position (Weighted) - k=%d', k));
            xlim(bounds.x); ylim(bounds.y);
            grid on; axis square;

        case 'hmm_posterior'
            % HMM Posterior Heatmap (weighted average)
            hold on;
            try
                N_p = rbpf.N_p;
                npx = rbpf.particles{1}.hmm.npx;
                npx2 = npx * npx;
                avg_posterior = zeros(npx2, 1);

                for ii = 1:N_p
                    w_i = rbpf.particles{ii}.weight;
                    posterior_i = rbpf.particles{ii}.hmm.ptarget_prob;
                    avg_posterior = avg_posterior + w_i * full(posterior_i(:));
                end

                posterior_grid = reshape(avg_posterior, [npx, npx]);
                scene_x = rbpf.particles{1}.hmm.scenebounds_x;
                scene_y = rbpf.particles{1}.hmm.scenebounds_y;

                y_edges = linspace(scene_y(1), scene_y(2), npx + 1);
                row_start = find(y_edges(1:end-1) >= bounds.y(1), 1, 'first');
                if isempty(row_start), row_start = 1; end

                posterior_cropped = posterior_grid(row_start:end, :);
                y_plot = [max(scene_y(1), bounds.y(1)), scene_y(2)];

                imagesc(scene_x, y_plot, posterior_cropped);
                colormap(gca, hot(256)); colorbar;
                set(gca, 'YDir', 'normal');

                est = hist.estimate(1:2);
                plot(est(1), est(2), 'go', 'MarkerSize', 10, 'LineWidth', 2);
                if ~isempty(true_state)
                    plot(true_state(1), true_state(2), 'md', 'MarkerSize', 10, ...
                        'LineWidth', 2, 'MarkerFaceColor', 'm');
                end

                title(sprintf('HMM Posterior (Weighted Avg) - k=%d', k));
            catch
                text(0.5, 0.5, 'HMM Posterior not available', ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
                title(sprintf('HMM Posterior (k=%d)', k));
            end

            xlabel('X (m)'); ylabel('Y (m)');
            xlim(bounds.x); ylim(bounds.y);
            grid on; axis square;

        case 'entropy'
            % HMM Entropy distribution across particles
            hold on;
            if isfield(hist, 'particle_entropies') && ~isempty(hist.particle_entropies)
                entropies = hist.particle_entropies;
                mean_entropy = sum(particle_weights .* entropies);

                histogram(entropies, 20, 'Normalization', 'probability', ...
                    'FaceColor', [0.3 0.3 0.8], 'FaceAlpha', 0.7);
                xline(mean_entropy, 'g-', 'LineWidth', 2);

                xlabel('Entropy (nats)');
                ylabel('Probability');
                title(sprintf('HMM Entropy - k=%d (Mean: %.2f)', k, mean_entropy));
            else
                text(0.5, 0.5, 'Entropy data not available', ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
                title(sprintf('HMM Entropy (k=%d)', k));
            end
            grid on;

        case 'signal'
            % mWidar signal subplot
            if ~isempty(signal_data) && k <= length(signal_data)
                signal = signal_data{k};

                signal_y_range = [0, 4];
                crop_y_range = [0.5, 4];
                n_rows = size(signal, 1);

                row_indices = round((crop_y_range - signal_y_range(1)) / diff(signal_y_range) * n_rows);
                row_indices = max(1, min(n_rows, row_indices));
                row_start = row_indices(1);
                row_end = n_rows;

                signal_cropped = signal(row_start:row_end, :);
                signal_range = max(signal_cropped(:)) - min(signal_cropped(:));
                if signal_range > 0
                    signal_norm = (signal_cropped - min(signal_cropped(:))) / signal_range;
                else
                    signal_norm = zeros(size(signal_cropped));
                end

                imagesc(bounds.x, bounds.y, signal_norm);
                colormap(gca, hot(256)); colorbar;
                set(gca, 'YDir', 'normal');
                hold on;

                measurements = hist.measurements;
                if ~isempty(measurements)
                    plot(measurements(1, :), measurements(2, :), 'co', ...
                        'MarkerSize', 8, 'LineWidth', 2);
                end

                est = hist.estimate(1:2);
                plot(est(1), est(2), 'gx', 'MarkerSize', 15, 'LineWidth', 3);

                true_meas_flag = hist.true_meas_flag;
                if ~isempty(true_state) && true_meas_flag
                    plot(true_state(1), true_state(2), 'mp', 'MarkerSize', 15, ...
                        'LineWidth', 2, 'MarkerFaceColor', 'm');
                end

                xlabel('X (m)'); ylabel('Y (m)');
                title(sprintf('mWidar Signal (t=%d)', k));
                xlim(bounds.x); ylim(bounds.y);
                grid on; axis square;
            else
                text(0.5, 0.5, 'Signal data not available', ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
            end

        case 'association'
            % Association histogram subplot
            particle_assocs = hist.particle_associations;
            measurements = hist.measurements;
            true_meas_flag = hist.true_meas_flag;

            correct_assoc = [];
            if ~isempty(true_state) && true_meas_flag && ~isempty(measurements)
                true_pos = true_state(1:2);
                dists = vecnorm(measurements - true_pos, 2, 1);
                [~, closest_idx] = min(dists);
                correct_assoc = closest_idx;
            end

            N_meas = max(particle_assocs);
            N_meas_total = max([N_meas, size(measurements, 2)]);

            if N_meas_total > 0
                h = histogram(particle_assocs, 'BinEdges', -0.5:(N_meas_total + 0.5), ...
                    'Normalization', 'probability', 'FaceColor', [0.3 0.3 0.8]);

                set(gca, 'YScale', 'log');
                ylim([1e-4, 2]);

                if ~isempty(correct_assoc)
                    hold on;
                    xline(correct_assoc, 'r--', 'LineWidth', 2);
                end

                xlabel('Measurement Association Index');
                ylabel('Probability (log scale)');
                title(sprintf('Association Histogram (k=%d)', k));
                grid on;
            else
                text(0.5, 0.5, 'No measurements at this timestep', ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
            end

        case 'trajectory'
            % Trajectory tree subplot
            hold on;
            if isfield(hist, 'particle_trajectories') && ~isempty(hist.particle_trajectories)
                particle_traj = hist.particle_trajectories;
                n_particles = length(particle_traj);
                n_to_plot = min(n_particles, max_particles_to_plot);

                for pp = 1:n_to_plot
                    traj = particle_traj{pp};
                    if ~isempty(traj) && size(traj, 2) > 1
                        plot(traj(1, :), traj(2, :), '-', 'Color', [0.7 0.7 0.7 0.3], 'LineWidth', 0.5);
                    end
                end

                est_traj = zeros(2, k);
                for t = 1:k
                    est_traj(:, t) = rbpf.history(t).estimate(1:2);
                end
                plot(est_traj(1, :), est_traj(2, :), 'g-', 'LineWidth', 3, 'DisplayName', 'Estimate');

                if ~isempty(true_state)
                    true_traj = zeros(2, k);
                    for t = 1:k
                        if ~isempty(rbpf.history(t).true_state)
                            true_traj(:, t) = rbpf.history(t).true_state(1:2);
                        end
                    end
                    plot(true_traj(1, :), true_traj(2, :), 'm--', 'LineWidth', 3, 'DisplayName', 'True');
                end

                scatter(particle_states(1, :), particle_states(2, :), 10, particle_weights, ...
                    'filled', 'MarkerFaceAlpha', 0.6);

                xlabel('X (m)'); ylabel('Y (m)');
                title(sprintf('Trajectory Tree - k=%d', k));
                xlim(bounds.x); ylim(bounds.y);
                grid on; axis square;
            else
                text(0.5, 0.5, 'Trajectory data not available', ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
            end
    end

    drawnow;
end

%% =====================================================================
%%  HELPER: Covariance Ellipse
%% =====================================================================

function ellipse_points = computeCovarianceEllipse(mu, Sigma, num_sigma)
    % Compute ellipse points for plotting covariance
    [V, D] = eig(Sigma);
    a = num_sigma * sqrt(max(D(1, 1), 0));
    b = num_sigma * sqrt(max(D(2, 2), 0));
    theta = atan2(V(2, 1), V(1, 1));
    t = linspace(0, 2 * pi, 100);
    ellipse = [a * cos(t); b * sin(t)];
    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    ellipse_points = R * ellipse + mu(:);
end