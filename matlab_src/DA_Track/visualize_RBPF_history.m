function visualize_RBPF_history(rbpf, varargin)
    % VISUALIZE_RBPF_HISTORY Standalone visualization of RBPF filter history
    %
    % USAGE:
    %   visualize_RBPF_HISTORY(rbpf)
    %   visualize_RBPF_history(rbpf, 'Animate', true)
    %   visualize_RBPF_history(rbpf, 'SaveGIF', 'output.gif')
    %   visualize_RBPF_history(rbpf, 'SaveFinalFigure', 'final.png')
    %
    % INPUTS:
    %   rbpf - KF_RBPF object with populated history field
    %
    % OPTIONAL PARAMETERS:
    %   'Animate'         - Boolean: Show timestep-by-timestep animation (default: false)
    %   'AnimationSpeed'  - Pause duration between frames in seconds (default: 0.2)
    %   'SaveGIF'         - String: Filename to save animation as GIF (default: '')
    %   'SaveFinalFigure' - String: Filename to save final timestep figure (default: '')
    %   'PlotMargin'      - Margin percentage for plot bounds (default: 0.2 = 20%)
    %   'SignalData'      - Cell array of mWidar signals [128x128] per timestep (default: {})
    %   'PlotTrajectories'- Boolean: Plot all particle trajectory histories (default: false)
    %   'MaxParticlesToPlot' - Maximum number of particle trajectories to plot (default: inf = all)
    %
    % OUTPUTS:
    %   Creates figure with 2x3 subplot layout (if signal data):
    %   TOP ROW: Position (weighted) | Velocity (weighted) | Acceleration (weighted)
    %   BOTTOM ROW: mWidar Signal | Association histogram | Trajectory Tree

    % Parse optional arguments
    p = inputParser;
    addParameter(p, 'Animate', false, @islogical);
    addParameter(p, 'AnimationSpeed', 0.2, @(x) x > 0);
    addParameter(p, 'SaveGIF', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'SaveFinalFigure', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'PlotMargin', 0.2, @(x) x >= 0);
    addParameter(p, 'SignalData', {}, @iscell); % Cell array of mWidar signals [128x128] per timestep
    addParameter(p, 'PlotTrajectories', false, @islogical); % Plot particle trajectory tree
    addParameter(p, 'MaxParticlesToPlot', inf, @isnumeric); % Max particles to plot
    addParameter(p, 'SaveIndividualGIFs', false, @islogical); % Save each subplot as separate GIF
    addParameter(p, 'GIFDirectory', '', @(x) ischar(x) || isstring(x)); % Directory for individual GIFs
    parse(p, varargin{:});

    animate = p.Results.Animate;
    anim_speed = p.Results.AnimationSpeed;
    gif_filename = char(p.Results.SaveGIF); % Convert to char
    final_fig_filename = char(p.Results.SaveFinalFigure); % Convert to char
    margin = p.Results.PlotMargin;
    signal_data = p.Results.SignalData; % Cell array of mWidar signals
    plot_trajectories = p.Results.PlotTrajectories; % Plot trajectory tree
    max_particles_to_plot = p.Results.MaxParticlesToPlot;
    save_individual_gifs = p.Results.SaveIndividualGIFs;
    gif_directory = char(p.Results.GIFDirectory);

    % Check that history exists
    if isempty(rbpf.history)
        error('RBPF history is empty. Run filter first before visualizing.');
    end

    n_steps = length(rbpf.history);
    fprintf('Visualizing %d timesteps...\n', n_steps);

    % Compute static plot bounds from all data
    bounds = computePlotBounds(rbpf, margin);

    % Animation or just final frame?
    if animate
        frames_to_plot = 1:n_steps;
    else
        frames_to_plot = n_steps; % Only final frame
    end

    % Determine if we should generate individual GIFs
    if save_individual_gifs
        % Generate separate GIF for each subplot
        generateIndividualGIFs(rbpf, frames_to_plot, bounds, signal_data, ...
                              plot_trajectories, max_particles_to_plot, ...
                              gif_directory, anim_speed, animate);
    else
        % Original combined figure approach
        % Create figure - 2x3 layout if signal data, 1x4 if not
        if ~isempty(signal_data)
            fig = figure('Name', 'KF-RBPF Tracking Results', 'Position', [100, 100, 1600, 800]);
        else
            fig = figure('Name', 'KF-RBPF Tracking Results', 'Position', [100, 100, 1600, 400]);
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

function bounds = computePlotBounds(rbpf, margin)
    % Compute static plot bounds from all history data

    % n_steps = length(rbpf.history);

    % % Collect all positions
    % all_x = [];
    % all_y = [];
    % all_vx = [];
    % all_vy = [];

    % for k = 1:n_steps
    %     % Particle positions
    %     states = rbpf.history(k).particle_states;
    %     all_x = [all_x, states(1, :)];
    %     all_y = [all_y, states(2, :)];

    %     if size(states, 1) >= 4
    %         all_vx = [all_vx, states(3, :)];
    %         all_vy = [all_vy, states(4, :)];
    %     end

    %     % True state
    %     if ~isempty(rbpf.history(k).true_state)
    %         ts = rbpf.history(k).true_state;
    %         all_x = [all_x, ts(1)];
    %         all_y = [all_y, ts(2)];
    %         if length(ts) >= 4
    %             all_vx = [all_vx, ts(3)];
    %             all_vy = [all_vy, ts(4)];
    %         end
    %     end

    %     % Measurements
    %     if ~isempty(rbpf.history(k).measurements)
    %         meas = rbpf.history(k).measurements;
    %         all_x = [all_x, meas(1, :)];
    %         all_y = [all_y, meas(2, :)];
    %     end
    % end

    % % Compute bounds with margin
    % bounds.x = [min(all_x), max(all_x)];
    % bounds.y = [min(all_y), max(all_y)];

    % x_range = diff(bounds.x);
    % y_range = diff(bounds.y);
    % bounds.x = bounds.x + [-margin, margin] * x_range;
    % bounds.y = bounds.y + [-margin, margin] * y_range;

    % if ~isempty(all_vx)
    %     bounds.vx = [min(all_vx), max(all_vx)];
    %     bounds.vy = [min(all_vy), max(all_vy)];
    %     vx_range = diff(bounds.vx);
    %     vy_range = diff(bounds.vy);
    %     bounds.vx = bounds.vx + [-margin, margin] * vx_range;
    %     bounds.vy = bounds.vy + [-margin, margin] * vy_range;
    % else
    %     bounds.vx = [];
    %     bounds.vy = [];
    % end

    % Manual bounds for mWidar simulations
    bounds.x = [-2, 2];
    bounds.y = [0.5, 4]; % Crop out detector at y=0
    bounds.vx = [-2, 2]; % Velocity bounds for mWidar
    bounds.vy = [-2, 6]; % Velocity bounds for mWidar

    % Acceleration bounds (if 6-state)
    bounds.ax = [-2, 2];
    bounds.ay = [-2, 2];

end

function plotTimestep(rbpf, k, bounds, fig, signal_data, plot_trajectories, max_particles_to_plot)
    % Plot a single timestep with 2x3 layout

    figure(fig);

    % Extract data for this timestep
    hist = rbpf.history(k);
    particle_states = hist.particle_states;
    particle_assocs = hist.particle_associations;
    particle_weights = hist.particle_weights;
    measurements = hist.measurements;
    true_state = hist.true_state;
    true_meas_flag = hist.true_meas_flag;
    x_est = hist.estimate;
    P_est = hist.covariance;
    ESS = hist.ESS;

    N_x = size(particle_states, 1);

    % Determine layout
    has_signal = ~isempty(signal_data) && k <= length(signal_data) && ~isempty(signal_data{k});

    %% ========== TOP ROW: Position | Velocity | Acceleration ==========

    %% SUBPLOT 1: Position (weighted)
    if has_signal
        subplot(2, 3, 1);
    else
        subplot(1, 4, 1);
    end

    cla; hold on;

    scatter(particle_states(1, :), particle_states(2, :), 20, particle_weights, ...
        'filled', 'MarkerFaceAlpha', 0.8);
    colormap(gca, hot(256));
    colorbar;

    % Plot mean estimate with covariance
    plot(x_est(1), x_est(2), 'go', 'MarkerSize', 12, 'LineWidth', 3);

    if N_x >= 2
        pos_cov = P_est(1:2, 1:2);
        ellipse_1sigma = computeCovarianceEllipse(x_est(1:2), pos_cov, 1);
        plot(ellipse_1sigma(1, :), ellipse_1sigma(2, :), 'g-', 'LineWidth', 2);
    end

    % Plot true state
    if ~isempty(true_state)
        plot(true_state(1), true_state(2), 'md', 'MarkerSize', 10, ...
            'LineWidth', 2, 'MarkerFaceColor', 'm');
    end

    title(sprintf('Position (Weighted)\\newlineESS: %.1f', ESS), 'Interpreter', 'tex');
    xlabel('X (m)'); ylabel('Y (m)');
    xlim(bounds.x); ylim(bounds.y);
    axis square; grid on;

    %% SUBPLOT 2: Velocity (weighted)
    if has_signal
        subplot(2, 3, 2);
    else
        subplot(1, 4, 2);
    end

    cla; hold on;

    if N_x >= 4
        scatter(particle_states(3, :), particle_states(4, :), 20, ...
            particle_weights, 'filled', 'MarkerFaceAlpha', 0.6);
        colormap(gca, hot(256));
        colorbar;

        % Plot mean with covariance
        plot(x_est(3), x_est(4), 'go', 'MarkerSize', 12, 'LineWidth', 3);
        vel_cov = P_est(3:4, 3:4);
        ellipse_1sigma = computeCovarianceEllipse(x_est(3:4), vel_cov, 1);
        plot(ellipse_1sigma(1, :), ellipse_1sigma(2, :), 'g-', 'LineWidth', 2);

        % Plot true velocity
        if ~isempty(true_state) && length(true_state) >= 4
            plot(true_state(3), true_state(4), 'md', 'MarkerSize', 10, ...
                'LineWidth', 2, 'MarkerFaceColor', 'm');
        end

        if ~isempty(bounds.vx)
            xlim(bounds.vx); ylim(bounds.vy);
        end

        title('Velocity (Weighted)', 'Interpreter', 'tex');
        xlabel('Vx (m/s)'); ylabel('Vy (m/s)');
    else
        text(0.5, 0.5, 'No velocity states', 'HorizontalAlignment', 'center');
        title('Velocity');
    end

    axis square; grid on;

    %% SUBPLOT 3: Acceleration (weighted)
    if has_signal
        subplot(2, 3, 3);
    else
        subplot(1, 4, 3);
    end

    cla; hold on;

    if N_x >= 6
        scatter(particle_states(5, :), particle_states(6, :), 20, ...
            particle_weights, 'filled', 'MarkerFaceAlpha', 0.6);
        colormap(gca, hot(256));
        colorbar;

        % Plot mean with covariance
        plot(x_est(5), x_est(6), 'go', 'MarkerSize', 12, 'LineWidth', 3);
        acc_cov = P_est(5:6, 5:6);
        ellipse_1sigma = computeCovarianceEllipse(x_est(5:6), acc_cov, 1);
        plot(ellipse_1sigma(1, :), ellipse_1sigma(2, :), 'g-', 'LineWidth', 2);

        % Plot true acceleration
        if ~isempty(true_state) && length(true_state) >= 6
            plot(true_state(5), true_state(6), 'md', 'MarkerSize', 10, ...
                'LineWidth', 2, 'MarkerFaceColor', 'm');
        end

        % Set bounds if available
        xlim(bounds.ax); ylim(bounds.ay);

        title('Acceleration (Weighted)', 'Interpreter', 'tex');
        xlabel('Ax (m/s^2)'); ylabel('Ay (m/s^2)');
    else
        text(0.5, 0.5, 'No acceleration states', 'HorizontalAlignment', 'center');
        title('Acceleration');
    end

    axis square; grid on;

    %% ========== BOTTOM ROW: Signal | Association | Trajectory ==========

    %% SUBPLOT 4: mWidar Signal with measurements
    if has_signal
        subplot(2, 3, 4);
        cla; hold on;

        signal = signal_data{k};  % Full signal [128x128]
        
        % Crop signal to y: 0.5-4 (remove detector region at y=0-0.5)
        % Assuming signal represents y from 0 to 4 uniformly
        signal_y_range = [0, 4];  % Full range of signal
        crop_y_range = [0.5, 4];  % Desired range
        
        % Calculate which rows to keep (signal is [y, x], rows are y)
        n_rows = size(signal, 1);
        row_indices = round((crop_y_range - signal_y_range(1)) / diff(signal_y_range) * n_rows);
        row_indices = max(1, min(n_rows, row_indices));  % Clamp to valid range
        row_start = row_indices(1);
        row_end = n_rows;  % Keep to the end
        
        % Crop the signal
        signal_cropped = signal(row_start:row_end, :);
        
        % Normalize for better contrast
        signal_norm = (signal_cropped - min(signal_cropped(:))) / (max(signal_cropped(:)) - min(signal_cropped(:)));

        % Plot cropped signal
        imagesc(bounds.x, bounds.y, signal_norm);
        colormap(gca, hot(256));
        colorbar;
        set(gca, 'YDir', 'normal');

        % Plot measurements
        if ~isempty(measurements)
            plot(measurements(1, :), measurements(2, :), 'co', ...
                'MarkerSize', 8, 'LineWidth', 2);
        end

        % Plot true measurement if available
        if ~isempty(true_state) && true_meas_flag
            true_meas_pos = true_state(1:2);
            plot(true_meas_pos(1), true_meas_pos(2), 'mp', 'MarkerSize', 15, ...
                'LineWidth', 2, 'MarkerFaceColor', 'm');
        end

        xlim(bounds.x);
        ylim(bounds.y);
        xlabel('X Position (m)');
        ylabel('Y Position (m)');
        title(sprintf('mWidar Signal (t=%d)', k));
        grid on;
    end

    %% SUBPLOT 5: Association histogram
    if has_signal
        subplot(2, 3, 5);
    else
        subplot(1, 4, 4);
    end

    cla; hold on;

    % Determine correct association
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

        % Log scale to see small probabilities
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
        xlabels{1} = 'Clutter';

        for i = 1:N_meas_total

            if ~isempty(correct_assoc) && i == correct_assoc
                xlabels{i + 1} = sprintf('%d ★', i);
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
        xticklabels({'Clutter'});
        ylim([0 1.1]);
        grid on;
    end

    %% SUBPLOT 6: Trajectory hypothesis tree
    if has_signal
        subplot(2, 3, 6);
        cla; hold on;

        if plot_trajectories && k > 1
            % Check if particle_trajectories field exists
            if isfield(hist, 'particle_trajectories') && ~isempty(hist.particle_trajectories)
                % Plot particle trajectories up to current timestep
                particle_trajs = hist.particle_trajectories;
                N_particles = length(particle_trajs);
                N_to_plot = min(N_particles, max_particles_to_plot);

                % Plot all particle trajectories with transparency
                for p = 1:N_to_plot
                    traj = particle_trajs{p};

                    if ~isempty(traj) && size(traj, 2) >= 2
                        plot(traj(1, :), traj(2, :), 'Color', [0.7 0.7 0.7 0.3], 'LineWidth', 0.5);
                    end

                end

            else
                % Trajectory data not available
                text(0.5, 0.5, sprintf('Trajectory Tree\n(Data not available)\nRun with updated KF_RBPF'), ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized', 'FontSize', 10);
            end

            % Plot estimated trajectory (from history) - always available
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
            % legend('Location', 'best');
            grid on;
            axis square;
        else
            text(0.5, 0.5, sprintf('Trajectory Tree\n(k=%d)', k), ...
                'HorizontalAlignment', 'center', 'Units', 'normalized');
            xlim(bounds.x); ylim(bounds.y);
            axis square; grid on;
        end

    end

    drawnow;
end

function generateIndividualGIFs(rbpf, frames_to_plot, bounds, signal_data, ...
                                plot_trajectories, max_particles_to_plot, ...
                                gif_directory, anim_speed, animate)
    % Generate separate GIF for each subplot type
    
    % Define subplot types to generate
    subplot_types = {'position', 'velocity', 'acceleration', 'association'};
    if ~isempty(signal_data)
        subplot_types{end+1} = 'signal';
    end
    if plot_trajectories
        subplot_types{end+1} = 'trajectory';
    end
    
    fprintf('\nGenerating %d individual GIF files...\n', length(subplot_types));
    
    % Generate each GIF
    for i = 1:length(subplot_types)
        subplot_type = subplot_types{i};
        gif_filename = fullfile(gif_directory, [subplot_type '_weighted.gif']);
        
        fprintf('  Creating %s...\n', gif_filename);
        
        % Create individual figure for this subplot type
        fig = figure('Name', sprintf('RBPF %s', upper(subplot_type)), ...
                    'Position', [100, 100, 800, 600]);
        
        gif_first_frame = true;
        
        % Generate frames
        for k = frames_to_plot
            clf(fig); % Clear figure for new frame
            
            % Plot the specific subplot type
            plotIndividualSubplot(rbpf, k, bounds, subplot_type, signal_data, max_particles_to_plot);
            
            % Save GIF frame
            frame = getframe(fig);
            im = frame2im(frame);
            [imind, cm] = rgb2ind(im, 256);
            
            if gif_first_frame
                imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', anim_speed);
                gif_first_frame = false;
            else
                imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', anim_speed);
            end
            
            % Pause for animation
            if animate && k < frames_to_plot(end)
                pause(anim_speed);
            end
        end
        
        close(fig);
        fprintf('    Saved: %s\n', gif_filename);
    end
    
    fprintf('\nAll individual GIFs saved successfully!\n');
end

function plotIndividualSubplot(rbpf, k, bounds, subplot_type, signal_data, max_particles_to_plot)
    % Plot a specific subplot type for timestep k
    
    hist = rbpf.history(k);
    particle_states = hist.particle_states;
    particle_weights = hist.particle_weights;
    true_state = hist.true_state;
    
    % State dimension
    state_dim = size(particle_states, 1);
    
    switch subplot_type
        case 'position'
            % Position subplot (weighted)
            scatter(particle_states(1, :), particle_states(2, :), 50, particle_weights, ...
                'filled', 'MarkerFaceAlpha', 0.8);
            colormap(gca, hot(256)); colorbar;
            hold on;
            
            % Plot estimate with covariance
            est = hist.estimate(1:2);
            est_cov = hist.covariance(1:2, 1:2);
            plot(est(1), est(2), 'gx', 'MarkerSize', 15, 'LineWidth', 3);
            ellipse_pts = computeCovarianceEllipse(est, est_cov, 3);
            plot(ellipse_pts(1, :), ellipse_pts(2, :), 'g-', 'LineWidth', 2);
            
            if ~isempty(true_state)
                plot(true_state(1), true_state(2), 'mo', 'MarkerSize', 12, 'LineWidth', 3);
            end
            
            xlabel('X (m)'); ylabel('Y (m)');
            title(sprintf('Position (Weighted) - k=%d', k));
            xlim(bounds.x); ylim(bounds.y);
            grid on; axis square;
            
        case 'velocity'
            % Velocity subplot (weighted)
            if state_dim >= 4
                scatter(particle_states(3, :), particle_states(4, :), 50, particle_weights, ...
                    'filled', 'MarkerFaceAlpha', 0.8);
                colormap(gca, hot(256)); colorbar;
                hold on;
                
                % Plot estimate with covariance
                est = hist.estimate(3:4);
                est_cov = hist.covariance(3:4, 3:4);
                plot(est(1), est(2), 'gx', 'MarkerSize', 15, 'LineWidth', 3);
                ellipse_pts = computeCovarianceEllipse(est, est_cov, 3);
                plot(ellipse_pts(1, :), ellipse_pts(2, :), 'g-', 'LineWidth', 2);
                
                if ~isempty(true_state) && length(true_state) >= 4
                    plot(true_state(3), true_state(4), 'mo', 'MarkerSize', 12, 'LineWidth', 3);
                end
                
                xlabel('V_x (m/s)'); ylabel('V_y (m/s)');
                title(sprintf('Velocity (Weighted) - k=%d', k));
                xlim(bounds.vx); ylim(bounds.vy);
                grid on; axis square;
            else
                text(0.5, 0.5, 'Velocity data not available', ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
            end
            
        case 'acceleration'
            % Acceleration subplot (weighted)
            if state_dim >= 6
                scatter(particle_states(5, :), particle_states(6, :), 50, particle_weights, ...
                    'filled', 'MarkerFaceAlpha', 0.8);
                colormap(gca, hot(256)); colorbar;
                hold on;
                
                % Plot estimate with covariance
                est = hist.estimate(5:6);
                est_cov = hist.covariance(5:6, 5:6);
                plot(est(1), est(2), 'gx', 'MarkerSize', 15, 'LineWidth', 3);
                ellipse_pts = computeCovarianceEllipse(est, est_cov, 3);
                plot(ellipse_pts(1, :), ellipse_pts(2, :), 'g-', 'LineWidth', 2);
                
                if ~isempty(true_state) && length(true_state) >= 6
                    plot(true_state(5), true_state(6), 'mo', 'MarkerSize', 12, 'LineWidth', 3);
                end
                
                xlabel('A_x (m/s^2)'); ylabel('A_y (m/s^2)');
                title(sprintf('Acceleration (Weighted) - k=%d', k));
                xlim(bounds.ax); ylim(bounds.ay);
                grid on; axis square;
            else
                text(0.5, 0.5, 'Acceleration data not available', ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
            end
            
        case 'signal'
            % mWidar signal subplot
            if ~isempty(signal_data) && k <= length(signal_data)
                signal = signal_data{k}; % 128x128 mWidar signal
                
                % Crop signal to y = 0.5-4 (remove detector at y=0-0.5)
                signal_y_range = [0, 4];  % Full range of signal
                crop_y_range = [0.5, 4];  % Desired range
                n_rows = size(signal, 1);
                
                % Calculate which rows to keep
                row_indices = round((crop_y_range - signal_y_range(1)) / diff(signal_y_range) * n_rows);
                row_indices = max(1, min(n_rows, row_indices));  % Clamp to valid range
                row_start = row_indices(1);
                row_end = n_rows;  % Keep to the end
                
                % Crop the signal
                signal_cropped = signal(row_start:row_end, :);
                
                % Normalize for better contrast
                signal_norm = (signal_cropped - min(signal_cropped(:))) / (max(signal_cropped(:)) - min(signal_cropped(:)));
                
                % Plot cropped signal
                imagesc(bounds.x, bounds.y, signal_norm);
                colormap(gca, hot(256));
                colorbar;
                set(gca, 'YDir', 'normal');
                hold on;
                
                % Plot measurements if available
                measurements = hist.measurements;
                if ~isempty(measurements)
                    plot(measurements(1, :), measurements(2, :), 'co', ...
                        'MarkerSize', 8, 'LineWidth', 2);
                end
                
                % Plot estimate with covariance ellipse
                est = hist.estimate(1:2);
                est_cov = hist.covariance(1:2, 1:2);
                plot(est(1), est(2), 'gx', 'MarkerSize', 15, 'LineWidth', 3);
                ellipse_pts = computeCovarianceEllipse(est, est_cov, 3);
                plot(ellipse_pts(1, :), ellipse_pts(2, :), 'g-', 'LineWidth', 2);
                
                % Plot true measurement if available
                true_meas_flag = hist.true_meas_flag;
                if ~isempty(true_state) && true_meas_flag
                    true_meas_pos = true_state(1:2);
                    plot(true_meas_pos(1), true_meas_pos(2), 'mp', 'MarkerSize', 15, ...
                        'LineWidth', 2, 'MarkerFaceColor', 'm');
                end
                
                xlabel('X Position (m)'); ylabel('Y Position (m)');
                title(sprintf('mWidar Signal (t=%d)', k));
                xlim(bounds.x); ylim(bounds.y);
                grid on;
            else
                text(0.5, 0.5, 'Signal data not available', ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
            end
            
        case 'association'
            % Association histogram subplot
            particle_assocs = hist.particle_associations;
            measurements = hist.measurements;
            true_meas_flag = hist.true_meas_flag;
            
            % Determine correct association
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
                
                % Log scale to see small probabilities
                set(gca, 'YScale', 'log');
                ylim([1e-4, 2]);
                
                % Highlight correct association if available
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
            if isfield(hist, 'particle_trajectories')
                particle_traj = hist.particle_trajectories; % Cell array of trajectories
                
                n_particles = length(particle_traj);
                n_to_plot = min(n_particles, max_particles_to_plot);
                
                hold on;
                
                % Plot particle trajectories (transparent gray)
                for p = 1:n_to_plot
                    traj = particle_traj{p}; % 2 x T matrix
                    if ~isempty(traj) && size(traj, 2) > 1
                        plot(traj(1, :), traj(2, :), '-', 'Color', [0.7 0.7 0.7 0.3], 'LineWidth', 0.5);
                    end
                end
                
                % Plot estimated trajectory
                est_traj = zeros(2, k);
                for t = 1:k
                    est_traj(:, t) = rbpf.history(t).estimate(1:2);
                end
                plot(est_traj(1, :), est_traj(2, :), 'g-', 'LineWidth', 3, 'DisplayName', 'Estimate');
                
                % Plot true trajectory
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
                title(sprintf('Trajectory Tree - k=%d', k));
                xlim(bounds.x); ylim(bounds.y);
                grid on; axis square;
            else
                text(0.5, 0.5, 'Trajectory tree data not available', ...
                    'HorizontalAlignment', 'center', 'Units', 'normalized');
            end
    end
    
    drawnow;
end

function ellipse_points = computeCovarianceEllipse(mu, Sigma, num_sigma)
    % Compute ellipse points for plotting covariance
    [V, D] = eig(Sigma);
    a = num_sigma * sqrt(D(1, 1));
    b = num_sigma * sqrt(D(2, 2));
    theta = atan2(V(2, 1), V(1, 1));
    t = linspace(0, 2 * pi, 100);
    ellipse = [a * cos(t); b * sin(t)];
    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    ellipse_points = R * ellipse + mu;
end
