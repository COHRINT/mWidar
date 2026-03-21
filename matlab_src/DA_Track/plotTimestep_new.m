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
        
        if ~isempty(bounds.ax)
            xlim(bounds.ax); ylim(bounds.ay);
        end
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
        
        signal = signal_data{k};
        % Normalize for better contrast
        signal_norm = (signal - min(signal(:))) / (max(signal(:)) - min(signal(:)));
        
        % Plot signal
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
            % Plot particle trajectories up to current timestep
            particle_trajs = hist.particle_trajectories;
            N_particles = length(particle_trajs);
            N_to_plot = min(N_particles, max_particles_to_plot);
            
            % Plot all particle trajectories with transparency
            for p = 1:N_to_plot
                traj = particle_trajs{p};
                if size(traj, 2) >= 2
                    plot(traj(1, :), traj(2, :), 'Color', [0.7 0.7 0.7 0.3], 'LineWidth', 0.5);
                end
            end
            
            % Plot estimated trajectory (from history)
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
            legend('Location', 'best');
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
