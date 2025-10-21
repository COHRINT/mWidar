function mWidar_AddDATrackToSubplot(parent_fig, subplot_position, tracker_obj, measurements, true_state, title_str)
    % MWIDAR_ADDDATRACKTOSUBPLOT Add DA_Track visualization to existing subplot
    %
    % SYNTAX:
    %   mWidar_AddDATrackToSubplot(parent_fig, subplot_position, tracker_obj, measurements, true_state, title_str)
    %
    % INPUTS:
    %   parent_fig       - Handle to parent figure containing subplots
    %   subplot_position - Subplot position (e.g., subplot(2,3,4) -> position = [2,3,4])
    %   tracker_obj      - DA_Track filter object (PDA_KF, GNN_KF, PDA_PF, etc.)
    %   measurements     - Current measurements [N_z x N_measurements]
    %   true_state       - (optional) True state for comparison
    %   title_str        - (optional) Custom title string
    %
    % OUTPUTS:
    %   None - modifies the specified subplot in parent_fig
    %
    % DESCRIPTION:
    %   Adds a DA_Track visualization directly to a specified subplot position
    %   within an existing figure. Uses pinned axis limits [-2 2] for x and [0 4] for y.
    %
    % EXAMPLE:
    %   figure(1); 
    %   mWidar_AddDATrackToSubplot(1, [2,2,1], pda_kf_obj, z_meas, x_true, 'PDA-KF');
    %   % Adds PDA-KF plot to position (2,2,1) in figure 1
    
    % Validate inputs
    if nargin < 4
        error('Usage: mWidar_AddDATrackToSubplot(parent_fig, subplot_position, tracker_obj, measurements, ...)');
    end
    
    if nargin < 5 || isempty(true_state)
        true_state = [];
    end
    
    if nargin < 6 || isempty(title_str)
        title_str = sprintf('%s Visualization', class(tracker_obj));
    end
    
    % Make parent figure current
    figure(parent_fig);
    
    % Create subplot at specified position
    if length(subplot_position) == 3
        subplot(subplot_position(1), subplot_position(2), subplot_position(3));
    else
        error('subplot_position must be [rows, cols, index] format');
    end
    
    % Clear current axes and set up for DA_Track plotting
    cla; hold on; grid on;
    
    % Extract filter-specific visualization components and plot manually
    % This approach gives us more control than calling the full visualize method
    
    switch class(tracker_obj)
        case {'PDA_KF', 'GNN_KF'}
            % Kalman Filter visualization
            [x_est, P_est] = tracker_obj.getGaussianEstimate();
            
            % Plot state estimate (red circle)
            plot(x_est(1), x_est(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3, ...
                'DisplayName', 'State Estimate');
            
            % Plot covariance ellipses if covariance available
            if ~isempty(P_est) && size(P_est, 1) >= 2
                try
                    P_pos = P_est(1:2, 1:2);
                    
                    % 1-sigma ellipse
                    [X_1sig, Y_1sig] = calc_gsigma_ellipse_plotpoints(x_est(1:2), P_pos, 1, 100);
                    plot(X_1sig, Y_1sig, 'r-', 'LineWidth', 2, 'DisplayName', '1σ');
                    
                    % 2-sigma ellipse  
                    [X_2sig, Y_2sig] = calc_gsigma_ellipse_plotpoints(x_est(1:2), P_pos, 2, 100);
                    plot(X_2sig, Y_2sig, 'r--', 'LineWidth', 1.5, 'DisplayName', '2σ');
                catch
                    % Skip ellipses if calculation fails
                end
            end
            
        case {'PDA_PF', 'GNN_PF'}
            % Particle Filter visualization
            if isprop(tracker_obj, 'particles') && isprop(tracker_obj, 'weights')
                particles = tracker_obj.particles;
                weights = tracker_obj.weights;
                
                % Set light gray background
                set(gca, 'Color', [0.94, 0.94, 0.94]);
                
                % Plot particles colored by weights (position only)
                scatter(particles(1, :), particles(2, :), 20, weights, 'filled', ...
                    'MarkerFaceAlpha', 0.6);
                
                % Get mean estimate for overlay
                [x_est, ~] = tracker_obj.getGaussianEstimate();
                plot(x_est(1), x_est(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3, ...
                    'DisplayName', 'Mean Estimate');
            end
            
        case {'PDA_HMM', 'GNN_HMM'}
            % HMM visualization - simplified probability distribution
            if isprop(tracker_obj, 'posterior_prob')
                % Create spatial grid (matching HMM grid)
                xgrid = linspace(-2, 2, tracker_obj.grid_size);
                ygrid = linspace(0, 4, tracker_obj.grid_size);
                
                % Reshape posterior probability
                prob_2d = reshape(tracker_obj.posterior_prob, [tracker_obj.grid_size, tracker_obj.grid_size]);
                
                % Plot as image
                imagesc(xgrid, ygrid, prob_2d);
                set(gca, 'YDir', 'normal');
                colormap('parula');
                
                % Get mean estimate for overlay
                [x_est, ~] = tracker_obj.getGaussianEstimate();
                plot(x_est(1), x_est(2), 'wo', 'MarkerSize', 8, 'LineWidth', 2, ...
                    'MarkerFaceColor', 'w', 'DisplayName', 'Mean Estimate');
            end
            
        otherwise
            % Generic fallback - just plot state estimate
            [x_est, ~] = tracker_obj.getGaussianEstimate();
            plot(x_est(1), x_est(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
    end
    
    % Plot measurements if provided
    if ~isempty(measurements)
        plot(measurements(1, :), measurements(2, :), '+', 'Color', [1 0.5 0], ...
            'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'Measurements');
    end
    
    % Plot true state if provided
    if ~isempty(true_state)
        plot(true_state(1), true_state(2), 'd', 'Color', 'm', ...
            'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'm', ...
            'DisplayName', 'True State');
    end
    
    % Apply pinned axis limits and formatting
    xlim([-2 2]);
    ylim([0 4]);
    axis square;
    
    xlabel('X (m)', 'Interpreter', 'latex');
    ylabel('Y (m)', 'Interpreter', 'latex');
    title(title_str, 'Interpreter', 'latex');
    
    % Add legend if there are named plot elements
    legend_entries = findobj(gca, 'DisplayName');
    if ~isempty(legend_entries)
        legend('Location', 'best', 'Interpreter', 'latex');
    end
    
    hold off;
end

%% Helper function for ellipse plotting (same as in other plotting functions)
function [X, Y] = calc_gsigma_ellipse_plotpoints(muin, Sigma, g, npoints)
    [R, D, thetalocx] = subf_rotategaussianellipse(Sigma, g);
    
    if Sigma(1) < Sigma(4) 
        a = 1 / sqrt(D(4));
        b = 1 / sqrt(D(1));
    elseif Sigma(1) >= Sigma(4)
        a = 1 / sqrt(D(1)); 
        b = 1 / sqrt(D(4));
    end

    mux = muin(1);
    muy = muin(2);

    if Sigma(2) ~= 0
        [X, Y] = calculateEllipse(mux, muy, a, b, rad2deg(thetalocx), npoints);
    else 
        [X, Y] = calculateEllipse(mux, muy, a, b, 0, npoints);
    end

    function [R, D, thetalocx] = subf_rotategaussianellipse(Sigma, g)
        P = inv(Sigma);
        P = 0.5 * (P + P'); 

        a11 = P(1);
        a12 = P(2);
        a22 = P(4);
        c = -g ^ 2;

        mu = 1 / (-c); 
        m11 = mu * a11;
        m12 = mu * a12;
        m22 = mu * a22;

        lambda1 = 0.5 * (m11 + m22 + sqrt((m11 - m22) .^ 2 + 4 * m12 .^ 2));
        lambda2 = 0.5 * (m11 + m22 - sqrt((m11 - m22) .^ 2 + 4 * m12 .^ 2));
        
        D = diag([lambda2, lambda1]); 
        
        if m11 >= m22
            u11 = lambda1 - m22;
            u12 = m12;
        elseif m11 < m22
            u11 = m12;
            u12 = lambda1 - m11;
        end

        norm1 = sqrt(u11 .^ 2 + u12 .^ 2);
        U1 = ([u11; u12]) / norm1; 
        U2 = [-u12; u11]; 
        R = [U1, U2];

        thetalocx = 0.5 * atan(-2 * a12 / (a22 - a11));
    end
end

function [X, Y] = calculateEllipse(x, y, a, b, angle, steps)
    if nargin < 6, steps = 36; end

    beta = angle * (pi / 180);
    sinbeta = sin(beta);
    cosbeta = cos(beta);

    alpha = linspace(0, 360, steps)' .* (pi / 180);
    sinalpha = sin(alpha);
    cosalpha = cos(alpha);

    X = x + (a * cosalpha * cosbeta - b * sinalpha * sinbeta);
    Y = y + (a * cosalpha * sinbeta + b * sinalpha * cosbeta);

    if nargout == 1, X = [X Y]; end
end
