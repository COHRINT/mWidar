function [] = mWidar_GenerateAnimationFrame(performance, Data, current_k, filter_type, validation_sigma, gif_path)
    % MWIDAR_GENERATEANIMATIONFRAME Generate single animation frame for tuning
    %
    % SYNTAX:
    %   mWidar_GenerateAnimationFrame(performance, Data, current_k, filter_type, validation_sigma, gif_path)
    %
    % INPUTS:
    %   performance      - Cell array of performance metrics up to current timestep
    %   Data             - Data struct with GT, signal, and measurements
    %   current_k        - Current timestep
    %   filter_type      - String: 'KF', 'HMM', or 'HybridPF'
    %   validation_sigma - Validation sigma for ellipses
    %   gif_path         - Path to save GIF animation
    %
    % DESCRIPTION:
    %   Generates a single frame for animation during tuning experiments.
    %   More efficient than calling the full distribution plotting function.
    
    % Unpack data
    GT = Data.GT;
    sim_signal = Data.signal;
    y = Data.y;
    
    % Define spatial grid
    Lscene = 4;
    npx = 128;
    xgrid = linspace(-2, 2, npx);
    ygrid = linspace(0, Lscene, npx);
    
    % Extract current state
    if isfield(performance{current_k}, 'x')
        x_current = performance{current_k}.x;
    else
        return; % Skip if no state available
    end
    
    % Clear and setup figure
    clf;
    
    %% LEFT SUBPLOT: mWidar Signal with Estimates
    subplot(1, 2, 1); cla; hold on; grid on;
    
    % Set light gray background
    set(gca, 'Color', [0.94, 0.94, 0.94]);
    
    % Plot the mWidar signal as 2D image
    if ~isempty(sim_signal{current_k})
        imagesc(xgrid, ygrid, sim_signal{current_k} / (max(max(sim_signal{current_k}))));
        set(gca, 'YDir', 'normal');
        colormap('parula');
    end
    
    % Plot true target location (magenta diamond)
    plot(GT(1, current_k), GT(2, current_k), 'd', 'Color', 'm', 'MarkerSize', 8, 'LineWidth', 2);
    
    % Plot measurements
    if isfield(performance{current_k}, 'measurements_original') && ~isempty(performance{current_k}.measurements_original)
        scatter(performance{current_k}.measurements_original(1, :), performance{current_k}.measurements_original(2, :), ...
            30, [1 0.5 0], '+', 'LineWidth', 2);
    end
    
    if isfield(performance{current_k}, 'measurements_used') && ~isempty(performance{current_k}.measurements_used)
        scatter(performance{current_k}.measurements_used(1, :), performance{current_k}.measurements_used(2, :), ...
            50, 'r', '+', 'LineWidth', 2);
    end
    
    % Plot mean estimate (red circle)
    plot(x_current(1), x_current(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
    
    % Plot covariance ellipses
    if isfield(performance{current_k}, 'P') && ~isempty(performance{current_k}.P)
        try
            innovCov = performance{current_k}.P(1:2, 1:2);
            muin = x_current(1:2);
            
            % Plot 1-sigma ellipse
            [Xellip1, Yellip1] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);
            plot(Xellip1, Yellip1, 'k-', 'LineWidth', 2);
            
            % Plot validation region ellipse
            [Xellip2, Yellip2] = calc_gsigma_ellipse_plotpoints(muin, innovCov, validation_sigma, 100);
            plot(Xellip2, Yellip2, 'k:', 'LineWidth', 2);
        catch
            % Skip if ellipse calculation fails
        end
    end
    
    xlim([-2 2]);
    ylim([0 4]);
    
    % Count measurements for title
    n_measurements = size(y{current_k}, 2);
    title(sprintf('%d Measurements', n_measurements), 'Interpreter', 'latex');
    
    xlabel('X (m)', 'Interpreter', 'latex');
    ylabel('Y (m)', 'Interpreter', 'latex');
    axis square;
    
    %% RIGHT SUBPLOT: Filter-Specific Distribution
    subplot(1, 2, 2); cla; hold on; grid on;
    
    switch filter_type
        case 'KF'
            % Create 2D Gaussian distribution
            if isfield(performance{current_k}, 'P') && ~isempty(performance{current_k}.P)
                innovCov = performance{current_k}.P(1:2, 1:2);
                muin = x_current(1:2);
                
                % Create meshgrid for evaluation
                [Xmesh, Ymesh] = meshgrid(xgrid, ygrid);
                
                % Evaluate 2D Gaussian
                gaussian_2d = zeros(size(Xmesh));
                for idx = 1:numel(Xmesh)
                    point = [Xmesh(idx); Ymesh(idx)];
                    diff = point - muin;
                    gaussian_2d(idx) = exp(-0.5 * diff' * (innovCov \ diff));
                end
                
                % Plot as heatmap
                imagesc(xgrid, ygrid, gaussian_2d);
                set(gca, 'YDir', 'normal');
                colormap('parula');
                colorbar;
                
                % Overlay estimates
                plot(x_current(1), x_current(2), 'wo', 'MarkerSize', 10, 'MarkerFaceColor', 'w', 'LineWidth', 2);
                plot(GT(1, current_k), GT(2, current_k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
                
                title(sprintf('KF: Gaussian (%d Meas)', n_measurements), 'Interpreter', 'latex');
            end
            
        case 'HybridPF'
            % Plot particles
            if isfield(performance{current_k}, 'particles') && isfield(performance{current_k}, 'weights')
                particles = performance{current_k}.particles;
                weights = performance{current_k}.weights;
                
                % Set light gray background
                set(gca, 'Color', [0.94, 0.94, 0.94]);
                
                % Plot particles colored by weights
                scatter(particles(1, :), particles(2, :), 20, weights, 'filled', 'MarkerFaceAlpha', 0.6);
                colorbar;
                
                % Overlay estimates
                plot(x_current(1), x_current(2), 'ro', 'MarkerSize', 10, 'LineWidth', 3);
                plot(GT(1, current_k), GT(2, current_k), 'd', 'Color', 'm', 'MarkerSize', 8, 'LineWidth', 2);
                
                title(sprintf('Particles (%d Meas)', n_measurements), 'Interpreter', 'latex');
            end
            
        case 'HMM'
            % Plot probability distribution
            if isfield(performance{current_k}, 'posterior_prob')
                prob_dist = performance{current_k}.posterior_prob;
                prob_2d = reshape(prob_dist, [npx, npx]);
                
                imagesc(xgrid, ygrid, prob_2d);
                set(gca, 'YDir', 'normal');
                colormap('parula');
                colorbar;
                
                % Overlay estimates
                plot(x_current(1), x_current(2), 'wo', 'MarkerSize', 8, 'MarkerFaceColor', 'w', 'LineWidth', 2);
                plot(GT(1, current_k), GT(2, current_k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
                
                title(sprintf('HMM: Probability (%d Meas)', n_measurements), 'Interpreter', 'latex');
            end
    end
    
    xlim([-2 2]);
    ylim([0 4]);
    xlabel('X (m)', 'Interpreter', 'latex');
    ylabel('Y (m)', 'Interpreter', 'latex');
    axis square;
    
    % Overall title
    sgtitle([filter_type, ' Filter @ k=', num2str(current_k)], 'FontSize', 12, 'Interpreter', 'latex');
    
    % Save frame to GIF
    try
        frame = getframe(gcf);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        
        if current_k == 2  % First actual timestep (k=1 is initialization)
            imwrite(imind, cm, gif_path, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
            fprintf('Started saving animation GIF: %s\n', gif_path);
        else
            imwrite(imind, cm, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
        end
        
        if mod(current_k, 10) == 0  % Print progress every 10 steps
            fprintf('Animation frame %d saved\n', current_k);
        end
    catch ME
        warning('Failed to save animation frame %d: %s', current_k, ME.message);
    end
end

%% Helper function for ellipse plotting
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
