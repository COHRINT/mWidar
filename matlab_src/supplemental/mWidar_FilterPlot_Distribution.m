function [] = mWidar_FilterPlot_Distribution(Filter, Data, tvec, filter_type, gif_path)

    %%%%%%% mWidar_FilterPlot_Distribution %%%%%%%%%%%%%%%
%{

Given a Filter Struct, Data struct from trajectory data, and filter type, 
function will plot the trajectory over the mWidar image alongside the 
filter-specific distribution visualization.

INPUTS:
Filter - Struct containing all the relevant filter data
Data - Data struct with GT, signal, and measurements
tvec - time vector
filter_type - String: 'KF', 'HMM', or 'HybridPF'
gif_path - (optional) String path to save GIF animation. If empty or not provided, no GIF is saved.

LEFT SUBPLOT: Current mWidar signal with true target, mean estimate, 
              covariance ellipse, and detections
RIGHT SUBPLOT: Filter-specific distribution:
    - KF: 2D Gaussian distribution as heatmap
    - HMM: Full probability distribution as heatmap
    - HybridPF: All particles colored by weights + mean + covariance

%}

    % Validate inputs
    if nargin < 4
        error('Usage: mWidar_FilterPlot_Distribution(Filter, Data, tvec, filter_type, [gif_path])');
    end
    
    % Handle optional gif_path parameter
    if nargin < 5 || isempty(gif_path)
        save_gif = false;
        gif_path = '';
    else
        save_gif = true;
        % Ensure gif_path has .gif extension
        [~, ~, ext] = fileparts(gif_path);
        if ~strcmpi(ext, '.gif')
            gif_path = [gif_path, '.gif'];
        end
    end
    
    valid_types = {'KF', 'HMM', 'HybridPF'};
    if ~ismember(filter_type, valid_types)
        error('filter_type must be one of: %s', strjoin(valid_types, ', '));
    end

    % Unpack data structs
    GT = Data.GT;
    sim_signal = Data.signal;
    y = Data.y; % Filtered measurements (for filter processing)
    
    % Get original measurements and filtering info for visualization
    if isfield(Data, 'y_original') && isfield(Data, 'y_filtered_indices')
        y_original = Data.y_original; % All original measurements
        filtered_indices = Data.y_filtered_indices; % Which ones were kept
        show_filtering = true;
    else
        y_original = y; % Fallback to filtered measurements
        filtered_indices = [];
        show_filtering = false;
    end

    n_k = size(GT, 2); % # of timesteps

    % Define spatial grid (hardcoded as specified)
    Lscene = 4;
    npx = 128;
    xgrid = linspace(-2, 2, npx);
    ygrid = linspace(0, Lscene, npx);
    [pxgrid, pygrid] = meshgrid(xgrid, ygrid);

    %% Process filter data based on type
    switch filter_type
        case 'KF'
            % Unpack Kalman Filter data
            X = zeros(6, n_k); % State History
            P = cell(1, n_k); % State Cov
            
            for k = 1:n_k
                X(:, k) = Filter{k}.x;
                P{k} = Filter{k}.P;
            end
            
        case 'HMM'
            % HMM data is probability distributions
            % Filter{k} should contain the probability distribution
            % We'll extract the mean and covariance for consistency
            X = zeros(2, n_k); % Position only for HMM
            pxyvec = [pxgrid(:), pygrid(:)];
            
            for k = 1:n_k
                prob_dist = Filter{k};
                % Compute mean position from probability distribution
                X(:, k) = sum(pxyvec .* repmat(prob_dist, [1, 2]), 1)';
            end
            
        case 'HybridPF'
            % Particle Filter data
            X = zeros(6, n_k); % State History (mean)
            P = cell(1, n_k); % State Covariance
            particles_hist = cell(1, n_k);
            weights_hist = cell(1, n_k);
            
            for k = 1:n_k
                particles_hist{k} = Filter{k}.particles;
                weights_hist{k} = Filter{k}.weights;
                % Use precomputed Gaussian estimates
                X(:, k) = Filter{k}.x;
                P{k} = Filter{k}.P;
            end
    end

    %% Animation Loop
    
    % Create figure once with larger size for better spacing
    figure(66); clf;
    set(gcf, 'Position', [100, 100, 1400, 600], 'Visible', 'off');
    
    for k = 1:n_k
        clf;
        
        
        %% LEFT SUBPLOT: mWidar Signal with Estimates
        subplot(1, 2, 1); cla; hold on; grid on;
        
        % Plot the mWidar signal as 2D image instead of 3D surf
        imagesc(xgrid, ygrid, sim_signal{k} / (max(max(sim_signal{k}))));
        set(gca, 'YDir', 'normal');
        colormap('parula');
        
        % Plot true target location (2D)
        plot(GT(1, k), GT(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
        
        % Plot detections (2D instead of 3D scatter)
        if show_filtering && ~isempty(y_original{k})
            % Plot all original measurements in light red (rejected)
            scatter(y_original{k}(1, :), y_original{k}(2, :), 30, [1 0.7 0.7], '*');
            
            % Plot filtered (kept) measurements in bright red on top
            if ~isempty(y{k})
                scatter(y{k}(1, :), y{k}(2, :), 50, '*r');
            end
        else
            % Fallback: just plot the measurements we have
            if ~isempty(y{k})
                scatter(y{k}(1, :), y{k}(2, :), 50, '*r');
            end
        end
        
        % Plot mean estimate (2D)
        if strcmp(filter_type, 'HMM')
            plot(X(1, k), X(2, k), 'ms', 'MarkerSize', 12, 'LineWidth', 1.2);
        else
            plot(X(1, k), X(2, k), 'ms', 'MarkerSize', 12, 'LineWidth', 1.2);
        end
        
        % Plot covariance ellipses (2D instead of 3D)
        if strcmp(filter_type, 'KF')
            % Extract position covariance
            innovCov = [P{k}(1, 1) P{k}(1, 2); P{k}(2, 1) P{k}(2, 2)];
            muin = X(1:2, k);
            
            % Plot 1-sigma ellipse (68% confidence)
            [Xellip1, Yellip1] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);
            plot(Xellip1, Yellip1, '--k', 'LineWidth', 2);
            
            % Plot 2-sigma ellipse (95% confidence)
            [Xellip2, Yellip2] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 2, 100);
            plot(Xellip2, Yellip2, ':k', 'LineWidth', 1.5);
            
        elseif strcmp(filter_type, 'HybridPF')
            % Use precomputed position covariance
            innovCov = [P{k}(1, 1) P{k}(1, 2); P{k}(2, 1) P{k}(2, 2)];
            muin = X(1:2, k);
            
            % Plot 1-sigma ellipse (68% confidence)
            [Xellip1, Yellip1] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);
            plot(Xellip1, Yellip1, '--k', 'LineWidth', 2);
            
            % Plot 2-sigma ellipse (95% confidence)
            [Xellip2, Yellip2] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 2, 100);
            plot(Xellip2, Yellip2, ':k', 'LineWidth', 1.5);
        end
        
        xlim([-2 2]);
        ylim([0 4]);
        title(['mWidar Signal @ k=', num2str(k)], 'Interpreter', 'latex');
        xlabel('X (m)', 'Interpreter', 'latex');
        ylabel('Y (m)', 'Interpreter', 'latex');
        axis square;
        
        % Add legend for left subplot
        if show_filtering
            legend('mWidar Signal', 'True Target', 'Rejected Measurements', 'Valid Measurements', 'Filter Estimate', 'Location', 'northeast', 'Interpreter', 'latex');
        else
            legend('mWidar Signal', 'True Target', 'Measurements', 'Filter Estimate', 'Location', 'northeast', 'Interpreter', 'latex');
        end
        
        %% RIGHT SUBPLOT: Filter-Specific Distribution
        subplot(1, 2, 2); cla; hold on; grid on;
        
        switch filter_type
            case 'KF'
                % Create 2D Gaussian distribution from mean and covariance
                innovCov = [P{k}(1, 1) P{k}(1, 2); P{k}(2, 1) P{k}(2, 2)];
                muin = X(1:2, k);
                
                % Create meshgrid for evaluation
                [Xmesh, Ymesh] = meshgrid(xgrid, ygrid);
                
                % Evaluate 2D Gaussian at each grid point
                gaussian_2d = zeros(size(Xmesh));
                for i = 1:numel(Xmesh)
                    point = [Xmesh(i); Ymesh(i)];
                    diff = point - muin;
                    gaussian_2d(i) = exp(-0.5 * diff' * (innovCov \ diff));
                end
                
                % Plot as heatmap
                imagesc(xgrid, ygrid, gaussian_2d);
                set(gca, 'YDir', 'normal');
                colormap('parula');
                colorbar;
                
                % Overlay mean estimate
                h1 = plot(X(1, k), X(2, k), 'wo', 'MarkerSize', 10, 'MarkerFaceColor', 'w', 'LineWidth', 2);
                
                % Plot 1-sigma covariance ellipse
                [Xellip1, Yellip1] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);
                h2 = plot(Xellip1, Yellip1, '--w', 'LineWidth', 2);
                
                % Plot 2-sigma covariance ellipse
                [Xellip2, Yellip2] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 2, 100);
                h4 = plot(Xellip2, Yellip2, ':w', 'LineWidth', 1.5);
                
                % Plot true position for reference
                h3 = plot(GT(1, k), GT(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
                
                title('KF: Gaussian Distribution', 'Interpreter', 'latex');
                
                % Add legend for KF
                legend([h1, h2, h4, h3], 'KF Mean Estimate', '1$\sigma$ Ellipse', '2$\sigma$ Ellipse', 'True Position', 'Location', 'northeast', 'Interpreter', 'latex');
                
            case 'HMM'
                % Plot full probability distribution as heatmap
                prob_dist = Filter{k};
                prob_2d = reshape(prob_dist, [npx, npx]);
                
                imagesc(xgrid, ygrid, prob_2d);
                set(gca, 'YDir', 'normal');
                colormap('parula');
                colorbar;
                
                % Overlay mean estimate
                h1 = plot(X(1, k), X(2, k), 'wo', 'MarkerSize', 8, 'MarkerFaceColor', 'w', 'LineWidth', 2);
                
                % Plot true position for reference
                h2 = plot(GT(1, k), GT(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
                
                title('HMM: Probability Distribution', 'Interpreter', 'latex');
                
                % Add legend for HMM
                legend([h1, h2], 'HMM Mean Estimate', 'True Position', 'Location', 'northeast', 'Interpreter', 'latex');
                
            case 'HybridPF'
                % Plot all particles colored by weights (position only)
                particles = particles_hist{k};
                weights = weights_hist{k};
                
                % Plot particles using weights directly (no normalization)
                h1 = scatter(particles(1, :), particles(2, :), 20, weights, 'filled', 'MarkerFaceAlpha', 0.6);
                colormap('parula');
                colorbar;
                
                % Overlay mean estimate
                h2 = plot(X(1, k), X(2, k), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);
                
                % Plot covariance ellipses using precomputed covariance
                innovCov = [P{k}(1, 1) P{k}(1, 2); P{k}(2, 1) P{k}(2, 2)];
                muin = X(1:2, k);
                
                % Plot 1-sigma ellipse
                [Xellip1, Yellip1] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);
                h3 = plot(Xellip1, Yellip1, '--k', 'LineWidth', 2);
                
                % Plot 2-sigma ellipse
                [Xellip2, Yellip2] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 2, 100);
                h5 = plot(Xellip2, Yellip2, ':k', 'LineWidth', 1.5);
                
                % Plot true position for reference
                h4 = plot(GT(1, k), GT(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
                
                title('Hybrid PF: Particles + Mean + Covariance', 'Interpreter', 'latex');
                
                % Add legend for right subplot
                legend([h1, h2, h3, h5, h4], 'Particles (by weight)', 'PF Mean Estimate', '1$\sigma$ Ellipse', '2$\sigma$ Ellipse', 'True Position', 'Location', 'northeast', 'Interpreter', 'latex');
        end
        
        xlim([-2 2]);
        ylim([0 4]);
        xlabel('X (m)', 'Interpreter', 'latex');
        ylabel('Y (m)', 'Interpreter', 'latex');
        axis square;
        
        %% Overall figure settings
        sgtitle([filter_type, ' Filter @ k=', num2str(k)], 'FontSize', 12, 'Interpreter', 'latex');
        
        % Save frame to GIF if requested
        if save_gif
            try
                frame = getframe(gcf);
                im = frame2im(frame);
                [imind, cm] = rgb2ind(im, 256);
                
                if k == 1
                    imwrite(imind, cm, gif_path, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
                    fprintf('Started saving GIF: %s\n', gif_path);
                else
                    imwrite(imind, cm, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
                end
                fprintf('Frame %d/%d saved\n', k, n_k);
            catch ME
                warning('Failed to capture frame %d for GIF: %s', k, ME.message);
                % Continue with animation even if GIF capture fails
            end
        end
        
        pause(0.1);
    end
    
    % Print completion message if GIF was saved
    if save_gif
        fprintf('GIF animation saved successfully: %s\n', gif_path);
    end

end

%% Helper Functions (copied from original mWidar_FilterPlot.m)

%calc_gsigma_ellipse_plotpoints.m
%%Does all the computations needed to plot 2D Gaussian ellipses properly.
%%Takes in the Gaussian mean (muin), cov matrix (Sigma, positive definite),
%%g-sigma value, and number of points to generate for plotting.
function [X, Y] = calc_gsigma_ellipse_plotpoints(muin, Sigma, g, npoints)
    %%align the Gaussian along its principal axes
    [R, D, thetalocx] = subf_rotategaussianellipse(Sigma, g);
    %%pick semi-major and semi-minor "axes" (lengths)
    if Sigma(1) < Sigma(4) %use if sigxx<sigyy
        a = 1 / sqrt(D(4));
        b = 1 / sqrt(D(1));
    elseif Sigma(1) >= Sigma(4)
        a = 1 / sqrt(D(1)); %use if sigxx<sigyy
        b = 1 / sqrt(D(4));
    end

    %%calculate points of ellipse:
    mux = muin(1);
    muy = muin(2);

    if Sigma(2) ~= 0
        [X Y] = calculateEllipse(mux, muy, a, b, rad2deg(thetalocx), npoints);
    else %if there are no off-diagonal terms, then no rotation needed
        [X Y] = calculateEllipse(mux, muy, a, b, 0, npoints);
    end

    %%call the rotate gaussian ellipse thingy as a local subfunction (there is
    %%also a separate fxn m-file for this, too)
    function [R, D, thetalocx] = subf_rotategaussianellipse(Sigma, g)
        P = inv(Sigma);
        P = 0.5 * (P + P'); %symmetrize

        a11 = P(1);
        a12 = P(2);
        a22 = P(4);
        c = -g ^ 2;

        mu = 1 / (-c); %can define mu this way since b1=0,b2=0 b/c we are mean centered
        m11 = mu * a11;
        m12 = mu * a12;
        m22 = mu * a22;

        %%solve for eigenstuff
        lambda1 = 0.5 * (m11 + m22 + sqrt((m11 - m22) .^ 2 + 4 * m12 .^ 2));
        lambda2 = 0.5 * (m11 + m22 - sqrt((m11 - m22) .^ 2 + 4 * m12 .^ 2));
        % % b = 1/sqrt(lambda1); %semi-minor axis for standard ellipse (length)
        % % a = 1/sqrt(lambda2); %semi-major axis for standard ellipse (length)
        D = diag([lambda2, lambda1]); %elements are 1/a^2 and 1/b^2, respectively
        %%Choose the mahor axis direction of the ellipse
        if m11 >= m22
            u11 = lambda1 - m22;
            u12 = m12;
        elseif m11 < m22
            u11 = m12;
            u12 = lambda1 - m11;
        end

        norm1 = sqrt(u11 .^ 2 + u12 .^ 2);
        U1 = ([u11; u12]) / norm1; %major axis direction
        U2 = [-u12; u11]; %minor axis direction
        R = [U1, U2];
        % if sum(sum(isnan(R)))>0
        %    R = eye(2); %default hack for now in case of degenerate stuff
        % end

        thetalocx = 0.5 * atan(-2 * a12 / (a22 - a11));
        % if isnan(thetalocx)
        %     thetalocx = 0; %default hack for now...
        % end

    end

end

function [X Y] = calculateEllipse(x, y, a, b, angle, steps)
    %# This functions returns points to draw an ellipse
    %#
    %#  @param x     X coordinate
    %#  @param y     Y coordinate
    %#  @param a     Semimajor axis
    %#  @param b     Semiminor axis
    %#  @param angle Angle of the ellipse (in degrees)
    %#

    %error(nargchk(5, 6, nargin));
    narginchk(5, 6);
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
