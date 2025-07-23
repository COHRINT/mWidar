function [] = mWidar_FilterPlot_Distribution(Filter, Data, tvec, filter_type)

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

LEFT SUBPLOT: Current mWidar signal with true target, mean estimate, 
              covariance ellipse, and detections
RIGHT SUBPLOT: Filter-specific distribution:
    - KF: Mean as dot + covariance ellipse (position only)
    - HMM: Full probability distribution as heatmap
    - HybridPF: All particles colored by weights (position only)

%}

    % Validate inputs
    if nargin < 4
        error('Usage: mWidar_FilterPlot_Distribution(Filter, Data, tvec, filter_type)');
    end
    
    valid_types = {'KF', 'HMM', 'HybridPF'};
    if ~ismember(filter_type, valid_types)
        error('filter_type must be one of: %s', strjoin(valid_types, ', '));
    end

    % Unpack data structs
    GT = Data.GT;
    sim_signal = Data.signal;
    y = Data.y;

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
            particles_hist = cell(1, n_k);
            weights_hist = cell(1, n_k);
            
            for k = 1:n_k
                particles_hist{k} = Filter{k}.particles;
                weights_hist{k} = Filter{k}.weights;
                % Compute weighted mean
                X(:, k) = particles_hist{k} * weights_hist{k}';
            end
    end

    %% Animation Loop
    for k = 1:n_k
        figure(66); clf;
        
        %% LEFT SUBPLOT: mWidar Signal with Estimates
        subplot(1, 2, 1); cla; hold on; grid on;
        
        % Plot the mWidar signal
        surf(pxgrid, pygrid, sim_signal{k} / (max(max(sim_signal{k}))), 'EdgeColor', 'none');
        
        % Plot true target location
        plot3(GT(1, k), GT(2, k), ones(1, 1), 'mx', 'MarkerSize', 10, 'LineWidth', 10);
        
        % Plot detections
        if ~isempty(y{k})
            scatter3(y{k}(1, :), y{k}(2, :), ones(length(y{k}(1, :)), 1), 50, '*r');
        end
        
        % Plot mean estimate
        if strcmp(filter_type, 'HMM')
            plot3(X(1, k), X(2, k), ones(1, 1), 'ms', 'MarkerSize', 12, 'LineWidth', 1.2);
        else
            plot3(X(1, k), X(2, k), ones(1, 1), 'ms', 'MarkerSize', 12, 'LineWidth', 1.2);
        end
        
        % Plot covariance ellipse (for KF and HybridPF)
        if strcmp(filter_type, 'KF')
            % Extract position covariance
            innovCov = [P{k}(1, 1) P{k}(1, 2); P{k}(2, 1) P{k}(2, 2)];
            muin = X(1:2, k);
            [Xellip, Yellip] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);
            plot3(Xellip, Yellip, ones(length(Xellip), 1), '--k', 'LineWidth', 2);
        elseif strcmp(filter_type, 'HybridPF')
            % Compute empirical covariance from particles
            particles = particles_hist{k};
            weights = weights_hist{k};
            mean_pos = X(1:2, k);
            
            % Weighted covariance calculation
            pos_particles = particles(1:2, :);
            diff_particles = pos_particles - mean_pos;
            empirical_cov = (diff_particles .* weights) * diff_particles' / sum(weights);
            
            [Xellip, Yellip] = calc_gsigma_ellipse_plotpoints(mean_pos, empirical_cov, 1, 100);
            plot3(Xellip, Yellip, ones(length(Xellip), 1), '--k', 'LineWidth', 2);
        end
        
        xlim([-2 2]);
        ylim([0 4]);
        title(['mWidar Signal @ k=', num2str(k)]);
        xlabel('X (m)');
        ylabel('Y (m)');
        axis square;
        view(2);
        
        %% RIGHT SUBPLOT: Filter-Specific Distribution
        subplot(1, 2, 2); cla; hold on; grid on;
        
        switch filter_type
            case 'KF'
                % Plot mean as dot and covariance ellipse (position only)
                plot(X(1, k), X(2, k), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
                
                % Plot covariance ellipse
                innovCov = [P{k}(1, 1) P{k}(1, 2); P{k}(2, 1) P{k}(2, 2)];
                muin = X(1:2, k);
                [Xellip, Yellip] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);
                plot(Xellip, Yellip, '--k', 'LineWidth', 2);
                
                % Plot true position for reference
                plot(GT(1, k), GT(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
                
                title('KF: Mean + Covariance');
                
            case 'HMM'
                % Plot full probability distribution as heatmap
                prob_dist = Filter{k};
                prob_2d = reshape(prob_dist, [npx, npx]);
                
                imagesc(xgrid, ygrid, prob_2d);
                set(gca, 'YDir', 'normal');
                colormap('hot');
                colorbar;
                
                % Overlay mean estimate
                plot(X(1, k), X(2, k), 'wo', 'MarkerSize', 8, 'MarkerFaceColor', 'w', 'LineWidth', 2);
                
                % Plot true position for reference
                plot(GT(1, k), GT(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
                
                title('HMM: Probability Distribution');
                
            case 'HybridPF'
                % Plot all particles colored by weights (position only)
                particles = particles_hist{k};
                weights = weights_hist{k};
                
                % Normalize weights for better color visualization
                weights_norm = weights / max(weights);
                
                scatter(particles(1, :), particles(2, :), 20, weights_norm, 'filled', 'MarkerFaceAlpha', 0.6);
                colormap('parula');
                colorbar;
                
                % Overlay mean estimate
                plot(X(1, k), X(2, k), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'LineWidth', 2);
                
                % Plot true position for reference
                plot(GT(1, k), GT(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
                
                title('Hybrid PF: Particles (colored by weight)');
        end
        
        xlim([-2 2]);
        ylim([0 4]);
        xlabel('X (m)');
        ylabel('Y (m)');
        axis square;
        
        %% Overall figure settings
        sgtitle([filter_type, ' Filter @ k=', num2str(k)], 'FontSize', 14);
        
        pause(0.1);
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
