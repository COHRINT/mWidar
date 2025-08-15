function [] = mWidar_FilterPlot_multiObj_Distribution(Filter, Data, tvec, gif_path)

    %%%%%%% mWidar_FilterPlot_multiObj_Distribution %%%%%%%%%%%%%%%
%{

Extension of mWidar_FilterPlot_Distribution to handle multiple objects in
scene

Given a Filter Struct, Data struct from trajectory data, and filter type, 
function will plot the trajectory over the mWidar image alongside the 
filter-specific distribution visualization. 

Currently, the only distribution visialization will be a gaussian for
JPDAF, funciton will need to be modified to support future work

t -> number of targets
k -> number of timesteps

INPUTS:
Filter - Struct containing all the relevant filter data (Stored as a t x k
cell array.
Data - Data struct with GT, signal, and measurements
tvec - time vector
gif_path - (optional) String path to save GIF animation. If empty or not provided, no GIF is saved.

LEFT SUBPLOT: Current mWidar signal with true target

RIGHT SUBPLOTS: t number of plots to the right of the mWidar signal for
each targets gaussian distribution as a heatmap.

%}

    % Validate inputs
    if nargin < 3
        error('Usage: mWidar_FilterPlot_Distribution(Filter, Data, tvec, [gif_path])');
    end

     % Handle optional gif_path parameter
    if nargin < 4 || isempty(gif_path)
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
    
    % Unpack Data Struct
    GT = Data.GT; % Stored as a cell array, 1 x t, where each cell is 6 x k
    sim_signal = Data.signal;
    y = Data.y; % All measurments, no gating considered for plotting
    
    n_t = size(GT, 2); % # of objects
    n_k = size(GT{1}, 2); % # of timesteps

    % Define spatial grid (hardcoded as specified)
    Lscene = 4;
    npx = 128;
    xgrid = linspace(-2, 2, npx);
    ygrid = linspace(0, Lscene, npx);
    [pxgrid, pygrid] = meshgrid(xgrid, ygrid);
    
    % Unpack Filter Struct
    X = cell(n_t, n_k); % State History
    P = cell(n_t, n_k); % State Cov

    for t = 1:n_t
        for k = 1:n_k
            X{t,k} = Filter{k}.x(:,t);
            P{t,k} = Filter{k}.P{t};
        end
    end

    %% Animation Loop
    
    % Create figure once with larger size for better spacing
    figure(66); clf;
    set(gcf, 'Position', [100, 100, 1400, 600], 'Visible', 'off');

    for k = 1:n_k
        clf;
        
        %% LEFT SUBPLOT: mWidar Signal with detections and true Obj

        subplot(1, n_t+1, 1); cla; hold on; grid on;

        % Plot the mWidar signal as 2D image instead of 3D surf
        imagesc(xgrid, ygrid, sim_signal{k} / (max(max(sim_signal{k}))));
        set(gca, 'YDir', 'normal');
        colormap('parula');

        % Plot true target location (2D) & estimated target location (no
        % ellipse)
        for t = 1:n_t
            plot(GT{t}(1, k), GT{t}(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
            plot(X{t,k}(1), X{t,k}(2), 'ms', 'MarkerSize', 10, 'LineWidth', 3);
        end
        % Plot detections
        scatter(y{k}(1, :), y{k}(2, :), 50, '*r');
        
        % Format
        xlim([-2 2]);
        ylim([0 4]);
        title(['mWidar Signal @ k=', num2str(k)], 'Interpreter', 'latex');
        xlabel('X (m)', 'Interpreter', 'latex');
        ylabel('Y (m)', 'Interpreter', 'latex');
        axis square;

        %% RIGHT SUBPLOT(s): Each objects individual distribution
        for t = 1:n_t
            subplot(1, n_t+1, t+1); cla; hold on; grid on;

            % Create 2D Gaussian distribution from mean and covariance
            innovCov = [P{t,k}(1, 1) P{t,k}(1, 2); P{t,k}(2, 1) P{t,k}(2, 2)];
            muin = X{t,k}(1:2);
            
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
            h1 = plot(X{t,k}(1), X{t,k}(2), 'wo', 'MarkerSize', 10, 'MarkerFaceColor', 'w', 'LineWidth', 2);
            
            % Plot 1-sigma covariance ellipse
            [Xellip1, Yellip1] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);
            h2 = plot(Xellip1, Yellip1, '--w', 'LineWidth', 2);
            
            % Plot 2-sigma covariance ellipse
            [Xellip2, Yellip2] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 2, 100);
            h4 = plot(Xellip2, Yellip2, ':w', 'LineWidth', 1.5);
            
            % Plot true position for reference
            h3 = plot(GT{t}(1, k), GT{t}(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
            
            title('KF: Gaussian Distribution', 'Interpreter', 'latex');
            xlim([-2 2]);
            ylim([0 4]);
            legend([h1, h2, h4, h3], 'KF Mean Estimate', '1$\sigma$ Ellipse', '2$\sigma$ Ellipse', 'True Position', 'Location', 'northeast', 'Interpreter', 'latex');


        end

        %% Overall figure settings
        sgtitle(['Filter @ k=', num2str(k)], 'FontSize', 12, 'Interpreter', 'latex');
        
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
