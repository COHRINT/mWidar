function [] = mWidar_FilterPlot_Interactive(Filter, Data, tvec, filter_type, gif_path)

    %%%%%%% mWidar_FilterPlot_Interactive %%%%%%%%%%%%%%%
%{

Interactive version of mWidar_FilterPlot_Distribution with slider control.

Given a Filter Struct, Data struct from trajectory data, and filter type, 
function will plot the trajectory over the mWidar image alongside the 
filter-specific distribution visualization with interactive time controls.

INPUTS:
Filter - Struct containing all the relevant filter data
Data - Data struct with GT, signal, and measurements
tvec - time vector
filter_type - String: 'KF', 'HMM', or 'HybridPF'
gif_path - (optional) String path to save GIF animation. If empty or not provided, no GIF is saved.

FEATURES:
- Time slider to manually navigate through timesteps
- Play/Pause button for automatic animation
- Speed control for animation playback

LEFT SUBPLOT: Current mWidar signal with true target, mean estimate, 
              covariance ellipse, and detections
RIGHT SUBPLOT: Filter-specific distribution:
    - KF: 2D Gaussian distribution as heatmap
    - HMM: Full probability distribution as heatmap
    - HybridPF: All particles colored by weights + mean + covariance

%}

    % Validate inputs
    if nargin < 4
        error('Usage: mWidar_FilterPlot_Interactive(Filter, Data, tvec, filter_type, [gif_path])');
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

    %% Create Interactive GUI
    fig = figure(67); 
    clf(fig);
    set(fig, 'Position', [100, 100, 1400, 600], 'Name', [char(filter_type), ' Interactive Visualization']);
    
    % Initialize GUI state
    gui_data = struct();
    gui_data.current_k = 1;
    gui_data.is_playing = false;
    gui_data.animation_speed = 0.1; % seconds between frames
    gui_data.timer_obj = [];
    
    % Create UI controls at the bottom
    control_panel = uipanel(fig, 'Position', [0.05, 0.02, 0.9, 0.08], 'Title', 'Time Controls');
    
    % Time slider
    gui_data.time_slider = uicontrol(control_panel, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.1, 0.4, 0.6, 0.3], ...
        'Min', 1, 'Max', n_k, 'Value', 1, 'SliderStep', [1/(n_k-1), 10/(n_k-1)], ...
        'Callback', @slider_callback);
    
    % Time display
    gui_data.time_text = uicontrol(control_panel, 'Style', 'text', ...
        'Units', 'normalized', 'Position', [0.72, 0.4, 0.08, 0.3], ...
        'String', sprintf('k=%d/%d', 1, n_k), 'FontSize', 10);
    
    % Play/Pause button
    gui_data.play_button = uicontrol(control_panel, 'Style', 'pushbutton', ...
        'Units', 'normalized', 'Position', [0.82, 0.3, 0.06, 0.5], ...
        'String', 'Play', 'FontSize', 10, 'Callback', @play_callback);
    
    % Speed control
    uicontrol(control_panel, 'Style', 'text', ...
        'Units', 'normalized', 'Position', [0.01, 0.6, 0.08, 0.3], ...
        'String', 'Speed:', 'FontSize', 9);
    
    gui_data.speed_slider = uicontrol(control_panel, 'Style', 'slider', ...
        'Units', 'normalized', 'Position', [0.01, 0.1, 0.08, 0.3], ...
        'Min', 0.01, 'Max', 1.0, 'Value', 0.1, ...
        'Callback', @speed_callback);
    
    % Reset button
    uicontrol(control_panel, 'Style', 'pushbutton', ...
        'Units', 'normalized', 'Position', [0.9, 0.3, 0.08, 0.5], ...
        'String', 'Reset', 'FontSize', 10, 'Callback', @reset_callback);
    
    % Store GUI data in figure
    guidata(fig, gui_data);
    
    % Create initial plot
    update_plot(1);
    
    %% Callback Functions
    function slider_callback(src, ~)
        gui_data = guidata(fig);
        gui_data.current_k = round(get(src, 'Value'));
        guidata(fig, gui_data);
        update_plot(gui_data.current_k);
        set(gui_data.time_text, 'String', sprintf('k=%d/%d', gui_data.current_k, n_k));
    end
    
    function play_callback(~, ~)
        gui_data = guidata(fig);
        
        if ~gui_data.is_playing
            % Start playing
            gui_data.is_playing = true;
            set(gui_data.play_button, 'String', 'Pause');
            
            % Create timer for animation
            gui_data.timer_obj = timer('TimerFcn', @timer_callback, ...
                'Period', gui_data.animation_speed, 'ExecutionMode', 'fixedRate');
            start(gui_data.timer_obj);
        else
            % Stop playing
            stop_animation();
        end
        
        guidata(fig, gui_data);
    end
    
    function timer_callback(~, ~)
        gui_data = guidata(fig);
        
        if gui_data.current_k < n_k
            gui_data.current_k = gui_data.current_k + 1;
        else
            % Reached end, stop animation
            stop_animation();
            return;
        end
        
        % Update slider and plot
        set(gui_data.time_slider, 'Value', gui_data.current_k);
        set(gui_data.time_text, 'String', sprintf('k=%d/%d', gui_data.current_k, n_k));
        update_plot(gui_data.current_k);
        
        guidata(fig, gui_data);
    end
    
    function stop_animation()
        gui_data = guidata(fig);
        gui_data.is_playing = false;
        set(gui_data.play_button, 'String', 'Play');
        
        if ~isempty(gui_data.timer_obj) && isvalid(gui_data.timer_obj)
            stop(gui_data.timer_obj);
            delete(gui_data.timer_obj);
            gui_data.timer_obj = [];
        end
        
        guidata(fig, gui_data);
    end
    
    function speed_callback(src, ~)
        gui_data = guidata(fig);
        gui_data.animation_speed = get(src, 'Value');
        
        % Update timer if playing
        if gui_data.is_playing && ~isempty(gui_data.timer_obj) && isvalid(gui_data.timer_obj)
            stop(gui_data.timer_obj);
            set(gui_data.timer_obj, 'Period', gui_data.animation_speed);
            start(gui_data.timer_obj);
        end
        
        guidata(fig, gui_data);
    end
    
    function reset_callback(~, ~)
        gui_data = guidata(fig);
        
        % Stop animation if playing
        if gui_data.is_playing
            stop_animation();
        end
        
        % Reset to first frame
        gui_data.current_k = 1;
        set(gui_data.time_slider, 'Value', 1);
        set(gui_data.time_text, 'String', sprintf('k=%d/%d', 1, n_k));
        update_plot(1);
        
        guidata(fig, gui_data);
    end
    
    %% Main Plotting Function
    function update_plot(k)
        
        %% LEFT SUBPLOT: mWidar Signal with Estimates
        subplot(1, 2, 1); cla; hold on; grid on;
        
        % Plot the mWidar signal
        surf(pxgrid, pygrid, sim_signal{k} / (max(max(sim_signal{k}))), 'EdgeColor', 'none');
        
        % Plot true target location
        plot3(GT(1, k), GT(2, k), ones(1, 1), 'mx', 'MarkerSize', 10, 'LineWidth', 10);
        
        % Plot detections (show filtered vs unfiltered in different colors)
        if show_filtering && ~isempty(y_original{k})
            % Plot all original measurements in light red (rejected)
            scatter3(y_original{k}(1, :), y_original{k}(2, :), ones(size(y_original{k}, 2), 1), 30, [1 0.7 0.7], '*');
            
            % Plot filtered (kept) measurements in bright red on top
            if ~isempty(y{k})
                scatter3(y{k}(1, :), y{k}(2, :), ones(size(y{k}, 2), 1), 50, '*r');
            end
        else
            % Fallback: just plot the measurements we have
            if ~isempty(y{k})
                scatter3(y{k}(1, :), y{k}(2, :), ones(length(y{k}(1, :)), 1), 50, '*r');
            end
        end
        
        % Plot mean estimate
        if strcmp(filter_type, 'HMM')
            plot3(X(1, k), X(2, k), ones(1, 1), 'ms', 'MarkerSize', 12, 'LineWidth', 1.2);
        else
            plot3(X(1, k), X(2, k), ones(1, 1), 'ms', 'MarkerSize', 12, 'LineWidth', 1.2);
        end
        
        % Plot covariance ellipses (for KF and HybridPF)
        if strcmp(filter_type, 'KF')
            % Extract position covariance
            innovCov = [P{k}(1, 1) P{k}(1, 2); P{k}(2, 1) P{k}(2, 2)];
            muin = X(1:2, k);
            
            % Plot 1-sigma ellipse
            [Xellip1, Yellip1] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);
            plot3(Xellip1, Yellip1, ones(length(Xellip1), 1), '--k', 'LineWidth', 2);
            
            % Plot 2-sigma ellipse
            [Xellip2, Yellip2] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 2, 100);
            plot3(Xellip2, Yellip2, ones(length(Xellip2), 1), ':k', 'LineWidth', 1.5);
            
        elseif strcmp(filter_type, 'HybridPF')
            % Use precomputed position covariance
            innovCov = [P{k}(1, 1) P{k}(1, 2); P{k}(2, 1) P{k}(2, 2)];
            muin = X(1:2, k);
            
            % Plot 1-sigma ellipse
            [Xellip1, Yellip1] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);
            plot3(Xellip1, Yellip1, ones(length(Xellip1), 1), '--k', 'LineWidth', 2);
            
            % Plot 2-sigma ellipse
            [Xellip2, Yellip2] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 2, 100);
            plot3(Xellip2, Yellip2, ones(length(Xellip2), 1), ':k', 'LineWidth', 1.5);
        end
        
        xlim([-2 2]);
        ylim([0 4]);
        title(['mWidar Signal @ k=', num2str(k)], 'Interpreter', 'latex');
        xlabel('X (m)', 'Interpreter', 'latex');
        ylabel('Y (m)', 'Interpreter', 'latex');
        axis square;
        view(2);
        
        % Add legend for left subplot
        if show_filtering
            legend('mWidar Signal', 'True Target', 'Rejected Measurements', 'Valid Measurements', 'Filter Estimate', '1$\sigma$ Ellipse', '2$\sigma$ Ellipse', 'Location', 'northeast', 'Interpreter', 'latex');
        else
            legend('mWidar Signal', 'True Target', 'Measurements', 'Filter Estimate', '1$\sigma$ Ellipse', '2$\sigma$ Ellipse', 'Location', 'northeast', 'Interpreter', 'latex');
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
                colormap('hot');
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
                colormap('hot');
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
                
                % Auto-zoom to show all particles while keeping square aspect ratio
                x_particles = particles(1, :);
                y_particles = particles(2, :);
                
                % Get particle bounds with small margin
                margin = 0.1; % 10cm margin
                x_min = min(x_particles) - margin;
                x_max = max(x_particles) + margin;
                y_min = min(y_particles) - margin;
                y_max = max(y_particles) + margin;
                
                % Calculate ranges
                x_range = x_max - x_min;
                y_range = y_max - y_min;
                
                % Make square by expanding the smaller range
                if x_range > y_range
                    % Expand y range
                    y_center = (y_min + y_max) / 2;
                    y_min = y_center - x_range / 2;
                    y_max = y_center + x_range / 2;
                else
                    % Expand x range
                    x_center = (x_min + x_max) / 2;
                    x_min = x_center - y_range / 2;
                    x_max = x_center + y_range / 2;
                end
                
                xlim([x_min, x_max]);
                ylim([y_min, y_max]);
        end
        
        % Set standard axis limits for KF and HMM (HybridPF sets its own limits above)
        if ~strcmp(filter_type, 'HybridPF')
            xlim([-2 2]);
            ylim([0 4]);
        end
        xlabel('X (m)', 'Interpreter', 'latex');
        ylabel('Y (m)', 'Interpreter', 'latex');
        axis square;
        
        %% Overall figure settings
        sgtitle([char(filter_type), ' Filter @ k=', num2str(k), ' (Interactive Mode)'], 'FontSize', 14, 'Interpreter', 'latex');
        
        % Adjust subplot positions for better spacing
        h1 = subplot(1, 2, 1);
        pos1 = get(h1, 'Position');
        h2 = subplot(1, 2, 2);
        pos2 = get(h2, 'Position');
        
        % Ensure tight layout with proper margins
        set(gcf, 'Units', 'normalized');
        tight_layout_margin = 0.02;
        set(h1, 'Position', [tight_layout_margin, 0.2, 0.42, 0.65]);
        set(h2, 'Position', [0.52, 0.2, 0.42, 0.65]);
        
        % Save frame to GIF if requested and currently playing
        gui_data = guidata(fig);
        if save_gif && gui_data.is_playing
            frame = getframe(gcf);
            im = frame2im(frame);
            [imind, cm] = rgb2ind(im, 256);
            
            if k == 1
                imwrite(imind, cm, gif_path, 'gif', 'Loopcount', inf, 'DelayTime', gui_data.animation_speed);
                fprintf('Started saving GIF: %s\n', gif_path);
            else
                imwrite(imind, cm, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', gui_data.animation_speed);
            end
        end
        
        drawnow;
    end

    % Clean up timer when figure is closed
    set(fig, 'CloseRequestFcn', @close_figure);
    
    function close_figure(~, ~)
        gui_data = guidata(fig);
        if ~isempty(gui_data.timer_obj) && isvalid(gui_data.timer_obj)
            stop(gui_data.timer_obj);
            delete(gui_data.timer_obj);
        end
        delete(fig);
    end
    
    % Wait for user interaction (figure remains open until closed)
    fprintf('Interactive visualization is ready!\n');
    fprintf('Use the slider to navigate timesteps, or click Play for animation.\n');
    fprintf('Close the figure window when finished.\n');
    
    % Keep figure open until user closes it
    waitfor(fig);

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
