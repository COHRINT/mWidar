function [] = mWidar_FilterPlot_multiObj_RBPF(Filter, Data, tvec, gif_path)

    %%%%%%% mWidar_FilterPlot_multiObj_RBPF %%%%%%%%%%%%%%%
%{

Visualization for RBPF multi-object tracking showing particle state estimates.

At each timestep, plots:
- LEFT: mWidar signal with ground truth, measurements, and mean estimate
- RIGHT: Scatter plot of all particle state estimates (position from all KFs in all particles)

This allows visualization of how particle states evolve over time, showing the
distribution of beliefs about target positions.

INPUTS:
Filter - Cell array {1 x k} where each Filter{k} contains RBPF output with 'particles' field
Data - Data struct with GT, signal, and measurements
tvec - time vector
gif_path - (optional) String path to save GIF animation. If empty or not provided, no GIF is saved.

%}

    % Validate inputs
    if nargin < 3
        error('Usage: mWidar_FilterPlot_multiObj_RBPF(Filter, Data, tvec, [gif_path])');
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
    GT = Data.GT; % Cell array, 1 x n_t, where each cell is 6 x n_k
    sim_signal = Data.signal;
    y = Data.y; % All measurements
    
    n_t = size(GT, 2); % # of targets
    n_k = size(GT{1}, 2); % # of timesteps

    % Define spatial grid (hardcoded as specified)
    Lscene = 4;
    npx = 128;
    xgrid = linspace(-2, 2, npx);
    ygrid = linspace(0, Lscene, npx);
    
    %% Animation Loop
    
    % Create figure once with 2 subplots
    figure(67); clf;
    set(gcf, 'Position', [100, 100, 1400, 600], 'Visible', 'off');

    for k = 1:n_k
        clf;
        
        %% LEFT SUBPLOT: mWidar Signal with detections and true objects
        subplot(1, 2, 1); cla; hold on; grid on;

        % Plot the mWidar signal as 2D image
        imagesc(xgrid, ygrid, sim_signal{k} / (max(max(sim_signal{k}))));
        set(gca, 'YDir', 'normal');
        colormap('parula');

        % Plot true target locations (2D) & mean estimated locations
        for t = 1:n_t
            plot(GT{t}(1, k), GT{t}(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3);
            
            % Plot mean estimate from Filter{k}.x
            if isfield(Filter{k}, 'x')
                x_est = Filter{k}.x;
                if iscell(x_est)
                    plot(x_est{t}(1), x_est{t}(2), 'ms', 'MarkerSize', 10, 'LineWidth', 3);
                else
                    plot(x_est(1, t), x_est(2, t), 'ms', 'MarkerSize', 10, 'LineWidth', 3);
                end
            end
        end
        
        % Plot detections
        if ~isempty(y{k})
            scatter(y{k}(1, :), y{k}(2, :), 50, '*r');
        end
        
        % Format
        xlim([-2 2]);
        ylim([0 4]);
        title(['mWidar Signal @ k=', num2str(k)], 'Interpreter', 'latex');
        xlabel('X (m)', 'Interpreter', 'latex');
        ylabel('Y (m)', 'Interpreter', 'latex');
        axis square;

        %% RIGHT SUBPLOT: Scatter of all particle state estimates
        subplot(1, 2, 2); cla; hold on; grid on;

        % Extract particle states
        % For each particle, extract position (x,y) from all KFs
        if isfield(Filter{k}, 'particles')
            particles = Filter{k}.particles;
            N_p = length(particles);
            
            % Collect all particle state positions
            particle_x = [];
            particle_y = [];
            
            for p = 1:N_p
                % Each particle has .kfs which is a cell array {1 x n_t}
                if isfield(particles{p}, 'kfs')
                    kfs = particles{p}.kfs;
                    for t = 1:length(kfs)
                        % Extract position from each KF
                        kf_state = kfs{t}.x; % Should be [N_x x 1]
                        particle_x = [particle_x; kf_state(1)];
                        particle_y = [particle_y; kf_state(2)];
                    end
                end
            end
            
            % Plot all particle states as scatter
            if ~isempty(particle_x)
                scatter(particle_x, particle_y, 20, [0.3 0.3 0.8], 'filled', ...
                    'MarkerFaceAlpha', 0.5, 'DisplayName', 'Particle States');
            end
        end
        
        % Plot ground truth for reference
        for t = 1:n_t
            plot(GT{t}(1, k), GT{t}(2, k), 'mx', 'MarkerSize', 10, 'LineWidth', 3, ...
                'DisplayName', sprintf('GT Target %d', t));
        end
        
        % Plot mean estimate
        if isfield(Filter{k}, 'x')
            x_est = Filter{k}.x;
            if iscell(x_est)
                for t = 1:length(x_est)
                    plot(x_est{t}(1), x_est{t}(2), 'ms', 'MarkerSize', 12, ...
                        'MarkerFaceColor', 'm', 'LineWidth', 2, ...
                        'DisplayName', 'Mean Estimate');
                end
            else
                for t = 1:n_t
                    plot(x_est(1, t), x_est(2, t), 'ms', 'MarkerSize', 12, ...
                        'MarkerFaceColor', 'm', 'LineWidth', 2, ...
                        'DisplayName', 'Mean Estimate');
                end
            end
        end
        
        % Format
        xlim([-2 2]);
        ylim([0 4]);
        title(['RBPF: Particle State Estimates @ k=', num2str(k)], 'Interpreter', 'latex');
        xlabel('X (m)', 'Interpreter', 'latex');
        ylabel('Y (m)', 'Interpreter', 'latex');
        legend('Location', 'northeast', 'Interpreter', 'latex');
        axis square;
        grid on;

        %% Overall figure settings
        sgtitle(['RBPF Multi-Target Tracking @ k=', num2str(k)], 'FontSize', 14, 'Interpreter', 'latex');
        
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
