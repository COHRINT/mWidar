function plot_single_trajectory(Data, trajectory_name, save_filename, dt)
    %PLOT_SINGLE_TRAJECTORY Creates an animation for a single trajectory
    %   Data: struct with fields GT (ground truth), y (measurements), signal
    %   trajectory_name: string for plot title
    %   save_filename: filename for saving the animation (without extension)
    %   dt: time step in seconds
    
    if nargin < 4
        dt = 0.1; % Default time step
    end
    
    % Extract data
    GT = Data.GT;
    y_meas = Data.y;
    signals = Data.signal;
    
    % Scene parameters
    npx = 128;
    xgrid = linspace(-2, 2, npx);
    ygrid = linspace(0, 4, npx);
    [pxgrid, pygrid] = meshgrid(xgrid, ygrid);
    
    % Get number of timesteps
    n_t = size(GT, 2);
    
    % Setup figure
    figure; clf;
    set(gcf, 'Position', [200, 200, 800, 600]);
    
    % Animation parameters
    animation_filename = [save_filename '.gif'];
    frame_delay = 0.3; % seconds between frames
    
    fprintf('Creating animation for %s...\n', trajectory_name);
    
    for i = 1:n_t
        clf;
        
        current_time = (i-1) * dt;
        
        % Get current position from ground truth
        px = GT(1, i);  % x position
        py = GT(2, i);  % y position (2nd element in state vector)

        % Plot the signal as background
        if ~isempty(signals{i})
            imagesc(xgrid, ygrid, signals{i});
            set(gca, 'YDir', 'normal');
            colormap(parula);
            hold on;
        else
            % Create empty plot if no signal
            imagesc(xgrid, ygrid, zeros(npx, npx));
            set(gca, 'YDir', 'normal');
            colormap(parula);
            hold on;
        end
        
        % Plot ground truth position
        h_gt = plot(px, py, 'ro', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'r');
        
        % Plot measurements if they exist
        if ~isempty(y_meas{i})
            if size(y_meas{i}, 1) >= 2
                h_det = scatter(y_meas{i}(1, :), y_meas{i}(2, :), 60, 'g+', 'LineWidth', 2);
                n_detections = size(y_meas{i}, 2);
            else
                n_detections = 0;
                h_det = [];
            end
        else
            n_detections = 0;
            h_det = [];
        end
        
        % Plot trajectory history up to current point
        if i > 1
            h_traj = plot(GT(1, 1:i), GT(2, 1:i), 'r--', 'LineWidth', 1.5);
        else
            h_traj = [];
        end
        
        % Formatting
        xlim([-2.2, 2.2]);
        ylim([-0.2, 4.2]);
        xlabel('X (m)');
        ylabel('Y (m)');
        title(sprintf('%s (t=%.1fs, step %d/%d, %d detections)', trajectory_name, current_time, i, n_t, n_detections));
        grid on;
        set(gca, 'GridAlpha', 0.3);
        colorbar;
        
        % Add legend with proper handles
        legend_handles = [];
        legend_labels = {};
        
        % Always include ground truth
        legend_handles(end+1) = h_gt;
        legend_labels{end+1} = 'Ground Truth';
        
        % Include detections if they exist
        if n_detections > 0 && ~isempty(h_det)
            legend_handles(end+1) = h_det;
            legend_labels{end+1} = sprintf('Detections (%d)', n_detections);
        end
        
        % Include trajectory if it exists
        if ~isempty(h_traj)
            legend_handles(end+1) = h_traj;
            legend_labels{end+1} = 'Trajectory';
        end
        
        if ~isempty(legend_handles)
            legend(legend_handles, legend_labels, 'Location', 'northeast');
        end
        
        drawnow;
        
        % Capture frame for animation
        try
            frame = getframe(gcf);
            im = frame2im(frame);
            [imind, cm] = rgb2ind(im, 256);
            
            if i == 1
                imwrite(imind, cm, animation_filename, 'gif', 'Loopcount', inf, 'DelayTime', frame_delay);
            else
                imwrite(imind, cm, animation_filename, 'gif', 'WriteMode', 'append', 'DelayTime', frame_delay);
            end
        catch ME
            fprintf('Warning: Could not save frame %d to animation: %s\n', i, ME.message);
        end
        
        pause(0.02); % Small pause for display
    end
    
    fprintf('Animation saved to: %s\n', animation_filename);
    fprintf('Animation complete for %s\n\n', trajectory_name);
end
