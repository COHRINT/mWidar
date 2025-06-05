% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mWidar Simulator implementation in MATLAB
%
% Anthony La Barca
%
% script to Simulate mWidar radar signals given object trajectories
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;
clc;

% Default plotting parameters 
% Set default font size to be larger
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultTextFontSize', 14);
set(0, 'DefaultLineLineWidth', 2);

% Set default text interpreter to LaTeX
set(0, 'DefaultTextInterpreter', 'latex');
set(0, 'DefaultAxesTickLabelInterpreter', 'latex');
set(0, 'DefaultLegendInterpreter', 'latex');

% Make titles bold
set(0, 'DefaultAxesTitleFontWeight', 'bold');

% Set PLOT_FLAG to 1 to plot the simulation
PLOT_FLAG = 1;
SAVE_FLAG = 0;

addpath("matlab_functions/")

% Import sampling.mat (M), recovery.mat (G)
load("sampling.mat");
load("recovery.mat");


% OBJECT is defined as a vector -- a POSITION, VELOCITY, and ACCELERATION. Since
% there is no object oriented structures / no structs in MATLAB (to my limited
% knowledge), this is how each will be represented. Multiple objects will be
% columns in a vector.

% object_example = [25; 25; 5; 5; 0; 0]; % [x;y;vx;vy;ax;ay]
% objects = [66, 63;
%     66, 63;
%     -1, -1;
%     1, -1;
%     0, 0;
%     0, 0];

% Object appear from sides of screen and move across the screen 
% Decreased dt by 4, so multiply velocity and acceleration by 4 for equivalent motion per timestep
obj1 = [1; 60; 2; 0; 0; 0]; % [x;y;vx;vy;ax;ay]
obj2 = [127; 127; -2; -1.7; 0; 0];
obj3 = [2; 127; 2.5; -7; 0; 0.32];


TARGET_STRING = "Triple";
objects = [];

if TARGET_STRING == "Single"
    objects = [obj1];
elseif TARGET_STRING == "Double"
    objects = [obj1, obj2];
elseif TARGET_STRING == "Triple"
    objects = [obj1, obj2, obj3];
else
    error("TARGET_STRING must be Single, Double, or Triple");
end

% objects = [obj1, obj2, obj3];



checkbounds = @(coordinate) coordinate > 0 && coordinate < 128;

if PLOT_FLAG
    figure("Position", [100, 100, 600, 300])
    tile_sim = tiledlayout(1, 2, 'TileSpacing', 'Compact');
end

% Simulation parameters
dt = 1;
timesteps = 120;
end_timestep = timesteps;
GT = zeros(128, 128, timesteps);
simulated_signal = zeros(128, 128, timesteps);
objects_traj = zeros(2, size(objects, 2), timesteps);
for a = 1:dt:timesteps
    S = zeros(128,128);
    for obj = objects
        if checkbounds(obj(1)) && checkbounds(obj(2))
            % cast to int
            obj(1) = int32(obj(1));
            obj(2) = int32(obj(2));
            S(obj(1), obj(2)) = 1;
        end
    end
    if all(S == 0)
        end_timestep = a-1;
        break;
    end

    % Now we have the signal, we can propogate the objects to get the simulated
    % signal -- just do it here but will be a function with global variables G,M
    % propogateDistribution(signal, G, M);

    % Row-major flatten the signal 
    signal_flat = S;
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';

    GT(:, :, a) = S;
    simulated_signal(:,:,a) = sim_signal;
    objects_traj(:, :, a) = objects(1:2, :);

    CUTSCALED = 1;
    if PLOT_FLAG
        if CUTSCALED
            PLOTSIGNAL = sim_signal(20:end, :);
            % PLOTSIGNAL = imgaussfilt(PLOTSIGNAL, 2);
            PLOTSIGNAL = (PLOTSIGNAL - min(PLOTSIGNAL(:))) / ...
                       (max(PLOTSIGNAL(:)) - min(PLOTSIGNAL(:)));
        else
            PLOTSIGNAL = sim_signal;
        end
        
        % PLOTSIGNAL = PLOTSIGNAL - 
        ax1 = nexttile(1);
        cla(ax1)
        hold on
        grid on
        % contourf(S');
        for obj = objects
            if checkbounds(obj(1)) && checkbounds(obj(2))
                % cast to int
                obj(1) = int32(obj(1));
                if CUTSCALED
                    obj(2) = int32(obj(2)-20);
                else
                    obj(2) = int32(obj(2));
                end
                plot(obj(1), obj(2), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
            end
        end
        hold off
        axis equal
        xlim(ax1, [1, size(PLOTSIGNAL, 2)]);
        ylim(ax1, [1, size(PLOTSIGNAL, 1)]);
        title("Ground Truth Object Locations")

        ax2 = nexttile(2);
        grid(ax2, 'off')
        % Clear ax2 
        cla(ax2)
        % s = surface(sim_signal, 'FaceAlpha', 0.5);
        [X, Y] = meshgrid(1:size(PLOTSIGNAL, 2), 1:size(PLOTSIGNAL, 1));
        s = surf(ax2, X, Y, PLOTSIGNAL);
        s.EdgeColor = 'none';
        colormap(ax2, 'gray');
        c = colorbar(ax2, 'eastoutside');
        % s.EdgeColor = 'none';
        % colormap(ax2, 'jet')

        %  c = colorbar(ax1);
        c.Label.String = 'Signal Intensity';
        c.Label.Interpreter = 'latex';
        axis(ax2, 'equal');
        xlim(ax2, [1, size(PLOTSIGNAL, 2)]);
        ylim(ax2, [1, size(PLOTSIGNAL, 1)]);
        zlim(ax2, [min(PLOTSIGNAL(:)), max(PLOTSIGNAL(:))]);
        view(ax2, 2); % Set the view to 2D
        % title(ax1, "Radar Signal at Time " + t, 'Interpreter', 'latex', 'FontSize', 14);
        % legend(ax1, 'Location', 'northeast', "Interpreter", "latex", 'FontSize', 12);
        % colorbar(ax2, 'eastoutside');
        % axis equal
        % xlim([0, 128])
        % ylim([0,128])
        
        title("Simulated mWidar Signal")

        pause(.5)
    end

    objects = update_objects(objects, dt);
end


% Save the simulated signal up to end_timestep
if SAVE_FLAG
    fprintf("Simulation complete. Saving to simulated_signal.mat and GT.mat\n");
    fprintf("Simulation ran for %d timesteps\n", end_timestep);
    simulated_signal = simulated_signal(:,:,1:end_timestep);
    GT = GT(:,:,1:end_timestep);
    objects_traj = objects_traj(:,:,1:end_timestep);
    save(sprintf("../data_tracks/%s_simulated_signal.mat", TARGET_STRING), "simulated_signal");
    save(sprintf("../data_tracks/%s_GT.mat", TARGET_STRING), "GT");
    save(sprintf("../data_tracks/%s_objects_traj.mat", TARGET_STRING), "objects_traj");
end

% Create diagram of the simulated signal

figure(Position=[100, 100, 300, 300])
% 3 colors for the 3 objects
% Colorblind-friendly matte/pastel palette
color_palette = [
    0.6, 0.6, 0.8;  % Soft blue/lavender
    0.9, 0.6, 0.6;  % Soft salmon/pink
    0.6, 0.8, 0.6   % Soft green
];
ax1 = nexttile(1);
cla(ax1)
hold on
grid on

% Plot trajectory of each object in the frame over 17 timestep
% Display the trajectories for each object
for obj_idx = 1:size(objects_traj,2)
    x_traj = squeeze(objects_traj(1,obj_idx,1:end_timestep));
    y_traj = squeeze(objects_traj(2,obj_idx,1:end_timestep));
    h = plot(x_traj, y_traj, 'LineWidth', 4, 'Color', color_palette(obj_idx, :));
    % Add label at the starting point of each trajectory
    % Store plot handle for legend
    plot_handles(obj_idx) = h;
    
    % Draw arrows along the trajectory to indicate direction
    arrow_step = max(1, floor(length(x_traj)/10)); % Place up to 10 arrows
    for k = 1:arrow_step:length(x_traj)-1
        dx = x_traj(k+1) - x_traj(k);
        dy = y_traj(k+1) - y_traj(k);
        q = quiver(x_traj(k), y_traj(k), dx, dy, 2, 'MaxHeadSize', 5, 'LineWidth', 2, 'Color', 'k');
    end
    

end
% Put last quiver into the plot_handles
if ~isempty(q)
    plot_handles(end+1) = q; % Add the last quiver to the legend
end


axis equal
xlim([0, 128])
ylim([0, 128])
title("Sample Trajectories")
xlabel("X Position")
ylabel("Y Position")
legend(plot_handles, {'Object 1', 'Object 2', 'Object 3', 'Direction'}, 'Location', 'best');

% Save figure in data_tracks
if SAVE_FLAG
    saveas(gcf, sprintf("../data_tracks/%s_fulltrajectory.png", TARGET_STRING), "png");
end