%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mWidar Simulator implementation in MATLAB
%
% Anthony La Barca
%
% script to Simulate mWidar radar signals given object trajectories
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;
clc;

% Set PLOT_FLAG to 1 to plot the simulation
PLOT_FLAG = 1;

addpath("matlab_functions/")

% Import sampling.mat (M), recovery.mat (G)
load("sampling.mat");
load("recovery.mat");


% OBJECT is defined as a vector -- a POSITION, VELOCITY, and ACCELERATION. Since
% there is no object oriented structures / no structs in MATLAB (to my limited
% knowledge), this is how each will be represented. Multiple objects will be
% columns in a vector.

object_example = [25; 25; 5; 5; 0; 0]; % [x;y;vx;vy;ax;ay]
objects = [66, 63;
    66, 63;
    -1, -1;
    1, -1;
    0, 0;
    0, 0];

checkbounds = @(coordinate) coordinate > 0 && coordinate < 128;

if PLOT_FLAG
    figure()
    tile_sim = tiledlayout(1, 2, 'TileSpacing', 'Compact');
end

% Simulation parameters
dt = 1;
timesteps = 100;
end_timestep = timesteps;
GT = zeros(128, 128, timesteps);
simulated_signal = zeros(128, 128, timesteps);
for a = 1:dt:dt*timesteps
    S = zeros(128,128);
    for obj = objects
        if checkbounds(obj(1)) && checkbounds(obj(2))
            S(obj(1), obj(2)) = 1;
        end
    end
    if all(S == 0)
        end_timestep = a;
        break;
    end

    % Now we have the signal, we can propogate the objects to get the simulated
    % signal -- just do it here but will be a function with global variables G,M
    % propogateDistribution(signal, G, M);

    % Row-major flatten the signal 
    signal_flat = S';
    signal_flat = signal_flat(:);
    signal_flat = M * signal_flat;
    signal_flat = G' * signal_flat;
    sim_signal = reshape(signal_flat, 128, 128)';

    GT(:, :, a) = S;
    simulated_signal(:,:,a) = sim_signal;

    % surf(signal)
    if PLOT_FLAG
        ax1 = nexttile(1);
        contourf(S);

        xlim([0, 128])
        ylim([0,128])
        title("Ground Truth Object Locations")

        ax2 = nexttile(2);
        s = surface(sim_signal, 'FaceAlpha', 0.5);
        s.EdgeColor = 'none';
        colormap(ax2, 'jet')
        xlim([0, 128])
        ylim([0,128])
        title("Simulated Radar Signal")

        pause(.1)
    end

    objects = update_objects(objects, dt);
end


% Save the simulated signal up to end_timestep
fprintf("Simulation complete. Saving to simulated_signal.mat and GT.mat\n");
fprintf("Simulation ran for %d timesteps\n", end_timestep);
simulated_signal = simulated_signal(:,:,1:end_timestep);
GT = GT(:,:,1:end_timestep);
save("simulated_signal.mat", "simulated_signal");
save("GT.mat", "GT");
