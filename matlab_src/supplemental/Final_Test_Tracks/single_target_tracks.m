clear; clc; close all

%{
    Script to generate/save mWidar images for the following 4 single object trajectories:
        Near array (Accelerating)
        Far from array (Accelerating)
        Along the border (Decelerating)
        Parabolic trajectory (bottom of scene to the top back to the
        bottom)

    State vector, x = [x, xdot, xdoubledot, y, ydot, ydoubledot]'
%}

% Load mWidar simulation matrices
load(fullfile('..', 'recovery.mat'))
load(fullfile('..', 'sampling.mat'))

fprintf('=== GENERATING SINGLE TARGET TRACKS ===\n');

dt = 0.1; % [sec]
tvec = 0:dt:10; % 10 seconds for each trajectory, 100 timesteps
n_t = size(tvec, 2); % # of tsteps
toplot_during = 0; 
toplot = 1; % Plot signal at end of script
% Select detector
%detector = "peaks2";
detector = "CFAR";

% CA-CFAR parameters
% Pfa = 0.35; % Probability of false alarm
% Ng = 5; % Guard cells
% Nr = 20; % Training cells
Pfa = 0.5; % Probability of false alarm -- 36 WORKS THE BEST
Ng = 2;    % Guard cells
Nr = 20;   % Training cells

fprintf('Duration: %.1f seconds, %d timesteps\n', tvec(end), n_t);
fprintf('Detector: %s\n', detector);

% Create output directory if it doesn't exist
if ~exist('SingleObj', 'dir')
    mkdir('SingleObj');
end

%% Traj 1: Near the array
fprintf('\n--- Generating Trajectory 1: Near Array (Accelerating) ---\n');

% X will be our ground truth state time history

A = [0, 0, 1, 0, 0, 0;
     0, 0, 0, 1, 0, 0;
     0, 0, 0, 0, 1, 0;
     0, 0, 0, 0, 0, 1;
     0, 0, 0, 0, 0, 0;
     0, 0, 0, 0, 0, 0];

% IC -> Near array, y ~ 0
x0 = [-1.75,  1,  0, 0, 0.065, 0]'; % No y acceleration

X_1 = generate_track(x0, A, tvec);
fprintf('Generated track with %d timesteps\n', size(X_1, 2));

% Generate an mWidar image for each timestep, save the signal, each
% detection, and ground truth into one .mat file
[y_1, Signal_1] = sim_mWidar_image(n_t, X_1, M, G, detector, false);
fprintf('Generated signals and detections\n');

% Count valid detections
n_detections = sum(~cellfun(@isempty, y_1));
fprintf('Timesteps with detections: %d/%d\n', n_detections, n_t);

Data.GT = X_1;
Data.y = y_1;
Data.signal = Signal_1;

save(fullfile('SingleObj', 'T1_near.mat'), 'Data', '-mat')
fprintf('Saved: T1_near.mat\n');

% Create animation for this trajectory
if toplot_during
    plot_single_trajectory(Data, 'T1: Near Array (Accelerating)', fullfile('SingleObj', 'T1_near_animation'), dt);
end
%% Traj 2: Far from array
fprintf('\n--- Generating Trajectory 2: Far from Array (Accelerating) ---\n');

X_2 = zeros(6, n_t); % Preallocate

% IC -> Far from array, y ~ 4
x0 = [-1.75 3.5 0 0 0.065 0]'; % No y acceleration

X_2 = generate_track(x0, A, tvec);
fprintf('Generated track with %d timesteps\n', size(X_2, 2));

% Generate an mWidar image for each timestep, save the signal, each
% detection, and ground truth into one .mat file
[y_2, Signal_2] = sim_mWidar_image(n_t, X_2, M, G, detector, false);
fprintf('Generated signals and detections\n');

% Count valid detections
n_detections = sum(~cellfun(@isempty, y_2));
fprintf('Timesteps with detections: %d/%d\n', n_detections, n_t);

Data.GT = X_2;
Data.y = y_2;
Data.signal = Signal_2;

save(fullfile('SingleObj', 'T2_far.mat'), 'Data', '-mat')
fprintf('Saved: T2_far.mat\n');

% Create animation for this trajectory
if toplot_during
    plot_single_trajectory(Data, 'T2: Far from Array (Accelerating)', fullfile('SingleObj', 'T2_far_animation'), dt);
end
%% Traj 3: Along Border (deaccelerate)
fprintf('\n--- Generating Trajectory 3: Along Border (Decelerating) ---\n');

X_3 = zeros(6, n_t); % Preallocate

% IC -> Along the border
x0 = [-1.75 3.75 0 -0.9 0 0.14]';

X_3 = generate_track(x0, A, tvec);
fprintf('Generated track with %d timesteps\n', size(X_3, 2));

% Generate an mWidar image for each timestep, save the signal, each
% detection, and ground truth into one .mat file
[y_3, Signal_3] = sim_mWidar_image(n_t, X_3, M, G, detector, false);
fprintf('Generated signals and detections\n');

% Count valid detections
n_detections = sum(~cellfun(@isempty, y_3));
fprintf('Timesteps with detections: %d/%d\n', n_detections, n_t);

Data.GT = X_3;
Data.y = y_3;
Data.signal = Signal_3;

save(fullfile('SingleObj', 'T3_border.mat'), 'Data', '-mat')
fprintf('Saved: T3_border.mat\n');

% Create animation for this trajectory
if toplot_during
    plot_single_trajectory(Data, 'T3: Along Border (Decelerating)', fullfile('SingleObj', 'T3_border_animation'), dt);
end

%% Traj 4: Parabolic traj
fprintf('\n--- Generating Trajectory 4: Parabolic Trajectory ---\n');

X_4 = zeros(6, n_t); % Preallocate

% IC
x0 = [-1.75 0.25 0.25 1.25 0 -0.25]';

X_4 = generate_track(x0, A, tvec);
fprintf('Generated track with %d timesteps\n', size(X_4, 2));

[y_4, Signal_4] = sim_mWidar_image(n_t, X_4, M, G, detector, false);
fprintf('Generated signals and detections\n');

% Count valid detections
n_detections = sum(~cellfun(@isempty, y_4));
fprintf('Timesteps with detections: %d/%d\n', n_detections, n_t);

Data.GT = X_4;
Data.y = y_4;
Data.signal = Signal_4;

save(fullfile('SingleObj', 'T4_parab.mat'), 'Data', '-mat')
fprintf('Saved: T4_parab.mat\n');

% Create animation for this trajectory
if toplot_during
    plot_single_trajectory(Data, 'T4: Parabolic Trajectory', fullfile('SingleObj', 'T4_parab_animation'), dt);
end

%% Track 5: Increased noise onto mWidar signal
fprintf('\n--- Generating Trajectory 5: Parabolic with Noise ---\n');

% IC
x0 = [-1.75 0.25 0.25 1.25 0 -0.25]';

X_5 = generate_track(x0, A, tvec);
fprintf('Generated track with %d timesteps\n', size(X_5, 2));

[y_5, Signal_5] = sim_mWidar_image(n_t, X_5, M, G, detector, true);
fprintf('Generated signals and detections with noise\n');

% Count valid detections
n_detections = sum(~cellfun(@isempty, y_5));
fprintf('Timesteps with detections: %d/%d\n', n_detections, n_t);

Data.GT = X_5;
Data.y = y_5;
Data.signal = Signal_5;

save(fullfile('SingleObj', 'T5_parab_noise.mat'), 'Data', '-mat')
fprintf('Saved: T5_parab_noise.mat\n');

% Create animation for this trajectory
if toplot_during
    plot_single_trajectory(Data, 'T5: Parabolic with Noise', fullfile('SingleObj', 'T5_parab_noise_animation'), dt);
end

%% Create Comprehensive Animation Showing All Five Trajectories
fprintf('\n--- Creating Comprehensive Single Target Tracks Animation ---\n');

if toplot
    % Setup figure for comprehensive animation
    figure(99); clf;
    set(gcf, 'Position', [150, 150, 1400, 900]);
    
    % Scene parameters
    npx = 128;
    xgrid = linspace(-2, 2, npx);
    ygrid = linspace(0, 4, npx);
    [pxgrid, pygrid] = meshgrid(xgrid, ygrid);
    
    % Collect all trajectory data
    GT_all = {X_1, X_2, X_3, X_4, X_5};
    signals_all = {Signal_1, Signal_2, Signal_3, Signal_4, Signal_5};
    y_all = {y_1, y_2, y_3, y_4, y_5};
    traj_names = {'T1: Near Array', 'T2: Far Array', 'T3: Border', 'T4: Parabolic', 'T5: Parabolic+Noise'};
    
    % Initialize comprehensive animation saving
    comprehensive_animation_filename = fullfile('SingleObj', 'All_Single_Trajectories_Animation.gif');
    comp_frame_delay = 0.4; % Slightly slower for better visibility
    
    for i = 1:n_t
        clf;
        
        current_time = (i-1) * dt;
        
        % Create subplots for each trajectory
        for j = 1:5
            subplot(2, 3, j);
            
            % Get current trajectory data
            GT = GT_all{j};
            y_meas = y_all{j};
            signals = signals_all{j};
            
            % Plot the signal as background
            if ~isempty(signals{i})
                imagesc(xgrid, ygrid, signals{i});
                set(gca, 'YDir', 'normal');
                colormap(gca, parula);
                hold on;
            else
                % Create empty plot if no signal
                imagesc(xgrid, ygrid, zeros(npx, npx));
                set(gca, 'YDir', 'normal');
                colormap(gca, parula);
                hold on;
            end
            
            % Get current position from ground truth
            px = GT(1, i);  % x position
            py = GT(2, i);  % y position (2nd element in state vector)

            % Plot ground truth position
            plot(px, py, 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
            
            % Plot measurements if they exist
            if ~isempty(y_meas{i}) && size(y_meas{i}, 1) >= 2
                scatter(y_meas{i}(1, :), y_meas{i}(2, :), 40, 'g+', 'LineWidth', 2);
                n_detections = size(y_meas{i}, 2);
            else
                n_detections = 0;
            end
            
            % Plot trajectory history up to current point
            if i > 1
                plot(GT(1, 1:i), GT(2, 1:i), 'r--', 'LineWidth', 1.5);
            end
            
            % Formatting
            xlim([-2.2, 2.2]);
            ylim([-0.2, 4.2]);
            xlabel('X (m)');
            ylabel('Y (m)');
            title(sprintf('%s (%d det)', traj_names{j}, n_detections));
            grid on;
            set(gca, 'GridAlpha', 0.3);
        end
        
        % Add overall title
        sgtitle(sprintf('Single Target Trajectories Comparison (t=%.1fs, step %d/%d)', current_time, i, n_t), 'FontSize', 14, 'FontWeight', 'bold');
        
        % Ensure tight layout
        drawnow;
        
        % Capture frame for comprehensive animation
        try
            frame = getframe(gcf);
            im = frame2im(frame);
            [imind, cm] = rgb2ind(im, 256);
            
            % Write to GIF file
            if i == 1
                imwrite(imind, cm, comprehensive_animation_filename, 'gif', 'Loopcount', inf, 'DelayTime', comp_frame_delay);
            else
                imwrite(imind, cm, comprehensive_animation_filename, 'gif', 'WriteMode', 'append', 'DelayTime', comp_frame_delay);
            end
        catch ME
            fprintf('Warning: Could not save frame %d to comprehensive animation: %s\n', i, ME.message);
        end
        
        % Small pause for display
        pause(0.03);
    end
    
    fprintf('Comprehensive animation saved to: %s\n', comprehensive_animation_filename);
end

%% Completion Summary
fprintf('\n=== SINGLE TARGET TRACKS GENERATION COMPLETE ===\n');
fprintf('All datasets saved to: SingleObj/\n');
fprintf('Dataset files:\n');
fprintf('  - T1_near.mat\n');
fprintf('  - T2_far.mat\n');
fprintf('  - T3_border.mat\n');
fprintf('  - T4_parab.mat\n');
fprintf('  - T5_parab_noise.mat\n');
if toplot
    fprintf('\nIndividual animations created:\n');
    fprintf('  - T1_near_animation.gif\n');
    fprintf('  - T2_far_animation.gif\n');
    fprintf('  - T3_border_animation.gif\n');
    fprintf('  - T4_parab_animation.gif\n');
    fprintf('  - T5_parab_noise_animation.gif\n');
    fprintf('\nComprehensive animation:\n');
    fprintf('  - All_Single_Trajectories_Animation.gif (all 5 trajectories)\n');
end
fprintf('==========================================\n');
