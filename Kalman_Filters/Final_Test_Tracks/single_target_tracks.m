clear; clc; close all

load Kalman_Filters\recovery.mat
load Kalman_Filters\sampling.mat

%{
    Script to generate/save mWidar images for the following 4 single object trajectories:
        Near array (Accelerating)
        Far from array (Accelerating)
        Along the border (Decelerating)
        Parabolic trajectory (bottom of scene to the top back to the
        bottom)
    
    State vector, x = [x, xdot, xdoubledot, y, ydot, ydoubledot]'
%}

dt = 0.1; % [sec]
tvec = 0:dt:10; % 10 seconds for each trajectory, 100 timesteps
n_t = size(tvec,2); % # of tsteps
toplot = 1; % Plot signal at end of script
% Select detector
detector = "peaks2";
%detector = "CFAR";

%% Traj 1: Near the array

% X will be our ground truth state time history
X_1 = zeros(6,n_t); % Preallocate

A = [0 0 1 0 0 0;
    0 0 0 1 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0;
    0 0 0 0 0 0];

% IC -> Near array, y ~ 0
X_1(:,1) = [-1.75 0.5 0 0 0.065 0]'; % No y acceleration

for t = 2:n_t
    X_1(:,t) = expm(A*dt)*X_1(:,t-1);
end

% Generate an mWidar image for each timestep, save the signal, each
% detection, and ground truth into one .mat file
[y_1, Signal_1] = sim_mWidar_image(n_t,X_1,M,G,detector);

Data.GT = X_1;
Data.y = y_1;
Data.signal = Signal_1;

save Kalman_Filters/Final_Test_Tracks/SingleObj/T1_near.mat Data -mat
%% Traj 2: Far from array

X_2 = zeros(6,n_t); % Preallocate

% IC -> Far from array, y ~ 4
X_2(:,1) = [-1.75 3.5 0 0 0.065 0]'; % No y acceleration

for t = 2:n_t
    X_2(:,t) = expm(A*dt)*X_2(:,t-1);
end

% Generate an mWidar image for each timestep, save the signal, each
% detection, and ground truth into one .mat file
[y_2, Signal_2] = sim_mWidar_image(n_t,X_2,M,G,detector);

Data.GT = X_2;
Data.y = y_2;
Data.signal = Signal_2;

save Kalman_Filters/Final_Test_Tracks/SingleObj/T2_far.mat Data -mat
%% Traj 3: Along Border (deaccelerate)

X_3 = zeros(6,n_t); % Preallocate


% IC -> Along the border
X_3(:,1) = [-1.75 3.75 0 -0.9 0 0.14]'; 

for t = 2:n_t
    X_3(:,t) = expm(A*dt)*X_3(:,t-1);
end

% Generate an mWidar image for each timestep, save the signal, each
% detection, and ground truth into one .mat file
[y_3, Signal_3] = sim_mWidar_image(n_t,X_3,M,G,detector);

Data.GT = X_3;
Data.y = y_3;
Data.signal = Signal_3;

save Kalman_Filters/Final_Test_Tracks/SingleObj/T3_border.mat Data -mat

%% Traj 4: Parabolic traj

X_4 = zeros(6,n_t); % Preallocate

% IC
X_4(:,1) = [-1.75 0.25 0.25 1.25 0 -0.25]'; 

for t = 2:n_t
    X_4(:,t) = expm(A*dt)*X_4(:,t-1);
end

[y_4, Signal_4] = sim_mWidar_image(n_t,X_4,M,G,detector);

Data.GT = X_4;
Data.y = y_4;
Data.signal = Signal_4;

save Kalman_Filters/Final_Test_Tracks/SingleObj/T4_parab.mat Data -mat
%% Plot mWidar image for each traj along w detections and GT
npx = 128;
xgrid = linspace(-2,2,npx);
ygrid = linspace(0,4,npx);
[pxgrid,pygrid] = meshgrid(xgrid,ygrid);

GT = {X_1, X_2, X_3, X_4};
sim_signal = {Signal_1, Signal_2, Signal_3, Signal_4};
Y = {y_1, y_2, y_3, y_4};
if toplot
        
    for i = 1:n_t
        figure(99), clf, hold on

        for j = 1:4
            ax = subplot(2,2,j); hold on
            
            X = GT{j};
            y = Y{j};
            px = X(1,i);
            py = X(2,i);
    
            plot3(px,py,10000*ones(length(GT),1),'mx','MarkerSize',10,'LineWidth',10)
            plot3(y{i}(1,:),y{i}(2,:),10000*ones(length(y{i}),1),'ms','MarkerSize',12,'LineWidth',1.2)
            surface(pxgrid,pygrid,sim_signal{j}{i},'EdgeColor','none')
            view(ax,2)
        end
        pause(0.1)
    end
end