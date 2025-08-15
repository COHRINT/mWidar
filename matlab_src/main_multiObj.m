function main_multiObj(varargin)
    % MAIN for multiObj trackers

    % Keeping functionality basic until we have more trackers

    DATASET = "JPDAF_test_traj"; % Default dataset

        %% --- Environment Configuration ---
    clc; close all
    % Add paths for MATLAB functions
    addpath(fullfile('DA_Track'))
    addpath(fullfile('supplemental'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks'))
    addpath(fullfile('supplemental', 'Final_Test_Tracks', 'MultiObj'))

    load(fullfile('supplemental', 'Final_Test_Tracks', 'MultiObj', DATASET + '.mat'), 'Data');

     %% --- Initialize Variables ---
    % Load Data
    GT = Data.GT;
    z = Data.y;

    dt = 0.1;

    n_k = size(GT{1}, 2);
    nt = size(GT,2);
    performance = cell(1, n_k);
    %% --- Define Kalman Filter Matrices ---
    % Define KF Matrices state vector - {x,y,vx,vy,ax,ay}
    A = [0 0 1 0 0 0;
         0 0 0 1 0 0;
         0 0 0 0 1 0;
         0 0 0 0 0 1;
         0 0 0 0 0 0;
         0 0 0 0 0 0];

    F_KF = expm(A * dt);

        %% --- Define Noise Matrices ---
    Q = 1e-2 * eye(6);
    Q(3,3) = 1e-6; % Set process noise for acceleration

    Q(6,6) = 1e-6; % Set process noise for acceleration

    R = 0.1 * eye(2);

    %% --- Define Observation Matrix ---
    H = [1 0 0 0 0 0;
         0 1 0 0 0 0];

    %% --- Define Initial Covariance ---
    P0 = diag([0.1 0.1 0.25 0.25 0.5 0.5]);

    initial_states = [];
    initial_covs = {};
    for t = 1:nt
        initial_states = [initial_states, GT{t}(:,1)];
        initial_covs = [initial_covs, P0];

    end

    current_class = JPDA_KF(initial_states, initial_covs, F_KF,Q,R,H,nt,'Debug', false);

    [performance{1}.x, performance{1}.P] = current_class.getGaussianEstimate(); % Initial Gaussian estimate

    for k = 2:n_k
        fprintf('Processing time step %d/%d\n',k,n_k);

        current_meas = z{k};

        current_class.timestep(current_meas)
        
        [performance{k}.x, performance{k}.P] = current_class.getGaussianEstimate();
    end

    %% PLOT

    mWidar_FilterPlot_multiObj_Distribution(performance, Data, 0:dt:10, DATASET + '.gif')

end