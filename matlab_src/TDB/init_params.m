function params = init_params()
    %{
        init_params(): Initialize TBD params
    %}
    params = struct();
    

    %%% Sensor Model parameters
    params.nx = 25; % # of x pixels
    params.ny = 25; % # of y pixels
    params.NoiseSTD = 0.5; % std in noise
    params.NoiseMean = 0; % mean value of noise
    params.Sigma = 0.75; % PSF Blurring parameter
    params.dt = 0.1; % 10 Hz sampling time
    params.Ip = 1.5;

    %%% PF parameters
    params.N = 5000; % # of particles
    params.Pb = 0.03; % Birth prob
    params.Ps = 0.98; % Survival prob
    params.pEthresh = 0.5;             % existence threshold for "declaring" a track
    params.PI = [1-params.Pb, params.Pb;
                 1-params.Ps, params.Ps];
    params.p = 3;
    
    %%% Target dynamics
    params.A = [0 1 0 0;0 0 0 0;0 0 0 1;0 0 0 0];
    params.F = expm(params.A * params.dt);
    params.x0 = [10; 3.5; 10; 7];  % [px; vx; py; vy] at kBirth
    params.Qtrue = 1e-2*[0.01 0 0 0;0 0.5 0 0;0 0 0.01 0;0 0 0 0.5];
    % Filter process noise, deliberately inflated 100x over Qtrue: the PF
    % needs enough jitter to keep particle diversity after resampling, and
    % Qtrue's 0.01 px position noise is invisible against a 0.75 px PSF.
    params.Q = [0.01 0 0 0;0 0.5 0 0;0 0 0.01 0;0 0 0 0.5];
    params.v_max = 10;
    params.v_min = -10;
    
    %%% General
    params.kSteps = 30; % 100 frame sim
    params.kBirth = 10; % Target appears at t = 33 
    params.kDeath = 25; % Target dies at t = 85 
    params.seed = 67420;
    params.gamma = 0.5; % For new born particles, sample pos states uniformly from image that passes this threshold value
    rng(params.seed)


end