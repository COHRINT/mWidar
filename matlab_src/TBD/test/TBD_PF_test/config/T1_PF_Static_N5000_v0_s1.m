
function cfg = T1_PF_Static_N5000_v0_s1()

    % initialize struct
    cfg = struct();

    cfg.description = " Particle Filter for Track Before Detect on a single static target. 5000 particles, and no additive noise added onto mWidar signal. This will be a fairly easy scenario mainly used to debug new changes";
    cfg.filter = "TBD_PF"; % Select our filter
    
    %% Higher level config fields. Determines how many trials we will be running. Needs more work to implement, for now this doesn't do anything
    cfg.Trials = 1; % The idea is, for cfg.Trials > 1, there will be separate fields for whatever field we want to change for different trials.

    % For example: Let's say we want two trials, one comparing the performance of the gauss and model likelihoods.
    
    %% Fields for the environment class

    cfg.env = struct(); % make it it's own struct to organize things
    cfg.env.objs = 1;
    cfg.env.trajectories = {'static'};
    cfg.env.rw = [false];
    cfg.env.kEnd = 100;
    cfg.env.kBirth = 10;
    cfg.env.kDeath = 90;
    cfg.env.Seed = 1;
    cfg.env.Noise = false;
    cfg.env.Var = 0;
    cfg.env.Debug = true; % will enable these since this is primarily for debugging purposes
    cfg.env.Units = 'meters';
    cfg.env.dt = 0.1;
    cfg.env.start = [];
    cfg.env.final = [];
    cfg.env.Normalize = false;
    cfg.env.Blur = false;
    cfg.env.Sigma = 2;


    %% Fields for the TBD class
    cfg.TBD = struct();
    cfg.TBD.Debug = true;
    cfg.TBD.Units = 'meters';
    cfg.TBD.Seed = cfg.env.Seed;
    cfg.TBD.STD = 0.75;
    cfg.TBD.Sigma = 0.5;
    cfg.TBD.dt = cfg.env.dt;
    cfg.TBD.Ip = 1;
    cfg.TBD.Background = 'none';
    cfg.TBD.N = 5000;
    cfg.TBD.Pb = 0.03;
    cfg.TBD.Ps = 0.98;
    cfg.TBD.pEthresh = 0.5;
    cfg.TBD.pDist = 5;

    % Choosing a good Q for a static target
    Q = 1e-3 * diag([0.01 0.005 0.01 0.005 1e-3]);
    cfg.TBD.Q = Q;
    cfg.TBD.v_max = 0.05;
    cfg.TBD.v_min = -0.05;
    cfg.TBD.gamma = 0.75;

    %% Fields for TBD_Pf.run()
    cfg.run.Bootstrap = false; % try
    cfg.run.ESS = 0.3;
    cfg.run.Likelihood = 'model';
    cfg.run.Var = cfg.env.Var;
    cfg.run.DebugPlots = true;
    cfg.run.DebugFrames = [9 11 20 40 60 80 89 91];
    %% Fields for TBD_PF.show() (plotting)
    cfg.show.Animate = true;
    cfg.show.Save = sprintf('TBD/test/TBD_PF_test/figures/PF_Static_N5000_v0_s1');
    cfg.show.ShowBirths = false;


end