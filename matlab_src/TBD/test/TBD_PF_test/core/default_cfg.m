%%% Single source of truth for every parameter of a trial.
%%%
%%% Every block (env, TBD, run, show) is forwarded as-is as name/value pairs
%%% to the matching call, so a field here must be a real argument of:
%%%   env  -> environment(...)
%%%   TBD  -> TBD_PF(...)        (the block for cfg.filter, see build_filter)
%%%   run  -> f.run(scenario, ...)
%%%   show -> f.show(R, scenario, ...)
%%% Test-level controls (plotting on/off, saving, ...) live in the test
%%% definition, never here. See default_test.
%%%
%%% [] marks a DERIVED field, filled in by resolve_cfg (or by run_test for
%%% output paths). Setting it explicitly in a config breaks the link.

function cfg = default_cfg()

    cfg = struct();
    cfg.filter = "TBD_PF";

    %% environment(...)
    cfg.env = struct();
    cfg.env.objs         = 1;
    cfg.env.trajectories = {'static'};
    cfg.env.rw           = false;
    cfg.env.kEnd         = 100;
    cfg.env.kBirth       = 10;
    cfg.env.kDeath       = 90;
    cfg.env.Seed         = 1;
    cfg.env.Noise        = false;
    cfg.env.Var          = 0;
    cfg.env.Debug        = false;
    cfg.env.Units        = 'meters';
    cfg.env.dt           = 0.1;
    cfg.env.start        = [];
    cfg.env.final        = [];
    cfg.env.Normalize    = false;
    cfg.env.Blur         = false;
    cfg.env.Sigma        = 2;

    %% TBD_PF(...)
    cfg.TBD = struct();
    cfg.TBD.Debug      = false;
    cfg.TBD.Units      = 'meters';
    cfg.TBD.Seed       = [];        % derived: env.Seed
    cfg.TBD.STD        = 0.75;
    cfg.TBD.Sigma      = 0.5;
    cfg.TBD.dt         = [];        % derived: env.dt
    cfg.TBD.K          = [];        % derived (run_test): scenario.K, the # of generated frames
    cfg.TBD.Ip         = 1;
    cfg.TBD.Background = 'none';
    cfg.TBD.N          = 5000;
    cfg.TBD.Pb         = 0.1;
    cfg.TBD.Ps         = 0.98;
    cfg.TBD.pEthresh   = 0.5;
    cfg.TBD.pDist      = 5;
    cfg.TBD.Q          = diag([0.01 0.005 0.01 0.005]);
    cfg.TBD.v_max      = 0.15;
    cfg.TBD.v_min      = -0.15;
    cfg.TBD.I_max      = 3;
    cfg.TBD.I_min      = 0.1;
    cfg.TBD.gamma      = 0.75;

    %% TBD_PF.run(...)
    cfg.run = struct();
    cfg.run.Bootstrap    = false;
    cfg.run.ESS          = 0.3;
    cfg.run.Likelihood   = 'model';
    cfg.run.Var          = [];      % derived: env.Var
    cfg.run.DebugPlots   = false;
    cfg.run.DebugFrames  = [];
    cfg.run.DebugName    = '';      % derived (run_test): trial file label
    cfg.run.DebugDir     = '';      % derived (run_test): <outDir>/debug
    cfg.run.DebugVisible = false;

    %% TBD_PF.show(...)
    cfg.show = struct();
    cfg.show.Animate        = false;
    cfg.show.Save           = '';   % derived (run_test): <outDir>/figures/<label>
    cfg.show.Trail          = 15;
    cfg.show.Title          = '';
    cfg.show.LikelihoodMap  = true;
    cfg.show.ShowBirths     = false;
    cfg.show.WeightColormap = '';
    cfg.show.WeightScale    = 'log';
    cfg.show.Intensity      = false;
    cfg.show.IntensityBins  = 24;

end
