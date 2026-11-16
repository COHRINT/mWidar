%%% Run one (environment, filter, N) job for run_compare and time it.
%%%
%%%   out = run_compare_job(job)
%%%
%%% job fields (built in run_compare)
%%%   envIdx, trial, traj, seed     which scenario this is
%%%   filter                        "KF_RBPF" | "TBD_PF_gauss" | "TBD_PF_model"
%%%   variant                       TBD-PF variant name in config/variants.m ("" for RBPF)
%%%   N                             # particles
%%%   noiseSTD                      TBD-PF run NoiseSTD (NaN for RBPF)
%%%   tbdESS                        TBD-PF resample threshold (run.ESS)
%%%   rbpfAssoc                     KF_RBPF association_strategy
%%%   env, scenario                 the environment and the scenario it generated
%%%
%%% out fields
%%%   job metadata (no env/scenario, those stay on the client), plus
%%%   runtime   wall time [s] of the filtering loop only (construction excluded,
%%%             same as run_test)
%%%   R         TBD-PF: the struct returned by TBD_PF.run
%%%             KF_RBPF: .x_est [6 x K] ([x y vx vy ax ay]), .P_est {1 x K},
%%%                      .ess [1 x K], .nMeas [1 x K], .N, .K

function out = run_compare_job(job)

    scenario = job.scenario;

    switch job.filter
        case "KF_RBPF"
            [R, runtime] = run_rbpf(job, scenario);
        otherwise
            [R, runtime] = run_tbd(job, scenario);
    end

    out = rmfield(job, {'env', 'scenario'});
    out.runtime = runtime;
    out.R = R;

end


function [R, runtime] = run_tbd(job, scenario)

    V = variants();
    cfg = apply_overrides(default_cfg(), V.(job.variant), "run_compare " + job.variant);

    cfg.TBD.N      = job.N;
    cfg.TBD.Seed   = job.seed;
    cfg.TBD.dt     = scenario.dt;
    cfg.TBD.K      = scenario.K;
    cfg.run.ESS      = job.tbdESS;
    cfg.run.NoiseSTD = job.noiseSTD;

    f = build_filter(cfg);
    runArgs = name_value(cfg.run);

    tStart = tic;
    R = f.run(scenario, job.env, runArgs{:});
    runtime = toc(tStart);

end


function [R, runtime] = run_rbpf(job, scenario)

    % KF_RBPF draws from the global stream, seed it so runs are repeatable
    rng(job.seed);

    K = scenario.K;
    hp  = FilterHyperParams('KF_RBPF');
    cfg = FilterConfig('KF_RBPF', 'dt', scenario.dt, 'N_particles', job.N, ...
                       'store_full_history', false);

    % Blind start: UniformInit spreads particles over the scene, x0 only sets N_x
    x0 = zeros(6, 1);
    filt = KF_RBPF(x0, cfg.N_particles, cfg.F, cfg.Q, cfg.H, cfg.R, ...
                   'PD', cfg.PD, 'PFA', cfg.PFA, 'ESSThreshold', hp.ESS_threshold, ...
                   'UniformInit', true);
    % Set after construction: the constructor only accepts 'uniform' |
    % 'optimal' | 'likelihood', but timestep() only implements 'uniform',
    % 'gating' (optimal importance dist.) and 'likelihood' (= uniform for now).
    filt.association_strategy = job.rbpfAssoc;

    x_est = nan(6, K);
    P_est = cell(1, K);
    ess   = nan(1, K);
    nMeas = zeros(1, K);

    tStart = tic;
    for k = 1:K
        z = scenario.detections{k};
        filt.timestep(z);
        [x_est(:, k), P_est{k}] = filt.getGaussianEstimate();
        ess(k)   = filt.current_ESS;
        nMeas(k) = size(z, 2);
    end
    runtime = toc(tStart);

    R = struct();
    R.filter = "KF_RBPF";
    R.N      = job.N;
    R.K      = K;
    R.x_est  = x_est;
    R.P_est  = P_est;
    R.ess    = ess;
    R.nMeas  = nMeas;
    R.association_strategy = job.rbpfAssoc;

end
