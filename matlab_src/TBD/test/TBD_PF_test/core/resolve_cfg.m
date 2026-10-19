%%% Fill derived fields and sanity check a fully overridden cfg.
%%% Runs once per trial, after every override (see expand_trials).
%%% Derived fields are the [] entries in default_cfg; set one explicitly in a
%%% config to break its link.

function cfg = resolve_cfg(cfg)

    %% Links
    if isempty(cfg.TBD.Seed), cfg.TBD.Seed = cfg.env.Seed; end
    if isempty(cfg.TBD.dt),   cfg.TBD.dt   = cfg.env.dt;   end
    if isempty(cfg.run.Var),  cfg.run.Var  = cfg.env.Var;  end

    %% Checks that span blocks or would otherwise fail deep in a class
    assert(isequal(size(cfg.TBD.Q), [4 4]), "cfg.TBD.Q must be 4x4, got %s", mat2str(size(cfg.TBD.Q)));
    assert(cfg.TBD.v_min <= cfg.TBD.v_max, "cfg.TBD.v_min (%g) must be <= v_max (%g)", cfg.TBD.v_min, cfg.TBD.v_max);
    assert(numel(cfg.env.trajectories) == cfg.env.objs, ...
        "cfg.env.trajectories has %d entries but objs = %d", numel(cfg.env.trajectories), cfg.env.objs);
    assert(cfg.env.Noise || cfg.env.Var == 0, ...
        "cfg.env.Var = %g but Noise is off; set env.Noise = true or Var = 0", cfg.env.Var);
    assert(ismember(cfg.run.Likelihood, {'gauss', 'model'}), "cfg.run.Likelihood must be 'gauss' or 'model'");

end
