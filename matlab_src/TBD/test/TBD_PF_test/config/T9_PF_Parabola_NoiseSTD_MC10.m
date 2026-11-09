%%% NoiseSTD tuning sweep on the parabola trajectory: additive noise Var = 0, 15,
%%% 30, 45, NoiseSTD = 100:100:10000 (coarse first pass), 10 MC seeds, default_cfg
%%% filter with ESS = 0.3. Same seeds at every NoiseSTD, so values are compared on
%%% the same scenarios. Run with run_sweep('T9'), see run_sweep.m.

function t = T9_PF_Parabola_NoiseSTD_MC10()

    traj = {'parabola'};

    t.description = "NoiseSTD tuning sweep, parabola trajectory. Var = 0, 15, 30, 45; " + ...
                    "NoiseSTD = 100:100:10000; 10 MC seeds; 5000 particles; model likelihood; ESS = 0.3.";
    t.scenarios = { ...
        "v0",  struct('env', struct('trajectories', {traj}, 'Noise', false, 'Var', 0)), ...
        "v15", struct('env', struct('trajectories', {traj}, 'Noise', true,  'Var', 15)), ...
        "v30", struct('env', struct('trajectories', {traj}, 'Noise', true,  'Var', 30)), ...
        "v45", struct('env', struct('trajectories', {traj}, 'Noise', true,  'Var', 45))};

    t.base.run = struct('ESS', 0.3);
    t.sweep    = struct('path', {'run.NoiseSTD'}, 'values', {100:100:10000});
    t.mc       = struct('n', 10, 'seed0', 1);
    t.output   = struct('perTrial', false, 'keepR', false);

end
