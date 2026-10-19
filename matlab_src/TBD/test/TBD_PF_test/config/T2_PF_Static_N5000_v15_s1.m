%%% Single static target with additive noise (Var = 15), model likelihood.

function t = T2_PF_Static_N5000_v15_s1()

    t.description = "Particle Filter for Track Before Detect on a single static target. 5000 particles, " + ...
                    "additive noise (Var = 15) on the mWidar signal, model likelihood.";
    t.scenarios = {"static_noisy"};

    t.base.env = struct('Debug', true);
    t.base.TBD = struct('Debug', true, 'Pb', 0.03, 'v_max', 0.05, 'v_min', -0.05);
    t.base.run = struct('ESS', 0.0005, 'DebugFrames', [9 11 20 40 60 80 91]);
    t.base.show = struct('Animate', true);

end
