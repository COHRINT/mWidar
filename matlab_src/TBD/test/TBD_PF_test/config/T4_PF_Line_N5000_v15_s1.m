%%% Single target on a line with additive noise (Var = 15), model likelihood.

function t = T4_PF_Line_N5000_v15_s1()

    t.description = "Particle Filter for Track Before Detect on a single target moving on a line. 5000 particles, " + ...
                    "additive noise (Var = 15) on the mWidar signal, model likelihood.";
    t.scenarios = {"line_noisy"};

    t.base.env = struct('Seed', 5);
    t.base.TBD = struct('Debug', true);
    t.base.show = struct('Animate', true);

end
