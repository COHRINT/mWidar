%%% Single static target, no additive noise. Easy case, mainly for debugging
%%% new changes. Debug printing is on.

function t = T1_PF_Static_N5000_v0_s1()

    t.description = "Particle Filter for Track Before Detect on a single static target. 5000 particles, " + ...
                    "no additive noise on the mWidar signal. Fairly easy scenario, mainly used to debug new changes.";
    t.scenarios = {"static"};

    t.base.env = struct('Debug', false);
    t.base.TBD = struct('Debug', true, 'Pb', 0.13, 'v_max', 0.05, 'v_min', -0.05, 'gamma', 0.99);
    t.base.run = struct('ESS', 0.300, 'NoiseSTD', 10000);
    t.base.show = struct('Animate', true, 'AnimateWeights', true);

end
