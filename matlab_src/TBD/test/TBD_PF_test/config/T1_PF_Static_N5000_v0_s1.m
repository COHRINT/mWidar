%%% Single static target, no additive noise. Easy case, mainly for debugging
%%% new changes. Debug printing is on.

function t = T1_PF_Static_N5000_v0_s1()

    t.description = "Particle Filter for Track Before Detect on a single static target. 5000 particles, " + ...
                    "no additive noise on the mWidar signal. Fairly easy scenario, mainly used to debug new changes.";
    t.scenarios = {"static"};

    t.base.env = struct('Debug', true);
    t.base.TBD = struct('Debug', true, 'Pb', 0.13, 'v_max', 0.05, 'v_min', -0.05);
    t.base.run = struct('ESS', 0.0005, 'DebugFrames', [9 11 20 40 60 80 91]);
    t.base.show = struct('Animate', true);

end
