%%% Single static target with additive noise (Var = 15), gaussian likelihood
%%% and a much tighter process noise.

function t = T3_PF_Static_N5000_v15_s1()

    t.description = "Particle Filter for Track Before Detect on a single static target. 5000 particles, " + ...
                    "additive noise (Var = 15) on the mWidar signal, gaussian likelihood, small Q.";
    t.scenarios = {"static_noisy"};
    t.variants  = {"pf_gauss"};

    t.base.env = struct('Debug', true);
    t.base.TBD = struct('Debug', true, 'Pb', 0.03, 'v_max', 0.05, 'v_min', -0.05, ...
                        'Q', 1e-3 * diag([0.01 0.005 0.01 0.005]));
    t.base.show = struct('Animate', true);

end
