%%% 100 trial MC run of the line trajectory with rw enabled, additive noise
%%% (Var = 15), model likelihood. Seeds 10..109; compared through the
%%% summary table/plot instead of per-trial dashboards.

function t = T6_PF_Line_RW_MC100_s10()

    t.description = "Particle Filter for Track Before Detect on a single target moving on a line with " + ...
                    "random walk enabled. 5000 particles, additive noise (Var = 15) on the mWidar signal, " + ...
                    "model likelihood. 100 MC trials, seeds 10..109.";
    t.scenarios = {"line_noisy_rw"};
    t.mc        = struct('n', 100, 'seed0', 10);

    t.output = struct('perTrial', false, 'keepR', false);

end
