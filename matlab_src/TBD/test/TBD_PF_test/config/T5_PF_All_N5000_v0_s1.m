%%% Every trajectory type (static, line, parabola, scurve), no additive noise,
%%% no random walk. Same filter settings on all four; compared through the
%%% summary table/plot instead of per-trial dashboards.

function t = T5_PF_All_N5000_v0_s1()

    t.description = "Particle Filter for Track Before Detect on a single target, run on every trajectory " + ...
                    "(static, line, parabola, scurve). 5000 particles, no additive noise, no rw process noise.";
    t.scenarios = {"static", "line", "parabola", "scurve"};

    t.base.TBD = struct('Debug', true);
    t.output = struct('perTrial', true);

end
