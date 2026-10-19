%%% Named filter-variant library. A variant picks the filter (cfg.filter)
%%% and overrides its parameters. Reference by name in a test definition:
%%%   t.variants = {"pf_model", "pf_gauss"};
%%% A one-off variant can be given inline: {"mine", struct('TBD', ...)}

function V = variants()
    V.pf_model = struct('filter', "TBD_PF", 'run', struct('Likelihood', 'model'));
    V.pf_gauss = struct('filter', "TBD_PF", 'run', struct('Likelihood', 'gauss'));
    V.pf_slow  = struct('filter', "TBD_PF", 'TBD', struct('v_max', 0.05, 'v_min', -0.05));
    % future filters...
end
