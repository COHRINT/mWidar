%%% Every field a test definition (config/T*.m) may set, with its default.
%%% load_test merges a test definition onto this strictly, so a misspelled
%%% field errors instead of being silently ignored.
%%%
%%% Axes of the trial grid (all optional, see expand_trials):
%%%   scenarios  names from scenarios.m, or {name, overrides} pairs
%%%   variants   names from variants.m,  or {name, overrides} pairs
%%%   sweep      struct array, one element per swept parameter:
%%%                struct('path', {'TBD.Pb'}, 'values', {[0.05 0.1]})
%%%              values may be a numeric vector or a cell array
%%%   sweep_mode "grid" (cross product of all sweep params) or "zip" (paired)
%%%   mc         struct('n', 10, 'seed0', 1) -> seeds seed0 : seed0+n-1.
%%%              Empty = no MC axis, the seed in the cfg is used.

function t = default_test()

    t = struct();
    t.name        = "";
    t.description = "";

    t.base       = struct();                         % overrides on default_cfg, every trial
    t.scenarios  = {};
    t.variants   = {};
    t.sweep      = struct('path', {}, 'values', {});
    t.sweep_mode = "grid";
    t.mc         = [];

    %% Output controls (never forwarded to a class)
    t.output = struct();
    t.output.perTrial    = true;    % call f.show() for every trial
    t.output.closeFigs   = false;   % close per-trial figures once saved
    t.output.summaryPlot = true;    % metric comparison plot (only if > 1 group or MC)
    t.output.save        = true;    % write results/<name>_<timestamp>/
    t.output.saveR       = false;   % include filter output R in trials.mat (large)
    t.output.keepR       = true;    % keep R in the returned struct (memory heavy for big MC)
    t.output.rootDir     = "";      % "" -> TBD_PF_test/results

end
