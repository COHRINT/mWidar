%%% Named scenario library. Each entry is a partial cfg (overrides on
%%% default_cfg), normally just env.* fields. Reference them by name in a
%%% test definition:  t.scenarios = {"static", "line"};
%%% A one-off scenario can be given inline instead: {"mine", struct('env', ...)}

function S = scenarios()
    %% Clean signal
    S.static   = struct('env', struct('trajectories', {{'static'}}));
    S.line     = struct('env', struct('trajectories', {{'line'}}));
    S.parabola = struct('env', struct('trajectories', {{'parabola'}}));
    S.scurve   = struct('env', struct('trajectories', {{'scurve'}}));

    %% Additive noise, Var = 15
    S.static_noisy = struct('env', struct('trajectories', {{'static'}}, 'Noise', true, 'Var', 15));
    S.line_noisy   = struct('env', struct('trajectories', {{'line'}},   'Noise', true, 'Var', 15));

    %% Additive noise, Var = 15, random walk on the truth trajectory
    S.line_noisy_rw = struct('env', struct('trajectories', {{'line'}}, 'rw', true, 'Noise', true, 'Var', 15));
end
