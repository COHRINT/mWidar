addpath("TBD/Includes")

trajs = {'parabola'};
rw = [true];
v = 5;
e = environment('objs', 1,'trajectories', trajs, 'rw', rw, 'kEnd', 100, ...
                'kBirth', 10, 'kDeath', 90, 'Seed', 123456, 'Noise', true, 'Var', v);

scenario = e.setup();
%fig = e.show(scenario,'Animate', true);

Q = 1e-1 * [0.01 0 0 0 0;
0 0.05 0 0 0;
0 0 0.01 0 0;
0 0 0 0.05 0;
0 0 0 0 1e-3]; % keep the intensity random walk tight, see TBD.m

f = TBD_PF('gamma', 0.85, 'Q', Q, 'N', 5000, 'Debug', true, 'pDist',10);
R = f.run(scenario, 'ESS', 0.3);
figpath = sprintf('TBD/Figures/T_noise_v%i_SIS_%s', v, trajs{1});
[f1, f2] = f.show(R,scenario, 'Animate', true, 'Save', figpath);