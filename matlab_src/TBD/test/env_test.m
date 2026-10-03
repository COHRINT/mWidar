addpath("TBD/Includes")

trajs = {'line'};
rw = [true];
v = 0;
e = environment('objs', 1,'trajectories', trajs, 'rw', rw, 'kEnd', 100, ...
                'kBirth', 15, 'kDeath', 85, 'Seed', 1234, 'Noise', true, 'Var', v);

scenario = e.setup();
%fig = e.show(scenario,'Animate', true);

Q = 1e-1 * [0.01 0 0 0;0 0.5 0 0;0 0 0.01 0;0 0 0 0.5];

f = TBD_PF('gamma', 0.75, 'Q', Q, 'N', 5000);
R = f.run(scenario);
figpath = sprintf('TBD/Figures/T_noise_v%i_SIS', v);
[f1, f2] = f.show(R,scenario, 'Animate', true, 'Save', figpath);