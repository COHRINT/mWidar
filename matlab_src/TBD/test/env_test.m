addpath("TBD/Includes")

trajs = {'line'};
rw = [true];
e = environment('objs', 1,'trajectories', trajs, 'rw', rw, 'kEnd', 100, ...
                'kBirth', 2, 'kDeath', 98, 'Seed', 6967420);

scenario = e.setup();
%fig = e.show(scenario,'Animate', true);

Q = 1e-1 * [0.01 0 0 0;0 0.5 0 0;0 0 0.01 0;0 0 0 0.5];

f = TBD_PF('gamma', 0.75, 'Q', Q);
R = f.run(scenario);

[f1, f2] = f.show(R,scenario, 'Animate', true, 'Save', 'TBD/Figures/T4');