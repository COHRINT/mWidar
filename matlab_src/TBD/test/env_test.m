addpath("TBD/Includes")

trajs = { 'parabola'};
rw = [false];
e = environment('objs', 1,'trajectories', trajs, 'rw', rw, 'kEnd', 100,'Seed', 6967420);

scenario = e.setup();
%fig = e.show(scenario,'Animate', true);

f = TBD_PF('gamma', 0.9);
R = f.run(scenario);

[f1, f2] = f.show(R,scenario, 'Animate', true, 'Save', 'TBD/Figures/T2');