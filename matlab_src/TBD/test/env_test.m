addpath("TBD/Includes")

trajs = {'line', 'scurve', 'parabola'};
rw = [false, false true];
e = environment('objs', 3,'trajectories', trajs, 'rw', rw, 'kEnd', 200);

scenario = e.setup();
fig = e.show(scenario,'Animate', true);