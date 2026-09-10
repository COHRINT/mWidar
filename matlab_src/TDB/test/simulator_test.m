
addpath("TDB/Includes")

sim = simulator('Debug', true);

s = sim.generate_mWidar_image([20, 50],'pixels', true);
