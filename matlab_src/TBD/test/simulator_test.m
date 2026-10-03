
addpath("TBD/Includes")

sim = simulator('Var', 10,'Debug', true);
v = visualize();

[s, snr] = sim.generate_mWidar_image({[20, 50]},'pixels', true);

f = v.signal_frame(s, 'SNR', snr)