%%% environment class will create GT tracks using generator, and then create signals with simulator.
%%% The main function scenario = environment.setup() will return the signal at each timestep, as well as the truth data at each timestep 
%%% In a scenario struct for simple access
%%% scenario struct example:
%%% scenario.object_count = n, # of objects present in scenario
%%% scenario.signal = {1 x K} cell array, where each cell contains the signal (128x128) at time k
%%% scenario.t1...n = 4 x K, track (1,2,...n) ground truth. Will be n t# fields 
%%% This should allow things to flow down easily

classdef environment < mWidar



end