function [ImageGridmWidar] = genmWidarImage(posArray, mWidarParams)
% Generate mWidar image grid given occupancy grid and mWidar parameters
%
% Inputs
%   posArray: Nx128x128 array of (x,y) positions to evaluate -- 1 indicating occupied, 0 indicating free
%.            N is number of different occupancy grids to evaluate (e.g., from MC sim or different time steps)
%   mWidarParams: struct of mWidar parameters 
%.                should contain fields:
%.                - sampling: sampling matrix 
%.                - recovery: recovery matrix
%
% Outputs
%   ImageGridmWidar: 128x128 array of mWidar image grid values



        % Get dimensions from input
        [N, height, width] = size(posArray);

        if height ~= 128 || width ~= 128
            error('Input posArray must have dimensions Nx128x128');
        end

        % Initialize output array
        ImageGridmWidar = zeros(N, height, width);

        % Extract sampling and recovery matrices from params
        M = mWidarParams.sampling;
        G = mWidarParams.recovery;

        % Process each grid
        for i = 1:N
            % Get current occupancy grid and flatten it
            S = posArray(i,:,:);
            S = reshape(S, height, width);
            
            % Apply transformation (same as in original code)
            signal_flat = S';
            signal_flat = signal_flat(:);
            signal_flat = M * signal_flat;
            signal_flat = G' * signal_flat;
            
            % Reshape and store in output array
            ImageGridmWidar(i,:,:) = reshape(signal_flat, width, height)';
        end

end