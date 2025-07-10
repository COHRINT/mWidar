function [pks, locs_y, locs_x] = CA_CFAR(signal, Pfa, Ng, Nr)
% CA_CFAR_2D  Two-dimensional cell-averaging CFAR
%
%   [pks, y, x] = CA_CFAR_2D(signal, Pfa, Ng, Nr) returns the peak
%   heights `pks` and their row/col coordinates `y,x` in the 2D matrix
%   `signal`.  `Ng` is the guard-cell radius, `Nr` is the noise-cell radius.
%   `Pfa` is the desired probability of false alarm.
%
%   This implements a square sliding window.  For each cell it:
%     1. Excludes a (2*Ng+1)-by-(2*Ng+1) guard area around the cell.
%     2. Averages the remaining noise cells in a (2*Nr+1)-by-(2*Nr+1) window.
%     3. Multiplies by the threshold factor α to get the detection threshold.
%     4. Flags a CFAR detection if signal > threshold.
%     5. Picks sub-pixel peaks with imregionalmax.

    % --- compute CFAR threshold multiplier α for CA-CFAR given Pfa and Nc ---
    Nc = (2*Nr+1)^2 - (2*Ng+1)^2;          % number of noise cells
    alpha = Nc * (Pfa^(-1/Nc) - 1);         % see standard CA-CFAR derivation

    % --- build a binary mask for the noise window (1=noise, 0=guard+cell) ---
    M = 2*Nr + 1;
    [X,Y] = meshgrid(1:M, 1:M);
    center = Nr+1;
    dist = max(abs(X-center), abs(Y-center));
    noiseMask = dist > Ng;   % ones for noise cells

    % --- convolve to get local sum and count of noise cells at each pixel ---
    noiseSum   = conv2(signal, noiseMask, 'same');
    noiseCount = conv2(ones(size(signal)), noiseMask, 'same');

    % avoid division by zero (edge pixels)
    noiseCount(noiseCount==0) = 1;

    % --- compute local average noise and threshold map ---
    noiseAvg  = noiseSum ./ noiseCount;
    threshMap = alpha * noiseAvg;

    % --- apply threshold ---
    cfarMask = (signal > threshMap);

    % --- refine to local maxima within a small neighborhood ---
    % use a 3×3 neighborhood (or larger if desired)
    localMaxMask = imregionalmax(signal);

    % --- final detection mask ---
    detMask = cfarMask & localMaxMask;

    % --- extract peak locations and values ---
    [locs_y, locs_x] = find(detMask);
    pks = signal(sub2ind(size(signal), locs_y, locs_x));

    % sort by descending amplitude
    [pks, idx] = sort(pks, 'descend');
    locs_y = locs_y(idx);
    locs_x = locs_x(idx);
end