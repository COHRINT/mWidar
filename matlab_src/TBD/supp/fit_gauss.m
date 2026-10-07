addpath('TBD/Includes')

sim = simulator('Var', 0);                 % match Blur/Sigma to the scenario
z = sim.generate_mWidar_image({[0, 2]}, 'meters', true);
z = (z - median(z(:))) / (max(z(:)) - median(z(:)));    % same as TBD_PF.preprocess

[r0, c0] = find(z == max(z(:)), 1);

% Quick: FWHM through the peak. sigma = FWHM / 2.3548
sx = nnz(z(r0, :) >= 0.5) / 2.3548;        % px, along x
sy = nnz(z(:, c0) >= 0.5) / 2.3548;        % px, along y

% Better: least-squares fit in a window around the peak (no toolboxes needed)
h = 15;
rr = max(1, r0-h):min(sim.npx, r0+h);
cc = max(1, c0-h):min(sim.npx, c0+h);
[C, Rw] = meshgrid(cc, rr);
Zw = z(rr, cc);
model = @(b) b(1) * exp(-(C-b(2)).^2/(2*b(4)^2) - (Rw-b(3)).^2/(2*b(5)^2));
b = fminsearch(@(b) sum((model(b) - Zw).^2, 'all'), [1, c0, r0, 3, 3]);
b
% b = [amplitude, x0, y0, sigma_x, sigma_y]   (pixels)
