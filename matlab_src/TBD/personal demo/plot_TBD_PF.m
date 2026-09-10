function plot_TBD_PF(xt, Et, Y_hist, p, im)
%PLOT_TBD_PF Summary plots (+ optional animation) for the TBD PF demo.
%
%   plot_TBD_PF(xt, Et, Y_hist, p) plots, over all k = 1:kSteps:
%       - estimated existence probability P(E_k) vs. true existence
%       - true vs. estimated track in the (px,py) plane
%       - true vs. estimated px and py individually vs. time
%
%   plot_TBD_PF(xt, Et, Y_hist, p, im) additionally animates the particle
%   cloud (particles with E=1) over the raw measurement images.
%
%   Inputs
%       xt     - 4 x kSteps ground truth state [px;vx;py;vy] (NaN when absent)
%       Et     - 1 x kSteps true existence flag
%       Y_hist - 5 x N x kSteps particle history [px;vx;py;vy;E] per particle
%       p      - params struct from init_params()
%       im     - (optional) nx x ny x kSteps measurement image stack

    kSteps = size(Y_hist,3);

    pE   = nan(1,kSteps);
    xHat = nan(1,kSteps);
    yHat = nan(1,kSteps);

    for k = 1:kSteps
        alive = squeeze(Y_hist(5,:,k)) ~= 0;
        pE(k)   = mean(alive);
        xHat(k) = mean(Y_hist(1,alive,k));
        yHat(k) = mean(Y_hist(3,alive,k));
    end

    declared = pE > p.pEthresh;

    figure('Name','TBD Particle Filter Results','Color','w');
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    % --- Existence probability ---
    nexttile;
    stairs(1:kSteps, Et, 'k--', 'LineWidth', 1.2); hold on;
    plot(1:kSteps, pE, 'b-', 'LineWidth', 1.5);
    yline(p.pEthresh, 'r:', 'LineWidth', 1);
    xlabel('k'); ylabel('P(exists)'); ylim([-0.05 1.05]); grid on;
    legend('True existence','Estimated P(E_k)','Declare threshold','Location','best');
    title('Target Existence');

    % --- XY plane trajectory ---
    nexttile;
    plot(xt(1,:), xt(3,:), 'k-', 'LineWidth', 1.5); hold on;
    scatter(xHat(declared), yHat(declared), 20, find(declared), 'filled');
    cb = colorbar; cb.Label.String = 'k';
    xlabel('p_x'); ylabel('p_y'); axis equal;
    xlim([1 p.nx]); ylim([1 p.ny]); grid on;
    legend('True track','Estimated track (declared)','Location','best');
    title('True vs. Estimated Track');

    % --- X position over time ---
    nexttile;
    plot(1:kSteps, xt(1,:), 'k-', 'LineWidth', 1.2); hold on;
    plot(1:kSteps, xHat, 'b.', 'MarkerSize', 8);
    xlabel('k'); ylabel('p_x'); grid on;
    legend('True','Estimated','Location','best');
    title('X Position');

    % --- Y position over time ---
    nexttile;
    plot(1:kSteps, xt(3,:), 'k-', 'LineWidth', 1.2); hold on;
    plot(1:kSteps, yHat, 'b.', 'MarkerSize', 8);
    xlabel('k'); ylabel('p_y'); grid on;
    legend('True','Estimated','Location','best');
    title('Y Position');

    % --- Optional particle-cloud animation over the raw images ---
    if nargin >= 5 && ~isempty(im)
        figure('Name','TBD Particle Cloud','Color','w');
        for k = 1:kSteps
            alive = squeeze(Y_hist(5,:,k)) ~= 0;
            imagesc(1:p.nx, 1:p.ny, im(:,:,k)'); axis xy equal tight;
            colormap(gca, 'gray'); hold on;
            plot(Y_hist(1,alive,k), Y_hist(3,alive,k), 'r.', 'MarkerSize', 4);
            if ~isnan(xt(1,k))
                plot(xt(1,k), xt(3,k), 'go', 'MarkerSize', 10, 'LineWidth', 1.5);
            end
            hold off;
            xlabel('p_x'); ylabel('p_y');
            title(sprintf('k = %d,  P(E) = %.2f', k, pE(k)));
            drawnow;
            pause(0.03);
        end
    end

end
