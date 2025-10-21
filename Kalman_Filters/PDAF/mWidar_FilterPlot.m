function [] = mWidar_FilterPlot(Filter, Data, tvec)

    %%%%%%% mWidar_FilterPlot %%%%%%%%%%%%%%%
%{

Given a Filter Struct, and a Data struct from trajectory data, function will
plot the trajectory over the mWidar image, standard error plots, and RMSE.
Can be adapted to add more plots for future metrics

INPUTS:
Filter - Struct containing all the relevant filter data. Ideally plotting
fxn works for any filter (HMM/KF/Hybrid PF, need to make sure things are consistent)
tvec - time vector

Function is a mess I will fix it later
%}

    % Unpack data structs
    GT = Data.GT;
    sim_signal = Data.signal;
    y = Data.y;

    n_k = size(GT, 2); % # of timesteps

    % Unpack Filter Struct
    X = zeros(6, n_k); % State History
    P = cell(1, n_k); % State Cov
    S = cell(1, n_k); % Innov Cov

    std_xp = zeros(1, n_k);
    std_xv = zeros(1, n_k);
    std_xa = zeros(1, n_k);

    std_yp = zeros(1, n_k);
    std_yv = zeros(1, n_k);
    std_ya = zeros(1, n_k);

    for k = 1:n_k
        X(:, k) = Filter{k}.x;
        P{k} = Filter{k}.P;
        
        std_xp(k) = sqrt(P{k}(1, 1));
        std_xv(k) = sqrt(P{k}(3, 3));
        std_xa(k) = sqrt(P{k}(5, 5));

        std_yp(k) = sqrt(P{k}(2, 2));
        std_yv(k) = sqrt(P{k}(4, 4));
        std_ya(k) = sqrt(P{k}(6, 6));
    end

    Lscene = 4;
    npx = 128;
    xgrid = linspace(-2, 2, npx);
    ygrid = linspace(0, Lscene, npx);
    [pxgrid, pygrid] = meshgrid(xgrid, ygrid);

    %% Trajectory

    for k = 1:n_k

        figure(66); clf; hold on; grid on
        % Calculate/plot covariance ellipse, currently using innovation
        % covariance to visualize gating
        innovCov = [P{k}(1, 1) P{k}(1, 2); P{k}(2, 1) P{k}(2, 2)];
        % Innovation mean is predicted meas at time k
        muin = Filter{k}.H * X(:, k); % This may not work for all filters, as they likely don't have same meas fxn
        [Xellip, Yellip] = calc_gsigma_ellipse_plotpoints(muin, innovCov, 1, 100);

        %%%% NOTE: This assumes obj positions are at index 1 and 4
        plot3(X(1, k), X(2, k), ones(length(GT), 1), 'ms', 'MarkerSize', 12, 'LineWidth', 1.2)
        plot3(GT(1, k), GT(2, k), ones(length(GT), 1), 'mx', 'MarkerSize', 10, 'LineWidth', 10)
        scatter3(y{k}(1, :), y{k}(2, :), ones(length(y{k}(1, :)), 1), '*r')
        plot3(Xellip, Yellip, ones(length(Xellip), 1), '--k')
        surf(pxgrid, pygrid, sim_signal{k} / (max(max(sim_signal{k}))), 'EdgeColor', 'none')

        xlim([-2 2])
        ylim([0 4])
        title(['Object @ k=', num2str(k)], 'Interpreter', 'latex')

        %     if save_traj
        %         %%Turn fig into a gif
        %         fig = figure(66);
        %         frame = getframe(fig);
        %         Mov{k} = frame2im(frame);
        %         [Fr, map] = rgb2ind(Mov{k},256);
        %         if k == 1
        %             imwrite(Fr,map,filename,"gif","LoopCount",Inf,"DelayTime",0.1);
        %         else
        %             imwrite(Fr,map,filename,"gif","WriteMode","append","DelayTime",0.1);
        %         end
        %
        %     end

    end

    %% RMSE

    E = rmse(X, GT);

    figure(77); hold on; grid on
    plot(tvec, E, 'k-.', LineWidth = 1)
    title('RMSE over time', 'Interpreter', 'latex')
    xlabel('Time [s]', 'Interpreter', 'latex')
    ylabel('RMSE', 'Interpreter', 'latex')

    %% Normal error plot
    err = X - GT;

    figure(88); hold on; grid on
    subplot(3, 2, 1); hold on; grid on
    plot(tvec, err(1, :), 'b', LineWidth = 1)
    plot(tvec, std_xp, 'b--', LineWidth = 1)
    plot(tvec, -std_xp, 'b--', LineWidth = 1)

    subplot(3, 2, 2); hold on; grid on
    plot(tvec, err(4, :), 'b', LineWidth = 1)
    plot(tvec, std_yp, 'b--', LineWidth = 1)
    plot(tvec, -std_yp, 'b--', LineWidth = 1)

    subplot(3, 2, 3); hold on; grid on
    plot(tvec, err(2, :), 'b', LineWidth = 1)
    plot(tvec, std_xv, 'b--', LineWidth = 1)
    plot(tvec, -std_xv, 'b--', LineWidth = 1)

    subplot(3, 2, 4); hold on; grid on
    plot(tvec, err(5, :), 'b', LineWidth = 1)
    plot(tvec, std_yv, 'b--', LineWidth = 1)
    plot(tvec, -std_yv, 'b--', LineWidth = 1)

    subplot(3, 2, 5); hold on; grid on
    plot(tvec, err(3, :), 'b', LineWidth = 1)
    plot(tvec, std_xa, 'b--', LineWidth = 1)
    plot(tvec, -std_xa, 'b--', LineWidth = 1)

    subplot(3, 2, 6); hold on; grid on
    plot(tvec, err(6, :), 'b', LineWidth = 1)
    plot(tvec, std_ya, 'b--', LineWidth = 1)
    plot(tvec, -std_ya, 'b--', LineWidth = 1)

end

%% More Functions for plotting

%calc_gsigma_ellipse_plotpoints.m
%%Does all the computations needed to plot 2D Gaussian ellipses properly.
%%Takes in the Gaussian mean (muin), cov matrix (Sigma, positive definite),
%%g-sigma value, and number of points to generate for plotting.
function [X, Y] = calc_gsigma_ellipse_plotpoints(muin, Sigma, g, npoints)
    %%align the Gaussian along its principal axes
    [R, D, thetalocx] = subf_rotategaussianellipse(Sigma, g);
    %%pick semi-major and semi-minor "axes" (lengths)
    if Sigma(1) < Sigma(4) %use if sigxx<sigyy
        a = 1 / sqrt(D(4));
        b = 1 / sqrt(D(1));
    elseif Sigma(1) >= Sigma(4)
        a = 1 / sqrt(D(1)); %use if sigxx<sigyy
        b = 1 / sqrt(D(4));
    end

    %%calculate points of ellipse:
    mux = muin(1);
    muy = muin(2);

    if Sigma(2) ~= 0
        [X Y] = calculateEllipse(mux, muy, a, b, rad2deg(thetalocx), npoints);
    else %if there are no off-diagonal terms, then no rotation needed
        [X Y] = calculateEllipse(mux, muy, a, b, 0, npoints);
    end

    %%call the rotate gaussian ellipse thingy as a local subfunction (there is
    %%also a separate fxn m-file for this, too)
    function [R, D, thetalocx] = subf_rotategaussianellipse(Sigma, g)
        P = inv(Sigma);
        P = 0.5 * (P + P'); %symmetrize

        a11 = P(1);
        a12 = P(2);
        a22 = P(4);
        c = -g ^ 2;

        mu = 1 / (-c); %can define mu this way since b1=0,b2=0 b/c we are mean centered
        m11 = mu * a11;
        m12 = mu * a12;
        m22 = mu * a22;

        %%solve for eigenstuff
        lambda1 = 0.5 * (m11 + m22 + sqrt((m11 - m22) .^ 2 + 4 * m12 .^ 2));
        lambda2 = 0.5 * (m11 + m22 - sqrt((m11 - m22) .^ 2 + 4 * m12 .^ 2));
        % % b = 1/sqrt(lambda1); %semi-minor axis for standard ellipse (length)
        % % a = 1/sqrt(lambda2); %semi-major axis for standard ellipse (length)
        D = diag([lambda2, lambda1]); %elements are 1/a^2 and 1/b^2, respectively
        %%Choose the mahor axis direction of the ellipse
        if m11 >= m22
            u11 = lambda1 - m22;
            u12 = m12;
        elseif m11 < m22
            u11 = m12;
            u12 = lambda1 - m11;
        end

        norm1 = sqrt(u11 .^ 2 + u12 .^ 2);
        U1 = ([u11; u12]) / norm1; %major axis direction
        U2 = [-u12; u11]; %minor axis direction
        R = [U1, U2];
        % if sum(sum(isnan(R)))>0
        %    R = eye(2); %default hack for now in case of degenerate stuff
        % end

        thetalocx = 0.5 * atan(-2 * a12 / (a22 - a11));
        % if isnan(thetalocx)
        %     thetalocx = 0; %default hack for now...
        % end

    end

end

function [X Y] = calculateEllipse(x, y, a, b, angle, steps)
    %# This functions returns points to draw an ellipse
    %#
    %#  @param x     X coordinate
    %#  @param y     Y coordinate
    %#  @param a     Semimajor axis
    %#  @param b     Semiminor axis
    %#  @param angle Angle of the ellipse (in degrees)
    %#

    %error(nargchk(5, 6, nargin));
    narginchk(5, 6);
    if nargin < 6, steps = 36; end

    beta = angle * (pi / 180);
    sinbeta = sin(beta);
    cosbeta = cos(beta);

    alpha = linspace(0, 360, steps)' .* (pi / 180);
    sinalpha = sin(alpha);
    cosalpha = cos(alpha);

    X = x + (a * cosalpha * cosbeta - b * sinalpha * sinbeta);
    Y = y + (a * cosalpha * sinbeta + b * sinalpha * cosbeta);

    if nargout == 1, X = [X Y]; end
end
