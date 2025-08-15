function [] = mWidar_TuningPlot(performance, Data, current_k, filter_type, dataset_name, da_method, save_dir)
    % MWIDAR_TUNINGPLOT Plot trajectory and state deviation for tuning
    %
    % SYNTAX:
    %   mWidar_TuningPlot(performance, Data, current_k, filter_type, dataset_name, da_method, save_dir)
    %
    % INPUTS:
    %   performance  - Cell array of performance metrics up to current timestep
    %   Data         - Data struct with GT, signal, and measurements
    %   current_k    - Current timestep
    %   filter_type  - String: 'KF', 'HMM', or 'HybridPF'
    %   dataset_name - String: dataset name for plot title
    %   da_method    - String: 'PDA' or 'GNN'
    %   save_dir     - String: directory to save plots
    %
    % OUTPUTS:
    %   None - saves plots to specified directory
    %
    % DESCRIPTION:
    %   Creates two subplots:
    %   a) State trajectory (true and estimated)
    %   b) State estimate deviation (x-xhat) with covariances
    %   Saves plots with timestep information for tuning analysis

    % Validate inputs
    if nargin < 7
        error('Usage: mWidar_TuningPlot(performance, Data, current_k, filter_type, dataset_name, da_method, save_dir)');
    end

    % Ensure save directory exists
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % Unpack data
    GT = Data.GT;

    % Determine state dimensionality based on filter type
    switch lower(filter_type)
        case 'hmm'
            state_dim = 2; % HMM only estimates position
        case {'kf', 'hybridpf'}
            state_dim = 6; % KF and PF estimate full state (pos, vel, acc)
        otherwise
            state_dim = 6; % Default to full state
    end

    % Extract trajectory data up to current timestep
    x_est = zeros(state_dim, current_k);
    P_est = cell(1, current_k);

    for k = 1:current_k

        if isfield(performance{k}, 'x')
            x_est(:, k) = performance{k}.x;
        end

        if isfield(performance{k}, 'P')
            P_est{k} = performance{k}.P;
        end

    end

    % Create figure with appropriate size based on filter type
    figure(999); clf;

    % Determine layout based on filter type
    switch lower(filter_type)
        case {'kf', 'hybridpf'}
            % 1 trajectory row + 3 state error rows
            fig_height = 1000;
            num_state_rows = 4; % 1 for trajectory + 3 for states
        case 'hmm'
            % 1 state (pos) x 2 dimensions + 1 trajectory plot
            % Use 2 rows: top row for trajectory, bottom row for X and Y errors
            fig_height = 600;
            num_state_rows = 2;
        otherwise
            fig_height = 600;
            num_state_rows = 2;
    end

    set(gcf, 'Position', [100, 100, 1400, fig_height], 'Visible', 'off');

    %% SUBPLOT A: State Trajectory (True vs Estimated)
    if strcmp(lower(filter_type), 'hmm')
        subplot(num_state_rows, 3, 1:2); hold on; grid on;
    else
        % KF/PF: trajectory spans all 3 columns of first row
        subplot(num_state_rows, 3, 1:3); hold on; grid on;
    end

    % Plot true trajectory
    plot(GT(1, 1:current_k), GT(2, 1:current_k), 'k-', 'LineWidth', 2, 'DisplayName', 'True Trajectory');
    plot(GT(1, current_k), GT(2, current_k), 'ko', 'MarkerSize', 8, 'DisplayName', 'Current True Position');

    % Plot estimated trajectory
    plot(x_est(1, 1:current_k), x_est(2, 1:current_k), 'r--', 'LineWidth', 2, 'DisplayName', 'Estimated Trajectory');
    plot(x_est(1, current_k), x_est(2, current_k), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Current Estimate');

    % Plot confidence ellipse for current estimate (if available)
    if current_k > 1 && ~isempty(P_est{current_k})

        try
            % Extract position covariance
            P_pos = P_est{current_k}(1:2, 1:2);
            x_pos = x_est(1:2, current_k);

            % Plot 2-sigma ellipse
            [X_ellip, Y_ellip] = calc_gsigma_ellipse_plotpoints(x_pos, P_pos, 2, 100);
            plot(X_ellip, Y_ellip, 'r:', 'LineWidth', 1.5, 'DisplayName', '2\sigma Confidence');
        catch
            % Skip if ellipse calculation fails
        end

    end

    xlabel('X (m)', 'Interpreter', 'latex');
    ylabel('Y (m)', 'Interpreter', 'latex');
    title('Trajectory Comparison', 'Interpreter', 'latex');
    legend('Location', 'best', 'Interpreter', 'latex');

    % Set pinned axis limits for consistent positioning
    xlim([-2 2]);
    ylim([0 4]);
    axis square; grid on;

    %% SUBPLOT B: State Estimate Deviation - Individual State Components

    % Determine the number of states based on filter type
    switch lower(filter_type)
        case {'kf', 'hybridpf'}
            % KF and PF estimate position, velocity, acceleration in x and y
            state_names = {'pos', 'vel', 'acc'};
            state_labels = {'Position (m)', 'Velocity (m/s)', 'Acceleration (m/s²)'};
            state_indices = [1 3 5; 2 4 6]; % [x_states; y_states]
            gt_indices = [1 3 5; 2 4 6]; % Ground truth indices for KF/PF
            num_states = 3;
        case 'hmm'
            % HMM only estimates position in x and y
            state_names = {'pos'};
            state_labels = {'Position (m)'};
            state_indices = [1; 2]; % [x_pos; y_pos]
            gt_indices = [1; 2]; % Ground truth indices for HMM
            num_states = 1;
        otherwise
            error('Unknown filter type: %s', filter_type);
    end

    % Time vector
    tvec = 0:0.1:(current_k - 1) * 0.1; % Assuming dt = 0.1

    % Create subplot layout: num_states rows, 2 columns (x and y)
    for state_idx = 1:num_states

        for xy_idx = 1:2 % 1=x, 2=y
            % Calculate subplot position based on filter type
            if strcmp(lower(filter_type), 'hmm')
                % HMM: Row 1 = trajectory (cols 1:2), Row 2 = errors (X in col 1, Y in col 2)
                row = 2; % Always second row for errors
                col = xy_idx; % Column 1 for X, column 2 for Y
                subplot_idx = (row - 1) * 3 + col;
            else
                % KF/PF: Row 1 = trajectory (cols 1:3), Rows 2+ = error plots
                % Map state and xy to subplot positions starting from row 2
                row = state_idx + 1; % Start from row 2 (after trajectory)
                col = xy_idx; % Column 1 for X, column 2 for Y
                subplot_idx = (row - 1) * 3 + col;
            end

            subplot(num_state_rows, 3, subplot_idx); hold on; grid on;

            % Get state indices for estimation and ground truth
            est_state_idx = state_indices(xy_idx, state_idx);
            gt_state_idx = gt_indices(xy_idx, state_idx);

            % Calculate errors
            state_error = zeros(1, current_k);
            sigma_bounds = zeros(1, current_k);

            for k = 1:current_k

                if strcmp(lower(filter_type), 'hmm')
                    % HMM: GT has only position, x_est has only position
                    if est_state_idx <= size(x_est, 1) && gt_state_idx <= size(GT, 1)
                        state_error(k) = GT(gt_state_idx, k) - x_est(est_state_idx, k);
                    end

                else
                    % KF/PF: Both GT and x_est have full state
                    if gt_state_idx <= size(GT, 1) && est_state_idx <= size(x_est, 1)
                        state_error(k) = GT(gt_state_idx, k) - x_est(est_state_idx, k);
                    end

                end

                % Extract covariance bounds if available
                if ~isempty(P_est{k}) && est_state_idx <= size(P_est{k}, 1)
                    sigma_bounds(k) = sqrt(P_est{k}(est_state_idx, est_state_idx));
                end

            end

            % Plot the error
            if xy_idx == 1
                color = 'r'; % Red for x components
            else
                color = 'b'; % Blue for y components
            end

            plot(tvec, state_error, 'Color', color, 'LineWidth', 1.5, 'DisplayName', sprintf('%s Error', upper(char('X' + xy_idx - 1))));

            % Plot 2-sigma bounds if available
            if any(sigma_bounds > 0)
                plot(tvec, 2 * sigma_bounds, '--', 'Color', color, 'LineWidth', 1, 'DisplayName', '2\sigma');
                plot(tvec, -2 * sigma_bounds, '--', 'Color', color, 'LineWidth', 1);

                % Fill the confidence region
                fill([tvec, fliplr(tvec)], [2 * sigma_bounds, fliplr(-2 * sigma_bounds)], ...
                    color, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
            end

            % Labels and formatting
            if state_idx == num_states
                xlabel('Time (s)', 'Interpreter', 'latex');
            end

            ylabel(sprintf('%s %s', upper(char('X' + xy_idx - 1)), state_labels{state_idx}), 'Interpreter', 'latex');

            % Title for each subplot
            title_str = sprintf('%s %s Error', upper(char('X' + xy_idx - 1)), state_names{state_idx});
            title(title_str, 'Interpreter', 'latex');

            grid on;

            % Set consistent y-limits based on error magnitude
            if any(abs(state_error) > 0)
                max_error = max(abs(state_error));

                if any(sigma_bounds > 0)
                    max_bound = max(2 * sigma_bounds);
                    y_lim = max(max_error * 1.1, max_bound * 1.1);
                else
                    y_lim = max_error * 1.2;
                end

                ylim([-y_lim, y_lim]);
            end

        end

    end

    %% Overall title and save
    % Escape underscores for LaTeX interpreter
    dataset_name_escaped = strrep(dataset_name, '_', '\_');
    filter_type_escaped = strrep(filter_type, '_', '\_');
    da_method_escaped = strrep(da_method, '_', '\_');

    sgtitle(sprintf('%s-%s %s Tuning: k=%d/%d', filter_type_escaped, da_method_escaped, dataset_name_escaped, current_k, size(GT, 2)), ...
        'FontSize', 14, 'Interpreter', 'latex');

    % Save plot
    filename = sprintf('%s_%s_%s.png', filter_type, da_method, dataset_name);
    filepath = fullfile(save_dir, filename);

    exportgraphics(gcf, filepath, 'Resolution', 150);

    if mod(current_k, 10) == 0 || current_k == 1 % Print progress every 10 steps
        fprintf('Saved tuning plot: %s\n', filename);
    end

end

%% Helper Functions
function [X, Y] = calc_gsigma_ellipse_plotpoints(muin, Sigma, g, npoints)
    % Calculate ellipse points for plotting confidence regions
    % Same implementation as in mWidar_FilterPlot_Distribution.m

    [R, D, thetalocx] = subf_rotategaussianellipse(Sigma, g);

    if Sigma(1) < Sigma(4)
        a = 1 / sqrt(D(4));
        b = 1 / sqrt(D(1));
    elseif Sigma(1) >= Sigma(4)
        a = 1 / sqrt(D(1));
        b = 1 / sqrt(D(4));
    end

    mux = muin(1);
    muy = muin(2);

    if Sigma(2) ~= 0
        [X Y] = calculateEllipse(mux, muy, a, b, rad2deg(thetalocx), npoints);
    else
        [X Y] = calculateEllipse(mux, muy, a, b, 0, npoints);
    end

    function [R, D, thetalocx] = subf_rotategaussianellipse(Sigma, g)
        P = inv(Sigma);
        P = 0.5 * (P + P');

        a11 = P(1);
        a12 = P(2);
        a22 = P(4);
        c = -g ^ 2;

        mu = 1 / (-c);
        m11 = mu * a11;
        m12 = mu * a12;
        m22 = mu * a22;

        lambda1 = 0.5 * (m11 + m22 + sqrt((m11 - m22) .^ 2 + 4 * m12 .^ 2));
        lambda2 = 0.5 * (m11 + m22 - sqrt((m11 - m22) .^ 2 + 4 * m12 .^ 2));

        D = diag([lambda2, lambda1]);

        if m11 >= m22
            u11 = lambda1 - m22;
            u12 = m12;
        elseif m11 < m22
            u11 = m12;
            u12 = lambda1 - m11;
        end

        norm1 = sqrt(u11 .^ 2 + u12 .^ 2);
        U1 = ([u11; u12]) / norm1;
        U2 = [-u12; u11];
        R = [U1, U2];

        thetalocx = 0.5 * atan(-2 * a12 / (a22 - a11));
    end

end

function [X Y] = calculateEllipse(x, y, a, b, angle, steps)
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
