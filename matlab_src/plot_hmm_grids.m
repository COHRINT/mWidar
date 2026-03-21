function plot_hmm_grids(result_file, varargin)
% PLOT_HMM_GRIDS  Visualise the weighted HMM mixture grid at every timestep.
%
% USAGE:
%   plot_hmm_grids('data/TUNING_DATASET1/HMM_RBPF_N150.mat')
%   plot_hmm_grids(..., 'animate', true)
%   plot_hmm_grids(..., 'animate', true, 'delay', 0.15)
%   plot_hmm_grids(..., 'animate', true, 'gif', 'out.gif')
%
% INPUTS:
%   result_file - Path to .mat file saved by run_experiment (contains 'R').
%                 If omitted or '', a file picker dialog opens.
%
% OPTIONS (name-value pairs):
%   'animate'  - true/false (default false): single-frame animation view
%   'delay'    - seconds between animation frames (default 0.2)
%   'gif'      - filename string: save animation as GIF (animate must be true)
%
% The result struct must contain:
%   R.hmm_mixture  [128 x 128 x N_k]  - weighted mixture grids
%   R.GT           [6 x N_k]          - ground truth (rows 1-2 = position)
%   R.x_est        [N_x x N_k]        - MMSE estimates
%   R.x_map        [2 x N_k]          - MAP estimates (optional)
%   R.measurements {1 x N_k}          - raw measurements
%   R.params.dt    scalar              - timestep duration

%% ---- Parse arguments -------------------------------------------------------
p = inputParser;
addOptional(p, 'result_file_arg', '', @ischar);
addParameter(p, 'animate', false, @islogical);
addParameter(p, 'delay',   0.2,   @(x) isnumeric(x) && x >= 0);
addParameter(p, 'gif',     '',    @ischar);
parse(p, varargin{:});
opt = p.Results;

if nargin >= 1 && ~isempty(result_file)
    mat_path = result_file;
else
    [f, d] = uigetfile('*.mat', 'Select run_experiment result file');
    if isequal(f, 0)
        error('plot_hmm_grids:NoFile', 'No file selected.');
    end
    mat_path = fullfile(d, f);
end

%% ---- Load result -----------------------------------------------------------
tmp = load(mat_path, 'R');
R   = tmp.R;

if ~isfield(R, 'hmm_mixture') || isempty(R.hmm_mixture)
    error('plot_hmm_grids:NoGrid', ...
        'R.hmm_mixture is empty or missing. Re-run run_experiment with an HMM_RBPF filter.');
end

grids  = R.hmm_mixture;         % [128 x 128 x N_k]
GT     = R.GT;                   % [6 x N_k]
x_est  = R.x_est;               % [N_x x N_k]
z_all  = R.measurements;        % {1 x N_k}
n_k    = size(grids, 3);
dt     = R.params.dt;

has_map = isfield(R, 'x_map') && ~all(isnan(R.x_map(:)));
if has_map
    x_map = R.x_map;            % [2 x N_k]
end

%% ---- Grid coordinates (match HMM.m defaults) --------------------------------
xgrid = linspace(-2, 2, 128);
ygrid = linspace( 0, 4, 128);

%% ---- Shared colour limit ----------------------------------------------------
% Use 99th-percentile of all grids to avoid one bright frame dominating
all_vals = grids(:);
clim_max = prctile(all_vals(all_vals > 0), 99);
if clim_max == 0, clim_max = 1; end
clim_val = [0, clim_max];

%% ---- Branch: animate or static subplot matrix ------------------------------
if opt.animate
    plot_animate(grids, GT, x_est, z_all, n_k, dt, xgrid, ygrid, clim_val, ...
                 has_map, x_map, opt);
else
    plot_static(grids, GT, x_est, z_all, n_k, dt, xgrid, ygrid, clim_val, ...
                has_map, x_map);
end

end % main function

%% ============================================================================
%  STATIC SUBPLOT VIEW
%% ============================================================================
function plot_static(grids, GT, x_est, z_all, n_k, dt, xgrid, ygrid, clim_val, has_map, x_map)

n_cols = 9;
n_rows = 6;
n_cells = n_rows * n_cols;   % 54 cells, use first min(n_k, 54)
n_show  = min(n_k, n_cells);

fig = figure('Name', 'HMM Mixture Grids', 'Color', 'k', ...
             'Position', [50, 50, 1600, 900]);
colormap(fig, hot);

for k = 1:n_show
    ax = subplot(n_rows, n_cols, k);

    imagesc(ax, xgrid, ygrid, grids(:,:,k));
    set(ax, 'YDir', 'normal');
    clim(ax, clim_val);
    axis(ax, 'tight');
    set(ax, 'XTick', [], 'YTick', []);

    hold(ax, 'on');

    % Ground truth
    plot(ax, GT(1,k), GT(2,k), 'g+', 'MarkerSize', 6, 'LineWidth', 1.5);

    % MMSE estimate
    plot(ax, x_est(1,k), x_est(2,k), 'co', 'MarkerSize', 4, 'LineWidth', 1.2);

    % MAP estimate
    if has_map && ~any(isnan(x_map(:,k)))
        plot(ax, x_map(1,k), x_map(2,k), 'ms', 'MarkerSize', 4, 'LineWidth', 1.2);
    end

    % Measurements
    z_k = z_all{k};
    if ~isempty(z_k)
        plot(ax, z_k(1,:), z_k(2,:), 'wx', 'MarkerSize', 4, 'LineWidth', 1);
    end

    title(ax, sprintf('k=%d  t=%.1fs', k, (k-1)*dt), ...
          'Color', 'w', 'FontSize', 6, 'FontWeight', 'normal');

    hold(ax, 'off');
end

% Shared colorbar on the right
cb = colorbar('eastoutside');
cb.Color = 'w';
cb.Label.String = 'Mixture probability';
cb.Label.Color  = 'w';
% Move colorbar outside subplot grid
cb.Position(1) = 0.93;

% Legend as text in figure
annotation(fig, 'textbox', [0.005, 0.003, 0.35, 0.025], ...
    'String', '  +  GT    o  MMSE    s  MAP    x  Measurements', ...
    'Color', 'w', 'EdgeColor', 'none', 'FontSize', 8, 'BackgroundColor', 'none');

sgtitle(fig, 'HMM RBPF — Weighted Mixture Grid per Timestep', 'Color', 'w');

end

%% ============================================================================
%  ANIMATION VIEW
%% ============================================================================
function plot_animate(grids, GT, x_est, z_all, n_k, dt, xgrid, ygrid, clim_val, ...
                      has_map, x_map, opt)

fig = figure('Name', 'HMM Mixture Animation', 'Color', 'k', ...
             'Position', [200, 100, 700, 600]);
colormap(fig, hot);
ax = axes(fig);

do_gif = ~isempty(opt.gif);

for k = 1:n_k
    cla(ax);

    imagesc(ax, xgrid, ygrid, grids(:,:,k));
    set(ax, 'YDir', 'normal');
    clim(ax, clim_val);
    axis(ax, 'tight');
    colorbar(ax);

    hold(ax, 'on');

    plot(ax, GT(1,k), GT(2,k), 'g+', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'GT');
    plot(ax, x_est(1,k), x_est(2,k), 'co', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'MMSE');

    if has_map && ~any(isnan(x_map(:,k)))
        plot(ax, x_map(1,k), x_map(2,k), 'ms', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'MAP');
    end

    z_k = z_all{k};
    if ~isempty(z_k)
        plot(ax, z_k(1,:), z_k(2,:), 'wx', 'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName', 'Meas');
    end

    hold(ax, 'off');
    legend(ax, 'Location', 'northeast', 'TextColor', 'w', 'Color', 'k');
    title(ax, sprintf('k = %d / %d    t = %.2f s', k, n_k, (k-1)*dt), 'Color', 'w');
    xlabel(ax, 'X (m)', 'Color', 'w');
    ylabel(ax, 'Y (m)', 'Color', 'w');
    ax.XColor = 'w'; ax.YColor = 'w';
    drawnow;

    if do_gif
        frame = getframe(fig);
        im    = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        if k == 1
            imwrite(imind, cm, opt.gif, 'gif', 'Loopcount', inf, 'DelayTime', opt.delay);
        else
            imwrite(imind, cm, opt.gif, 'gif', 'WriteMode', 'append', 'DelayTime', opt.delay);
        end
    end

    pause(opt.delay);
end

if do_gif
    fprintf('GIF saved: %s\n', opt.gif);
end

end
