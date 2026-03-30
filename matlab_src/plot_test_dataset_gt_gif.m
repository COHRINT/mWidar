function fig = plot_test_dataset_gt_gif(data_dir,track)
%PLOT_TEST_DATASET_GT_GIF Plot Data.GT trajectories from .mat files in a 2x5 grid.
%   fig = PLOT_TEST_DATASET_GT_GIF(data_dir, track) loads all .mat files in data_dir and
%   plots each Data.GT trajectory in a 2x5 subplot layout.

    if nargin < 2 || (isstring(track) && strlength(track) == 0) || ...
            (ischar(track) && isempty(track))
        error('plot_test_dataset_gt_gif:MissingInput', ...
            'Provide a track name.');
    end

    if nargin < 1 || (isstring(data_dir) && strlength(data_dir) == 0) || ...
            (ischar(data_dir) && isempty(data_dir))
        error('plot_test_dataset_gt_gif:MissingInput', ...
            'Provide a directory containing .mat files.');
    end

    data_dir = char(data_dir);

    if ~isfolder(data_dir)
        error('plot_test_dataset_gt_gif:InvalidDirectory', ...
            'Directory does not exist: %s', data_dir);
    end

    mat_files = dir(fullfile(data_dir, '*.mat'));
    if isempty(mat_files)
        error('plot_test_dataset_gt_gif:NoMatFiles', ...
            'No .mat files found in: %s', data_dir);
    end

    [~, order] = sort(lower({mat_files.name}));
    mat_files = mat_files(order);

    file_numbers = nan(numel(mat_files), 1);
    for k = 1:numel(mat_files)
        tokens = regexp(mat_files(k).name, '\d+', 'match');
        if ~isempty(tokens)
            file_numbers(k) = str2double(tokens{end});
        end
    end

    [~, order] = sortrows([isnan(file_numbers), file_numbers]);
    mat_files = mat_files(order);

    if numel(mat_files) > 10
        warning('plot_test_dataset_gt:TooManyFiles', ...
            'Found %d .mat files. Plotting the first 10 in sorted order.', numel(mat_files));
        mat_files = mat_files(1:10);
    end

    fig = figure('Name', 'Test Dataset Ground Truth', 'Color', 'w');
    tl = tiledlayout(fig, 2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('Ground Truth Trajectories: %s', track), 'Interpreter', 'none');

    % Pre-load data and find maximum time steps
    max_steps = 0;
    datasets = cell(10, 1);
    for k = 1:min(10, numel(mat_files))
        S = load(fullfile(mat_files(k).folder, mat_files(k).name), 'Data');
        if ~isfield(S, 'Data') || ~isfield(S.Data, 'GT')
            error('plot_test_dataset_gt:MissingGT', ...
                'File does not contain Data.GT: %s', mat_files(k).name);
        end
        datasets{k} = S.Data;
        max_steps = max(max_steps, size(S.Data.GT, 2));
    end

    % Setup subplots (static like plot_test_dataset_gt)
    for k = 1:10
        ax = nexttile(tl, k);

        if k > numel(datasets) || isempty(datasets{k})
            axis(ax, 'off');
            continue;
        end

        grid(ax, 'on');
        axis(ax, 'equal');

        if isfield(datasets{k}, 'params') && isfield(datasets{k}.params, 'xgrid_range') && ...
                isfield(datasets{k}.params, 'ygrid_range')
            xlim(ax, datasets{k}.params.xgrid_range);
            ylim(ax, datasets{k}.params.ygrid_range);
        else
            xlim(ax, [-2, 2]);
            ylim(ax, [0, 4]);
        end

        title(ax, mat_files(k).name, 'Interpreter', 'none');
        xlabel(ax, 'x [m]');
        ylabel(ax, 'y [m]');
        hold(ax, 'on');

        GT = datasets{k}.GT;
        plot(ax, GT(1, :), GT(2, :), 'b-', 'LineWidth', 1.8);
        plot(ax, GT(1, 1),   GT(2, 1),   'go', 'MarkerFaceColor', 'g', 'MarkerSize', 5);
        plot(ax, GT(1, end), GT(2, end), 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    end

    % Save tiled PNG like plot_test_dataset_gt
    exportgraphics(fig, fullfile(data_dir, char(track) + ".png"), 'Resolution', 200);

    % ---- Per-file GIFs with signal + detections + growing trajectory ----
    for k = 1:numel(datasets)
        if isempty(datasets{k})
            continue;
        end

        Data = datasets{k};
        GT = Data.GT;
        Signal = [];
        if isfield(Data, 'signal'), Signal = Data.signal; end
        meas = [];
        if isfield(Data, 'y'), meas = Data.y; end

        n_steps = size(GT, 2);
        dt = 0.1;
        if isfield(Data, 'params') && isfield(Data.params, 'dt')
            dt = Data.params.dt;
        end

        x_range = [-2, 2]; y_range = [0, 4];
        if isfield(Data, 'params')
            if isfield(Data.params, 'xgrid_range'), x_range = Data.params.xgrid_range; end
            if isfield(Data.params, 'ygrid_range'), y_range = Data.params.ygrid_range; end
        end

        gif_name = fullfile(mat_files(k).folder, replace(mat_files(k).name, '.mat', '.gif'));
        if exist(gif_name, 'file'), delete(gif_name); end

        hfig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 700 600]);
        ax = axes('Parent', hfig);
        colormap(ax, parula);

        % Determine signal grid size for proper axes scaling
        npx_y = 128; npx_x = 128;
        if ~isempty(Signal)
            first_sig_idx = find(~cellfun(@isempty, Signal), 1, 'first');
            if ~isempty(first_sig_idx)
                [npx_y, npx_x] = size(Signal{first_sig_idx});
            end
        end
        x_lin = linspace(x_range(1), x_range(2), npx_x);
        y_lin = linspace(y_range(1), y_range(2), npx_y);

        % Pre-create artists
        imgH = imagesc(ax, x_lin, y_lin, zeros(npx_y, npx_x));
        set(ax, 'YDir', 'normal');
        hold(ax, 'on');
        detH = scatter(ax, NaN, NaN, 25, 'm', 'filled', 'MarkerFaceAlpha', 0.7, 'MarkerEdgeColor', 'k');
        trajH = plot(ax, NaN, NaN, 'c-', 'LineWidth', 2.0);
        currH = plot(ax, NaN, NaN, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);

        xlim(ax, x_range); ylim(ax, y_range);
        grid(ax, 'on'); axis(ax, 'equal');
        title(ax, sprintf('%s | %s', mat_files(k).name, track), 'Interpreter', 'none');
        xlabel(ax, 'x [m]'); ylabel(ax, 'y [m]');

        for t = 1:n_steps
            % Update background signal if present
            if ~isempty(Signal) && numel(Signal) >= t && ~isempty(Signal{t})
                imgH.CData = Signal{t};
            else
                imgH.CData = zeros(size(imgH.CData));
            end

            % Update detections (all clutter included)
            if ~isempty(meas) && numel(meas) >= t && ~isempty(meas{t})
                detH.XData = meas{t}(1, :);
                detH.YData = meas{t}(2, :);
            else
                detH.XData = NaN; detH.YData = NaN;
            end

            % Growing trajectory
            trajH.XData = GT(1, 1:t);
            trajH.YData = GT(2, 1:t);
            currH.XData = GT(1, t);
            currH.YData = GT(2, t);

            drawnow;
            frame = getframe(hfig);
            [im_ind, cmap] = rgb2ind(frame2im(frame), 256, 'nodither');
            if t == 1
                imwrite(im_ind, cmap, gif_name, 'gif', 'LoopCount', inf, 'DelayTime', dt);
            else
                imwrite(im_ind, cmap, gif_name, 'gif', 'WriteMode', 'append', 'DelayTime', dt);
            end
        end

        close(hfig);
    end

end
