function fig = plot_test_dataset_gt(data_dir,track)
%PLOT_TEST_DATASET_GT Plot Data.GT trajectories from .mat files in a 2x5 grid.
%   fig = PLOT_TEST_DATASET_GT(data_dir) loads all .mat files in data_dir and
%   plots each Data.GT trajectory in a 2x5 subplot layout.

    if nargin < 1 || (isstring(data_dir) && strlength(data_dir) == 0) || ...
            (ischar(data_dir) && isempty(data_dir))
        error('plot_test_dataset_gt:MissingInput', ...
            'Provide a directory containing .mat files.');
    end

    data_dir = char(data_dir);

    if ~isfolder(data_dir)
        error('plot_test_dataset_gt:InvalidDirectory', ...
            'Directory does not exist: %s', data_dir);
    end

    mat_files = dir(fullfile(data_dir, '*.mat'));
    if isempty(mat_files)
        error('plot_test_dataset_gt:NoMatFiles', ...
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

    for k = 1:10
        ax = nexttile(tl, k);

        if k > numel(mat_files)
            axis(ax, 'off');
            continue;
        end

        S = load(fullfile(mat_files(k).folder, mat_files(k).name), 'Data');
        if ~isfield(S, 'Data') || ~isfield(S.Data, 'GT')
            error('plot_test_dataset_gt:MissingGT', ...
                'File does not contain Data.GT: %s', mat_files(k).name);
        end

        GT = S.Data.GT;
        if size(GT, 1) < 2
            error('plot_test_dataset_gt:InvalidGTSize', ...
                'Data.GT must have at least two rows in file: %s', mat_files(k).name);
        end

        plot(ax, GT(1, :), GT(2, :), 'b-', 'LineWidth', 1.8);
        hold(ax, 'on');
        plot(ax, GT(1, 1),   GT(2, 1),   'go', 'MarkerFaceColor', 'g', 'MarkerSize', 5);
        plot(ax, GT(1, end), GT(2, end), 'rs', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
        hold(ax, 'off');

        grid(ax, 'on');
        axis(ax, 'equal');

        if isfield(S.Data, 'params') && isfield(S.Data.params, 'xgrid_range') && ...
                isfield(S.Data.params, 'ygrid_range')
            xlim(ax, S.Data.params.xgrid_range);
            ylim(ax, S.Data.params.ygrid_range);
        else
            xlim(ax, [-2, 2]);
            ylim(ax, [0, 4]);
        end

        title(ax, mat_files(k).name, 'Interpreter', 'none');
        xlabel(ax, 'x [m]');
        ylabel(ax, 'y [m]');

        
    end

    exportgraphics(gcf, data_dir + "\" + track + ".png", 'Resolution', 200);
    
end
