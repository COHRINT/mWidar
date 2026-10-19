%%% Collect per-trial metrics into tables and optionally plot a comparison.
%%%
%%%   S.perTrial  one row per trial: scenario, variant, sweep, seed, metrics
%%%   S.grouped   one row per group (scenario|variant|sweep), mean and std of
%%%               every metric over the MC seeds, plus GroupCount
%%%   S.fig       comparison figure, or [] when not drawn

function S = summarize(trials, t)

    M = struct2table([trials.metrics], 'AsArray', true);
    info = table(string({trials.group})', string({trials.scenario})', string({trials.variant})', ...
                 string({trials.sweep})', [trials.seed]', ...
                 'VariableNames', {'group', 'scenario', 'variant', 'sweep', 'seed'});
    S.perTrial = [info, M];

    metricVars = M.Properties.VariableNames;
    S.grouped = groupsummary(S.perTrial, 'group', {'mean', 'std'}, metricVars);

    fprintf("\n---------------- %s: summary (%d trials, %d groups) ----------------\n", ...
        t.name, height(S.perTrial), height(S.grouped));
    show = S.grouped(:, [{'group', 'GroupCount'}, strcat('mean_', metricVars)]);
    show.Properties.VariableNames = erase(show.Properties.VariableNames, 'mean_');
    disp(show);

    S.fig = [];
    nGroups = height(S.grouped);
    if t.output.summaryPlot && (nGroups > 1 || any(S.grouped.GroupCount > 1))
        S.fig = plot_summary(S.perTrial, t.name);
    end

end

function fig = plot_summary(T, name)

    show = {'rmse', 'Position RMSE [m]';
            'detect', 'Detection fraction';
            'delay', 'Detection delay [frames]';
            'false_alarm', 'False alarm fraction';
            'ess_med', 'Median ESS / N';
            'runtime', 'Runtime [s]'};

    g = categorical(T.group, unique(T.group, 'stable'));
    fig = figure('Name', name + " summary", 'Color', 'w', 'Position', [100 100 1200 700]);
    tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, strrep(name, '_', '\_') + " (" + height(T) + " trials)");
    for i = 1:size(show, 1)
        ax = nexttile(tl);
        y = T.(show{i, 1});
        if all(isnan(y)), axis(ax, 'off'); title(ax, show{i, 2} + " (n/a)"); continue; end
        boxchart(ax, g, y);
        hold(ax, 'on');
        swarmchart(ax, g, y, 12, 'filled', 'MarkerFaceAlpha', 0.5);
        hold(ax, 'off');
        title(ax, show{i, 2});
        ax.TickLabelInterpreter = 'none';
        ax.Toolbar.Visible = 'off';
        grid(ax, 'on');
    end

end
