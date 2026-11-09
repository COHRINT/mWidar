%%% Parallel, resumable runner for the NoiseSTD tuning sweeps (T7-T10).
%%%
%%%   run_sweep('T7')                                 % coarse pass from the config
%%%   run_sweep('T7', 'Var', 15, 'NoiseSTD', 300:10:800)
%%%                                                   % refine one noise level
%%%   run_sweep('T7', 'ReportOnly', true)             % rebuild tables/plots only
%%%
%%% Options
%%%   'NoiseSTD'    values to run, replacing the config's list
%%%   'Var'         only these additive noise variances (default: all in the config)
%%%   'Seeds'       only these MC seeds (default: all in the config)
%%%   'Workers'     parallel pool size (default: the NumWorkers limit of the
%%%                 'Processes' profile; raise that limit in Parallel Preferences
%%%                 to use more)
%%%   'RootDir'     results root (default: TBD_PF_test/results)
%%%   'ReportOnly'  skip running; rebuild the outputs from what has finished
%%%
%%% Stopping and resuming: every finished run is written to results/<test>/progress
%%% straight away. Run the same command again and it picks up where it stopped.
%%% Refining uses the same folder, so new NoiseSTD values join the existing tables.
%%% A run that errors is logged in errors.log and retried on the next call.
%%%
%%% Output, in results/<test>/
%%%   summary.csv     one row per Var + NoiseSTD: mean and std over seeds
%%%   summary.png     metric vs NoiseSTD, one column per Var
%%%   summary.fig     same figure; open it to zoom and read off values
%%%   trials.csv      one row per run
%%%   test.mat        the test definition
%%%
%%% Same seed reuses the same scenario at every NoiseSTD, and the filter RNG is
%%% reset for each run, so differences between NoiseSTD values are not sampling
%%% noise from different scenarios. Metrics are from core/compute_metrics.

function out = run_sweep(name, varargin)

    p = inputParser;
    addParameter(p, 'NoiseSTD', [], @(x) isempty(x) || (isnumeric(x) && all(x > 0)));
    addParameter(p, 'Var', [], @isnumeric);
    addParameter(p, 'Seeds', [], @isnumeric);
    addParameter(p, 'Workers', [], @(x) isempty(x) || (isscalar(x) && x >= 1));
    addParameter(p, 'RootDir', "");
    addParameter(p, 'ReportOnly', false, @islogical);
    parse(p, varargin{:});
    opt = p.Results;

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(fileparts(here)));       % .../matlab_src
    paths = {fullfile(root, 'TBD', 'Includes'), fullfile(here, 'core'), fullfile(here, 'config')};
    addpath(paths{:});
    oldDir = cd(root);
    restoreDir = onCleanup(@() cd(oldDir));

    t = load_test(name);
    assert(numel(t.sweep) == 1 && string(t.sweep.path) == "run.NoiseSTD", ...
        "run_sweep: test '%s' does not sweep run.NoiseSTD", t.name);

    rootDir = string(opt.RootDir);
    if rootDir == "", rootDir = fullfile(here, 'results'); end
    outDir = char(fullfile(rootDir, t.name));
    progDir = fullfile(outDir, 'progress');
    if ~isfolder(progDir), mkdir(progDir); end
    save(fullfile(outDir, 'test.mat'), 't');

    fprintf("\n================ %s ================\n%s\noutput: %s\n", t.name, t.description, outDir);

    errs = strings(0, 1);
    if ~opt.ReportOnly
        work = plan_work(t, opt, progDir);
        if isempty(work)
            fprintf("nothing to run, every requested combination has finished\n");
        else
            errs = run_work(work, opt.Workers, root, paths);
        end
        write_errors(outDir, errs);
    end

    [T, S] = write_report(outDir, t);
    out = struct('test', t, 'trials', T, 'summary', S, 'dir', outDir, 'errors', errs);

end

%% ------------------------------------------------------------------
%% Planning
%% ------------------------------------------------------------------

%%% One work item per scenario + seed, holding the NoiseSTD values still to run.
%%% A worker handles a whole item, so it only generates the scenario once.
function work = plan_work(t, opt, progDir)

    values = t.sweep.values;
    if ~isempty(opt.NoiseSTD), values = opt.NoiseSTD; end
    values = unique(values(:)');

    %%% expand_trials once per scenario + seed; NoiseSTD is set by the worker
    one = t;
    one.sweep.values = values(1);
    groups = expand_trials(one);

    allVars = arrayfun(@(g) g.cfg.env.Var, groups);
    allSeeds = [groups.seed];
    assert(all(ismember(opt.Var, allVars)), "run_sweep: 'Var' must be from %s", mat2str(unique(allVars)));
    assert(all(ismember(opt.Seeds, allSeeds)), "run_sweep: 'Seeds' must be from %s", mat2str(unique(allSeeds)));

    keep = true(size(groups));
    if ~isempty(opt.Var),   keep = keep & ismember(allVars, opt.Var);    end
    if ~isempty(opt.Seeds), keep = keep & ismember(allSeeds, opt.Seeds); end
    groups = groups(keep);

    work = struct('trial', {}, 'todo', {}, 'file', {});
    for g = 1:numel(groups)
        file = fullfile(progDir, sprintf('%s_s%d.csv', groups(g).scenario, groups(g).seed));
        done = [];
        if isfile(file)
            repair_progress(file);
            T = read_progress(file);
            if ~isempty(T), done = T.NoiseSTD'; end
        end
        todo = setdiff(values, done);
        if ~isempty(todo)
            work(end+1) = struct('trial', groups(g), 'todo', todo, 'file', file); %#ok<AGROW>
        end
    end

end

%% ------------------------------------------------------------------
%% Running
%% ------------------------------------------------------------------

function errs = run_work(work, nWorkers, root, paths)

    if isempty(nWorkers)
        nWorkers = parcluster('Processes').NumWorkers;
    end

    pool = gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers ~= nWorkers
        delete(pool);
        pool = [];
    end
    if isempty(pool)
        pool = parpool('Processes', nWorkers);
    end

    nRuns = sum(cellfun(@numel, {work.todo}));
    fprintf("%d run(s) across %d scenario/seed group(s) on %d worker(s)\n", nRuns, numel(work), pool.NumWorkers);

    q = parallel.pool.DataQueue;
    state = containers.Map({'done', 'total', 't0'}, {0, nRuns, tic});
    afterEach(q, @(msg) print_progress(msg, state));

    %%% One group at a time per worker, so a slow group cannot hold others back
    popts = parforOptions(pool, 'RangePartitionMethod', 'fixed', 'SubrangeSize', 1);
    errCell = cell(1, numel(work));
    parfor (g = 1:numel(work), popts)
        errCell{g} = run_group(work(g), root, paths, q);
    end
    errs = vertcat(strings(0, 1), errCell{:});

    fprintf("finished: %d run(s) attempted, %d error(s)\n", nRuns, numel(errs));

end

%%% Runs on a worker. Mirrors the trial loop in run_test.
function errs = run_group(w, root, paths, q)

    errs = strings(0, 1);
    tag = sprintf('%s s%d', w.trial.scenario, w.trial.seed);

    cd(root);
    addpath(paths{:});

    %%% Workers default to a different RNG; match a normal MATLAB session so a
    %%% sweep result can be reproduced outside the pool.
    rng(0, 'twister');

    cfg = w.trial.cfg;
    try
        envArgs = name_value(cfg.env);
        e = environment(envArgs{:});
        scenario = e.setup();
    catch ME
        errs(end+1, 1) = sprintf("%s: scenario setup failed: %s", tag, ME.message);
        for v = w.todo
            send(q, struct('tag', tag, 'NoiseSTD', v, 'metrics', []));
        end
        return
    end
    if isempty(cfg.TBD.K), cfg.TBD.K = scenario.K; end

    for v = w.todo
        cfg.run.NoiseSTD = v;
        try
            f = build_filter(cfg);
            runArgs = name_value(cfg.run);
            tStart = tic;
            R = f.run(scenario, e, runArgs{:});
            m = compute_metrics(R, scenario, toc(tStart));
            append_progress(w.file, cfg.env.Var, v, cfg.env.Seed, m);
            send(q, struct('tag', tag, 'NoiseSTD', v, 'metrics', m));
        catch ME
            errs(end+1, 1) = sprintf("%s NoiseSTD=%g: %s (%s line %d)", tag, v, ME.message, ...
                ME.stack(1).name, ME.stack(1).line); %#ok<AGROW>
            send(q, struct('tag', tag, 'NoiseSTD', v, 'metrics', []));
        end
    end

end

function print_progress(msg, state)

    state('done') = state('done') + 1;
    done = state('done');
    total = state('total');
    elapsed = toc(state('t0'));
    eta = elapsed / done * (total - done);

    if isempty(msg.metrics)
        result = "ERROR, see errors.log";
    else
        result = sprintf("rmse=%.3f detect=%.2f false_alarm=%.2f run=%.0fs", ...
            msg.metrics.rmse, msg.metrics.detect, msg.metrics.false_alarm, msg.metrics.runtime);
    end
    fprintf("[%d/%d] %s NoiseSTD=%g  %s  (ETA %s)\n", done, total, msg.tag, msg.NoiseSTD, result, ...
        char(duration(0, 0, eta)));

end

function write_errors(outDir, errs)
    file = fullfile(outDir, 'errors.log');
    if isempty(errs)
        if isfile(file), delete(file); end
    else
        writelines(errs, file);
        fprintf("%d run(s) failed, see %s. Run the same command again to retry them.\n", numel(errs), file);
    end
end

%% ------------------------------------------------------------------
%% Progress files: one per scenario + seed, only ever written by one worker
%% ------------------------------------------------------------------

function append_progress(file, var, noiseSTD, seed, m)

    vals = [var, noiseSTD, seed, cell2mat(struct2cell(m))'];
    line = strjoin(compose("%.17g", vals), ",") + newline;

    if ~isfile(file)
        line = strjoin(["Var", "NoiseSTD", "seed", string(fieldnames(m))'], ",") + newline + line;
    end
    fid = fopen(file, 'a');
    assert(fid > 0, "run_sweep: cannot open %s", file);
    fwrite(fid, char(line));
    fclose(fid);

end

%%% A run killed mid-write can leave a partial last line, which would corrupt the
%%% next row appended after it. Cut the file back to its last complete line.
function repair_progress(file)

    txt = fileread(file);
    if isempty(txt) || txt(end) == newline, return; end

    last = find(txt == newline, 1, 'last');
    if isempty(last)
        delete(file);
        return
    end
    fid = fopen(file, 'w');
    fwrite(fid, txt(1:last));
    fclose(fid);

end

function T = read_progress(file)

    T = table();
    if ~isfile(file), return; end
    lines = splitlines(strtrim(string(fileread(file))));
    if numel(lines) < 2, return; end

    names = split(lines(1), ',')';
    data = sscanf(char(strjoin(lines(2:end), ',')), '%f,');
    data = reshape(data, numel(names), [])';
    T = array2table(data, 'VariableNames', cellstr(names));

end

%% ------------------------------------------------------------------
%% Report
%% ------------------------------------------------------------------

function [T, S] = write_report(outDir, t)

    files = dir(fullfile(outDir, 'progress', '*.csv'));
    parts = cell(1, numel(files));
    for i = 1:numel(files)
        file = fullfile(files(i).folder, files(i).name);
        repair_progress(file);
        parts{i} = read_progress(file);
    end
    T = vertcat(table(), parts{:});

    if isempty(T)
        S = table();
        fprintf("no finished runs to report yet\n");
        return
    end

    T = sortrows(T, {'Var', 'NoiseSTD', 'seed'});
    writetable(T, fullfile(outDir, 'trials.csv'));

    S = summarize_sweep(T);
    writetable(S, fullfile(outDir, 'summary.csv'));

    fig = plot_sweep(S, t.name);
    exportgraphics(fig, fullfile(outDir, 'summary.png'), 'Resolution', 150);
    %%% Saved hidden, so make it show itself when opened from disk
    set(fig, 'CreateFcn', 'set(gcbo, ''Visible'', ''on'')');
    savefig(fig, fullfile(outDir, 'summary.fig'));
    close(fig);

    fprintf("\n%d finished run(s), %d Var + NoiseSTD point(s), seeds per point: %d to %d\n", ...
        height(T), height(S), min(S.n_seeds), max(S.n_seeds));
    fprintf("start with %s\n", fullfile(outDir, 'summary.csv'));

end

%%% One row per Var + NoiseSTD. n_no_track counts seeds that never declared a
%%% track: those have no rmse or delay, so the means below only cover the others.
function S = summarize_sweep(T)

    metrics = {'rmse', 'detect', 'false_alarm', 'delay'};
    extras  = {'ess_med', 'resamples', 'runtime'};

    [gid, keys] = findgroups(T(:, {'Var', 'NoiseSTD'}));
    S = keys;
    S.n_seeds    = splitapply(@numel, T.seed, gid);
    S.n_no_track = splitapply(@(d) nnz(isnan(d)), T.delay, gid);
    for m = metrics
        S.(m{1} + "_mean") = splitapply(@(x) mean(x, 'omitnan'), T.(m{1}), gid);
        S.(m{1} + "_std")  = splitapply(@(x) std(x, 'omitnan'),  T.(m{1}), gid);
    end
    for m = extras
        S.(m{1} + "_mean") = splitapply(@(x) mean(x, 'omitnan'), T.(m{1}), gid);
    end
    S = sortrows(S, {'Var', 'NoiseSTD'});

    %%% Five decimals is plenty here, and keeps the CSV readable
    vars = S.Properties.VariableNames(5:end);
    S{:, vars} = round(S{:, vars}, 5);

end

%%% Small multiples: rows are metrics, columns are Var, each row sharing a y axis.
%%% Log x because the likelihood scales with 1/NoiseSTD^2. Band is +- 1 std over seeds.
function fig = plot_sweep(S, name)

    rows = {'rmse',        'Position RMSE [m]',       [];
            'detect',      'Detection fraction',      [0 1];
            'false_alarm', 'False alarm fraction',    [0 1];
            'delay',       'Detection delay [frames]', []};

    vars = unique(S.Var)';
    nr = size(rows, 1);
    nc = numel(vars);

    surface = [252 252 251] / 255;
    ink     = [11 11 11] / 255;
    inkSoft = [82 81 78] / 255;
    series  = [42 120 214] / 255;
    grid_c  = [225 224 220] / 255;

    fig = figure('Visible', 'off', 'Color', surface, 'Position', [50 50 300*nc + 90 210*nr + 70]);
    tl = tiledlayout(fig, nr, nc, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, name + " (mean, band = +/- 1 std over seeds)", 'Color', ink, 'FontSize', 13, 'Interpreter', 'none');
    xlabel(tl, 'NoiseSTD (log scale)', 'Color', inkSoft, 'Interpreter', 'none');

    axs = gobjects(nr, nc);
    for r = 1:nr
        for c = 1:nc
            ax = nexttile(tl);
            axs(r, c) = ax;
            hold(ax, 'on');

            sub = S(S.Var == vars(c), :);
            x = sub.NoiseSTD';
            y = sub.(rows{r, 1} + "_mean")';
            d = sub.(rows{r, 1} + "_std")';
            d(isnan(d)) = 0;

            draw_band(ax, x, y, d, series, rows{r, 3});
            plot(ax, x, y, '-', 'Color', series, 'LineWidth', 1.5, 'Marker', '.', 'MarkerSize', 8);

            set(ax, 'XScale', 'log', 'Box', 'off', 'Color', surface, 'XColor', inkSoft, 'YColor', inkSoft, ...
                'GridColor', grid_c, 'GridAlpha', 1, 'XGrid', 'on', 'YGrid', 'on', 'XMinorGrid', 'off', ...
                'XTick', [10 100 1000 10000], 'XTickLabel', {'10', '100', '1000', '10000'}, ...
                'TickLabelInterpreter', 'none', 'FontSize', 9);
            xlim(ax, [min(10, min(S.NoiseSTD)) max(10000, max(S.NoiseSTD))]);

            if r == 1
                title(ax, sprintf('Var = %g', vars(c)), 'Color', ink, 'FontSize', 11, 'Interpreter', 'none');
            end
            if c == 1
                ylabel(ax, rows{r, 2}, 'Color', ink, 'FontSize', 10, 'Interpreter', 'none');
            end
        end

        %%% Same y range across each row, so noise levels compare directly
        if ~isempty(rows{r, 3})
            set(axs(r, :), 'YLim', rows{r, 3} + [-0.02 0.02]);
        else
            linkaxes(axs(r, :), 'y');
            lim = ylim(axs(r, 1));
            ylim(axs(r, 1), [0 max(lim(2), eps)]);   % these metrics cannot go negative
        end
    end

end

%%% Shaded band, broken wherever the mean is NaN (no seed had a value there) and
%%% clipped to the metric's valid range: [0 1] for fractions, >= 0 otherwise.
function draw_band(ax, x, y, d, color, range)

    lo = max(y - d, 0);
    hi = y + d;
    if ~isempty(range), hi = min(hi, range(2)); end

    ok = ~isnan(y);
    edges = diff([false ok false]);
    starts = find(edges == 1);
    stops = find(edges == -1) - 1;
    for s = 1:numel(starts)
        i = starts(s):stops(s);
        fill(ax, [x(i) fliplr(x(i))], [lo(i), fliplr(hi(i))], color, ...
            'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end

end
