%%% Run a TBD test case from TBD_PF_test/config.
%%%
%%%   out = run_test('T5')        % unique prefix of a config/T*.m file
%%%   out = run_test(1)
%%%   out = run_test('T4_PF_Line_N5000_v15_s1')
%%%   out = run_test()            % list the available tests and prompt
%%%   out = run_test('T1', 'perTrial', false, 'save', false)
%%%                               % override any t.output field for this call
%%%
%%% Pipeline (see TEST_FRAMEWORK.md):
%%%   load_test -> expand_trials -> for each trial:
%%%       environment.setup (reused while env cfg is unchanged)
%%%       build_filter -> f.run -> compute_metrics -> f.show (optional)
%%%   -> summarize -> save to results/<name>_<timestamp>/
%%%
%%% out.test      the merged test definition
%%% out.trials    trial struct array (cfg, labels, metrics, R if output.keepR)
%%% out.summary   per-trial and grouped metric tables (see summarize)
%%% out.dir       output folder ('' if output.save is false)
%%%
%%% Runs from matlab_src (the filter loads supplemental/*.mat by relative
%%% path); the working directory is restored afterwards.

function out = run_test(name, varargin)

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(fileparts(here)));       % .../matlab_src
    addpath(fullfile(root, 'TBD', 'Includes'), fullfile(here, 'core'), fullfile(here, 'config'));
    oldDir = cd(root);
    restoreDir = onCleanup(@() cd(oldDir));

    if nargin == 0 || isempty(name)
        name = prompt_for_test(fullfile(here, 'config'));
    end

    %% Test definition -> trials
    t = load_test(name);
    if ~isempty(varargin)
        t.output = apply_overrides(t.output, struct(varargin{:}), "run_test output");
    end
    trials = expand_trials(t);
    nT = numel(trials);

    fprintf("\n================ %s ================\n", t.name);
    fprintf("%s\n", t.description);
    fprintf("%d trial(s)\n", nT);

    %% Output folder
    outDir = '';
    if t.output.save
        rootDir = t.output.rootDir;
        if rootDir == "", rootDir = fullfile(here, 'results'); end
        outDir = fullfile(rootDir, sprintf('%s_%s', t.name, char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))));
        mkdir(fullfile(outDir, 'figures'));
        fprintf("output: %s\n", outDir);
    end

    %% Run
    envKey = '';
    scenario = [];
    [trials.metrics] = deal([]);
    [trials.R] = deal([]);

    for n = 1:nT
        cfg = trials(n).cfg;
        fprintf("\n--- trial %d/%d: %s ---\n", n, nT, trials(n).label);

        % Per-trial output paths, unless the config set them explicitly
        if ~isempty(outDir)
            if isempty(cfg.show.Save),    cfg.show.Save    = fullfile(outDir, 'figures', trials(n).file); end
            if isempty(cfg.run.DebugDir), cfg.run.DebugDir = fullfile(outDir, 'debug'); end
        end
        if isempty(cfg.run.DebugName), cfg.run.DebugName = trials(n).file; end
        trials(n).cfg = cfg;

        % Scenario: the same generated data for every trial with the same env cfg
        key = jsonencode(cfg.env);
        if ~strcmp(key, envKey)
            envArgs = name_value(cfg.env);
            e = environment(envArgs{:});
            scenario = e.setup();
            envKey = key;
        else
            fprintf("reusing scenario\n");
        end

        if isempty(cfg.TBD.K)
            cfg.TBD.K = scenario.K;
            trials(n).cfg = cfg;
        end

        f = build_filter(cfg);
        runArgs = name_value(cfg.run);
        tStart = tic;
        R = f.run(scenario, runArgs{:});
        runtime = toc(tStart);

        trials(n).metrics = compute_metrics(R, scenario, runtime);
        fprintf("rmse=%.3f m  detect=%.2f  false_alarm=%.2f  delay=%g  runtime=%.1f s\n", ...
            trials(n).metrics.rmse, trials(n).metrics.detect, trials(n).metrics.false_alarm, ...
            trials(n).metrics.delay, runtime);

        if t.output.perTrial
            showArgs = name_value(cfg.show);
            [fig, figAnim] = f.show(R, scenario, showArgs{:});
            if t.output.closeFigs
                close([fig, figAnim]);
            end
        end

        if t.output.keepR || t.output.saveR
            trials(n).R = R;
        end
    end

    %% Summary and save
    summary = summarize(trials, t);

    if ~isempty(outDir)
        if ~isempty(summary.fig)
            exportgraphics(summary.fig, fullfile(outDir, 'summary.png'));
        end
        writetable(summary.perTrial, fullfile(outDir, 'trials.csv'));
        writetable(summary.grouped,  fullfile(outDir, 'summary.csv'));

        saved = trials;
        if ~t.output.saveR, saved = rmfield(saved, 'R'); end
        test = t; %#ok<NASGU>
        tables = rmfield(summary, 'fig'); %#ok<NASGU>
        save(fullfile(outDir, 'trials.mat'), 'test', 'saved', 'tables', '-v7.3');
        fprintf("\nsaved results to %s\n", outDir);
    end

    if ~t.output.keepR, trials = rmfield(trials, 'R'); end

    out = struct('test', t, 'trials', trials, 'summary', summary, 'dir', outDir);

end

function name = prompt_for_test(cfgDir)
    files = dir(fullfile(cfgDir, 'T*.m'));
    fprintf("Available tests:\n");
    fprintf("  %s\n", files.name);
    name = input('Select test (e.g. T1 or 1): ', 's');
end
