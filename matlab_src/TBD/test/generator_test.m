%%% generator_test.m
%%% Exercises generator.m end to end: constructor + validators, every
%%% trajectory type with and without random walk, multi-object generation,
%%% the input-validation guards, and the out-of-bounds fallback.
%%% Object count is capped at 3 throughout.
%%%
%%% Set SAVE_FIGS = true before running to write PNGs into TBD/test/figs.

close all;
if ~exist('SAVE_FIGS','var'), SAVE_FIGS = false; end
clearvars -except SAVE_FIGS;


% Resolve TBD/Includes relative to this file so the test runs from any cwd
TESTDIR = fileparts(mfilename('fullpath'));
if isempty(TESTDIR), TESTDIR = fullfile(pwd,'TBD','test'); end
addpath(fullfile(TESTDIR,'..','Includes'));

rng(7);  % reproducible random walks and random fallbacks

FIGDIR = fullfile(TESTDIR,'figs');
if SAVE_FIGS && ~exist(FIGDIR,'dir'), mkdir(FIGDIR); end

T.pass = 0; T.fail = 0; T.failures = {};

v = visualize('Units','meters');

%% ===================================================================
%% 1. Constructor: defaults
%% ===================================================================
fprintf('\n=== 1. Constructor defaults ===\n');

g = generator();

T = check(T, strcmp(g.units,'meters'),        'default units are meters');
T = check(T, g.kEnd == 100,                   'default kEnd is 100');
T = check(T, abs(g.dt - 0.1) < eps,           'default dt is 0.1');
T = check(T, g.ct == 1,                       'default object count is 1');
T = check(T, isequal(g.trajectories,{'line'}),'default trajectory is a single line');
T = check(T, g.rw == false,                   'random walk is off by default');
T = check(T, g.debug == false,                'debug is off by default');
T = check(T, numel(g.tvec) == g.kEnd + 1,     'tvec has kEnd+1 samples');
T = check(T, abs(g.tvec(end) - g.kEnd*g.dt) < 1e-12, 'tvec ends at kEnd*dt');
T = check(T, isequal(size(g.A),[6 6]),        'A is 6x6');
T = check(T, isequal(size(g.F),[6 6]),        'F is 6x6');
T = check(T, norm(g.F - expm(g.A*g.dt)) < 1e-12, 'F is expm(A*dt)');
% inherited from mWidar (implicit superclass construction)
T = check(T, g.npx == 128 && g.Lscene == 4,   'mWidar scene properties are inherited');

%% ===================================================================
%% 2. Constructor: named arguments and validators
%% ===================================================================
fprintf('\n=== 2. Constructor named arguments ===\n');

g2 = generator('Debug',true, 'Units','pixels', 'kEnd',50, 'dt',0.05, ...
               'objs',3, 'trajectories',{'line','parabola','scurve'}, ...
               'rw',[false true false]);

T = check(T, g2.debug == true,                 'Debug flag is stored');
T = check(T, strcmp(g2.units,'pixels'),        'Units accepts "pixels"');
T = check(T, g2.kEnd == 50,                    'kEnd accepts a whole number');
T = check(T, abs(g2.dt - 0.05) < eps,          'dt is stored');
T = check(T, g2.ct == 3,                       'objs accepts 3');
T = check(T, numel(g2.trajectories) == 3,      'a 3-element trajectory list is accepted');
T = check(T, isequal(g2.rw,[false true false]),'rw accepts a logical vector');
T = check(T, numel(g2.tvec) == 51,             'tvec tracks kEnd/dt');

% Validators should reject bad input
T = check(T, throws(@() generator('trajectories',{'spiral'})), ...
              'unknown trajectory name is rejected');
T = check(T, throws(@() generator('kEnd', 10.5)), ...
              'non-integer kEnd is rejected');
T = check(T, throws(@() generator('objs', 2.7)), ...
              'non-integer objs is rejected');
T = check(T, throws(@() generator('Units','furlongs')), ...
              'unknown unit is rejected');
T = check(T, ~throws(@() generator('kEnd', 200)), ...
              'a valid whole-number kEnd is accepted');

%% ===================================================================
%% 3. Single object: every trajectory type, random walk off and on
%% ===================================================================
fprintf('\n=== 3. Single-object trajectory types ===\n');

types  = {'line','parabola','scurve'};
p0     = [-1.5, 0.5];
p1     = [ 1.5, 3.5];

fig1 = figure('Name','Trajectory types','Color','w', ...
              'Position',[80 80 1300 700]);
singles = cell(2, numel(types));

for r = 1:2                       % row 1 = analytic, row 2 = random walk
    rw_on = (r == 2);
    for c = 1:numel(types)
        name = sprintf('%s%s', types{c}, ternary(rw_on,' + rw',''));

        gi = generator('trajectories', types(c), 'rw', rw_on);
        X  = gi.generate_trajectories(p0, p1);

        T = check(T, iscell(X) && numel(X) == 1, '%s: returns a 1x1 cell', name);
        T = validate_track(T, X{1}, gi.tvec, p0, p1, name, 5e-2);

        singles{r,c} = X{1};

        ax = subplot(2, numel(types), (r-1)*numel(types) + c, 'Parent', fig1);
        v.trajectories(X, 'Axes', ax, 'Title', name, 'Labels', {name});
        place_legend(ax);
    end
end
sgtitle(fig1, 'generator: trajectory types (analytic top, random walk bottom)');

% The random walk must actually perturb the analytic path
for c = 1:numel(types)
    d = max(vecnorm(singles{2,c}([1 3],:) - singles{1,c}([1 3],:)));
    T = check(T, d > 1e-6, '%s: random walk deviates from the analytic path (max %.3f m)', types{c}, d);
end

%% ===================================================================
%% 4. Multi-object generation, up to 3 objects
%% ===================================================================
fprintf('\n=== 4. Multi-object generation (1 to 3 objects) ===\n');

starts_all = { [-1.5 0.5], [-1.8 3.0], [ 0.0 0.3] };
finals_all = { [ 1.5 3.5], [ 1.6 0.6], [ 1.2 3.8] };
types_all  = { 'line', 'parabola', 'scurve' };
rw_all     = [ false,  true,        false ];

fig2 = figure('Name','Multi-object scenes','Color','w', ...
              'Position',[100 100 1400 460]);
multi = cell(1,3);

for n = 1:3
    gm = generator('objs', n, 'trajectories', types_all(1:n), 'rw', rw_all(1:n));
    Xm = gm.generate_trajectories(starts_all(1:n), finals_all(1:n));

    T = check(T, iscell(Xm) && numel(Xm) == n, '%d object(s): returns a 1x%d cell', n, n);
    for i = 1:n
        nm = sprintf('%d obj / target %d (%s%s)', n, i, types_all{i}, ternary(rw_all(i),' + rw',''));
        T = validate_track(T, Xm{i}, gm.tvec, starts_all{i}, finals_all{i}, nm, 5e-2);
    end

    multi{n} = Xm;

    ax = subplot(1,3,n,'Parent',fig2);
    v.trajectories(Xm, 'Axes', ax, 'Title', sprintf('%d object(s)', n), ...
                   'Labels', types_all(1:n));
    place_legend(ax);
end
sgtitle(fig2, 'generator: 1 to 3 simultaneous objects');

% Targets must be distinct tracks, not copies of one another
X3 = multi{3};
T = check(T, norm(X3{1} - X3{2}) > 1e-6 && norm(X3{2} - X3{3}) > 1e-6, ...
          '3 objects: each target has its own trajectory');

%% ===================================================================
%% 5. Time histories for the 3-object case
%% ===================================================================
fprintf('\n=== 5. Time histories ===\n');

g3   = generator('objs',3,'trajectories',types_all,'rw',rw_all);
tvec = g3.tvec;
cols = lines(3);

fig3 = figure('Name','State time histories','Color','w', ...
              'Position',[120 120 1200 760]);
rows = {'$x$ [m]','$v_x$ [m/s]','$y$ [m]','$v_y$ [m/s]'};
for s = 1:4
    ax = subplot(2,2,s,'Parent',fig3); hold(ax,'on'); grid(ax,'on');
    for i = 1:3
        plot(ax, tvec, X3{i}(s,:), 'LineWidth', 1.4, 'Color', cols(i,:), ...
             'DisplayName', types_all{i});
    end
    xlabel(ax,'t [s]');
    ylabel(ax, rows{s}, 'Interpreter','latex');
    title(ax, rows{s}, 'Interpreter','latex');
    if s == 1, legend(ax,'Location','best'); end
end
sgtitle(fig3, 'generator: 3-object state time histories');

% Speeds should be physically sensible for a 10 s traverse of a 4 m scene
for i = 1:3
    sp = hypot(X3{i}(2,:), X3{i}(4,:));
    T = check(T, max(sp) < 10, 'target %d: peak speed %.2f m/s is sane', i, max(sp));
end

%% ===================================================================
%% 6. Analytic vs random walk, same endpoints
%% ===================================================================
fprintf('\n=== 6. Analytic vs random walk overlay ===\n');

fig4 = figure('Name','Analytic vs random walk','Color','w', ...
              'Position',[140 140 1300 420]);
for c = 1:numel(types)
    ax = subplot(1,3,c,'Parent',fig4); hold(ax,'on'); grid(ax,'on');
    plot(ax, singles{1,c}(1,:), singles{1,c}(3,:), '-',  'LineWidth',1.8, ...
         'Color',[0.10 0.45 0.85], 'DisplayName','analytic');
    plot(ax, singles{2,c}(1,:), singles{2,c}(3,:), '--', 'LineWidth',1.8, ...
         'Color',[0.85 0.10 0.10], 'DisplayName','random walk');
    plot(ax, p0(1), p0(2), 'ks', 'MarkerFaceColor','k', 'DisplayName','start');
    plot(ax, p1(1), p1(2), 'kd', 'MarkerFaceColor','k', 'DisplayName','final');
    axis(ax,'equal'); xlim(ax,[-2 2]); ylim(ax,[0 4]);
    xlabel(ax,'x [m]'); ylabel(ax,'y [m]'); title(ax, types{c});
    if c == 1, legend(ax,'Location','northwest'); end
end
sgtitle(fig4, 'generator: random walk perturbs the path but holds the endpoints');

%% ===================================================================
%% 7. Input-validation guards
%% ===================================================================
fprintf('\n=== 7. Input validation guards ===\n');
fprintf('(the ABORTING messages below are expected)\n');

% rw length does not match the number of trajectories
ga = generator('trajectories',{'line','parabola'}, 'rw', false);
Xa = ga.generate_trajectories({[-1 1],[0 1]}, {[1 3],[1 2]});
T = check(T, iscell(Xa) && isempty(Xa), 'rw/trajectory count mismatch aborts cleanly');

% start is a cell but final is not
gb = generator('trajectories',{'line'}, 'rw', false);
Xb = gb.generate_trajectories({[-1 1]}, [1 3]);
T = check(T, iscell(Xb) && isempty(Xb), 'mixed cell/vector start and final aborts cleanly');

% cell lengths do not match the number of trajectories
gc = generator('trajectories',{'line','parabola','scurve'}, 'rw',[false false false]);
Xc = gc.generate_trajectories({[-1 1],[0 1]}, {[1 3],[1 2]});
T = check(T, iscell(Xc) && isempty(Xc), 'short start/final cell aborts cleanly');

% plain vectors cannot describe more than one trajectory
gd = generator('trajectories',{'line','parabola'}, 'rw',[false false]);
Xd = gd.generate_trajectories([-1 1], [1 3]);
T = check(T, iscell(Xd) && isempty(Xd), 'vector start/final with 2 trajectories aborts cleanly');

% the valid single-object vector form still works
ge = generator();
Xe = ge.generate_trajectories([-1 1], [1 3]);
T = check(T, iscell(Xe) && numel(Xe) == 1 && ~isempty(Xe{1}), ...
          'vector start/final with 1 trajectory is accepted');

%% ===================================================================
%% 8. Out-of-bounds start/final falls back to a random in-scene pair
%% ===================================================================
fprintf('\n=== 8. Out-of-bounds fallback ===\n');
fprintf('(the DEBUG fallback message below is expected)\n');

gf = generator('Debug', true);
Xf = gf.generate_trajectories([-99 -99], [99 99]);
Sf = Xf{1};

T = check(T, all(Sf(1,:) >= -2-1e-9 & Sf(1,:) <= 2+1e-9), 'fallback keeps x in scene');
T = check(T, all(Sf(3,:) >=  0-1e-9 & Sf(3,:) <= 4+1e-9), 'fallback keeps y in scene');
T = check(T, norm([Sf(1,1) Sf(3,1)] - [-99 -99]) > 1, 'fallback ignores the out-of-bounds start');
T = check(T, all(isfinite(Sf(:))), 'fallback trajectory is finite');

% Only one endpoint out of bounds also triggers the fallback
Xg = gf.generate_trajectories([-1 1], [99 99]);
T = check(T, all(Xg{1}(1,:) >= -2-1e-9 & Xg{1}(1,:) <= 2+1e-9), ...
          'single bad endpoint also falls back in scene');

fig5 = figure('Name','Out-of-bounds fallback','Color','w');
ax = axes(fig5);
v.trajectories(Xf, 'Axes', ax, 'Title', 'Out-of-bounds request -> random in-scene path', ...
               'Labels', {'fallback'});

%% ===================================================================
%% 9. kEnd / dt propagate into the generated track length
%% ===================================================================
fprintf('\n=== 9. Track length follows kEnd and dt ===\n');

for kE = [20 60 150]
    gk = generator('kEnd', kE, 'dt', 0.2);
    Xk = gk.generate_trajectories([-1 1],[1 3]);
    T = check(T, size(Xk{1},2) == kE + 1, 'kEnd=%d gives %d samples', kE, kE+1);
    T = check(T, abs(gk.tvec(end) - kE*0.2) < 1e-12, 'kEnd=%d gives the right horizon', kE);
end

%% ===================================================================
%% Save figures / summary
%% ===================================================================
if SAVE_FIGS
    figs  = [fig1 fig2 fig3 fig4 fig5];
    names = {'types','multi_object','time_histories','rw_overlay','oob_fallback'};
    for i = 1:numel(figs)
        exportgraphics(figs(i), fullfile(FIGDIR, [names{i} '.png']), 'Resolution', 150);
    end
    fprintf('\nFigures written to %s\n', FIGDIR);
end

fprintf('\n===================================\n');
fprintf(' generator_test: %d passed, %d failed\n', T.pass, T.fail);
if T.fail > 0
    fprintf('---- failures ----\n');
    for i = 1:numel(T.failures)
        fprintf('  %s\n', T.failures{i});
    end
end
fprintf('===================================\n');

%% ===================================================================
%% Local helpers
%% ===================================================================

function T = check(T, cond, fmt, varargin)
    msg = sprintf(fmt, varargin{:});
    if cond
        T.pass = T.pass + 1;
        fprintf('  [PASS] %s\n', msg);
    else
        T.fail = T.fail + 1;
        T.failures{end+1} = msg;
        fprintf('  [FAIL] %s\n', msg);
    end
end

%%% Shape / physical sanity checks shared by every generated track
function T = validate_track(T, X, tvec, start, final, name, velTol)
    n_t = numel(tvec);
    T = check(T, isequal(size(X),[4 n_t]), '%s: state is 4 x %d', name, n_t);
    if ~isequal(size(X),[4 n_t]), return; end

    T = check(T, all(isfinite(X(:))), '%s: all states are finite', name);

    x = X(1,:); vx = X(2,:); y = X(3,:); vy = X(4,:);

    T = check(T, all(x >= -2-1e-9 & x <= 2+1e-9), '%s: x stays inside [-2, 2]', name);
    T = check(T, all(y >=  0-1e-9 & y <= 4+1e-9), '%s: y stays inside [0, 4]', name);

    % A path must be fitted inside the scene, never clamped at the edge:
    % clamping pins position, flattens the reported velocity to zero and
    % then steps it discontinuously on release, which no motion model can follow.
    pinned = abs(x + 2) < 1e-12 | abs(x - 2) < 1e-12 | abs(y) < 1e-12 | abs(y - 4) < 1e-12;
    T = check(T, ~any(pinned), '%s: no samples clamped to a scene edge (%d of %d)', ...
              name, sum(pinned), numel(pinned));

    if ~isempty(start)
        T = check(T, norm([x(1) y(1)] - start(:)') < 1e-9, '%s: starts at the requested point', name);
    end
    if ~isempty(final)
        T = check(T, norm([x(end) y(end)] - final(:)') < 1e-9, '%s: ends at the requested point', name);
    end

    dt  = tvec(2) - tvec(1);
    err = max([abs(gradient(x, dt) - vx), abs(gradient(y, dt) - vy)]);
    sc  = max([1, max(abs(vx)), max(abs(vy))]);
    T = check(T, err <= velTol*sc, ...
              '%s: velocity agrees with d(position)/dt (rel err %.2e)', name, err/sc);
end

%%% visualize.trajectories asks for 'best', which lands on the subplot title
function place_legend(ax)
    lg = legend(ax);
    if isgraphics(lg)
        lg.Location = 'southeast';
    end
end

function tf = throws(f)
    tf = false;
    try
        f();
    catch
        tf = true;
    end
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
