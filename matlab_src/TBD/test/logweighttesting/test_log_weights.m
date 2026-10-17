%%% Script used to validate the functions and methods for log weight stuff.
%%% Runs one weight update, on one fixed particle set and one frame, three ways:
%%%   1. TBD_PF's own methods: importance_weights per particle ('gauss'
%%%      likelihood), then normalize_log. This is exactly what timestep does
%%%      between the regime transition and the resample.
%%%   2. A hand-written log-domain loop (reference).
%%%   3. The same loop in the linear domain: product of per-pixel LRs.
%%% 1 and 2 should agree to round-off. 3 agrees with them wherever the
%%% linear-domain L doesn't over/underflow.
%%% Not covered: the 'model' likelihood, the regime transition, dynamics, the
%%% birth correction logBirthCorr and the resample. The particle set below
%%% plays the role of Y in timestep (already transitioned and propagated, no
%%% newborns), so none of those apply.
%%% Run from matlab_src/.

addpath("TBD/Includes")

%%% Configure a static environment
trajs = {'static'};
rw = [false];
v = 0;
e = environment('Debug', true,...
    'Seed', 1,...
    'kEnd', 2,...
    'trajectories', trajs,...
    'kBirth', 1,...
    'kDeath', 2,...
    'Noise', false);

%%% plot the environment
scenario = e.setup();
%fig = e.show(scenario, 'Animate', true, 'Save', "TBD/test/logweighttesting/figs");
z = scenario.signal{1};

%%% Filter object whose methods are under test. TBD_PF.run sets likelihood and
%%% NoiseSTD, so set them by hand the same way run would.
N = 1000; % # of particles;
pf = TBD_PF('N', N, 'Seed', 1);   % defaults: Sigma = 0.5 px, pDist = 5
pf.likelihood = 'gauss';
Var = 0;                           % matches the environment ('Noise', false)
if Var ~= 0                        % same rule as TBD_PF.run
    pf.NoiseSTD = sqrt(Var);
else
    pf.NoiseSTD = 0.1;
end

%%% Parameters for the hand-written reference, read off pf so both sides use
%%% the same numbers. The y = 0.25 m gate is hard-coded in importance_weights.
prm = struct('xgrid', pf.xgrid, 'ygrid', pf.ygrid, 'dx', pf.dx, 'dy', pf.dy, ...
    'npx', pf.npx, 'pDist', pf.pDist, ...
    'sig',  pf.Sigma * pf.dx, ...      % PSF width, Sigma is in pixels
    'sigN', pf.NoiseSTD, ...
    'min_x', pf.min_x, 'max_x', pf.max_x, 'yMin', 0.25, 'max_y', pf.max_y);

%%% Setup a single TBD_PF timestep.

%%% Prior particle set, going to have 50% of particles exist
%%% Starting states should be sampled around the true state, (x, y) = (0, 2) m.
%%% I is drawn the way sample_new draws it, so the I factor in the PSF
%%% amplitude is exercised (I = 1 everywhere would hide a missing I).
prior_pos_vec = [mvnrnd(0,1,N)'; mvnrnd(0,0.5,N)'; mvnrnd(2,1,N)'; mvnrnd(0,0.5,N)'; ...
                 pf.I_min + (pf.I_max - pf.I_min) * rand(1,N)];
prior_E = [true(1,N/2),false(1,N/2)];

prior = [prior_pos_vec;prior_E];   % 6 x N, [px; vx; py; vy; I; E]

% start with uniform log and normal weights
w_prior = 1/N * ones(1,N);
logw_prior = -log(N) * ones(1,N);

%%% 1. TBD_PF's own methods (TBD_PF.timestep, "Evaluate importance weights"
%%% and "Update")
logL_pf = zeros(1, N);
for n = 1:N
    logL_pf(n) = pf.importance_weights(prior(:,n), z);
end
[w_pf, logw_pf] = pf.normalize_log(logw_prior + logL_pf);

%%% 2. and 3. Hand-written reference, log domain and linear domain
[w_lin, logw_ref, logL_ref, L_ref] = like_test(prior, w_prior, logw_prior, z, prm);
w_ref    = exp(logw_ref);   % log domain, exp'd back
logw_lin = log(w_lin);      % -Inf where w underflowed to 0

%% Checks
%%% Error is |a - b| / max(1, |a|, |b|): relative for the large log values,
%%% absolute for weights in [0, 1] (so a tiny weight vs an exact 0 passes).
%%% Non-finite entries (-Inf from the gate, NaN from overflow) must match
%%% exactly.
tol = 1e-10;
fprintf('\n--- TBD_PF vs reference (tol %.0e) ---\n', tol);
okCore = [compare_vec('logL:  TBD_PF vs reference',      logL_pf, logL_ref, tol), ...
          compare_vec('logw:  TBD_PF vs reference',      logw_pf, logw_ref, tol), ...
          compare_vec('w:     TBD_PF vs reference',      w_pf,    w_ref,    tol)];
fprintf('--- linear domain (may fail where L over/underflows) ---\n');
compare_vec('w:     TBD_PF vs linear reference', w_pf,     w_lin,    tol);
compare_vec('logL:  log(L) vs reference',        log(L_ref), logL_ref, tol);
fprintf('linear w: %d NaN, %d Inf, %d zero | TBD_PF w: %d zero | L: %d Inf, %d zero\n', ...
    nnz(isnan(w_lin)), nnz(isinf(w_lin)), nnz(w_lin == 0), nnz(w_pf == 0), ...
    nnz(isinf(L_ref)), nnz(L_ref == 0));
fprintf('sum(w): TBD_PF %.15g, reference %.15g, linear %.15g\n', sum(w_pf), sum(w_ref), sum(w_lin));
if all(okCore)
    fprintf('CORE CHECKS PASS: TBD_PF matches the reference\n\n');
else
    fprintf('CORE CHECKS FAIL: see the lines above\n\n');
end

%% Plots: TBD_PF vs the hand-written reference
alive  = prior(6,:) ~= 0;
mixErr = @(a, b) abs(a - b) ./ max(1, max(abs(a), abs(b)));   % same error as compare_vec

figW = figure('Name', 'TBD_PF vs reference weights', 'Color', 'w', 'Position', [100 100 1500 850]);
tl = tiledlayout(figW, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, sprintf('TBD\\_PF vs hand-written weight update, $N = %d$, %d alive, $\\sigma_n = %g$', ...
    N, nnz(alive), pf.NoiseSTD), 'Interpreter', 'latex');

%%% (a) logL per alive particle, TBD_PF against reference. Should sit on y = x.
ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
finL = alive & isfinite(logL_pf) & isfinite(logL_ref);
plot(ax, logL_pf(finL), logL_ref(finL), 'o', 'MarkerSize', 5, 'DisplayName', 'reference');
if any(finL)
    lim = [min([logL_pf(finL), logL_ref(finL)]), max([logL_pf(finL), logL_ref(finL)])];
    plot(ax, lim, lim, 'k--', 'DisplayName', '$y = x$');
end
xlabel(ax, '$\log L$, TBD\_PF', 'Interpreter', 'latex');
ylabel(ax, '$\log L$, reference', 'Interpreter', 'latex');
title(ax, sprintf('(a) $\\log L$: %d gated in both, %d gate mismatches', ...
    nnz(alive & isinf(logL_pf) & isinf(logL_ref)), nnz(isfinite(logL_pf) ~= isfinite(logL_ref))), ...
    'Interpreter', 'latex');
legend(ax, 'Interpreter', 'latex', 'Location', 'northwest');

%%% (b) logL error per particle. Exact zeros don't show on a log axis, so the
%%% title counts them.
ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
eL = mixErr(logL_ref, logL_pf);
nz = finL & eL > 0;
plot(ax, find(nz), eL(nz), 'o', 'MarkerSize', 5, 'DisplayName', 'reference');
yline(ax, eps, 'k:', 'DisplayName', 'machine $\epsilon$');
set(ax, 'YScale', 'log');
xlim(ax, [1 N]);
xlabel(ax, 'particle index', 'Interpreter', 'latex');
ylabel(ax, '$|\Delta \log L| \,/\, \max(1, |\log L|)$', 'Interpreter', 'latex');
title(ax, sprintf('(b) $\\log L$ error: max %.2g, %d of %d exactly equal', ...
    max([eL(finL), 0]), nnz(finL & eL == 0), nnz(finL)), 'Interpreter', 'latex');
legend(ax, 'Interpreter', 'latex', 'Location', 'best');

%%% (c) Sorted normalized weights, all three versions.
ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
[~, ord] = sort(w_pf, 'descend');
plot(ax, 1:N, w_pf(ord),  '-',  'LineWidth', 1.5, 'DisplayName', 'TBD\_PF');
plot(ax, 1:N, w_ref(ord), 'o',  'MarkerSize', 5,  'DisplayName', 'reference, log');
plot(ax, 1:N, w_lin(ord), 'x',  'MarkerSize', 5,  'DisplayName', 'reference, linear');
set(ax, 'YScale', 'log');
xlabel(ax, 'particle rank, sorted by TBD\_PF $w$', 'Interpreter', 'latex');
ylabel(ax, 'normalized weight $w$', 'Interpreter', 'latex');
title(ax, sprintf('(c) sorted $w$, zeros: %d TBD\\_PF, %d log, %d linear', ...
    nnz(w_pf == 0), nnz(w_ref == 0), nnz(w_lin == 0)), 'Interpreter', 'latex');
legend(ax, 'Interpreter', 'latex', 'Location', 'southwest');

%%% (d) Reference weights against TBD_PF's, one point per particle.
ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
posR = w_pf > 0 & w_ref > 0;
posL = w_pf > 0 & isfinite(w_lin) & w_lin > 0;
plot(ax, w_pf(posR), w_ref(posR), 'o', 'MarkerSize', 5, 'DisplayName', 'reference, log');
plot(ax, w_pf(posL), w_lin(posL), 'x', 'MarkerSize', 5, 'DisplayName', 'reference, linear');
if any(posR)
    lim = [min([w_pf(posR), w_ref(posR)]), max([w_pf(posR), w_ref(posR)])];
    plot(ax, lim, lim, 'k--', 'DisplayName', '$y = x$');
end
set(ax, 'XScale', 'log', 'YScale', 'log');
xlabel(ax, '$w$, TBD\_PF', 'Interpreter', 'latex');
ylabel(ax, '$w$, reference', 'Interpreter', 'latex');
title(ax, sprintf('(d) per particle: %d log pairs, %d linear pairs $> 0$', nnz(posR), nnz(posL)), ...
    'Interpreter', 'latex');
legend(ax, 'Interpreter', 'latex', 'Location', 'northwest');

%%% (e) Weight error against weight size. Round-off sits near eps; anything
%%% well above it is a math mismatch or an under/overflow in the linear domain.
ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
eR  = mixErr(w_ref, w_pf);
eLn = mixErr(w_lin, w_pf);
nzR = w_pf > 0 & eR > 0;    % NaN > 0 is false, so overflowed entries drop out
nzL = w_pf > 0 & eLn > 0;
plot(ax, w_pf(nzR), eR(nzR),  'o', 'MarkerSize', 5, 'DisplayName', 'reference, log');
plot(ax, w_pf(nzL), eLn(nzL), 'x', 'MarkerSize', 5, 'DisplayName', 'reference, linear');
yline(ax, eps, 'k:', 'DisplayName', 'machine $\epsilon$');
set(ax, 'XScale', 'log', 'YScale', 'log');
xlabel(ax, '$w$, TBD\_PF', 'Interpreter', 'latex');
ylabel(ax, '$|\Delta w|$', 'Interpreter', 'latex');
title(ax, sprintf('(e) $w$ error: max %.2g log, %.2g linear', max([eR, 0]), max([eLn, 0])), ...
    'Interpreter', 'latex');
legend(ax, 'Interpreter', 'latex', 'Location', 'best');

%%% (f) Normalized log weights against TBD_PF's, where both are finite. The
%%% title counts particles that only TBD_PF kept finite.
ax = nexttile(tl); hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
finR = isfinite(logw_pf) & isfinite(logw_ref);
finLn = isfinite(logw_pf) & isfinite(logw_lin);
plot(ax, logw_pf(finR),  logw_ref(finR),  'o', 'MarkerSize', 5, 'DisplayName', 'reference, log');
plot(ax, logw_pf(finLn), logw_lin(finLn), 'x', 'MarkerSize', 5, 'DisplayName', 'reference, linear');
if any(finR)
    lim = [min([logw_pf(finR), logw_ref(finR)]), max([logw_pf(finR), logw_ref(finR)])];
    plot(ax, lim, lim, 'k--', 'DisplayName', '$y = x$');
end
xlabel(ax, '$\log w$, TBD\_PF', 'Interpreter', 'latex');
ylabel(ax, '$\log w$, reference', 'Interpreter', 'latex');
title(ax, sprintf('(f) $\\log w$: finite only in TBD\\_PF: %d log, %d linear', ...
    nnz(isfinite(logw_pf) & ~isfinite(logw_ref)), nnz(isfinite(logw_pf) & ~isfinite(logw_lin))), ...
    'Interpreter', 'latex');
legend(ax, 'Interpreter', 'latex', 'Location', 'northwest');

% exportgraphics(figW, "TBD/test/logweighttesting/w_vs_logw.png", 'Resolution', 150);

%%% Compare two vectors. Non-finite entries must match exactly; finite ones to
%%% within tol in |a - b| / max(1, |a|, |b|). Prints one line, returns pass/fail.
function ok = compare_vec(name, a, b, tol)
    a = reshape(double(a), 1, []);
    b = reshape(double(b), 1, []);
    finA = isfinite(a);
    finB = isfinite(b);
    patOK = isequal(finA, finB) && isequaln(a(~finA), b(~finB));
    fin = finA & finB;
    err = abs(a(fin) - b(fin)) ./ max(1, max(abs(a(fin)), abs(b(fin))));
    maxErr = max([err, 0]);
    ok = patOK && maxErr <= tol;
    res = 'FAIL';
    if ok, res = 'PASS'; end
    fprintf('%-36s max err %.3g over %d finite, %d non-finite, pattern %d -> %s\n', ...
        name, maxErr, nnz(fin), nnz(~finA), patOK, res);
end

%%% function just to calculate the particle weights in both log and regular domain
%%%   w     normalized weights from the linear-domain product L
%%%   logw  normalized log weights from the log-domain sum logL
function [w, logw, logL, L] = like_test(prior, w_prior, logw_prior, z, prm)

    N = size(prior, 2);
    logL = zeros(1, N);
    L    = ones(1, N);
    for n = 1:N
        [logL(n), L(n)] = like(prior(:,n), z, prm);
    end
    %%% Update from priors
    logw = logL + logw_prior;
    w = L.*w_prior;

    %%% Normalize. Log-sum-exp, so logw stays finite even where exp(logw - m)
    %%% underflows (log of the normalized exp would give -Inf there). Half the
    %%% particles are dead with logL = 0, so m is always finite here; TBD_PF's
    %%% all -Inf fallback is not reproduced.
    w = w./sum(w);
    m = max(logw);
    logw = logw - m - log(sum(exp(logw - m)));

end

%%% Mirrors TBD_PF.importance_weights for the 'gauss' likelihood
function [logL,L] = like(Y, z, prm)
    X = Y(1:5);
    E = Y(6);

    logL = 0;   % dead particle: LR = 1
    L = 1;

    if ~E, return; end

    px = X(1);
    py = X(3);
    I  = X(5);

    % Outside the scene, or below the y = 0.25 m gate: LR = 0
    if px > prm.max_x || px < prm.min_x || py > prm.max_y || py < prm.yMin
        logL = -Inf;
        L = 0;
        return
    end

    % meters -> pixel index, grid points are pixel centers
    ix = round((px - prm.xgrid(1)) / prm.dx) + 1;
    iy = round((py - prm.ygrid(1)) / prm.dy) + 1;

    xvec = max(1, ix-prm.pDist):min(prm.npx, ix+prm.pDist);
    yvec = max(1, iy-prm.pDist):min(prm.npx, iy+prm.pDist);

    xm = prm.xgrid(xvec);
    ym = prm.ygrid(yvec);

    % calculate logL
    for i = 1:numel(xvec)
        for j = 1:numel(yvec)
            logL = logL + log_gauss_like([px py], [xm(i), ym(j)], [xvec(i), yvec(j)], z, I, prm);
        end
    end

    % calculate L
    for i = 1:numel(xvec)
        for j = 1:numel(yvec)
            L = L * gauss_like([px py], [xm(i), ym(j)], [xvec(i), yvec(j)], z, I, prm);
        end
    end
end

%%% Per-pixel log LR, same expression as TBD_PF.guass_likelihood
%%%   pos [px py] meters, pix [x y] pixel center in meters, idx [col row]
function [ll] = log_gauss_like(pos, pix, idx, z, I, prm)
    px = pos(1);
    py = pos(2);

    ii = pix(1);
    jj = pix(2);

    hh = I * exp( - ((ii -px)^2 + ((jj -py)^2))/ (2 * prm.sig^2));
    zz = z(idx(2), idx(1)); % rows = y, cols = x
    ll = (-(hh*(hh - 2*zz))/(2 * prm.sigN^2));
end

function [ll] = gauss_like(pos, pix, idx, z, I, prm)
    ll = exp(log_gauss_like(pos, pix, idx, z, I, prm));
end
