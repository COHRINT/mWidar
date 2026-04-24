%% Tune lambda_arrival and lambda_depart for recursive object-count filter
% Method:
%   1) Initialize lambdas from labeled count trajectory Data.obj_ct
%   2) Run local grid search around those estimates
%   3) Evaluate each pair with recursive filtering on observed detections
%      m_k = size(Data.y{k}, 2)
%
% Required files:
%   supplemental/probObjCt_results.mat
%   supplemental/Final_Test_Tracks/TI_test_case.mat

clc; close all; clear

%% --- Paths --------------------------------------------------------------
addpath(fullfile('DA_Track'))
addpath(fullfile('DA_Track', 'multi'))
addpath(fullfile('supplemental'))
addpath(fullfile('supplemental', 'Final_Test_Tracks'))

results_file = fullfile('supplemental', 'probObjCt_results.mat');
data_file = fullfile('supplemental', 'Final_Test_Tracks', 'TI_test_case.mat');

if ~isfile(results_file)
    error('Missing %s. Run supplemental/probObjCt.m first.', results_file);
end
if ~isfile(data_file)
    error('Missing %s. Run supplemental/Final_Test_Tracks/TI_test_case.m first.', data_file);
end

S = load(results_file, 'results');
D = load(data_file, 'Data');
results = S.results;
Data = D.Data;

if ~isfield(results, 'P_m_given_N')
    error('results.P_m_given_N is missing in %s.', results_file);
end
if ~isfield(Data, 'y') || isempty(Data.y)
    error('Data.y missing/empty in %s.', data_file);
end

%% --- Axes and observations ---------------------------------------------
P_m_given_N = results.P_m_given_N;
n_states = size(P_m_given_N, 1);

if isfield(results, 'N_vals') && numel(results.N_vals) == n_states
    N_vals = results.N_vals(:).';
elseif isfield(results, 'cfg') && isfield(results.cfg, 'max_obj') ...
        && n_states == (results.cfg.max_obj + 1)
    N_vals = 0:results.cfg.max_obj;
else
    N_vals = 1:n_states;
end

if isfield(results, 'det_axis') && numel(results.det_axis) == size(P_m_given_N, 2)
    m_axis = results.det_axis(:).';
else
    m_axis = 0:(size(P_m_given_N, 2) - 1);
end

det_count_seq = zeros(1, numel(Data.y));
for k = 1:numel(Data.y)
    det_count_seq(k) = size(Data.y{k}, 2);
end

if isfield(Data, 'params') && isfield(Data.params, 'T')
    start_k = max(1, Data.params.T);
else
    start_k = 1;
end

obs_seq = det_count_seq(start_k:end);
K = numel(obs_seq);
if K < 2
    error('Need at least 2 observations after start index %d.', start_k);
end

has_true = isfield(Data, 'obj_ct') && numel(Data.obj_ct) >= start_k;
if has_true
    true_seq = Data.obj_ct(start_k:end);
else
    true_seq = [];
end

%% --- Initial MLE-style estimates from true trajectory ------------------
if ~has_true
    error('Data.obj_ct required for lambda initialization.');
end

prev_counts = true_seq(1:end-1);
next_counts = true_seq(2:end);
arrivals = max(next_counts - prev_counts, 0);
departures = max(prev_counts - next_counts, 0);

lambda_arrival_init = mean(arrivals);
lambda_depart_init = sum(departures) / max(sum(prev_counts), eps);

%% --- Grid configuration -------------------------------------------------
arr_step = 0.02;
dep_step = 0.01;
arr_half_width = 0.30;
dep_half_width = 0.15;

arr_lo = max(0.0, lambda_arrival_init - arr_half_width);
arr_hi = min(1.0, lambda_arrival_init + arr_half_width);
dep_lo = max(0.0, lambda_depart_init - dep_half_width);
dep_hi = min(0.5, lambda_depart_init + dep_half_width);

if arr_hi <= arr_lo
    arr_lo = 0.0; arr_hi = 1.0;
end
if dep_hi <= dep_lo
    dep_lo = 0.0; dep_hi = 0.5;
end

arrival_grid = arr_lo:arr_step:arr_hi;
depart_grid = dep_lo:dep_step:dep_hi;

if isempty(arrival_grid), arrival_grid = lambda_arrival_init; end
if isempty(depart_grid), depart_grid = lambda_depart_init; end

%% --- Grid search --------------------------------------------------------
nA = numel(arrival_grid);
nD = numel(depart_grid);
map_acc_grid = nan(nA, nD);
mae_grid = nan(nA, nD);
logp_true_grid = nan(nA, nD);
logev_grid = nan(nA, nD);

best_score = -inf;
best = struct();
best.post_hist = [];
best.N_map_seq = [];

for ia = 1:nA
    for id = 1:nD
        lambda_arrival = arrival_grid(ia);
        lambda_depart = depart_grid(id);

        T_Nk_given_prev = buildPoissonCountTransition(N_vals, lambda_arrival, lambda_depart);
        [post_hist, N_map_seq, log_evidence] = runRecursiveFilter( ...
            obs_seq, P_m_given_N, m_axis, N_vals, T_Nk_given_prev);

        [map_acc, mae, logp_true] = scoreAgainstTruth(post_hist, N_map_seq, N_vals, true_seq);

        map_acc_grid(ia, id) = map_acc;
        mae_grid(ia, id) = mae;
        logp_true_grid(ia, id) = logp_true;
        logev_grid(ia, id) = log_evidence;

        score = logp_true;
        if score > best_score
            best_score = score;
            best.lambda_arrival = lambda_arrival;
            best.lambda_depart = lambda_depart;
            best.map_acc = map_acc;
            best.mae = mae;
            best.logp_true = logp_true;
            best.log_evidence = log_evidence;
            best.post_hist = post_hist;
            best.N_map_seq = N_map_seq;
            best.T_Nk_given_prev = T_Nk_given_prev;
        end
    end
end

%% --- Report -------------------------------------------------------------
fprintf('\nInitial estimates from Data.obj_ct:\n');
fprintf('  lambda_arrival_init = %.4f\n', lambda_arrival_init);
fprintf('  lambda_depart_init  = %.4f\n', lambda_depart_init);

fprintf('\nBest grid-search parameters:\n');
fprintf('  lambda_arrival = %.4f\n', best.lambda_arrival);
fprintf('  lambda_depart  = %.4f\n', best.lambda_depart);
fprintf('  MAP accuracy   = %.2f%%\n', best.map_acc);
fprintf('  MAE            = %.4f counts\n', best.mae);
fprintf('  log P(true|M)  = %.4f\n', best.logp_true);
fprintf('  log evidence   = %.4f\n', best.log_evidence);

%% --- Plots --------------------------------------------------------------
figure('Color', 'w', 'Position', [100 100 1200 760]);
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
imagesc(ax1, depart_grid, arrival_grid, map_acc_grid);
axis(ax1, 'xy');
xlabel(ax1, '\lambda_{depart}');
ylabel(ax1, '\lambda_{arrival}');
title(ax1, 'MAP Accuracy (%)');
colorbar(ax1);

ax2 = nexttile;
imagesc(ax2, depart_grid, arrival_grid, mae_grid);
axis(ax2, 'xy');
xlabel(ax2, '\lambda_{depart}');
ylabel(ax2, '\lambda_{arrival}');
title(ax2, 'MAE (counts)');
colorbar(ax2);

ax3 = nexttile;
imagesc(ax3, depart_grid, arrival_grid, logp_true_grid);
axis(ax3, 'xy');
xlabel(ax3, '\lambda_{depart}');
ylabel(ax3, '\lambda_{arrival}');
title(ax3, 'log P(True Sequence | M)');
colorbar(ax3);

ax4 = nexttile;
hold(ax4, 'on');
plot(ax4, 1:K, best.N_map_seq, 'b-', 'LineWidth', 1.5);
plot(ax4, 1:K, true_seq, 'k--', 'LineWidth', 1.2);
xlim(ax4, [1, K]);
ylim(ax4, [min(N_vals)-0.2, max(N_vals)+0.2]);
xlabel(ax4, 'Frame Index (post-start)');
ylabel(ax4, 'Object Count');
title(ax4, sprintf('Best MAP Sequence (\\lambda_a=%.3f, \\lambda_d=%.3f)', ...
    best.lambda_arrival, best.lambda_depart));
legend(ax4, 'MAP estimate', 'True count', 'Location', 'best');
grid(ax4, 'on');
hold(ax4, 'off');

%% --- Save tuning output -------------------------------------------------
LambdaTuning = struct();
LambdaTuning.data_file = data_file;
LambdaTuning.results_file = results_file;
LambdaTuning.start_k = start_k;
LambdaTuning.N_vals = N_vals;
LambdaTuning.m_axis = m_axis;
LambdaTuning.obs_seq = obs_seq;
LambdaTuning.true_seq = true_seq;
LambdaTuning.lambda_arrival_init = lambda_arrival_init;
LambdaTuning.lambda_depart_init = lambda_depart_init;
LambdaTuning.arrival_grid = arrival_grid;
LambdaTuning.depart_grid = depart_grid;
LambdaTuning.map_acc_grid = map_acc_grid;
LambdaTuning.mae_grid = mae_grid;
LambdaTuning.logp_true_grid = logp_true_grid;
LambdaTuning.logev_grid = logev_grid;
LambdaTuning.best = best;

save(fullfile('supplemental', 'lambda_tuning_results.mat'), 'LambdaTuning');
fprintf('Saved tuning results to supplemental\\lambda_tuning_results.mat\n');


%% FUNCTIONS

function [post_hist, N_map_seq, log_evidence] = runRecursiveFilter(obs_seq, P_m_given_N, m_axis, N_vals, T_Nk_given_prev)
n_states = numel(N_vals);
K = numel(obs_seq);

P_prev = ones(n_states, 1) ./ n_states;  % uniform prior
post_hist = zeros(n_states, K);
log_evidence = 0;

for k = 1:K
    mk = obs_seq(k);
    mk_col = find(m_axis == mk, 1, 'first');
    if isempty(mk_col)
        mk_col = max(1, min(numel(m_axis), mk - m_axis(1) + 1));
    end

    likelihood_k = P_m_given_N(:, mk_col);
    pred_k = T_Nk_given_prev * P_prev;
    unnorm_post = likelihood_k .* pred_k;
    s = sum(unnorm_post);

    if s > 0
        P_post = unnorm_post ./ s;
        log_evidence = log_evidence + log(s);
    else
        P_post = P_prev;
        log_evidence = log_evidence + log(eps);
    end

    post_hist(:, k) = P_post;
    P_prev = P_post;
end

[~, map_idx] = max(post_hist, [], 1);
N_map_seq = N_vals(map_idx);
end

function [map_acc, mae, logp_true] = scoreAgainstTruth(post_hist, N_map_seq, N_vals, true_seq)
K = numel(true_seq);
if K ~= size(post_hist, 2)
    K = min(K, size(post_hist, 2));
    true_seq = true_seq(1:K);
    N_map_seq = N_map_seq(1:K);
    post_hist = post_hist(:, 1:K);
end

map_acc = 100 * mean(N_map_seq(:).' == true_seq(:).');
mae = mean(abs(N_map_seq(:).' - true_seq(:).'));

idx_true = zeros(1, K);
valid = false(1, K);
for k = 1:K
    idx = find(N_vals == true_seq(k), 1, 'first');
    if ~isempty(idx)
        idx_true(k) = idx;
        valid(k) = true;
    end
end

if any(valid)
    cols = find(valid);
    p = zeros(1, numel(cols));
    for i = 1:numel(cols)
        c = cols(i);
        p(i) = post_hist(idx_true(c), c);
    end
    logp_true = sum(log(max(p, eps)));
else
    logp_true = -inf;
end
end

function T_Nk_given_prev = buildPoissonCountTransition(N_vals, lambda_arrival, lambda_depart)
n_states = numel(N_vals);
T_Nk_given_prev = zeros(n_states, n_states); % rows: N_k, cols: N_{k-1}

for n = 1:n_states
    n_prev = N_vals(n);

    a_max = n_states + 10;
    a_vals = 0:a_max;
    pA = localPoissPmf(a_vals, lambda_arrival);
    pA(end) = pA(end) + max(0, 1 - sum(pA));
    pA = pA ./ max(sum(pA), eps);

    if n_prev > 0
        d_vals = 0:n_prev;
        pD = zeros(size(d_vals));
        if n_prev > 1
            pD(1:end-1) = localPoissPmf(0:n_prev-1, lambda_depart * n_prev);
        end
        pD(end) = max(0, 1 - sum(pD(1:end-1)));
        pD = pD ./ max(sum(pD), eps);
    else
        d_vals = 0;
        pD = 1;
    end

    for ia = 1:numel(a_vals)
        for id = 1:numel(d_vals)
            n_next = n_prev - d_vals(id) + a_vals(ia);
            n_next = min(max(n_next, N_vals(1)), N_vals(end));
            idx_next = n_next - N_vals(1) + 1;
            T_Nk_given_prev(idx_next, n) = T_Nk_given_prev(idx_next, n) + pA(ia) * pD(id);
        end
    end
end

T_Nk_given_prev = T_Nk_given_prev ./ max(sum(T_Nk_given_prev, 1), eps);
end

function p = localPoissPmf(k, lambda)
if lambda <= 0
    p = double(k == 0);
    return
end
p = exp(-lambda + k .* log(lambda) - gammaln(k + 1));
end
