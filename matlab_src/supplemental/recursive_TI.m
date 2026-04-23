%% Recursive object-count estimation on TI_test_case dataset
% Uses the likelihood table built in probObjCt_results.mat and runs
% recursive Bayesian filtering over the generated TI dataset detections.

clc; close all; clear

%% --- Environment configuration -----------------------------------------
addpath(fullfile('DA_Track'))
addpath(fullfile('DA_Track', 'multi'))
addpath(fullfile('supplemental'))
addpath(fullfile('supplemental', 'Final_Test_Tracks'))
addpath(fullfile('supplemental', 'Final_Test_Tracks', 'MultiObj'))

load(fullfile('supplemental', 'probObjCt_results.mat'), 'results');
load(fullfile('supplemental', 'Final_Test_Tracks', 'TI_test_case_const1_10s_dt0p010.mat'), 'Data');

cfg = results.cfg;
P_m_given_N = results.P_m_given_N;
det_counts_by_obj = results.det_counts_by_obj; 

% State and measurement axes inferred from saved probability table.
n_states = size(P_m_given_N, 1);
if isfield(results, 'N_vals') && numel(results.N_vals) == n_states
    N_vals = results.N_vals(:).';
elseif isfield(results, 'cfg') && isfield(results.cfg, 'max_obj') && n_states == (results.cfg.max_obj + 1)
    N_vals = 0:results.cfg.max_obj;
else
    N_vals = 1:n_states;
end

if isfield(results, 'det_axis') && numel(results.det_axis) == size(P_m_given_N, 2)
    m_axis = results.det_axis(:).';
else
    m_axis = 0:(size(P_m_given_N, 2) - 1);
end

%% --- Observation sequence from generated TI dataset --------------------
if ~isfield(Data, 'y') || isempty(Data.y)
    error('Data.y missing or empty in TI_test_case.mat');
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
if K == 0
    error('No observations available after start index %d.', start_k);
end

%% --- Full recursive update with Poisson count dynamics ------------------
% P(N_k | N_{k-1}) from Poisson arrivals/departures, with uniform prior.
lambda_arrival = 0.1;   % expected arrivals per step
lambda_depart = 0.2;    % expected departures per known object per step
T_Nk_given_prev = buildPoissonCountTransition(N_vals, lambda_arrival, lambda_depart);

P_prev = ones(n_states, 1) / n_states;
post_hist = zeros(n_states, K);
alpha_hist = zeros(1, K);

for k = 1:K
    mk = obs_seq(k);
    mk_col = find(m_axis == mk, 1, 'first');
    if isempty(mk_col)
        mk_col = max(1, min(numel(m_axis), mk - m_axis(1) + 1));
    end

    likelihood_k = P_m_given_N(:, mk_col);
    pred_k = T_Nk_given_prev * P_prev;
    unnorm_post_k = likelihood_k .* pred_k;
    s = sum(unnorm_post_k);

    if s > 0
        alpha_hist(k) = 1 / s;
        P_post = unnorm_post_k / s;
    else
        alpha_hist(k) = NaN;
        P_post = P_prev;
    end

    post_hist(:, k) = P_post;
    P_prev = P_post;
end

[~, map_idx] = max(post_hist, [], 1);
N_map_seq = N_vals(map_idx);
N_map_final = N_map_seq(end);

if isfield(Data, 'obj_ct') && numel(Data.obj_ct) >= start_k
    true_seq = Data.obj_ct(start_k:end);
    acc = 100 * mean(N_map_seq(:).' == true_seq(:).');
else
    true_seq = [];
    acc = NaN;
end

%% --- Report -------------------------------------------------------------
fprintf('\nRecursive TI estimate using TI_test_case observations\n');
fprintf('States N: %s\n', mat2str(N_vals));
fprintf('Observation range m: [%d, %d]\n', m_axis(1), m_axis(end));
fprintf('Frames used: %d (k=%d..%d)\n', K, start_k, start_k + K - 1);
fprintf('Final MAP object-count estimate: N = %d\n', N_map_final);
if ~isnan(acc)
    fprintf('MAP sequence accuracy vs Data.obj_ct over used frames: %.2f%%\n', acc);
end

%% --- Visualization ------------------------------------------------------
figure('Color', 'w', 'Position', [100 100 1100 700]);
tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
imagesc(ax1, 1:K, N_vals, post_hist);
axis(ax1, 'xy');
xlabel(ax1, 'Frame Index (post-start)');
ylabel(ax1, 'Object Count State N');
title(ax1, 'Posterior P(N_k | M_k)');
colorbar(ax1);

ax2 = nexttile;
hold(ax2, 'on');
plot(ax2, 1:K, N_map_seq, 'b-', 'LineWidth', 1.5);
if ~isempty(true_seq)
    plot(ax2, 1:K, true_seq, 'k--', 'LineWidth', 1.2);
    legend(ax2, 'MAP estimate', 'True object count', 'Location', 'best');
else
    legend(ax2, 'MAP estimate', 'Location', 'best');
end
xlim(ax2, [1, K]);
ylim(ax2, [N_vals(1)-0.2, N_vals(end)+0.2]);
xlabel(ax2, 'Frame Index (post-start)');
ylabel(ax2, 'Count');
if ~isnan(acc)
    title(ax2, sprintf('Recursive Object-Count Estimate (Accuracy: %.2f%%)', acc));
else
    title(ax2, 'Recursive Object-Count Estimate');
end
grid(ax2, 'on');
hold(ax2, 'off');

%% --- Save results -------------------------------------------------------
RecursiveTI = struct();
RecursiveTI.N_vals = N_vals;
RecursiveTI.m_axis = m_axis;
RecursiveTI.P_m_given_N = P_m_given_N;
RecursiveTI.lambda_arrival = lambda_arrival;
RecursiveTI.lambda_depart = lambda_depart;
RecursiveTI.T_Nk_given_prev = T_Nk_given_prev;
RecursiveTI.start_k = start_k;
RecursiveTI.obs_seq = obs_seq;
RecursiveTI.post_hist = post_hist;
RecursiveTI.alpha_hist = alpha_hist;
RecursiveTI.N_map_seq = N_map_seq;
RecursiveTI.N_map_final = N_map_final;
RecursiveTI.true_seq = true_seq;
RecursiveTI.map_accuracy_pct = acc;

save(fullfile('supplemental', 'recursive_TI_results.mat'), 'RecursiveTI');
fprintf('Saved recursive results to supplemental\\recursive_TI_results.mat\n');


function T_Nk_given_prev = buildPoissonCountTransition(N_vals, lambda_arrival, lambda_depart)
n_states = numel(N_vals);
T_Nk_given_prev = zeros(n_states, n_states); % rows: N_k, cols: N_{k-1}

for n = 1:n_states
    n_prev = N_vals(n);

    % Arrivals: A ~ Poisson(lambda_arrival), fold overflow tail into max state.
    a_max = n_states + 10;
    a_vals = 0:a_max;
    pA = localPoissPmf(a_vals, lambda_arrival);
    pA(end) = pA(end) + max(0, 1 - sum(pA));
    pA = pA ./ max(sum(pA), eps);

    % Departures: D ~ Poisson(lambda_depart * n_prev), truncated to [0, n_prev].
    if n_prev > 0
        d_vals = 0:n_prev;
        pD = zeros(size(d_vals));
        if n_prev > 1
            pD(1:end-1) = localPoissPmf(0:n_prev-1, lambda_depart * n_prev);
        end
        pD(end) = max(0, 1 - sum(pD(1:end-1))); % tail folded into D=n_prev
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
