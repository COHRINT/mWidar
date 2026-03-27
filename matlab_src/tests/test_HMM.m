% TEST_HMM  Oracle diagnostic tests for the standalone HMM inner filter.
%
% PURPOSE:
%   Isolates the HMM filter from the RBPF wrapper to determine whether a
%   high RMSE in HMM_RBPF is caused by the inner filter (transition matrix
%   too diffuse / broken likelihood tables) or by the RBPF association and
%   weighting logic.
%
% TESTS:
%   1. Oracle association — feed only the nearest measurement to GT each step.
%      If RMSE is still large, the problem is inside HMM.m.
%   2. Pure prediction (no measurement_update) — shows how fast entropy
%      reaches maximum (log(16384) ≈ 9.7 nats).  If entropy maxes in 1-2
%      steps the transition matrix is too diffuse.
%
% OUTPUTS:
%   - Printed RMSE (MMSE and MAP) for both tests
%   - Figure 1: GT vs MMSE vs MAP trajectory
%   - Figure 2: Position error over time (MMSE and MAP)
%   - Figure 3: HMM entropy over time (oracle vs pure-prediction)
%
% RUN FROM matlab_src/tests/:
%   cd tests
%   run('test_HMM.m')

clear; clc; close all;

addpath('../DA_Track');                   % base: DA_Filter, KF, HMM
addpath('../DA_Track/single');            % single-target filters
addpath('../supplemental');

fprintf('=== HMM Oracle Diagnostic Test ===\n\n');

%% ---- Load precomputed HMM tables -------------------------------------------
fprintf('Loading HMM tables...\n');
load('../supplemental/precalc_imagegridHMMEmLike.mat',   'pointlikelihood_image');
tmp = load('../supplemental/precalc_imagegridHMMSTMn5.mat', 'A');
A_transition = tmp.A;
fprintf('  A_transition:          %d x %d\n', size(A_transition));
fprintf('  pointlikelihood_image: %d x %d\n', size(pointlikelihood_image));

%% ---- Load dataset ----------------------------------------------------------
fprintf('Loading dataset...\n');
load('../data/TUNING_DATASET1/data.mat', 'Data');
GT   = Data.GT;      % [6 x N_k]
z_all = Data.y;      % {1 x N_k}
n_k  = size(GT, 2);
dt   = Data.params.dt;
fprintf('  %d timesteps, dt=%.2f s\n\n', n_k, dt);

x0 = GT(1:2, 1);     % Initialise at true position

%% ============================================================================
%% TEST 1: Oracle Association
%% ============================================================================
fprintf('--- Test 1: Oracle Association ---\n');

hmm1 = HMM(x0, A_transition, pointlikelihood_image);

mmse_oracle = NaN(2, n_k);
map_oracle  = NaN(2, n_k);
ent_oracle  = NaN(1, n_k);

[mmse_oracle(:,1), ~] = hmm1.getGaussianEstimate();
[map_oracle(:,1),  ~] = hmm1.getMAPEstimate();
ent_oracle(1)         = hmm1.getEntropy();

for k = 2:n_k
    z_k  = z_all{k};              % [2 x N_det]
    GT_k = GT(1:2, k);

    hmm1.prediction();

    if ~isempty(z_k)
        dists  = vecnorm(z_k - GT_k, 2, 1);
        [~, best] = min(dists);
        hmm1.measurement_update(z_k(:, best));
    end
    % Missed detection: no update, posterior = prior (prediction already done)

    [mmse_oracle(:,k), ~] = hmm1.getGaussianEstimate();
    [map_oracle(:,k),  ~] = hmm1.getMAPEstimate();
    ent_oracle(k)          = hmm1.getEntropy();

    if mod(k, 10) == 0
        err = norm(mmse_oracle(:,k) - GT_k);
        fprintf('  k=%3d  MMSE err=%.4f m  entropy=%.3f\n', k, err, ent_oracle(k));
    end
end

err_mmse_oracle = vecnorm(mmse_oracle - GT(1:2,:), 2, 1);
err_map_oracle  = vecnorm(map_oracle  - GT(1:2,:), 2, 1);
rmse_mmse_oracle = sqrt(mean(err_mmse_oracle(~isnan(err_mmse_oracle)).^2));
rmse_map_oracle  = sqrt(mean(err_map_oracle(~isnan(err_map_oracle)).^2));

fprintf('\nTest 1 Results (oracle association):\n');
fprintf('  RMSE(MMSE) = %.4f m\n', rmse_mmse_oracle);
fprintf('  RMSE(MAP)  = %.4f m\n', rmse_map_oracle);
fprintf('  Mean entropy (start): %.3f nats\n', mean(ent_oracle(1:5)));
fprintf('  Mean entropy (end):   %.3f nats   [max=%.2f]\n', ...
        mean(ent_oracle(end-4:end)), log(hmm1.npx2));

%% ============================================================================
%% TEST 2: Pure Prediction (no measurement updates)
%% ============================================================================
fprintf('\n--- Test 2: Pure Prediction (no updates) ---\n');

hmm2 = HMM(x0, A_transition, pointlikelihood_image);

mmse_pred = NaN(2, n_k);
map_pred  = NaN(2, n_k);
ent_pred  = NaN(1, n_k);

[mmse_pred(:,1), ~] = hmm2.getGaussianEstimate();
[map_pred(:,1),  ~] = hmm2.getMAPEstimate();
ent_pred(1)          = hmm2.getEntropy();

for k = 2:n_k
    hmm2.prediction();
    [mmse_pred(:,k), ~] = hmm2.getGaussianEstimate();
    [map_pred(:,k),  ~] = hmm2.getMAPEstimate();
    ent_pred(k)          = hmm2.getEntropy();
end

err_mmse_pred = vecnorm(mmse_pred - GT(1:2,:), 2, 1);
rmse_pred = sqrt(mean(err_mmse_pred(~isnan(err_mmse_pred)).^2));

fprintf('Test 2 Results (pure prediction):\n');
fprintf('  RMSE(MMSE) = %.4f m\n', rmse_pred);
fprintf('  Entropy k=1: %.3f\n', ent_pred(1));
fprintf('  Entropy k=2: %.3f\n', ent_pred(min(2,n_k)));
fprintf('  Entropy k=5: %.3f\n', ent_pred(min(5,n_k)));
fprintf('  Entropy k=10: %.3f\n', ent_pred(min(10,n_k)));
fprintf('  Max possible: %.3f\n', log(hmm2.npx2));

%% ---- Interpretation hint ---------------------------------------------------
fprintf('\n--- Interpretation ---\n');
if rmse_mmse_oracle < 0.3
    fprintf('  Oracle HMM RMSE < 0.3 m → inner filter OK; problem is in RBPF association/weighting.\n');
elseif rmse_mmse_oracle < 1.0
    fprintf('  Oracle HMM RMSE in [0.3, 1.0] m → inner filter marginal; review likelihood tables.\n');
else
    fprintf('  Oracle HMM RMSE > 1.0 m → inner filter problem (transition matrix or likelihood).\n');
end

max_ent = log(hmm2.npx2);
if ent_pred(min(2,n_k)) > 0.95 * max_ent
    fprintf('  Entropy max reached by k=2 → transition matrix is near-diffusive.\n');
elseif ent_pred(min(5,n_k)) > 0.95 * max_ent
    fprintf('  Entropy max reached by k=5 → transition matrix diffuses quickly.\n');
else
    fprintf('  Entropy spreads slowly → transition matrix OK.\n');
end

%% ============================================================================
%% FIGURES
%% ============================================================================
time = (0:n_k-1) * dt;

%% Figure 1: Trajectory
figure('Name', 'HMM Oracle — Trajectory', 'Position', [100, 100, 700, 550]);
plot(GT(1,:), GT(2,:), 'g-', 'LineWidth', 2, 'DisplayName', 'GT');
hold on;
plot(mmse_oracle(1,:), mmse_oracle(2,:), 'b--', 'LineWidth', 1.5, 'DisplayName', 'MMSE (oracle)');
plot(map_oracle(1,:),  map_oracle(2,:),  'm:',  'LineWidth', 1.5, 'DisplayName', 'MAP  (oracle)');
plot(GT(1,1), GT(2,1), 'go', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
xlabel('X (m)'); ylabel('Y (m)');
title(sprintf('HMM Oracle Trajectory  (RMSE_{MMSE}=%.3f m, RMSE_{MAP}=%.3f m)', ...
      rmse_mmse_oracle, rmse_map_oracle));
legend('Location', 'best'); grid on; axis equal;

%% Figure 2: Position error
figure('Name', 'HMM Oracle — Position Error', 'Position', [820, 100, 700, 400]);
plot(time, err_mmse_oracle, 'b-', 'LineWidth', 1.5, 'DisplayName', 'MMSE (oracle)');
hold on;
plot(time, err_map_oracle,  'm-', 'LineWidth', 1.5, 'DisplayName', 'MAP  (oracle)');
yline(0.3, 'k--', 'LineWidth', 1, 'Label', '0.3 m');
xlabel('Time (s)'); ylabel('Position error (m)');
title('HMM Oracle Association — Position Error vs Time');
legend('Location', 'best'); grid on;

%% Figure 3: Entropy
figure('Name', 'HMM — Entropy Diagnostic', 'Position', [100, 580, 1200, 380]);
subplot(1,2,1);
plot(time, ent_oracle, 'b-', 'LineWidth', 1.5);
yline(log(hmm1.npx2), 'r--', 'LineWidth', 1.5, 'Label', 'Max entropy');
xlabel('Time (s)'); ylabel('Entropy (nats)');
title('Oracle Association — HMM Entropy');
ylim([0, log(hmm1.npx2) * 1.05]); grid on;

subplot(1,2,2);
plot(time, ent_pred, 'r-', 'LineWidth', 1.5);
yline(log(hmm2.npx2), 'r--', 'LineWidth', 1.5, 'Label', 'Max entropy');
xlabel('Time (s)'); ylabel('Entropy (nats)');
title('Pure Prediction — Entropy Spread (No Updates)');
ylim([0, log(hmm2.npx2) * 1.05]); grid on;

fprintf('\nTest complete.\n');
