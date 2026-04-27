% test_slotpool_smoke.m
%
% Smoke test for the slot-pool / variable-cardinality API on PDA_PF_multi
% and KF_RBPF_multi. Does NOT exercise the count estimator or the full
% TrackManager — just checks that:
%   1. Filters can be constructed with NMax > N_t_init.
%   2. timestep() with constant N_t works (active mask honoured).
%   3. add_target() activates a slot, N_t increments, particles get the
%      new slot's KF/states, history rows align.
%   4. remove_target() deactivates a slot, N_t decrements, association_history
%      is preserved (not deleted), inactive frames append NaN going forward.
%   5. getGaussianEstimate returns active-only cells with active_idx mapping.
%   6. KF_RBPF_multi.getClutterRate returns NaN for inactive slots and
%      ignores NaN history entries for active slots that were just spawned.
%
% Run from matlab_src/.

clear; clc; close all;
script_dir     = fileparts(mfilename('fullpath'));     % .../matlab_src/tests
matlab_src_dir = fileparts(script_dir);                 % .../matlab_src
addpath(matlab_src_dir);
addpath(fullfile(matlab_src_dir, 'DA_Track'));
addpath(fullfile(matlab_src_dir, 'DA_Track', 'multi'));
addpath(fullfile(matlab_src_dir, 'supplemental'));
addpath(fullfile(matlab_src_dir, 'supplemental', 'track_init'));

%% --- Output directory --------------------------------------------------
out_dir = fullfile(matlab_src_dir, 'tests', 'figures', 'trackInit_testing');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% --- Test result tracking ---------------------------------------------
test_log = struct('section', {}, 'status', {}, 'time_ms', {});

rng(7);

%% --- System model ------------------------------------------------------
dt = 0.1;
F = [eye(2), dt*eye(2); zeros(2), eye(2)];      % 4-state CV
Q = blkdiag(1e-4*eye(2), 1e-3*eye(2));
H = [eye(2), zeros(2)];
R = 0.05^2 * eye(2);

N_p_pda = 200;
N_p_rbpf = 100;
NMax = 5;

x0_cell = {[ 0.5;  1.0; 0; 0], ...
           [-0.5;  2.0; 0; 0]};
N_t_init = numel(x0_cell);

fprintf('--- Constructing filters (NMax=%d, N_t_init=%d) ---\n', NMax, N_t_init);

t0 = tic;
pda  = PDA_PF_multi(x0_cell, N_p_pda, F, Q, H, R, ...
    'NMax', NMax, 'Debug', true, 'InitSigmaPos', 0.1, 'InitSigmaVel', 0.05);
rbpf = KF_RBPF_multi(x0_cell, N_p_rbpf, F, Q, H, R, ...
    'NMax', NMax, 'Debug', true, 'InitSigmaPos', 0.1, 'InitSigmaVel', 0.05);

assert(pda.N_max == NMax,        'PDA NMax wrong');
assert(rbpf.N_max == NMax,       'RBPF NMax wrong');
assert(pda.N_t == N_t_init,      'PDA N_t init wrong');
assert(rbpf.N_t == N_t_init,     'RBPF N_t init wrong');
assert(isequal(pda.active(1:N_t_init), true(1, N_t_init)),  'PDA initial active wrong');
assert(all(~rbpf.active(N_t_init+1:end)),                    'RBPF tail-inactive wrong');
test_log(end+1) = struct('section', 'Construction (NMax=5, N\_t=2)', 'status', 'PASS', 'time_ms', toc(t0)*1e3);

%% --- 5 steady frames ---------------------------------------------------
fprintf('\n--- 5 steady frames at N_t=%d ---\n', N_t_init);
t0 = tic;
for k = 1:5
    z = simulate_meas(F, x0_cell, k*dt, R);
    pda.timestep(z);
    rbpf.timestep(z);
end
[xc, ~, aidx] = pda.getGaussianEstimate();
assert(numel(xc)  == N_t_init, 'PDA estimate count');
assert(isequal(aidx, 1:N_t_init), 'PDA active_idx');

[xc, ~, aidx] = rbpf.getGaussianEstimate();
assert(numel(xc) == N_t_init, 'RBPF estimate count');
assert(isequal(aidx, 1:N_t_init), 'RBPF active_idx');

% RBPF association_history shape: [N_max x 5]
hist_shape = size(rbpf.particles{1}.association_history);
assert(isequal(hist_shape, [NMax, 5]), ...
    sprintf('RBPF history shape %s, expected [%d 5]', mat2str(hist_shape), NMax));

% Inactive rows must be NaN throughout
inactive_rows = rbpf.particles{1}.association_history(N_t_init+1:end, :);
assert(all(isnan(inactive_rows(:))), 'Inactive rows should be all-NaN');

% Active rows must be non-NaN
active_rows = rbpf.particles{1}.association_history(1:N_t_init, :);
assert(~any(isnan(active_rows(:))), 'Active rows should be all-numeric');
test_log(end+1) = struct('section', '5 steady frames', 'status', 'PASS', 'time_ms', toc(t0)*1e3);

%% --- add_target on both filters ----------------------------------------
fprintf('\n--- add_target ---\n');
t0 = tic;
x_init = [1.5; 1.5; 0; 0];
P_init = diag([0.5, 0.5, 1, 1].^2);
pda.add_target(x_init, P_init);
rbpf.add_target(x_init, P_init);

assert(pda.N_t  == N_t_init + 1, 'PDA N_t after add');
assert(rbpf.N_t == N_t_init + 1, 'RBPF N_t after add');
assert(pda.active(N_t_init + 1)  == true, 'PDA new slot active');
assert(rbpf.active(N_t_init + 1) == true, 'RBPF new slot active');

% Spawn slot's history row should still hold NaN for the past 5 frames
% (no fabrication).
new_slot = N_t_init + 1;
spawn_row = rbpf.particles{1}.association_history(new_slot, :);
assert(all(isnan(spawn_row)), 'Spawn slot pre-spawn history must be NaN');
test_log(end+1) = struct('section', 'add\_target', 'status', 'PASS', 'time_ms', toc(t0)*1e3);

%% --- 3 frames after add -----------------------------------------------
fprintf('\n--- 3 frames at N_t=%d ---\n', pda.N_t);
t0 = tic;
for k = 6:8
    z = simulate_meas(F, [x0_cell, {x_init}], k*dt, R);
    pda.timestep(z);
    rbpf.timestep(z);
end

% New slot now has 3 numeric history entries appended
spawn_row = rbpf.particles{1}.association_history(new_slot, :);
assert(numel(spawn_row) == 8, 'History length after 8 frames');
assert(all(isnan(spawn_row(1:5))), 'First 5 must remain NaN');
assert(~any(isnan(spawn_row(6:8))), 'Post-spawn frames must be numeric');

% getClutterRate: active slot just spawned, only 3 numeric entries; should
% NOT return ~1.0 just because of the NaN-padded past.
rate = rbpf.getClutterRate(10);
assert(isnan(rate(N_t_init+2)), 'Inactive slot rate should be NaN');
fprintf('Clutter rates after 3 post-spawn frames: %s\n', mat2str(rate', 3));
test_log(end+1) = struct('section', '3 frames post-spawn', 'status', 'PASS', 'time_ms', toc(t0)*1e3);

%% --- remove_target -----------------------------------------------------
fprintf('\n--- remove_target slot 1 (PDA) and slot 2 (RBPF) ---\n');
t0 = tic;
pda.remove_target(1);
rbpf.remove_target(2);

assert(pda.N_t  == N_t_init,     'PDA N_t after remove');
assert(rbpf.N_t == N_t_init,     'RBPF N_t after remove');
assert(pda.active(1)  == false,  'PDA slot 1 inactive');
assert(rbpf.active(2) == false,  'RBPF slot 2 inactive');

% RBPF removed-slot KF object should be released
assert(isempty(rbpf.particles{1}.kfs{2}), 'RBPF removed slot kf should be []');

% Removed slot's history must be PRESERVED (not deleted).
removed_row_before = rbpf.particles{1}.association_history(2, :);
assert(numel(removed_row_before) == 8, 'Removed slot history preserved');
assert(~all(isnan(removed_row_before)), 'Removed slot history not NaN-only');
test_log(end+1) = struct('section', 'remove\_target', 'status', 'PASS', 'time_ms', toc(t0)*1e3);

%% --- 2 frames after remove (history grows for active, NaN for inactive)
fprintf('\n--- 2 more frames at N_t=%d ---\n', pda.N_t);
t0 = tic;
for k = 9:10
    z = simulate_meas(F, [x0_cell(2), {x_init}], k*dt, R);
    pda.timestep(z);
    rbpf.timestep(z);
end

% Now the removed slot row should have 2 NEW NaN entries appended
removed_row = rbpf.particles{1}.association_history(2, :);
assert(numel(removed_row) == 10, 'History length 10');
assert(all(isnan(removed_row(9:10))), 'Post-remove frames must be NaN for removed slot');
test_log(end+1) = struct('section', '2 frames post-remove', 'status', 'PASS', 'time_ms', toc(t0)*1e3);

%% --- final estimate ----------------------------------------------------
t0 = tic;
[xc, Pc, aidx] = rbpf.getGaussianEstimate();
fprintf('\nFinal RBPF active_idx: %s, N_t=%d\n', mat2str(aidx), rbpf.N_t);
assert(isequal(aidx, [1 3]), 'RBPF active_idx [1 3] after removing slot 2');
assert(numel(xc) == 2, 'RBPF final estimate count');

[xc, Pc, aidx] = pda.getGaussianEstimate();
fprintf('Final PDA  active_idx: %s, N_t=%d\n', mat2str(aidx), pda.N_t);
assert(isequal(aidx, [2 3]), 'PDA active_idx [2 3] after removing slot 1');
assert(numel(xc) == 2, 'PDA final estimate count');
test_log(end+1) = struct('section', 'Final estimate check', 'status', 'PASS', 'time_ms', toc(t0)*1e3);

fprintf('\n*** Slot-pool smoke test PASSED ***\n');

%% --- LaTeX table -------------------------------------------------------
tex_path = fullfile(out_dir, 'test_slotpool_results.tex');
fid = fopen(tex_path, 'w');
fprintf(fid, '%% Auto-generated by test_slotpool_smoke.m\n');
fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '  \\centering\n');
fprintf(fid, '  \\caption{Slot-Pool / Variable-Cardinality API Smoke Test Results}\n');
fprintf(fid, '  \\label{tab:slotpool_smoke}\n');
fprintf(fid, '  \\begin{tabular}{lcc}\n');
fprintf(fid, '    \\toprule\n');
fprintf(fid, '    Test Section & Status & Time (ms) \\\\\n');
fprintf(fid, '    \\midrule\n');
for i = 1:numel(test_log)
    fprintf(fid, '    %s & %s & %.1f \\\\\n', ...
        test_log(i).section, test_log(i).status, test_log(i).time_ms);
end
fprintf(fid, '    \\bottomrule\n');
fprintf(fid, '  \\end{tabular}\n');
fprintf(fid, '\\end{table}\n');
fclose(fid);
fprintf('Saved: %s\n', tex_path);


%% =====================================================================
function z = simulate_meas(F, x_cell, t, R)
    N = numel(x_cell);
    z = zeros(2, N);
    for i = 1:N
        x_t = x_cell{i};
        for s = 1:round(t / 0.1)
            x_t = F * x_t;
        end
        z(:, i) = x_t(1:2) + chol(R, 'lower') * randn(2, 1);
    end
end
