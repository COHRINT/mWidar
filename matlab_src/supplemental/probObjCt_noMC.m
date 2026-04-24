%% Plot results from probObjCt.m
clc; close all; clear

addpath(fullfile('DA_Track'))
addpath(fullfile('DA_Track', 'multi'))
addpath(fullfile('supplemental'))
addpath(fullfile('supplemental', 'Final_Test_Tracks'))
addpath(fullfile('supplemental', 'Final_Test_Tracks', 'MultiObj'))

results_file = fullfile('supplemental', 'probObjCt_results.mat');
if ~isfile(results_file)
    error('Missing %s. Run supplemental/probObjCt.m first.', results_file);
end

S = load(results_file, 'results');
if ~isfield(S, 'results') || ~isfield(S.results, 'P_m_given_N')
    error('File %s does not contain results.P_m_given_N.', results_file);
end

results = S.results;
P_m_given_N = results.P_m_given_N;

% Axes fallback for compatibility with older/newer result structs.
if isfield(results, 'det_axis') && numel(results.det_axis) == size(P_m_given_N, 2)
    det_axis = results.det_axis(:).';
else
    det_axis = 0:(size(P_m_given_N, 2) - 1);
end

if isfield(results, 'N_vals') && numel(results.N_vals) == size(P_m_given_N, 1)
    N_vals = results.N_vals(:).';
elseif isfield(results, 'cfg') && isfield(results.cfg, 'max_obj') ...
        && size(P_m_given_N, 1) == (results.cfg.max_obj + 1)
    N_vals = 0:results.cfg.max_obj;
else
    N_vals = 1:size(P_m_given_N, 1);
end

figure('Color', 'w');
bar3(P_m_given_N);
xlabel('Detections m');
ylabel('Objects N');
zlabel('Probability');
title('P(m | N) from probObjCt.m');
xticks(1:numel(det_axis));
xticklabels(string(det_axis));
yticks(1:numel(N_vals));
yticklabels(string(N_vals));
zlim([0, max(P_m_given_N(:), [], 'omitnan') * 1.05 + eps]);
colorbar;
