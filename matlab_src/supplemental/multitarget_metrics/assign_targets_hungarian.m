function [assignment, unmatched_est, unmatched_gt, total_cost] = ...
    assign_targets_hungarian(est_xy, gt_xy, gate)
% ASSIGN_TARGETS_HUNGARIAN  Optimal estimate-to-truth track assignment.
%
%   [assignment, unmatched_est, unmatched_gt, total_cost] = ...
%       assign_targets_hungarian(est_xy, gt_xy, gate)
%
% Solves the rectangular assignment problem on the Euclidean cost matrix
% between estimated tracks and ground-truth targets. Pairs whose cost
% exceeds `gate` are left unmatched. Uses MATLAB's matchpairs when
% available (R2019a+); falls back to greedy nearest-neighbour otherwise.
%
% INPUTS:
%   est_xy - [2 x N_est] estimated positions
%   gt_xy  - [2 x N_gt]  ground-truth positions
%   gate   - scalar (default Inf): assignments with cost > gate are dropped
%
% OUTPUTS:
%   assignment    - [K x 2] each row [est_idx, gt_idx] of matched pairs
%   unmatched_est - column vector of est indices with no match
%   unmatched_gt  - column vector of gt  indices with no match
%   total_cost    - sum of matched-pair Euclidean distances

    if nargin < 3 || isempty(gate), gate = Inf; end

    N_est = size(est_xy, 2);
    N_gt  = size(gt_xy,  2);

    if N_est == 0 || N_gt == 0
        assignment    = zeros(0, 2);
        unmatched_est = (1:N_est)';
        unmatched_gt  = (1:N_gt)';
        total_cost    = 0;
        return
    end

    cost = zeros(N_est, N_gt);
    for i = 1:N_est
        for j = 1:N_gt
            cost(i, j) = norm(est_xy(:, i) - gt_xy(:, j));
        end
    end

    if exist('matchpairs', 'builtin') || exist('matchpairs', 'file')
        % matchpairs needs a non-matching cost; use gate or a value
        % larger than any plausible match cost.
        cost_unmatched = gate;
        if ~isfinite(cost_unmatched)
            cost_unmatched = max(cost(:)) + 1;
        end
        M = matchpairs(cost, cost_unmatched);
        assignment = M;
    else
        assignment = greedy_assignment(cost, gate);
    end

    if isfinite(gate) && ~isempty(assignment)
        keep = false(size(assignment, 1), 1);
        for r = 1:size(assignment, 1)
            keep(r) = cost(assignment(r,1), assignment(r,2)) <= gate;
        end
        assignment = assignment(keep, :);
    end

    matched_est = false(N_est, 1);
    matched_gt  = false(N_gt,  1);
    total_cost  = 0;
    for r = 1:size(assignment, 1)
        i = assignment(r, 1);
        j = assignment(r, 2);
        matched_est(i) = true;
        matched_gt(j)  = true;
        total_cost = total_cost + cost(i, j);
    end

    unmatched_est = find(~matched_est);
    unmatched_gt  = find(~matched_gt);
end


function assignment = greedy_assignment(cost, gate)
    [N_est, N_gt] = size(cost);
    assignment = zeros(0, 2);
    used_est = false(N_est, 1);
    used_gt  = false(N_gt,  1);

    [sorted_costs, ord] = sort(cost(:));
    for idx = 1:numel(ord)
        if sorted_costs(idx) > gate, break; end
        [i, j] = ind2sub([N_est, N_gt], ord(idx));
        if ~used_est(i) && ~used_gt(j)
            assignment(end+1, :) = [i, j]; %#ok<AGROW>
            used_est(i) = true;
            used_gt(j)  = true;
            if all(used_est) || all(used_gt), break; end
        end
    end
end
