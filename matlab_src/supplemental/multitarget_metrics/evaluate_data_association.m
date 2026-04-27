function stats = evaluate_data_association(est_assoc_seq, gt_assoc_seq, est_xy_seq, gt_xy_seq, gate)
% EVALUATE_DATA_ASSOCIATION  Score per-measurement association decisions.
%
%   stats = evaluate_data_association(est_assoc, gt_assoc, est_xy, gt_xy, gate)
%
% At each frame:
%   1. Solve Hungarian assignment between est tracks and GT targets
%      using est_xy, gt_xy and gate -> permutation pi: est_track -> gt_target.
%   2. For each measurement j, the filter assigned it to estimated track t_e.
%      Map to gt_target = pi(t_e). Compare to gt_assoc(j).
%
% INPUTS:
%   est_assoc_seq - {1 x K} cell, each cell is [N_meas_k x 1] vector
%                   giving the estimated track index per measurement
%                   (0 = clutter)
%   gt_assoc_seq  - {1 x K} cell, each cell is [N_meas_k x 1] vector
%                   giving the ground-truth target ID per measurement
%                   (0 = clutter)
%   est_xy_seq    - {1 x K} cell, each cell is [2 x N_est_k]
%   gt_xy_seq     - {1 x K} cell, each cell is [2 x N_gt_k]
%   gate          - scalar Hungarian gate (default Inf)
%
% OUTPUTS:
%   stats.acc          - overall accuracy (matches / total measurements)
%   stats.acc_nonclutter - accuracy on measurements gt-labelled non-zero
%   stats.confusion_overall - 2x2: rows={gt clutter, gt target}, cols={est clutter, est target}
%   stats.per_frame_acc - [1 x K] per-frame accuracy

    if nargin < 5, gate = Inf; end
    K = numel(est_assoc_seq);
    if numel(gt_assoc_seq) ~= K, error('length mismatch'); end

    total = 0; correct = 0;
    total_nc = 0; correct_nc = 0;
    confusion = zeros(2, 2);
    per_frame_acc = nan(1, K);

    for k = 1:K
        ea = est_assoc_seq{k};
        ga = gt_assoc_seq{k};
        if isempty(ea) && isempty(ga), per_frame_acc(k) = NaN; continue; end
        if numel(ea) ~= numel(ga)
            warning('Frame %d: assoc length mismatch (%d vs %d)', k, ...
                numel(ea), numel(ga));
            continue
        end

        % Hungarian permutation est_track -> gt_target_id (0 if unmatched)
        if k > numel(est_xy_seq) || k > numel(gt_xy_seq)
            est_to_gt = containers.Map('KeyType','double','ValueType','double');
        else
            assignment = assign_targets_hungarian(est_xy_seq{k}, gt_xy_seq{k}, gate);
            est_to_gt = containers.Map('KeyType','double','ValueType','double');
            for r = 1:size(assignment, 1)
                est_to_gt(assignment(r, 1)) = assignment(r, 2);
            end
        end

        frame_correct = 0; frame_total = 0;
        for j = 1:numel(ea)
            est_t = ea(j);
            gt_t  = ga(j);

            if est_t == 0
                mapped = 0;
            elseif isKey(est_to_gt, est_t)
                mapped = est_to_gt(est_t);
            else
                mapped = -1;  % est track exists but no GT match -> wrong
            end

            row = (gt_t > 0) + 1;        % 1=clutter, 2=target
            col = (mapped > 0) + 1;
            confusion(row, col) = confusion(row, col) + 1;

            ok = (mapped == gt_t);
            frame_total = frame_total + 1;
            if ok, frame_correct = frame_correct + 1; end
            if gt_t > 0
                total_nc = total_nc + 1;
                if ok, correct_nc = correct_nc + 1; end
            end
        end

        total = total + frame_total;
        correct = correct + frame_correct;
        if frame_total > 0
            per_frame_acc(k) = frame_correct / frame_total;
        end
    end

    stats.acc               = ternary(total > 0, correct / total, NaN);
    stats.acc_nonclutter    = ternary(total_nc > 0, correct_nc / total_nc, NaN);
    stats.confusion_overall = confusion;
    stats.per_frame_acc     = per_frame_acc;
    stats.total             = total;
    stats.total_nonclutter  = total_nc;
end


function v = ternary(cond, a, b)
    if cond, v = a; else, v = b; end
end
