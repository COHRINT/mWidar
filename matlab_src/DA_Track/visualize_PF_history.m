function visualize_PF_history(pf_tracker, varargin)
    % VISUALIZE_PF_HISTORY RBPF-style post-processing visualization for PF trackers
    %
    % This wrapper reuses visualize_RBPF_history with a standardized history schema.
    % Supported trackers: GNN_PF, PDA_PF, MC_PF, KF_RBPF, and any tracker with
    % a compatible .history struct.

    if ~isprop(pf_tracker, 'history')
        error('visualize_PF_history:InvalidTracker', ...
            'Tracker %s does not expose a history property.', class(pf_tracker));
    end

    if isempty(pf_tracker.history)
        error('visualize_PF_history:EmptyHistory', ...
            'Tracker history is empty. Run timesteps before visualizing.');
    end

    visualize_RBPF_history(pf_tracker, varargin{:});
end

