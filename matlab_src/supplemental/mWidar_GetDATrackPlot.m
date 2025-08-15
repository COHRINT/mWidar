function fig_handle = mWidar_GetDATrackPlot(tracker_obj, measurements, true_state, title_str)
    % MWIDAR_GETDATRACKPLOT Get figure handle from DA_Track dynamic plotting
    %
    % SYNTAX:
    %   fig_handle = mWidar_GetDATrackPlot(tracker_obj, measurements, true_state, title_str)
    %
    % INPUTS:
    %   tracker_obj  - DA_Track filter object (PDA_KF, GNN_KF, PDA_PF, etc.)
    %   measurements - Current measurements [N_z x N_measurements]
    %   true_state   - (optional) True state for comparison
    %   title_str    - (optional) Custom title string
    %
    % OUTPUTS:
    %   fig_handle   - Figure handle that can be used in subplot layouts
    %
    % DESCRIPTION:
    %   Creates a separate figure using the DA_Track class's visualize method
    %   with pinned axis limits [-2 2] for x and [0 4] for y. This figure
    %   can then be incorporated into larger subplot layouts.
    %
    % EXAMPLE:
    %   fig = mWidar_GetDATrackPlot(pda_kf_obj, z_meas, x_true, 'PDA-KF Step 5');
    %   % Then use fig in a larger subplot arrangement
    
    % Validate inputs
    if nargin < 2
        error('Usage: mWidar_GetDATrackPlot(tracker_obj, measurements, true_state, title_str)');
    end
    
    if nargin < 3 || isempty(true_state)
        true_state = [];
    end
    
    if nargin < 4 || isempty(title_str)
        title_str = sprintf('%s Visualization', class(tracker_obj));
    end
    
    % Create new figure for the DA_Track visualization
    fig_handle = figure('Visible', 'off', 'Units', 'pixels', 'Position', [100, 100, 600, 600]);
    
    % Call the tracker's visualize method
    try
        if ~isempty(true_state)
            tracker_obj.visualize(fig_handle, title_str, measurements, true_state);
        else
            tracker_obj.visualize(fig_handle, title_str, measurements);
        end
        
        % Ensure pinned axis limits are applied (in case visualize method doesn't set them)
        current_axes = gca;
        xlim(current_axes, [-2 2]);
        ylim(current_axes, [0 4]);
        axis(current_axes, 'square');
        
        % Make figure visible after plotting is complete
        set(fig_handle, 'Visible', 'on');
        
    catch ME
        % Close figure if visualization failed
        if isvalid(fig_handle)
            close(fig_handle);
        end
        
        % Re-throw error with context
        error('Failed to create DA_Track visualization: %s', ME.message);
    end
end
