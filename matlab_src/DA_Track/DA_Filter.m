classdef (Abstract) DA_Filter < handle
    % DA_Filter  Abstract base class for all Data Association filters.
    %
    % DESCRIPTION:
    %   Defines the common interface, shared properties, and concrete utility
    %   methods for every filter in the mWidar single- and multi-target
    %   tracking system.  All concrete filter subclasses must implement the
    %   abstract methods declared here; the concrete methods are available to
    %   every subclass for free.
    %
    % USAGE MODES:
    %   Implementation / demo  -- call timestep() then getGaussianEstimate().
    %                             No history is accumulated; zero overhead.
    %   Testing / analysis     -- call timestep(), then storeHistory() to
    %                             snapshot the full internal state into
    %                             obj.history.  Analyse obj.history offline.
    %
    % ABSTRACT PROPERTIES (subclasses must declare):
    %   debug                  - Enable verbose debug output
    %   DynamicPlot            - Enable real-time step-by-step visualization
    %   dynamic_figure_handle  - Figure handle used by DynamicPlot
    %
    % SHARED PUBLIC PROPERTIES:
    %   history                - Struct array populated by storeHistory().
    %                            Always has fields x_est and P_est; each
    %                            subclass adds its own filter-specific fields.
    %   store_full_history     - When true (default), storeHistory() saves
    %                            the complete internal state (e.g. full
    %                            particle clouds, full HMM grids per
    %                            particle).  When false, only lightweight
    %                            summary statistics are stored.  Set via
    %                            FilterConfig before constructing the filter.
    %   current_ESS            - Most-recently computed Effective Sample Size
    %                            (NaN for non-particle filters).  Updated
    %                            inside timestep() BEFORE resampling so that
    %                            storeHistory() can read it without the caller
    %                            having to pass it explicitly.
    %   Frames                 - Cell array of captured figure frames for
    %                            GIF / video export.
    %
    % ABSTRACT METHODS (subclasses must implement):
    %   timestep               - Run one full predict-update cycle
    %   getGaussianEstimate    - Return [mean, covariance] Gaussian summary
    %   prediction             - Prediction step only
    %   measurement_update     - Measurement-update step only
    %   visualize              - Render current filter state to a figure
    %   storeHistory           - Snapshot full internal state into obj.history
    %
    % CONCRETE METHODS (available to all subclasses):
    %   initializeDynamicPlot  - Create the real-time visualization figure
    %   updateDynamicPlot      - Refresh visualization after each timestep
    %   plotUpdate             - Manual visualization refresh (external call)
    %   getTimestepCount       - Return current timestep counter value
    %   resetTimestepCount     - Reset timestep counter to zero
    %   validateCommonInputs   - Basic measurement sanity checks
    %   printFilterInfo        - Print current configuration to console
    %   captureFrame           - Capture current figure frame for animation
    %   saveAnimation          - Save captured frames as animated GIF
    %   clearFrames            - Free frame buffer memory
    %   hasFrames              - True if any frames have been captured
    %   getFrameCount          - Number of captured frames
    %
    % STATIC METHODS:
    %   parseFilterOptions     - Parse common constructor name-value pairs
    %
    % SUBCLASSES:
    %   KF-based  : PDA_KF, GNN_KF
    %   PF-based  : PDA_PF, GNN_PF, MC_PF
    %   HMM-based : PDA_HMM, GNN_HMM
    %   RBPF      : KF_RBPF, HMM_RBPF
    %   Multi-obj : JPDA_KF, KF_RBPF_multi
    %
    % See also PDA_KF, PDA_PF, PDA_HMM, KF_RBPF, HMM_RBPF
    
    properties (Abstract)
        % Core filter properties that all subclasses must implement
        debug                   % Enable debug output and validation
        DynamicPlot            % Enable real-time visualization during timesteps
        dynamic_figure_handle  % Figure handle for dynamic plotting
    end
    
    properties (Access = protected)
        % Shared timestep counter for dynamic plotting
        timestep_counter = 0   % Internal counter for timestep numbering
    end
    
    properties (Access = public)
        % -----------------------------------------------------------------
        % History / state-snapshot storage
        % -----------------------------------------------------------------

        % history - Struct array populated by storeHistory().
        %   Each element history(k) corresponds to one timestep.
        %   Guaranteed fields (set by every subclass):
        %     .x_est        [N_x x 1]   - Gaussian mean state estimate
        %     .P_est        [N_x x N_x] - Gaussian covariance estimate
        %     .measurements [N_z x N_m] - Raw measurements passed in
        %     .true_state   [N_x x 1]   - Ground truth ([] if not provided)
        %     .timestep_num scalar       - Timestep index
        %   Additional filter-specific fields are appended by each subclass.
        history = struct([])

        % store_full_history - Controls verbosity of storeHistory().
        %   true  (default) : Save complete internal state -- full particle
        %                     clouds, full HMM grid distributions per
        %                     particle, all trajectory buffers, etc.
        %                     Use for offline analysis and visualisation.
        %   false           : Save only lightweight summary statistics
        %                     (x_est, P_est, ESS, association indices).
        %                     Use for large Monte-Carlo runs where memory
        %                     is a concern.
        store_full_history = true

        % current_ESS - Effective Sample Size computed in the most recent
        %   timestep(), captured BEFORE resampling.  Read by storeHistory()
        %   so the caller never needs to pass it explicitly.
        %   NaN for non-particle-based filters (KF, HMM variants).
        current_ESS = NaN

        % Frames - Cell array of figure frames for GIF / video animation.
        %   Populated by captureFrame(); saved by saveAnimation().
        Frames = {}
    end
    
    methods (Abstract)
        % Core filter interface that all subclasses must implement
        
        timestep(obj, measurements)
        % TIMESTEP Process single time step with measurements
        %
        % INPUTS:
        %   measurements - Current measurements [N_z x N_measurements]
        %
        % DESCRIPTION:
        %   Process a single time step of the filter algorithm including
        %   prediction and measurement update. Implementation varies by
        %   filter type (particle filter, grid-based, etc.).
        
        [x_est, P_est] = getGaussianEstimate(obj)
        % GETGAUSSIANESTIMATE Extract Gaussian state estimate
        %
        % OUTPUTS:
        %   x_est - State estimate mean [N_x x 1]
        %   P_est - State covariance estimate [N_x x N_x]
        %
        % DESCRIPTION:
        %   Extract a Gaussian approximation of the current state estimate.
        %   For particle filters, this involves computing weighted statistics.
        %   For grid-based filters, this involves moment computation.
        
        prediction(obj)
        % PREDICTION Perform prediction step
        %
        % DESCRIPTION:
        %   Apply motion model to propagate state estimate forward in time.
        %   Implementation varies by filter type.
        
        measurement_update(obj, measurements)
        % MEASUREMENT_UPDATE Perform measurement update step
        %
        % INPUTS:
        %   measurements - Current measurements [N_z x N_measurements]
        %
        % DESCRIPTION:
        %   Update state estimate based on current measurements using
        %   data association algorithm. Implementation varies by filter type.
        
        visualize(obj, varargin)
        % VISUALIZE Plot current filter state
        %
        % DESCRIPTION:
        %   Create visualization of current filter state. Implementation
        %   varies by filter type (particles, probability grid, etc.).

        storeHistory(obj, measurements, varargin)
        % STOREHISTORY  Snapshot the full internal filter state into obj.history.
        %
        % SYNTAX:
        %   obj.storeHistory(measurements)
        %   obj.storeHistory(measurements, true_state)
        %
        % INPUTS:
        %   measurements - Raw measurements passed to this timestep [N_z x N_m].
        %   true_state   - (optional) Ground-truth state [N_x x 1].
        %                  Pass [] or omit when GT is unavailable (e.g. live data).
        %
        % DESCRIPTION:
        %   Appends a new struct to obj.history containing the complete
        %   internal state of the filter at the current timestep.  The
        %   level of detail is controlled by obj.store_full_history:
        %     true  -- full state (particles, grids, trajectories, etc.)
        %     false -- lightweight summary only (x_est, P_est, ESS, etc.)
        %
        %   This method is intentionally NOT called inside timestep() so
        %   that filters can be used in live / demo contexts with zero
        %   history overhead.  The test bench calls it explicitly:
        %
        %     filter.timestep(z);
        %     filter.storeHistory(z, GT(:,k));   % testing only
        %
        %   Every subclass implementation MUST populate at minimum:
        %     history(k).x_est        - Gaussian mean
        %     history(k).P_est        - Gaussian covariance
        %     history(k).measurements - Raw measurements
        %     history(k).true_state   - Ground truth ([] if unavailable)
        %     history(k).timestep_num - Timestep counter value
        %
        % See also getGaussianEstimate, timestep, store_full_history
    end
    
    methods
        % Concrete methods shared by all subclasses
        
        function initializeDynamicPlot(obj, figure_name, figure_position)
            % INITIALIZEDYNAMICPLOT Initialize dynamic plotting figure
            %
            % SYNTAX:
            %   obj.initializeDynamicPlot(figure_name, figure_position)
            %
            % INPUTS:
            %   figure_name     - String name for the figure window
            %   figure_position - [x, y, width, height] figure position
            %
            % DESCRIPTION:
            %   Creates and configures the dynamic plotting figure if
            %   DynamicPlot is enabled. Called from subclass constructors.
            
            if obj.DynamicPlot
                obj.dynamic_figure_handle = figure('Name', figure_name, ...
                    'NumberTitle', 'off', 'Position', figure_position);
                
                if obj.debug
                    fprintf('[DYNAMIC PLOT] Initialized figure %d: %s\n', ...
                        obj.dynamic_figure_handle.Number, figure_name);
                end
            end
        end
        
        function updateDynamicPlot(obj, measurements, varargin)
            % UPDATEDYNAMICPLOT Update dynamic plot during timestep execution
            %
            % SYNTAX:
            %   obj.updateDynamicPlot(measurements)
            %   obj.updateDynamicPlot(measurements, true_state)
            %
            % INPUTS:
            %   measurements - Current measurements [N_z x N_measurements]
            %   true_state   - (optional) True state for comparison
            %
            % DESCRIPTION:
            %   Updates the dynamic visualization if enabled. Uses the
            %   subclass-specific visualize() method for rendering.
            
            if ~obj.DynamicPlot || isempty(obj.dynamic_figure_handle) || ...
               ~isvalid(obj.dynamic_figure_handle)
                return;
            end
            
            % Increment timestep counter
            obj.timestep_counter = obj.timestep_counter + 1;
            
            % Create title with timestep information
            title_str = sprintf('%s Real-time Tracking (Step %d)', ...
                class(obj), obj.timestep_counter);
            
            % Call subclass-specific visualization
            % TODO: Standardize visualize() method signature across subclasses
            if nargin > 2
                true_state = varargin{1};
                obj.visualize(obj.dynamic_figure_handle, title_str, measurements, true_state);
            else
                obj.visualize(obj.dynamic_figure_handle, title_str, measurements);
            end
            
            drawnow; % Force immediate update
            
            % Capture frame for animation after plot is updated
            obj.captureFrame();
            
            pause(0.01); % Small pause for smooth animation
        end
        
        function plotUpdate(obj, measurements, varargin)
            % PLOTUPDATE Update dynamic plot during timestep execution (manual call)
            %
            % SYNTAX:
            %   obj.plotUpdate(measurements)
            %   obj.plotUpdate(measurements, true_state)
            %
            % INPUTS:
            %   measurements - Current measurements [N_z x N_measurements]
            %   true_state   - (optional) True state for comparison [N_x x 1]
            %
            % DESCRIPTION:
            %   Updates the dynamic visualization if enabled. Uses the
            %   subclass-specific visualize() method for rendering.
            %   This is a separate function called manually from main loop,
            %   unlike updateDynamicPlot which is called automatically in timestep.
            
            if ~obj.DynamicPlot || isempty(obj.dynamic_figure_handle) || ...
               ~isvalid(obj.dynamic_figure_handle)
                return;
            end
            
            % Increment timestep counter
            obj.timestep_counter = obj.timestep_counter + 1;
            
            % Create title with timestep information
            title_str = sprintf('%s Real-time Tracking (Step %d)', ...
                class(obj), obj.timestep_counter);
            
            % Call subclass-specific visualization
            if nargin > 2
                true_state = varargin{1};
                obj.visualize(obj.dynamic_figure_handle, title_str, measurements, true_state);
            else
                obj.visualize(obj.dynamic_figure_handle, title_str, measurements);
            end
            
            drawnow; % Force immediate update
            pause(0.01); % Small pause for smooth animation
        end
        
        function count = getTimestepCount(obj)
            % GETTIMESTEPCOUNT Get current timestep count
            %
            % OUTPUTS:
            %   count - Current timestep number
            %
            % DESCRIPTION:
            %   Returns the current timestep count for display purposes.
            
            count = obj.timestep_counter;
        end
        
        function resetTimestepCount(obj)
            % RESETTIMESTEPCOUNT Reset timestep counter to zero
            %
            % DESCRIPTION:
            %   Resets the internal timestep counter. Useful when restarting
            %   tracking or processing multiple datasets.
            
            obj.timestep_counter = 0;
        end
        
        function success = validateCommonInputs(obj, measurements)
            % VALIDATECOMMONINPUTS Validate common input parameters
            %
            % SYNTAX:
            %   success = obj.validateCommonInputs(measurements)
            %
            % INPUTS:
            %   measurements - Current measurements [N_z x N_measurements]
            %
            % OUTPUTS:
            %   success - true if validation passes, false otherwise
            %
            % DESCRIPTION:
            %   Performs common input validation checks that apply to all
            %   filter types. Subclasses can extend with specific checks.
            
            success = true;
            
            % Check measurement format
            if ~isempty(measurements)
                if ~ismatrix(measurements) || size(measurements, 1) < 1
                    if obj.debug
                        fprintf('[VALIDATION ERROR] Invalid measurement format\n');
                    end
                    success = false;
                    return;
                end
                
                % Check for NaN or Inf values
                if any(isnan(measurements(:))) || any(isinf(measurements(:)))
                    if obj.debug
                        fprintf('[VALIDATION WARNING] NaN or Inf values in measurements\n');
                    end
                    % TODO: Decide whether to treat this as error or warning
                end
            end
            
            % TODO: Add more common validation checks:
            % - Scene boundary checks
            % - Measurement noise validation
            % - Temporal consistency checks
            
            if obj.debug && success
                fprintf('[VALIDATION] Common input validation passed\n');
            end
        end
        
        function printFilterInfo(obj)
            % PRINTFILTERINFO Print filter configuration information
            %
            % DESCRIPTION:
            %   Prints current filter settings and status for debugging.
            
            fprintf('\n=== %s FILTER INFO ===\n', class(obj));
            fprintf('Debug mode: %s\n', obj.debug);
            fprintf('Dynamic plotting: %s\n', obj.DynamicPlot);
            fprintf('Timestep count: %d\n', obj.timestep_counter);
            
            if obj.DynamicPlot
                if ~isempty(obj.dynamic_figure_handle) && isvalid(obj.dynamic_figure_handle)
                    fprintf('Dynamic figure: %d (valid)\n', obj.dynamic_figure_handle.Number);
                else
                    fprintf('Dynamic figure: invalid or closed\n');
                end
            end
            
            fprintf('========================\n\n');
        end
        
        function captureFrame(obj)
            % CAPTUREFRAME Capture current figure frame for animation
            %
            % SYNTAX:
            %   obj.captureFrame()
            %
            % DESCRIPTION:
            %   Captures the current figure frame and stores it in the Frames
            %   property for later saving as an animation. Only captures if
            %   DynamicPlot is enabled and figure handle exists.
            %
            % See also saveAnimation, clearFrames
            
            if obj.DynamicPlot && ~isempty(obj.dynamic_figure_handle) && isvalid(obj.dynamic_figure_handle)
                frame = getframe(obj.dynamic_figure_handle);
                obj.Frames{end+1} = frame;
            end
        end
        
        function saveAnimation(obj, filename, varargin)
            % SAVEANIMATION Save captured frames as video or GIF animation
            %
            % SYNTAX:
            %   obj.saveAnimation(filename)
            %   obj.saveAnimation(filename, 'FrameRate', 30, 'Quality', 75)
            %
            % INPUTS:
            %   filename - Output filename with extension (.avi, .mp4, .gif)
            %
            % OPTIONAL NAME-VALUE PAIRS:
            %   'FrameRate' - Video frame rate (default: 10 fps)
            %   'Quality'   - Video quality 0-100 (default: 75)
            %
            % DESCRIPTION:
            %   Saves all captured frames as an animated GIF.
            %   GIF format is most reliable and works universally.
            %
            % See also captureFrame, clearFrames
            
            if isempty(obj.Frames)
                warning('No frames captured. Enable DynamicPlot and ensure captureFrame() is called during timesteps.');
                return;
            end
            
            % Parse input arguments
            p = inputParser;
            addRequired(p, 'filename', @(x) ischar(x) || isstring(x));
            addParameter(p, 'FrameRate', 10, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'Quality', 75, @(x) isnumeric(x) && x >= 0 && x <= 100);
            parse(p, filename, varargin{:});
            
            % Convert string to char if needed
            filename = char(filename);
            
            [~, ~, ext] = fileparts(filename);
            
            % Default to GIF if no extension provided
            if isempty(ext)
                filename = [filename '.gif'];
                ext = '.gif';
            end
            
            if strcmpi(ext, '.gif')
                % Save as animated GIF (most reliable format)
                for k = 1:length(obj.Frames)
                    [A, map] = rgb2ind(frame2im(obj.Frames{k}), 256);
                    if k == 1
                        imwrite(A, map, filename, 'gif', 'LoopCount', inf, 'DelayTime', 1/p.Results.FrameRate);
                    else
                        imwrite(A, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1/p.Results.FrameRate);
                    end
                end
            else
                % For any other extension, save as GIF with warning
                gif_filename = strrep(filename, ext, '.gif');
                fprintf('Warning: Only GIF format supported. Saving as: %s\n', gif_filename);
                
                for k = 1:length(obj.Frames)
                    [A, map] = rgb2ind(frame2im(obj.Frames{k}), 256);
                    if k == 1
                        imwrite(A, map, gif_filename, 'gif', 'LoopCount', inf, 'DelayTime', 1/p.Results.FrameRate);
                    else
                        imwrite(A, map, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1/p.Results.FrameRate);
                    end
                end
            end
            
            fprintf('Animation saved: %s (%d frames)\n', filename, length(obj.Frames));
        end
        
        function clearFrames(obj)
            % CLEARFRAMES Clear all captured frames from memory
            %
            % SYNTAX:
            %   obj.clearFrames()
            %
            % DESCRIPTION:
            %   Clears the Frames property to free memory. Use this after
            %   saving an animation or when starting a new recording session.
            %
            % See also captureFrame, saveAnimation
            
            obj.Frames = {};
            fprintf('Captured frames cleared from memory.\n');
        end
        
        function hasFrames = hasFrames(obj)
            % HASFRAMES Check if frames have been captured
            %
            % SYNTAX:
            %   hasFrames = obj.hasFrames()
            %
            % OUTPUTS:
            %   hasFrames - True if frames exist, false otherwise
            %
            % DESCRIPTION:
            %   Utility method to check if any frames have been captured.
            %   Useful for external scripts to determine if animation can be saved.
            %
            % See also captureFrame, saveAnimation
            
            hasFrames = ~isempty(obj.Frames);
        end
        
        function count = getFrameCount(obj)
            % GETFRAMECOUNT Get number of captured frames
            %
            % SYNTAX:
            %   count = obj.getFrameCount()
            %
            % OUTPUTS:
            %   count - Number of captured frames
            %
            % See also hasFrames, captureFrame
            
            count = length(obj.Frames);
        end
    end
    
    methods (Static)
        function options = parseFilterOptions(varargin)
            % PARSEFILTEROPTIONS Parse common filter constructor options
            %
            % SYNTAX:
            %   options = DA_Filter.parseFilterOptions(varargin)
            %
            % INPUTS:
            %   varargin - Name-value pairs: 'Debug', true/false, 'DynamicPlot', true/false, 'ValidationSigma', numeric
            %
            % OUTPUTS:
            %   options - Struct with parsed options
            %
            % DESCRIPTION:
            %   Static utility method for parsing common constructor options
            %   across all filter subclasses. Ensures consistent behavior.
            
            p = inputParser;
            addParameter(p, 'Debug', false, @islogical);
            addParameter(p, 'DynamicPlot', false, @islogical);
            addParameter(p, 'ValidationSigma', 2, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'ESSThreshold', 0.5, @(x) isnumeric(x) && x > 0 && x <= 1);

            parse(p, varargin{:});
            options = p.Results;
        end
    end
end
