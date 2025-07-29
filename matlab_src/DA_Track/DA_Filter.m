classdef (Abstract) DA_Filter < handle
    % DA_Filter Abstract base class for Data Association filters
    %
    % DESCRIPTION:
    %   Abstract base class defining the common interface for all data association
    %   filters in the mWidar tracking system. Provides standard methods and
    %   properties that all concrete filter implementations must support.
    %
    % ABSTRACT PROPERTIES:
    %   All concrete subclasses must define these properties
    %
    % ABSTRACT METHODS:
    %   timestep                - Process single time step with measurements
    %   getGaussianEstimate     - Extract Gaussian state estimate 
    %   prediction              - Perform prediction step
    %   measurement_update      - Perform measurement update step
    %
    % CONCRETE METHODS:
    %   updateDynamicPlot       - Update real-time visualization
    %   getTimestepCount        - Get current timestep counter
    %   validateCommonInputs    - Input validation utilities
    %
    % SUBCLASSES:
    %   PDA_PF                  - Particle Filter with PDA data association
    %   PDA_HMM                 - Hidden Markov Model with PDA data association
    %   GNN_PF                  - Particle Filter with GNN data association
    %
    % See also PDA_PF, PDA_HMM, GNN_PF
    
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
            
            % TODO: Add more common options:
            % - 'Verbose', false
            % - 'ValidationLevel', 'basic'
            % - 'PlotStyle', 'default'
            
            parse(p, varargin{:});
            options = p.Results;
        end
    end
end
