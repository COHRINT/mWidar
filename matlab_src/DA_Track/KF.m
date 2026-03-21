classdef KF < handle
    % KF Kalman Filter base class
    %
    % DESCRIPTION:
    %   Implements standard Kalman filtering for
    %   single target tracking under measurement uncertainty.
    %
    % PROPERTIES:
    %   x, P    - State estimate and state estimate covariance
    %   F, Q      - Dynamics model and process noise matrices
    %   H, R      - Observation model and measurement noise matricies
    %   z, S, K  - Innovation, innovation covariance, Kalman gain
    %
    % METHODS:
    %   KF              - Constructor
    %   timestep            - Process single time step with KF algorithm
    %   prediction          - Standard Kalman prediction step
    %   measurement_update  - Standard Kalman measurement update
    %   Data_Association    - Compute PDA beta coefficients
    %   Validation          - Measurement gating using chi-squared test
    %
    % EXAMPLE:
    %   kf = KF(x0, P0, F, Q, R, H);
    %   [x_est, P_est] = kf.timestep(x_prev, P_prev, measurements);
    %

    properties
        % State estimate and covariance
        x % State estimate
        P % State estimate covariance

        % Dynamics model and process noise
        F % State transition matrix
        Q % Process noise covariance

        % Observation model and measurement noise
        H % Observation matrix
        R % Measurement noise covariance

        % Innovation, innovation covariance, Kalman gain
        z % Innovation
        S % Innovation covariance
        K % Kalman gain
    end

    methods

        function obj = KF(x0, P0, F, Q, H, R)
            % KF Constructor for Kalman Filter class
            %
            % INPUTS:
            %   x0  - Initial state estimate
            %   P0  - Initial state estimate covariance
            %   F   - State transition matrix
            %   Q   - Process noise covariance
            %   H   - Observation matrix
            %   R   - Measurement noise covariance
            
            obj.x = x0;
            obj.P = P0;
            obj.F = F;
            obj.Q = Q;
            obj.H = H;
            obj.R = R;

        end

        function [x_est, P_est] = timestep(obj, measurements)
            % timestep Process a single time step with Kalman Filter
            %
            % INPUTS:
            %   measurements - Set of measurements at current time step
            %
            % OUTPUTS:
            %   x_est - Updated state estimate
            %   P_est - Updated state estimate covariance

            % Prediction step
            obj.prediction();

            % Measurement update step
            obj.measurement_update(measurements);

            % Return updated estimates
            x_est = obj.x;
            P_est = obj.P;
        end

        function prediction(obj)
            % prediction Standard Kalman prediction step
            %
            % Updates the state estimate and covariance based on the
            % dynamics model and process noise.

            obj.x = obj.F * obj.x;
            obj.P = obj.F * obj.P * obj.F' + obj.Q;
        end

        function measurement_update(obj, measurements)
            % measurement_update Standard Kalman measurement update
            %
            % Incorporates the given measurements to update the state
            % estimate and covariance.

            % Innovation
            obj.z = measurements - obj.H * obj.x;

            % Innovation covariance
            obj.S = obj.H * obj.P * obj.H' + obj.R;

            % Kalman gain
            obj.K = obj.P * obj.H' / obj.S;

            % Update state estimate and covariance
            obj.x = obj.x + obj.K * obj.z;
            obj.P = obj.P - obj.K * obj.S * obj.K';
        end

        function [innov, innov_cov] = getInnovation(obj, z)
            % GETINNOVATION Compute innovation and innovation covariance
            %
            % NOT FOR USE IN STANDARD KF CYCLE -- ONLY USED TO CALCULATE LIKELIHOOD
            % OF MEASUREMENTS IN PDA AND RBPF CONTEXTS
            %
            % INPUTS:
            %   z - Measurement vector
            %
            % OUTPUTS:
            %   innov     - Innovation (measurement residual)
            %   innov_cov - Innovation covariance

            innov = z - obj.H * obj.x;
            innov_cov = obj.H * obj.P * obj.H' + obj.R;
        end



    end

    methods (Static)
        
        function new_kf = copyKF(original_kf)
            % COPYKF Deep copy a KF object
            %
            % INPUTS:
            %   original_kf - KF object to copy
            %
            % OUTPUTS:
            %   new_kf - Independent copy of KF object
            %
            % CRITICAL: This creates a new KF object with independent memory.
            % Without this, resampling would create particles sharing the
            % same KF object (disaster for particle filters!)
            
            new_kf = KF(original_kf.x, original_kf.P, ...
                        original_kf.F, original_kf.Q, ...
                        original_kf.H, original_kf.R);
            
            % Copy innovation state if it exists
            if ~isempty(original_kf.z)
                new_kf.z = original_kf.z;
                new_kf.S = original_kf.S;
                new_kf.K = original_kf.K;
            end
        end
        
    end

end
