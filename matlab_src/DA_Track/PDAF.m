
classdef PDAF
    properties
        x0 % Inital State
        P0 % Inital Cov
        F % Dynamics
        Q % Process Noise cov
        R % Measurment Noise oov
        H % Meas fxn
    end

    methods
        function obj = PDAF(x0,P0,F,Q,R,H)
            %%% CLASS CONSTRUCTOR

            assert(nargin == 6,"Invalid number of inputs. Args: {x0, P0, F, Q, R, H}")
            obj.x0 = x0; 
            obj.P0 =P0;
            obj.F = F;
            obj.Q = Q;
            obj.R = R;
            obj.H = H;
        end

        %% timestep: Iterate PDAF through a single timestep
        %{
            args:
                x_prior -> estimated state from previous tstep
                P_prior -> estimate state cov from previous tstep
                z -> all detections from simulated mWidar image
            outputs:
                X -> Updated state estimate
                P -> Updated state cov estimate
        %}
        function [X,P] = timestep(obj,x_prior, P_prior,z)
            
            beta = 0;
            beta0 = 0;

            [x_minus, z_hat, P_minus, S] = obj.prediction(x_prior, P_prior);
            [valid_z, meas] = obj.Validation(S, z_hat,z);
            if meas
                [beta, beta0] = obj.Data_Association(z_hat,valid_z, S);
            end

            [X, P] = obj.state_Estimation(beta, beta0, P_minus, x_minus, z_hat, meas, valid_z,S);

        end

        %% Prediction Step: Standard KF prediction step
        %{
            args:
                obj -> class object
                x_prior -> estimated state from previous tstep
                P_prior -> estimate state cov from previous tstep
            
            outputs:
                x_minus -> Predicted state
                z_hat -> Predicted measument
                P_minus -> Predicted state cov
                S -> innovation cov
        %}   

        function [x_minus, z_hat, P_minus, S] = prediction(obj, x_prior, P_prior)
            
            x_minus = obj.F*x_prior; % Dynamics update for state
            z_hat = obj.H*x_minus; % Pred meas
            P_minus = obj.F * P_prior * obj.F' + obj.Q; % state covariance prediction
            S = obj.H * P_minus * obj.H' + obj.R; % measurment innovation covariance

        end

        %% Validation (Gating) Step: Trims down detections to a set of detections that pass through the validation ellipse
        %{
            args:
                S -> innovation covariance
                z_hat -> predicted measurment
                z -> set of all detections
            outputs:
                valid_z -> set of detections that pass through validation ellipse
                meas -> boolean, set to false if no measurments pass through the ellipse (missed detection)
        %}

        function [valid_z, meas] = Validation(obj, S, z_hat,z)
            meas = true; % default
            gamma = chi2inv(0.05/2, 2); % Threshold
            valid_z = [];

            for j = 1:size(z, 2)
                detection = z(:, j);
                Nu = (detection - z_hat)' / S * (detection - z_hat); % Validation ellipse (NIS statistic)

                if Nu < gamma % Validation gate
                    valid_z = [valid_z detection]; % Append new measurment onto validated list
                end

            end

            if isempty(valid_z)
                fprintf('No Valid Measurments \n')
                meas = false; % missed detection
            end
        end

        %% PDA: Standard Probabilistic Data Association algorithm
        %{
            args:
                z_hat -> predicted measurment
                valid_z -> set of valid measurments
                S -> innovation covariance
            outputs:
                beta -> probability of detection i being the true target
                beta0 -> probability of none of the detections being the true target
        %}

        function [beta, beta0] = Data_Association(obj, z_hat,valid_z, S)

        % Tuning parameters
        lambda = 2.5;
        PD = 0.95;
        PG = 0.95;

        % Pre allocate space
        likelihood = zeros(1, size(valid_z, 2));
        beta = zeros(1, size(valid_z, 2));

        % Compute likelihood of each validated measurment
        for j = 1:size(valid_z, 2)
            likelihood(j) = (mvnpdf(valid_z(:, j), z_hat, S) * PD) / lambda;
        end

        sum_likelihood = sum(likelihood, 2); % sum

        for j = 1:size(valid_z, 2)
            beta(j) = likelihood(j) / (1 - PD * PG + sum_likelihood); % Compute beta values
        end

        beta0 = (1 - PD * PG) / (1 - PD * PG + sum_likelihood); % beta0 -> probability of no detections being true

        end

        %% state_Estimation: Modified KF measument update step for PDA
        %{
            args:
                beta -> probability of detection i being the true target
                beta0 -> probability of none of the detections being the true target
                P_minus -> predicted state cov
                x_minus -> predicted state
                z_hat -> predicted meas
                meas -> boolean, set to false if no measurments pass through the ellipse (missed detection, beta0 = 1)
                z -> set of validated measurments
                S -> innovation covariance
            outputs:
                X -> Updated state estimate
                P -> Updated state cov estimate
        %}
        function [X, P] = state_Estimation(obj, beta, beta0, P_minus, x_minus, z_hat, meas, z,S)

            KK = 0;
            innov = 0;
            Pc = 0;
            P_tilde = 0;
            
            if meas == false % Assign beta0 = 1 and ensure filter won't break
                z = z_hat;
                innov = zeros(size(z, 1), 1);

                %For Probability Data Association
                beta0 = 1; %switching value

                %Calc Kalman gain
                KK = (P_minus * obj.H') / (obj.H * P_minus * obj.H' + obj.R);
                Pc = P_minus - KK * S * KK'; % Not necessary since beta0 = 1
                P_tilde = 0; 

            else
                nu = zeros(2, size(z, 2));
                innov = [0; 0];

                for j = 1:size(z, 2)
                    nu(:, j) = z(:, j) - z_hat;
                    innov = innov + beta(j) * nu(:, j); % Combined innovations -> weighted sum of measurments
                end

                % Standard kalman state update
                KK = P_minus * obj.H' * S;

                Pc = P_minus - KK * S * KK';

                temp = 0;

                for j = 1:size(z, 2)
                    temp = temp + (beta(j) * (nu(j) * nu(j)'));
                end

                P_tilde = KK * (temp - innov * innov') * KK';

                
            end
            
            % Updated state/Covariance
            X = x_minus + KK * innov;
            P = beta0 * P_minus + (1 - beta0) * Pc + P_tilde; 
        end

    end

end