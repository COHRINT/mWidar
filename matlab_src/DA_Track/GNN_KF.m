
classdef GNN_KF
    properties
        x0 % Inital State
        P0 % Inital Cov
        F % Dynamics
        Q % Process Noise cov
        R % Measurment Noise oov
        H % Meas fxn

        % May not need these
        %S % Innov cov
        %innov % Measurment innovation
    end
    methods
        function obj = GNN_KF(x0,P0,F,Q,R,H) 
            %%% CLASS CONSTRUCTOR

            assert(nargin == 6,"Invalid number of inputs. Args: {x0, P0, F, Q, R, H}")
            obj.x0 = x0; 
            obj.P0 =P0;
            obj.F = F;
            obj.Q = Q;
            obj.R = R;
            obj.H = H;

        end

        function [X,P] = timestep(obj, x_prior, P_prior,z)
            [x_minus, z_hat, P_minus, S] = obj.prediction(x_prior, P_prior);
            [valid_z, meas] = obj.Validation(S, z_hat,z);

            y = z_hat; % Failsave, is no measurment detected use pred measurment (innov = [0;0])

            if meas
                y = obj.GNN(S, valid_z, z_hat);
            end

            [X, P, ~] = obj.Measurment_Step(y, P_minus, x_minus, z_hat, S);

        end

        %% Prediction Step:
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
    
            % Unpack
            
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
            meas = true;
            gamma = chi2inv(0.05/2, 2);
            valid_z = [];

            for j = 1:size(z, 2)
                detection = z(:, j);
                Nu = (detection - z_hat)' / S * (detection - z_hat);

                if Nu < gamma % Validation gate
                    valid_z = [valid_z detection]; % Append new measurment onto validated list
                end

            end

            if isempty(valid_z)
                fprintf('No Valid Measurments \n')
                meas = false;
            end
        end

        %% Mahalnobis Dist Function for GNN DA

        function [d2] = mahalanobis(obj, innov, S)
            d2 = sqrt(innov \ S * innov);
        end

        %% GNN Data Association

        function [y] = GNN(obj, S, valid_z, z_hat)

            d2 = zeros(size(valid_z, 2), 1);
            for i = 1:size(valid_z, 2)
                innov = valid_z(:, i) - z_hat;
                d2(i) = obj.mahalanobis(innov, S);
            end

            [~, idx] = min(d2);
            y = valid_z(:, idx);

        end

        %% Measurment Update

        function [X, P, innov] = Measurment_Step(obj,y, P_minus, x_minus, z_hat, S)

            KK = P_minus * obj.H' / S;
            innov = y - z_hat;
            X = x_minus + KK * innov;
            P = (eye(6) - KK * obj.H) * P_minus;

        end
    end
end