
classdef JPDAF
    properties
        x0 % Inital State
        P0 % Inital Cov
        F % Dynamics
        Q % Process Noise cov
        R % Measurment Noise oov
        H % Meas fxn
        PD % Probability of detection
        t % Number of tracks
    end

    methods
        function obj = JPDAF(x0,P0,F,Q,R,H,PD,t)
            assert(nargin == 8,"Invalid number of inputs. Args: {x0, P0, F, Q, R, H, PD, t}")
            obj.x0 = x0; 
            obj.P0 =P0;
            obj.F = F;
            obj.Q = Q;
            obj.R = R;
            obj.H = H;
            obj.PD = PD;
            obj.t = t;
        end
        
        function [X, P] = timestep(obj,x_prior,P_prior,z)
            [x_minus, z_hat, P_minus, S] = obj.prediction(x_prior, P_prior);
            [valid_z, meas, V] = obj.Validation(S, z_hat,z); % Currently not using valid_z to see if it works without gating
            hypothesis = obj.generate_hypotheses(valid_z);
            L = obj.meas_liklihood(z_hat,S,valid_z);
            joint_probs = obj.compute_joint_probs(hypothesis,L,valid_z,V); % Issues here?
            beta = obj.compute_marginal_probs(hypothesis,joint_probs,valid_z);
            [X,P] = obj.measurment_step(beta,P_minus, x_minus, z_hat, meas, valid_z,S);

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
        
        %%%% Assuming dynamics + measurment model is constant %%%%%%%

        function [x_minus, z_hat, P_minus, S] = prediction(obj, x_prior, P_prior)
            x_minus = cell(1,obj.t);
            P_minus = cell(1,obj.t);
            z_hat = cell(1,obj.t);
            S = cell(1,obj.t);

            for i = 1:obj.t
                x_minus{i} = obj.F*x_prior{i}; % Dynamics update for state
                z_hat{i} = obj.H*x_minus{i}; % Pred meas
                P_minus{i} = obj.F * P_prior{i} * obj.F' + obj.Q; % state covariance prediction
                S{i} = obj.H * P_minus{i} * obj.H' + obj.R; % measurment innovation covariance
            end
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
                V -> Volume of 2D validation ellipse
        %}

        function [valid_z, meas, V] = Validation(obj, S, z_hat,z)
            meas = true(obj.t); % default
            gamma = chi2inv(0.05/2, 2); % Threshold, same for all tracks
            valid_z = [];
            V = zeros(1,obj.t);

            for i = 1:obj.t
                for j = 1:size(z, 2)
                    detection = z(:, j);
                    Nu = (detection - z_hat{i})' / S{i} * (detection - z_hat{i}); % Validation ellipse (NIS statistic)
    
                    if Nu < gamma % Validation gate
                        valid_z = [valid_z detection]; % Append new measurment onto validated list
                    end
    
                end

                V = pi*gamma*sqrt(det(S{1})); % Volume of validation region
            end
            
            if size(valid_z,2) ~= obj.t
                    fprintf('At least one obj not detected \n')
                    meas(i) = false; % missed detection
            end
            
        end

        %% Data Association
        
        %%% Generate Hypotheses

        function hypothesis = generate_hypotheses(obj,z)
            m = size(z,2);

            idx = 0:m; % All possible measurment index
            [grid_combo{1:obj.t}] = ndgrid(idx); % t-dimensional grid for all possible combinations
            combo = cellfun(@(x) x(:), grid_combo, 'UniformOutput', false);
            all = [combo{:}]; % Flatten

            % Trim to only keep valid hypotheses
            valid = arrayfun(@(i) obj.isValidAssignment(all(i,:)), 1:size(all,1));
            hypothesis = all(valid,:);  % n_hyp given by (m choose t)*2 + 2m+1
        end

        %%% Valid hypotheses
        function valid = isValidAssignment(obj,assign)
            meas_used = assign(assign > 0);
            valid = length(meas_used) == length(unique(meas_used)); % Must be unique (can't have one measurment associate to two tracks, or vice versa)
        end
        
        %%% Compute Liklihoods and all innovations

        function L = meas_liklihood(obj,z_hat,S,z)
            L = zeros(obj.t,size(z,2));

            for i = 1:obj.t
                for j = 1:size(z,2)
                    L(i,j) = mvnpdf(z(:,j),z_hat{i},S{i});
                end
            end
        end

        %%% Compute Joint Association Probabilities
        function joint_probs = compute_joint_probs(obj,hypothesis,L,z,V)
            m = size(z,2);
            n_hyp = size(hypothesis,1);
            P_D = obj.PD;
            joint_probs = zeros(n_hyp,1);

            for h = 1:n_hyp % Loop through hypothesis
                theta = hypothesis(h,:); % Current hypothesis
                
                used_theta = theta(theta>0);
                f = numel(used_theta); % Number of measurments used
                
                joint_prob = 1; % Joint probability for this specific hyp
                lambda = 0.25;

                
                for i = 1:obj.t % Loop through tracks
                    j = theta(i); % Detection j assigned to track i
                    if j == 0 % Missed detection
                        joint_prob = joint_prob*(1-P_D); % Prob of missed detection (delta_t = 0)
                    else % Detected measurment (Tau = 1)
                        joint_prob = joint_prob * (lambda)^-1 * L(i,j) * obj.PD; % (delta_t = 1)
                    end

                end
                joint_probs(h) = joint_prob;
            end

            joint_probs = joint_probs/sum(joint_probs);
                
        end

        %%% Marginal association probabilities
        function beta = compute_marginal_probs(obj,hypothesis,joint_probs,z)
            m = size(z,2);
            beta = zeros(obj.t,m+1);
            n_hyp = size(hypothesis,1);

            for h = 1:n_hyp
                theta = hypothesis(h,:);
                for i = 1:obj.t
                    j = theta(i);
                    if j == 0
                        beta(i,m+1) = beta(i,m+1) + joint_probs(h); % missed detection -> goes in last spot
                    else
                        beta(i,j) = beta(i,j) + joint_probs(h);
                    end
                end
            end
        end

        %% Kalman measurment update
        function [X,P] = measurment_step(obj,beta,P_minus, x_minus, z_hat, meas, z,S)
            X = cell(obj.t,1);
            P = cell(obj.t,1);
            m = size(z,2);
            for i = 1:obj.t
                beta0 = beta(i,m+1);

                if false % Nothing passed through validation ellipse, only get prediction update
                    X{i} = x_minus{i};
                    P{i} = P_minus{i};
                else

                    nu = zeros(2, m);
                    innov = [0; 0];
    
                    for j = 1:m
                        nu(:, j) = z(:, j) - z_hat{i};
                        innov = innov + beta(i,j) * nu(:, j); % Combined innovations -> weighted sum of measurments
                    end
    
                    % Standard kalman state update
                    KK = P_minus{i} * obj.H' * S{i};
    
                    Pc = P_minus{i} - KK * S{i} * KK';
    
                    temp = 0;
    
                    for j = 1:size(z, 2)
                        temp = temp + (beta(i,j) * (nu(:,j) * nu(:,j)'));
                    end
    
                    P_tilde = KK * (temp - innov * innov') * KK';

                        % Updated state/Covariance
                        X{i} = x_minus{i} + KK * innov;
                        P{i} = beta0 * P_minus{i} + (1 - beta0) * Pc + P_tilde; 
                end
            end
        end

    end
end