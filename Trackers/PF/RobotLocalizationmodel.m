function [state] = RobotLocalizationmodel(state, Q, whist)
    dt = 0.02; %should be small when using Euler integration
    qhist = 0.5; % linear velocity profile, in m/s
    
    % Ensure whist is a scalar or matches number of particles
    if isscalar(whist)
        whist = whist * ones(1, size(state,2));
    end
    
    % Generate process noise for each particle
    v_time = mvnrnd(zeros(3,1), Q, size(state,2))';
    
    % Integrate each particle through the system model
    state(1,:) = state(1,:) + dt*qhist*cos(state(3,:)) + dt*v_time(1,:);
    state(2,:) = state(2,:) + dt*qhist*sin(state(3,:)) + dt*v_time(2,:);
    state(3,:) = state(3,:) + dt*whist + dt*v_time(3,:);
end