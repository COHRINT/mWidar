%% generate_track: 
%{
    Given an IC, x0 dynamics model, A and time vector, tvec; generate a track (GT) for n timesteps
%}


function GT = generate_track(x0,A,tvec)

    n = size(tvec,2); % # of timesteps
    dt = 0.1;
    GT = zeros(6,n);
    GT(:,1) = x0;

    for k = 2:n
        GT(:,k) = expm(A*dt)*GT(:,k-1); % Propogate dynamics model

    end

    GT = normalized_track(GT);
end

function GT_norm = normalized_track(GT)
% Normalizes a track to keep positions within mWidar sim bounds

x_bounds = [-2 2];
y_bounds = [0 4];

x_pos = GT(1,:);
y_pos = GT(2,:);

current_x_range = max(x_pos) - min(x_pos);
current_y_range = max(y_pos) - min(y_pos);

% 4x4 mwidar scene
desired_x_range = 3.8;
desired_y_range = 3.8;

x_scale = desired_x_range/current_x_range;
y_scale = desired_y_range/current_y_range;

GT_norm = GT; % Initialize

GT_norm(1,:) = (x_pos - min(x_pos)) * x_scale + -1.9;
GT_norm(3,:) = GT(3,:)*x_scale;
GT_norm(5,:) = GT(5,:)*x_scale;

GT_norm(2,:) = (y_pos - min(y_pos)) * y_scale + 0.1;
GT_norm(4,:) = GT(4,:)*y_scale;
GT_norm(6,:) = GT(6,:)*y_scale;

end
