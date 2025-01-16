function objects = update_objects(objects, dt)
% Function update_objects
%
% Parameters
% ----------
% objects: 6xn array
%       [x;y;vx;vy;ax;ay];
%       n objects to be updated
%
% dt: timestep
%
% Equivalent C++ Code:
% void update(float time_step) {
%     x += vx * time_step + 0.5 * ax * time_step * time_step;
%     y += vy * time_step + 0.5 * ay * time_step * time_step;
%     vx += ax * time_step;
%     vy += ay * time_step;
% }
%
% Returns
% -------
% objects: 6xn array

objects(1,:) = 0.5 * objects(5,:) * dt * dt +objects(3,:) * dt + objects(1,:);
objects(2,:) = 0.5 * objects(6,:) * dt * dt +objects(4,:) * dt + objects(2,:);
objects(3,:) = objects(5,:) * dt + objects(3,:);
objects(4,:) = objects(6,:) * dt + objects(4,:);

% r

end
