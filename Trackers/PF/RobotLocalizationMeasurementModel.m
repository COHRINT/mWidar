function predicted_measurements = RobotLocalizationMeasurementModel(particles, landmarks)
    % Compute range measurements for each particle to each landmark
    % Inputs:
    %   - particles: robot states (3 x n_particles)
    %   - landmarks: landmark positions (n_landmarks x 2)
    % Output:
    %   - predicted_measurements: range measurements (n_landmarks x n_particles)
    
    n_landmarks = size(landmarks, 1);
    n_particles = size(particles, 2);
    
    predicted_measurements = zeros(n_landmarks, n_particles);
    
    for ll = 1:n_landmarks
        landmark_pos = repmat(landmarks(ll, :)', 1, n_particles);
        predicted_measurements(ll, :) = sqrt(sum((landmark_pos - particles(1:2, :)).^2, 1));
    end
end

