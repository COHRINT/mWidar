function clustered_xy = clusterNearbyDetections(meas_xy, cluster_radius)
% CLUSTERNEARBYDETECTIONS  BFS clustering of 2D detections.
%
% Lifted from probObjCt.m local function. Returns the centroid of each
% connected component (radius-graph) as a column.

    if isempty(meas_xy)
        clustered_xy = zeros(2, 0);
        return
    end

    n_det = size(meas_xy, 2);
    visited = false(1, n_det);
    clustered_xy = zeros(2, 0);

    for i = 1:n_det
        if visited(i)
            continue
        end

        component = i;
        queue = i;
        visited(i) = true;

        while ~isempty(queue)
            current = queue(1);
            queue(1) = [];

            for j = 1:n_det
                if visited(j)
                    continue
                end

                if norm(meas_xy(:, current) - meas_xy(:, j)) <= cluster_radius
                    visited(j) = true;
                    queue(end + 1) = j; %#ok<AGROW>
                    component(end + 1) = j; %#ok<AGROW>
                end
            end
        end

        clustered_xy(:, end + 1) = mean(meas_xy(:, component), 2); %#ok<AGROW>
    end
end
