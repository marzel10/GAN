classdef attention_matrix
    methods(Static)
        function flag = doIntersect(p1, q1, p2, q2)
            o1 = attention_matrix.orientation(p1, q1, p2);
            o2 = attention_matrix.orientation(p1, q1, q2);
            o3 = attention_matrix.orientation(p2, q2, p1);
            o4 = attention_matrix.orientation(p2, q2, q1);
            flag = (o1 ~= o2) && (o3 ~= o4);
            if o1 == 0 || o2 == 0 || o3 == 0 || o4 == 0
                flag = 0.5; % Collinear case
            end
        end


        function o = orientation(p, q, r)
            val = (q(2)-p(2))*(r(1)-p(1)) - (q(1)-p(1))*(r(2)-p(2));
            abs_val = abs(val);
            if abs_val < 1e-10
                o = 0; % colinear
            elseif val > 0
                o = 1; % clockwise
            else
                o = 2; % counterclockwise
            end
        end


        function [attention, attention_norm] = build_attention()

            addpath(fullfile(pwd, 'src', 'utils')); % Use local utilities

            ns = 8;
            np = (ns-1)*ns/2;

            % Sensor positions from 1 to 8 (in meters)
            positions = 0.001*[25 25; 
                        100 25; 
                        140 215; 
                        65 215; 
                        25 120; 
                        140 120; 
                        82.5 80; 
                        82.5 160];

            panel_domain_x = 0.001*[0 0 165 165 0];
            panel_domain_y = 0.001*[0 240 240 0 0];
            
            
          

            % Plot positons of the sensors to see if everything is okay
            figure;
            hold on;
            plot(panel_domain_x, panel_domain_y, 'k-', 'LineWidth', 1.5); % Plot panel boundary
            scatter(positions(:,1), positions(:,2), 25, 'b', 'filled');  % Plot blue dots
            set(gca, 'XDir', 'reverse');
            for i = 1:size(positions,1)
                text(positions(i,1), positions(i,2), num2str(i), 'FontSize',8, 'VerticalAlignment','bottom', 'HorizontalAlignment','right');
            end
            axis equal;
            title('Sensor Positions');
            xlabel('X Position (m)');
            ylabel('Y Position (m)');
            grid on;
            hold off;

            % Generate all unique pairs/paths
            sensor_pairs = nchoosek(1:ns, 2);
            segments = positions(sensor_pairs,:);
            % Generate all unique sensor pairs 
            seg_pairs = nchoosek(1:np,2);
            nsp = size(seg_pairs,1); % number of segment pairs 


            A = positions(sensor_pairs(:,1),:); % Positions of the starting points 
            B = positions(sensor_pairs(:,2),:); % Positions od the end points 
            seg_vectors = A-B;
            lengths = sqrt(seg_vectors(:,1).^2+seg_vectors(:,2).^2);


            events = zeros(nsp,4);  % Each event: [first segment ID, second segment ID, do intersect?, cosine between the paths] 
            for i = 1:nsp % For every pair of segments

            p1 = [A(seg_pairs(i,1),:);B(seg_pairs(i,1),:)]; % Sensors positions for the Path 1 (2 points, 2 coordinates each -> 4)
            p2 = [A(seg_pairs(i,2),:);B(seg_pairs(i,2),:)]; % Sensors positions for the Path 2 

            % Cosine similarity between paths 
            v1 = seg_vectors(seg_pairs(i,1),:);
            v2 = seg_vectors(seg_pairs(i,2),:);
            cos = dot(v1/lengths(seg_pairs(i,1)),v2/lengths(seg_pairs(i,2)));

            % Intersection check 
            flag = attention_matrix.doIntersect(p1(1,:),p1(2,:),p2(1,:),p2(2,:));

            events(i, :) = [seg_pairs(i,1), seg_pairs(i,2),flag,abs(cos)];

            end

            % Attention score is Do_intersect? + 0.5 * cosine_similarity, normalized
            attention_score = [];
            attention_score = [attention_score; (events(:,3)+events(:,4))/2];

            rows =  seg_pairs(:,1);
            cols =  seg_pairs(:,2);
            values = attention_score(:);

            % Determine the size of the output matrix
            maxRow = max(rows);
            maxCol = max(cols);

            % Initialize the output matrix
            attention = zeros(maxRow, maxCol);

            % Fill the matrix using linear indexing
            ind = sub2ind([maxRow, maxCol], rows, cols);
            attention(ind) = values;
            attention = [attention;zeros(1,maxCol)]; % Add one row at the bottom of the attention matrix (to latter make it symmetric)
            attention = attention + attention'; % To fill in the entries that are below diagonal
            attention_threshold = attention>0.5; % No connection between paths that have attention less than 0.5
            attention = attention_threshold .* attention;

            A = attention + eye(size(attention));
            D = diag(sum(A,2));
            A_norm = D^(-1/2) * A * D^(-1/2);
            attention_norm = A_norm;
        end

    end
end


