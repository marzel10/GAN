function Rg = calculate_Rg(xi, yi, x, y, xj, yj)
% CALCULATE_RG Computes the Rg ratio based on the formula:
% Rg = (sqrt((xi-x)^2 + (yi-y)^2) + sqrt((xj-x)^2 + (yj-y)^2)) / sqrt((xi-xj)^2 + (yi-yj)^2)
%
% This formula calculates the ratio of the sum of distances from two points
% (xi,yi) and (xj,yj) to a central point (x,y), divided by the direct 
% distance between the two points.
%
% INPUTS:
%   xi, yi - coordinates of first point
%   x, y   - coordinates of central/reference point  
%   xj, yj - coordinates of second point
%
% OUTPUT:
%   Rg - the calculated ratio (scalar or array if inputs are arrays)


    % Calculate distance from point i to central point
    dist_i_to_center = sqrt((xi - x).^2 + (yi - y).^2);
    
    % Calculate distance from point j to central point  
    dist_j_to_center = sqrt((xj - x).^2 + (yj - y).^2);
    
    % Calculate direct distance between points i and j
    dist_i_to_j = sqrt((xi - xj).^2 + (yi - yj).^2);
    
    % Handle division by zero case
    if any(dist_i_to_j == 0, 'all')
        warning('calculate_Rg:DivisionByZero', ...
            'Distance between points i and j is zero. Setting Rg to Inf.');
        dist_i_to_j(dist_i_to_j == 0) = eps; % Replace zeros with small number
    end
    
    % Calculate Rg ratio
    Rg = (dist_i_to_center + dist_j_to_center) ./ dist_i_to_j;
    
end

function d = dimless_nor_dist(x,y,xi,yi,xj,yj)
    % Fixed version that handles meshgrid inputs properly
    % x, y can be meshgrids (matrices) or scalars
    
    % Vector from i to j (constant for all points)
    v_ij = [xj - xi, yj - yi];
    v_ij_norm = sqrt(sum(v_ij.^2));  % FIXED: use sqrt for actual distance
    
    % For each point in the meshgrid, calculate the cross product
    % Vector from i to point (x,y)
    v_ix = x - xi;
    v_iy = y - yi;
    
    % 2D cross product: v1 × v2 = v1x*v2y - v1y*v2x
    % Here: v_i_to_point × v_i_to_j
    cross_product = v_ix * (yj - yi) - v_iy * (xj - xi);
    
    % Perpendicular distance from point to line i-j
    d = abs(cross_product) / v_ij_norm;  % FIXED: divide by norm, not norm²

    if all(d(:) == 0)
        error('All values in dimless_nor_dist are zero.');
    end
    
end
function [x, y, rapid_distribution] = calculate_RAPID(xi, yi, xj, yj, beta)
% CALCULATE_RAPID Computes the RAPID distribution factor
    x = linspace(-20, 185, 400);
    y = linspace(-20, 260, 400);
    [X, Y] = meshgrid(x,y);
    Rg = calculate_Rg(xi, yi, X, Y, xj, yj);

    rapid_distribution = (beta - Rg)/(1-beta);
    rapid_distribution(Rg > beta) = 0; % Set values outside the ellipse to zero

end

function [x, y, normal_distribution] = calculate_normal_gaussian(xi, yi, xj, yj, DI, var, max_Rg)
% CALCULATE_NORMAL_GAUSSIAN Computes the normal Gaussian distribution factor
    x = linspace(-20, 185, 400);
    y = linspace(-20, 260, 400);

    [X, Y] = meshgrid(x,y);
    Rg = calculate_Rg(xi, yi, X, Y, xj, yj);
    
    % Debug: check the distance values
    distances = dimless_nor_dist(X,Y,xi,yi,xj,yj);
    fprintf('Distance range: [%.3f, %.3f]\n', min(distances(:)), max(distances(:)));
    fprintf('Target DI: %.3f, Variance: %.3f\n', DI, var);
    
    % Check if DI is within reasonable range of distances
    if DI > max(distances(:)) + 3*var
        warning('DI = %.3f is too large. Max distance = %.3f. Consider reducing DI.', DI, max(distances(:)));
    end
    if DI < min(distances(:)) - 3*var
        warning('DI = %.3f is too small. Min distance = %.3f. Consider increasing DI.', DI, min(distances(:)));
    end
    
    normal_distribution = 1/(var*sqrt(2*pi)) * exp(-0.5*((distances-DI)/var).^2);
    
    fprintf('Normal distribution range: [%.6e, %.6e]\n', min(normal_distribution(:)), max(normal_distribution(:)));
    normal_distribution(Rg > max_Rg) = 0; % Set values outside the ellipse to zero
    %normal_distribution = dimless_nor_dist(X,Y,xi,yi,xj,yj);
end

function plot_distributions(X,Y,distribution, x_i, y_i, x_j, y_j, Title)
    figure;
        hold on;
        imagesc(X,Y,distribution);
        colorbar;
        axis image;
        title(sprintf('%s', Title));
        xlabel('X-axis');
        ylabel('Y-axis');
        colormap(jet);
        set(gca, 'YDir', 'normal');

        % % Plot point i
        % plot(x_i, y_i, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'red', ...
        %     'LineWidth', 2, 'DisplayName', sprintf('Point i (%.1f, %.1f)', x_i, y_i));

        % % Plot point j  
        % plot(x_j, y_j, 'bs', 'MarkerSize', 6, 'MarkerFaceColor', 'red', ...
        %     'LineWidth', 2, 'DisplayName', sprintf('Point j (%.1f, %.1f)', x_j, y_j));

        pos = [25 25; 
            25 100; 
            215 140; 
            215 66; 
            120 25; 
            120 140; 
            90 82.5; 
            180 82.5];
        scatter(pos(:,2), pos(:,1), 50, 'g', 'filled', 'DisplayName', 'Sensors');
        % Add a line connecting the points
        plot([x_i, x_j], [y_i, y_j], 'k--', 'LineWidth', 2, ...
            'DisplayName', 'Connection i-j');

        % Add text labels
        text(x_i + 15, y_i + 15, 'i', 'FontSize', 10, 'FontWeight', 'bold', ...
            'Color', 'white');
        text(x_j + 15, y_j + 15, 'j', 'FontSize', 10, 'FontWeight', 'bold', ...
            'Color', 'white');

        rectangle('Position',[0 0 165 240], 'EdgeColor','k', 'LineStyle','--', 'LineWidth',1.5)
        % Add legend
        legend('Location', 'best');

        hold off;
end


x_i = 140;
y_i = 120;

x_j = 25;
y_j = 25;

v = 5000;
s = 9.995*10^-4 * v*10^3; % in mm
sen_dist = sqrt((x_i - x_j).^2 + (y_i - y_j).^2);
max_dist = s/sen_dist;
disp(max_dist);


betas = [max_dist]; 
if true
    for beta = betas
        [X,Y, rapid_distribution] = calculate_RAPID(x_i, y_i, x_j, y_j, beta);
    
        plot_distributions(X,Y, -rapid_distribution, x_i, y_i, x_j, y_j, sprintf('RAPID Distribution (beta=%.3f)', beta));
    end
end
DI = 0.5;    % Center the distribution ON the i-j line (distance = 0)
var = 5;  % Much wider variance to cover the large distance range

[X,Y, normal_distribution] = calculate_normal_gaussian(x_i, y_i, x_j, y_j, DI, var, max_dist);

plot_distributions(X,Y, normal_distribution, x_i, y_i, x_j, y_j, 'Normal Gaussian Distribution');
