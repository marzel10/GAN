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

d103 = 0.001*[50 80]; % position of the inpact for the panel 103
d104 = 0.001*[25 80]; % position of the inpact for the panel 104
d105 = 0.001*[165-50 240-80]; % position of the inpact for the panel 105
d109 = 0.001*[165-82.5 240-100]; % position of the inpact for the panel 109

sprintf("Which paths would you like to plot? (indexing according to SP.mat)")
paths = input('> ');

segments = load("data\SP.mat").SP_table;


figure;
hold on;
plot(panel_domain_x, panel_domain_y, 'k-', 'LineWidth', 1.5); % Plot panel boundary
scatter(positions(:,1), positions(:,2), 25, 'b', 'filled');  % Plot the sensors
set(gca, 'XDir', 'reverse');
for i = 1:size(positions,1)
    text(positions(i,1), positions(i,2), num2str(i), 'FontSize',8, 'VerticalAlignment','bottom', 'HorizontalAlignment','right');
end

% Plot selected paths
for p = paths
    start = segments.Sensor1_idx(p);
    ending = segments.Sensor2_idx(p);
    x = [positions(start,1), positions(ending,1)];
    y = [positions(start,2), positions(ending,2)];
    plot(x, y, 'r-', 'LineWidth', 1.2); % Plot the path segment
end

% Plot impact points
scatter(d103(1), d103(2), 50, 'r', 'filled', 'DisplayName', 'Impact 103');
scatter(d104(1), d104(2), 50, 'r', 'filled', 'DisplayName', 'Impact 104');
scatter(d105(1), d105(2), 50, 'r', 'filled', 'DisplayName', 'Impact 105');
scatter(d109(1), d109(2), 50, 'r', 'filled', 'DisplayName', 'Impact 109');
for i = 1:4
    text(d103(1), d103(2)+0.01, '103', 'FontSize',8, 'VerticalAlignment','bottom', 'HorizontalAlignment','center');
    text(d104(1), d104(2)+0.01, '104', 'FontSize',8, 'VerticalAlignment','bottom', 'HorizontalAlignment','center');
    text(d105(1), d105(2)+0.01, '105', 'FontSize',8, 'VerticalAlignment','bottom', 'HorizontalAlignment','center');
    text(d109(1), d109(2)+0.01, '109', 'FontSize',8, 'VerticalAlignment','bottom', 'HorizontalAlignment','center');
end

axis equal;
title('Sensor Positions');
xlabel('X Position (m)');
ylabel('Y Position (m)');
grid on;
hold off;