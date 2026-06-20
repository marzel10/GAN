positions = 0.001*[25 25; 
        100 25; 
        140 215; 
        65 215; 
        25 120; 
        140 120; 
        82.5 80; 
        82.5 160];

% Calculate all distances between sensors
numSensors = size(positions, 1);
distances = zeros(numSensors, numSensors);
for i = 1:numSensors
    for j = 1:numSensors
        distances(i, j) = sqrt((positions(i,1) - positions(j,1))^2 + (positions(i,2) - positions(j,2))^2);
    end
end

max_distance = max(distances(:));
min_distance = min(distances(distances>0)); % exclude zero distance

desired_beta = 1.001;
desired_beta = desired_beta - 1
a = sqrt(desired_beta^2+max_distance^2/4);
t = 9.995*10^-4;
v = 2*a/t;
v = 6e3
a = (t*v)/2;
b_max = sqrt(a^2 - min_distance^2/4)+1;
disp('Maximal beta for min distance:');
disp(b_max);
disp('Corresponding velocity:');
disp(v);
disp('Corresponding a: (panel size = (0.165 x 0.24))');
disp(a);

Wave_velocity = 2.5*10^2; % m/s
a = (t*Wave_velocity)/2;
% Calculate beta parameter for all sensor pairs
beta_to_plot = zeros(numSensors, numSensors);
for i = 1:numSensors
    for j = 1:numSensors
        if i ~= j
            dij = distances(i, j);
            beta_ij = 2*a/dij;
            fprintf('Beta for sensor pair (%d, %d) with distance %.4f m: %.8f\n', i, j, dij, beta_ij);
            beta_to_plot(i, j) = beta_ij;
        else
            beta_to_plot(i, j) = NaN; % or some other value to indicate no self-pair
        end
    end
end

disp('Beta values matrix:');
disp(beta_to_plot);
% heat map of beta values
figure;
imagesc(beta_to_plot);
colorbar;
title('Beta Values for Sensor Pairs');
xlabel('Sensor Index');
ylabel('Sensor Index');
axis equal tight;
% Set diagonal to NaN for better visualization