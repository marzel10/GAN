% =========================
% Input parameters
% =========================
xi = 0; yi = 0;      % Point i
xj = 5; yj = 0;      % Point j
beta = 1.1;          % Beta parameter

% =========================
% Grid definition
% =========================
xRange = linspace(-2, 7, 500);
yRange = linspace(-5, 5, 500);
[X, Y] = meshgrid(xRange, yRange);

% =========================
% Distance calculations
% =========================
di = sqrt((X - xi).^2 + (Y - yi).^2);
dj = sqrt((X - xj).^2 + (Y - yj).^2);
dij = sqrt((xi - xj)^2 + (yi - yj)^2);

% =========================
% R_ij(x,y)
% =========================
R = (di + dj) ./ dij;

% =========================
% s_ij(x,y)
% =========================
S = zeros(size(R));
mask = beta > R;
S(mask) = -(beta - R(mask)) ./ (1 - beta);

% =========================
% Normalize to probability
% =========================
S(S < 0) = 0;
P = S / sum(S(:));

% =========================
% Plot heat map
% =========================
figure;
imagesc(xRange, yRange, P);
set(gca, 'YDir', 'normal');
colorbar;
xlabel('x');
ylabel('y');
title('Probability Heat Map');
hold on;

% Plot reference points
plot(xi, yi, 'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
plot(xj, yj, 'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 8);
text(xi, yi, '  i', 'Color', 'w');
text(xj, yj, '  j', 'Color', 'w');

axis equal tight;


% =========================
% Fixed parameters
% =========================
xi = 0; yi = 0;
xj = 5; yj = 0;

% Beta range for animation
betaValues = linspace(1.01, 2.5, 80);   % must be > 1

% =========================
% Grid
% =========================
xRange = linspace(-2, 7, 400);
yRange = linspace(-5, 5, 400);
[X, Y] = meshgrid(xRange, yRange);

% =========================
% Precompute distances
% =========================
di = sqrt((X - xi).^2 + (Y - yi).^2);
dj = sqrt((X - xj).^2 + (Y - yj).^2);
dij = sqrt((xi - xj)^2 + (yi - yj)^2);
R = (di + dj) ./ dij;

% =========================
% Figure setup
% =========================
figure;
hImg = imagesc(xRange, yRange, zeros(size(R)));
set(gca, 'YDir', 'normal');
set(gca, 'XDir', 'reverse');
axis equal tight;
colorbar;
xlabel('x');
ylabel('y');

hold on;
plot(xi, yi, 'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 7);
plot(xj, yj, 'wo', 'MarkerFaceColor', 'w', 'MarkerSize', 7);

titleHandle = title('');

% =========================
% Animation loop
% =========================
for beta = betaValues

    % s_ij(x,y)
    S = zeros(size(R));
    mask = beta > R;
    S(mask) = -(beta - R(mask)) ./ (1 - beta);

    % Enforce non-negativity
    S(S < 0) = 0;

    % Normalize to probability
    if sum(S(:)) > 0
        P = S / sum(S(:));
    else
        P = S;
    end

    % Update heat map
    set(hImg, 'CData', P);
    titleHandle.String = sprintf('Probability Heat Map (\\beta = %.2f)', beta);

    drawnow;
end
