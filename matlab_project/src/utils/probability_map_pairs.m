function [Psum, xRange, yRange] = probability_map_pairs(pairs, varargin)
% probability_map_pairs  Plot summed probability map for given path indices.
%
% Usage:
%   Psum = probability_map_pairs([2 15 28], 'sHI', [1 0.8 1.2], 'Beta', 1.3);
%
% Inputs:
%   pairs  - vector of path indices as in SP.mat (SP_table rows)
%
% Name-Value options:
%   'sHI'            - scalar or vector of weights per provided pair (default ones)
%   'Wave_velocity'            - scalar > 1 controlling shape (default 1.3)
%   'GridSize'        - [Ny Nx] resolution of grid (default [400 400])
%   'NormalizePerPair'- logical, normalize each pair map before sum (true)
%   'SumNormalize'    - logical, normalize final sum to sum=1 (true)
%   'ShowPanel'       - logical, draw panel boundary and sensors (true)
%
% Output:
%   Psum - Ny-by-Nx probability map (summed and optionally normalized)
%   xRange, yRange - axes vectors used for the grid (optional outputs)

% Parse inputs
p = inputParser;
p.addRequired('pairs', @(v) isnumeric(v) && isvector(v));
p.addParameter('sHI', ones(size(pairs)), @(v) isnumeric(v) && isvector(v));
p.addParameter('Wave_velocity', 2.0*10^3, @(x) isnumeric(x) && isscalar(x) && x>1);
p.addParameter('Beta', 10e6, @(x) isnumeric(x) && isscalar(x) && x>1);
p.addParameter('GridSize', [400 400], @(v) isnumeric(v) && numel(v)==2 && all(v>0));
p.addParameter('NormalizePerPair', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('SumNormalize', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('ShowPanel', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('CLim', [], @(v) (isempty(v)) || (isnumeric(v) && numel(v)==2 && all(isfinite(v)) && v(1) < v(2)));
p.addParameter('Plot', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('Plot_demage', [0 0], @(v) isnumeric(v) && numel(v)==2);
p.parse(pairs, varargin{:});
beta = p.Results.Beta;
Wave_velocity = p.Results.Wave_velocity;
gr = p.Results.GridSize;
normPer = logical(p.Results.NormalizePerPair);
normSum = logical(p.Results.SumNormalize);
showPanel = logical(p.Results.ShowPanel);
sHI_in = p.Results.sHI(:)';
clim = p.Results.CLim;
doPlot = logical(p.Results.Plot);
plot_demage = p.Results.Plot_demage;



% Load segments table
S = load('data\SP.mat');
if ~isfield(S, 'SP_table')
    error('SP.mat must contain SP_table with fields Sensor1_idx and Sensor2_idx.');
end
SP_table = S.SP_table;

% Sensor positions (meters)
positions = 0.001*[25 25; 
        100 25; 
        140 215; 
        65 215; 
        25 120; 
        140 120; 
        82.5 80; 
        82.5 160];

% Panel domain (meters)
panel_domain_x = 0.001*[0 0 165 165 0];
panel_domain_y = 0.001*[0 240 240 0 0];

% Grid over panel extents with small margins
xmin = min(panel_domain_x); xmax = max(panel_domain_x);
ymin = min(panel_domain_y); ymax = max(panel_domain_y);
mx = 0.005; my = 0.005; % 5 mm margins
xRange = linspace(xmin-mx, xmax+mx, gr(2));
yRange = linspace(ymin-my, ymax+my, gr(1));
[X, Y] = meshgrid(xRange, yRange);

t = 9.995*10^-4;
a = (t*Wave_velocity)/2;

% Accumulator
Psum = zeros(size(X));

% Validate pairs
pairsInput = pairs(:)';
validMask = pairsInput>=1 & pairsInput<=height(SP_table);
if ~all(validMask)
    warning('Some pair indices are out of range and will be skipped: %s', mat2str(pairsInput(~validMask)));
end
pairs = pairsInput(validMask);


% Align sHI with filtered pairs
if isscalar(sHI_in)
    sHI = repmat(sHI_in, 1, numel(pairs));
else
    if numel(sHI_in) ~= numel(pairsInput)
        error('Length of sHI (%d) must be 1 or equal to the number of provided pairs (%d).', numel(sHI_in), numel(pairsInput));
    end
    sHI = sHI_in(validMask);
end

for k = 1:numel(pairs)
    % k: loop index over the provided (validated) pairs
    % idx: actual path index into SP_table for the k-th requested pair
    idx = pairs(k);
    s1 = SP_table.Sensor1_idx(idx);
    s2 = SP_table.Sensor2_idx(idx);
    if any(~ismember([s1 s2], 1:size(positions,1)))
        warning('Pair %d refers to invalid sensor indices [%d %d]; skipping.', idx, s1, s2);
        continue;
    end

    xi = positions(s1,1); yi = positions(s1,2);
    xj = positions(s2,1); yj = positions(s2,2);

    % Distances
    di = sqrt((X - xi).^2 + (Y - yi).^2);
    dj = sqrt((X - xj).^2 + (Y - yj).^2);
    dij = sqrt((xi - xj)^2 + (yi - yj)^2);
    if dij == 0
        warning('Sensors coincide for pair %d; skipping.', idx);
        continue;
    end

    % R and base s_ij (without weighting); ensure nonnegative support inside lens
    R = (di + dj) ./ dij;
    
    if p.Results.Beta == 10e6
        
        beta = 2*a/dij;
        
        fprintf('beta=%.8f\n', beta);
        if beta <= 1
            %warning('Computed beta <= 1 (%.4f) for pair %d with sensor distance %.4f m; increase Wave_velocity. For now seting beta to 1.001 ', beta, idx, dij);
            beta = 1.001;
        else

            fprintf('Computed beta for pair %d with sensor distance %.8f m: beta=%.8f\n', idx, dij, beta);
        end

    end

    Base = zeros(size(R));
    mask = beta > R;
    Base(mask) = -((beta - R(mask)) ./ (1 - beta)); % positive inside lens for beta>1
    Base(Base < 0) = 0;
    

    % Per-pair normalization then apply weight sHI(k); if not normalizing, apply weight directly
    if normPer
        square_area = (xRange(2)-xRange(1)) * (yRange(2)-yRange(1));
        ssum = sum(Base(:));
        ssum = ssum * square_area; % approximate integral
        if ssum > 0
            Pij = (Base / ssum);
        else
            Pij = Base;
        end
        Pij = (1.1494-sHI(k)) * Pij; % weight after normalization so it has effect
    else
        Pij = (1.1494-sHI(k)) * Base; % no per-pair normalization, direct weighting
    end

    % Accumulate
    Psum = Psum + Pij;
end

% Final normalization
if normSum
    total = sum(Psum(:));
    square_area = (xRange(2)-xRange(1)) * (yRange(2)-yRange(1));
    total = total * square_area; % approximate integral
    if total > 0
        Psum = Psum / total;
    end
end

% Plot (optional)
if doPlot
    figure; 
    imagesc(xRange, yRange, Psum);
    set(gca, 'XDir', 'reverse');
    set(gca, 'YDir', 'normal');
    if ~isempty(clim)
        caxis(clim);
    end
    axis equal tight;
    colormap(parula);
    colorbar;
    xlabel('x (m)'); ylabel('y (m)');
    title(sprintf('Summed Probability Map (\\beta=%.2f), pairs=%s', beta, mat2str(pairs)));
    hold on;

    % Plot demage points if specified
    if any(plot_demage ~= 0)
        scatter(plot_demage(1), plot_demage(2), 100, 'r', 'filled', 'DisplayName', 'Damage Point');
        text(plot_demage(1), plot_demage(2)+0.01, 'Impact', 'Color', 'r', 'FontSize',8, 'VerticalAlignment','bottom', 'HorizontalAlignment','center');
    end

    if showPanel
        plot(panel_domain_x, panel_domain_y, 'k-', 'LineWidth', 1.0);
        scatter(positions(:,1), positions(:,2), 25, 'w', 'filled');
        for i = 1:size(positions,1)
            text(positions(i,1), positions(i,2), num2str(i), 'Color', 'w', 'FontSize',8, 'VerticalAlignment','bottom', 'HorizontalAlignment','right');
        end
        % Draw selected path segments
        for k = 1:numel(pairs)
            idx = pairs(k);
            s1 = SP_table.Sensor1_idx(idx);
            s2 = SP_table.Sensor2_idx(idx);
            if any(~ismember([s1 s2], 1:size(positions,1)))
                continue;
            end
            plot([positions(s1,1) positions(s2,1)], [positions(s1,2) positions(s2,2)], 'r-', 'LineWidth', 1.2);
        end
    end
    hold off;
end

end
