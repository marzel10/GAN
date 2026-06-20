% Simple demos for probabilityImagescFromEllipses
% - Single pair: ring between ellipse k and k-1 gets p(k)
% - Small batch: aggregate across multiple pairs

%% Single-pair demo
if true
    try
        f1 = [25, 25];
        f2 = [120, 140];
        a  = [80 85 90 95 100 105 110];          % must be sorted ascending
        p  = rand(size(a));         % same length as a

        fprintf('Running single-pair demo...\n');
        probabilityImagescFromEllipses(f1, f2, a, p, ...
            'GridStep', 1, ...
            'Bounds', [-20 185 -20 260], ...
            'ShowContour', true, ...
            'Colormap', 'jet');
    catch ME
        fprintf(2, '[demo single][ERROR] %s: %s\n', ME.identifier, ME.message);
    end
end

%% Small batch demo (N=3)
try
    % Three foci pairs around the original, for illustration
    p = [25 25; 
            100 25;
            140 215; 
            66 215; 
            25 120; 
            140 120; 
            82.5 90; 
            82.5 180];
    
    [F1, F2] = meshgrid(1:size(p, 1), 1:size(p, 1));
    orig_size = size(F1, 1);
    F1 = tril(F1, -1);
    F2 = tril(F2, -1);
    F1(F1 == 0) = []; % Remove diagonal elements
    F2(F2 == 0) = []; % Remove diagonal elements
    F1 = reshape(F1, orig_size-1, []);
    F2 = reshape(F2, orig_size-1, []);
    F1 = p(F1(:), :);
    F2 = p(F2(:), :);
    
    %a  = 80:-1:75;          % sorted ascending
    a = 116:-0.5:112;
    rng(0);
    p  = rand(size(F1,1), numel(a)); % N x K probabilities
    disp(p)
    x_i = 140;
    y_i = 120;

    x_j = 25;
    y_j = 25;

    v = 5000;
    s = 9.995*10^-4 * v*10^3; % in mm
    sen_dist = sqrt((x_i - x_j).^2 + (y_i - y_j).^2);
    max_dist = s/sen_dist;

    focus1 = [140 215];
    focus2 = [82.5 180];
    distance = sqrt((focus1(1)-focus2(1))^2 + (focus1(2)-focus2(2))^2);
    probabilityImagescFromEllipses([140 215], [82.5 180], 1.1*distance, 0.6, ...
        'GridStep', 1, ...
        'Bounds', [-20 185 -20 260], ...
        'Aggregate', 'mean', ...
        'ShowContour', true, 'ContourPairs',1:6 , ...
        'Colormap', 'jet', ...
        'ContourLW', 0.7);

    rapidDistributionBatch(F1, F2, max_dist*ones(size(F1,1),1), ...
        'GridStep', 1, ...
        'Bounds', [-20 185 -20 260], ...
        'Aggregate', 'mean', ...
        'Colormap', 'jet');

    gaussianDistributionBatch(F1, F2, 0.5* rand(size(F1,1),1), 0.5*ones(size(F1,1),1), ...
        'GridStep', 1, ...
        'Bounds', [-20 185 -20 260], ...
        'Aggregate', 'mean', ...
        'Colormap', 'jet', ...
        'Enable_Rg_Mask', false);
catch ME
    fprintf(2, '[demo batch][ERROR] %s: %s\n', ME.identifier, ME.message);
end
