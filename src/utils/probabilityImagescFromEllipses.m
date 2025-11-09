function [X, Y, probMap, meta] = probabilityImagescFromEllipses(f1, f2, aVec, p, varargin)
% PROBABILITYIMAGESCFROMELLIPSES Build and visualize a probability map from concentric ellipses.
%
%   [X, Y, probMap, meta] = probabilityImagescFromEllipses(f1, f2, aVec, pVec, ...)
%
% Inputs
%   f1, f2 : either 1x2 (single pair) or Nx2 arrays of focus coordinates
%   aVec   : vector of semi-major axes a_k (must satisfy a_k >= |F1F2|/2)
%   p      : probabilities. Single mode: vector length(aVec). Batch mode: NxK matrix,
%            where N=size(f1,1) and K=length(aVec).
%
% Name-Value pairs (all optional):
%   'GridStep'   : grid resolution in x/y units (default 1)
%   'Bounds'     : [xmin xmax ymin ymax]; if empty, auto from outer ellipse (default [])
%   'Pad'        : fractional padding to enlarge auto-bounds (default 0.1)
%   'Show'       : logical, whether to plot imagesc (default true)
%   'Axes'       : axes handle to plot into; if empty and Show==true, creates a figure
%   'Colormap'   : colormap to apply when plotting (default 'parula')
%   'ShowContour': logical, overlay ellipse contours (default true)
%   'ContourLW'  : contour line width (default 1.2)
%   'FocusMarks' : logical, draw foci (and line for single pair) (default true)
%   'Aggregate'  : aggregation across pairs in batch mode: 'max'|'sum'|'mean' (default 'max')
%   'ContourPairs': indices of pairs to draw contours for (batch mode). [] means none (default [])
%
% Outputs
%   X, Y     : meshgrid arrays used for evaluation
%   probMap  : probability image. For single pair, each ring between ellipses k and k-1
%              is assigned probability p(k). Outside all ellipses: 0.
%              For batch mode, maps are computed per pair and aggregated by 'Aggregate'.
%   meta     : struct with geometry and bookkeeping
%
% Semantics
%   With aVec sorted ascending (a0 < a1 < ...), each point is assigned the
%   probability p_k corresponding to the OUTERMOST ellipse that still
%   contains the point. This ensures the region between ellipse_1 (p1) and
%   ellipse_0 (p0) has probability p1, as requested.
%
% Example
%   % Single pair
%   f1 = [25,25]; f2=[120,140]; a=[80 100 120 150]; p=[0.1 0.4 0.8 0.1];
%   probabilityImagescFromEllipses(f1, f2, a, p, 'GridStep', 1);
%
%   % Batch (N pairs). p is N x K
%   f1s = repmat([25 25],28,1); f2s = repmat([120 140],28,1);
%   a = [80 100 120 150]; p = rand(28, numel(a));
%   probabilityImagescFromEllipses(f1s, f2s, a, p, 'Aggregate','max');

try
    tStart = tic;
    % ---- Input validation
    validateattributes(f1, {'numeric'}, {'real','finite'}, mfilename, 'f1');
    validateattributes(f2, {'numeric'}, {'real','finite'}, mfilename, 'f2');
    validateattributes(aVec, {'numeric'}, {'vector','real','finite','positive'}, mfilename, 'aVec');
    validateattributes(p,   {'numeric'}, {}, mfilename, 'p');
    % Determine mode
    if isvector(f1) && numel(f1)==2 && isvector(f2) && numel(f2)==2
        modeBatch = false; Npairs = 1;
        f1 = f1(:).'; f2 = f2(:).';
        if numel(p) ~= numel(aVec)
            error('probabilityImagescFromEllipses:SizeMismatch', 'For single pair: length(p) must equal length(aVec).');
        end
        pVec = p(:).'; pMat = pVec; % 1 x K for uniform downstream logic
        F1 = f1; F2 = f2;
    else
        % Batch mode: f1,f2 are Nx2
        if size(f1,2)~=2 || size(f2,2)~=2 || size(f1,1)~=size(f2,1)
            error('probabilityImagescFromEllipses:InvalidFoci', 'In batch mode, f1 and f2 must be Nx2 with same N.');
        end
        Npairs = size(f1,1); modeBatch = true;
        if ~ismatrix(p) || size(p,1)~=Npairs || size(p,2)~=numel(aVec)
            error('probabilityImagescFromEllipses:SizeMismatch', 'In batch mode, p must be N x K where N=size(f1,1) and K=length(aVec).');
        end
        pMat = p; pVec = []; % not used in batch
        F1 = f1; F2 = f2;
    end

    ip = inputParser; ip.FunctionName = mfilename;
    addParameter(ip, 'GridStep', 1, @(v)isnumeric(v)&&isscalar(v)&&v>0);
    addParameter(ip, 'Bounds', [], @(v)isnumeric(v) && (isempty(v) || numel(v)==4));
    addParameter(ip, 'Pad', 0.1, @(v)isnumeric(v)&&isscalar(v)&&v>=0);
    addParameter(ip, 'Show', true, @(v)islogical(v)&&isscalar(v));
    addParameter(ip, 'Axes', [], @(v)isempty(v) || isgraphics(v,'axes'));
    addParameter(ip, 'Colormap', 'parula');
    addParameter(ip, 'ShowContour', true, @(v)islogical(v)&&isscalar(v));
    addParameter(ip, 'ContourLW', 1.2, @(v)isnumeric(v)&&isscalar(v)&&v>0);
    addParameter(ip, 'FocusMarks', true, @(v)islogical(v)&&isscalar(v));
    addParameter(ip, 'Aggregate', 'max', @(s)ischar(s) || (isstring(s)&&isscalar(s)));
    addParameter(ip, 'ContourPairs', [], @(v)isnumeric(v) && isvector(v));
    parse(ip, varargin{:});

    step      = ip.Results.GridStep;
    boundsIn  = ip.Results.Bounds;
    padFrac   = ip.Results.Pad;
    doShow    = ip.Results.Show;
    ax        = ip.Results.Axes;
    cmap      = ip.Results.Colormap;
    doContour = ip.Results.ShowContour;
    lw        = ip.Results.ContourLW;
    doMarks   = ip.Results.FocusMarks;
    aggregate = char(ip.Results.Aggregate);
    contourPairs = ip.Results.ContourPairs(:).';

    % Geometry basics per pair
    % Distances L (Nx1), c (Nx1)
    L = hypot(F2(:,1) - F1(:,1), F2(:,2) - F1(:,2));
    if any(L <= 0)
        error('probabilityImagescFromEllipses:CoincidentFoci', 'All foci pairs must be distinct.');
    end
    c = L/2;

    % Sort ellipses by ascending a, keep probabilities aligned
    aVec = aVec(:).';
    K = numel(aVec);
    % Validate a against each pair's c: require min(aVec) >= c(n) for all n
    viol = find(c > min(aVec));
    if ~isempty(viol)
        error('probabilityImagescFromEllipses:InvalidA', ['Some pairs have |F1F2|/2 greater than min(a).\n' ...
               'Min(a)=%.3f, max required c across pairs=%.3f. First violating pairs: %s'], ...
               min(aVec), max(c), mat2str(viol(1:min(numel(viol),5)).'));
    end

    % Determine bounds from the outermost ellipse if not supplied
    if isempty(boundsIn)
    aOut = aVec(end);
        % Compute union of oriented-ellipse boxes across pairs
        xmin= inf; ymin= inf; xmax= -inf; ymax= -inf;
        for n = 1:Npairs
            bOut = sqrt(max(aOut^2 - c(n)^2, 0));
            mx = (F1(n,1) + F2(n,1))/2; my = (F1(n,2) + F2(n,2))/2;
            theta = atan2(F2(n,2) - F1(n,2), F2(n,1) - F1(n,1));
            R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
            extents = [ aOut  0; -aOut  0; 0  bOut; 0 -bOut];
            pts = (extents * R.');
            pts(:,1) = pts(:,1) + mx; pts(:,2) = pts(:,2) + my;
            xmin = min(xmin, min(pts(:,1))); xmax = max(xmax, max(pts(:,1)));
            ymin = min(ymin, min(pts(:,2))); ymax = max(ymax, max(pts(:,2)));
        end
        dx = xmax - xmin; dy = ymax - ymin;
        bounds = [xmin - padFrac*dx, xmax + padFrac*dx, ...
                  ymin - padFrac*dy, ymax + padFrac*dy];
    else
        bounds = boundsIn;
    end

    % Grid
    x = bounds(1):step:bounds(2);
    y = bounds(3):step:bounds(4);
    [X, Y] = meshgrid(x, y);

    % Sum-of-distances per pair (level sets S = 2a are ellipses)
    H = size(X,1); W = size(X,2);
    % Expand foci to broadcast: 1x1xN
    fx1 = reshape(F1(:,1), 1,1,[]); fy1 = reshape(F1(:,2), 1,1,[]);
    fx2 = reshape(F2(:,1), 1,1,[]); fy2 = reshape(F2(:,2), 1,1,[]);
    d1 = sqrt( (X - fx1).^2 + (Y - fy1).^2 );  % H x W x N
    d2 = sqrt( (X - fx2).^2 + (Y - fy2).^2 );  % H x W x N
    S  = d1 + d2;                               % H x W x N

    % For each pair, count how many ellipses contain the pixel: idx in [0..K]
    idxHWN = zeros(H, W, Npairs, 'uint16');
    for k = 1:K
        idxHWN = idxHWN + uint16(S <= 2*aVec(k));
    end

    % Map idx -> probability per pair using per-row lookup
    % Build lookup L: N x (K+1), with leading zeros for idx==0
    if modeBatch
        L = [pMat,zeros(Npairs,1,'like',pMat)];
    else
        L = [ pVec, 0]; % 1 x (K+1)
    end
    disp("L: "), disp(L);
    % Prepare linear indices for L(n, idx+1)
    idxHWN = K + 1 - idxHWN;                        % invert to [K+1..1]
    % Reorder dims to N x H x W for convenient sub2ind
    idxNHW = permute(idxHWN, [3,1,2]);          % N x H x W
    nGrid = repmat((1:Npairs).', 1, H, W);      % N x H x W
    lin = sub2ind([Npairs, K+1], nGrid, idxNHW);
    PNHW = reshape(L(lin), [Npairs, H, W]);     % N x H x W
    probPerPair = permute(PNHW, [2,3,1]);       % H x W x N

    % Aggregate across pairs
    switch lower(aggregate)
        case 'max'
            probMap = max(probPerPair, [], 3);
        case 'sum'
            probMap = sum(probPerPair, 3);
        case 'mean'
            probMap = mean(probPerPair, 3);
            nonzero = sum(probPerPair>0, 3);
            disp("nonzero: "), disp(size(nonzero));
            disp("probMap before: "), disp(size(probMap));
            probMap = sum(probPerPair, 3) ./ max(nonzero, 1); % avoid div0
            
        otherwise
            error('probabilityImagescFromEllipses:Aggregate', 'Unknown Aggregate mode: %s', aggregate);
    end
    

    % Meta information
    meta = struct();
    meta.foci = struct('F1',F1,'F2',F2);
    meta.c = c; meta.L = L;
    meta.aVec = aVec;
    if modeBatch, meta.pMat = pMat; else, meta.pVec = pVec; end
    meta.probPerPair = probPerPair; % H x W x N (useful if aggregation not desired)
    meta.modeBatch = modeBatch; meta.Npairs = Npairs; meta.K = K;
    meta.bounds = bounds; meta.step = step;
    meta.elapsedSec = toc(tStart);

    % ---- Debug prints
    fprintf('[probabilityImagescFromEllipses] Grid %dx%d, step=%.3g, bounds=[%.2f %.2f %.2f %.2f]\n', ...
        size(X,1), size(X,2), step, bounds(1), bounds(2), bounds(3), bounds(4));
    fprintf('[probabilityImagescFromEllipses] Npairs=%d, K=%d\n', Npairs, K);
    fprintf('[probabilityImagescFromEllipses] aVec: %s\n', mat2str(aVec,3));

    % ---- Plot
    if doShow
        if isempty(ax) || ~isvalid(ax)
            figure; ax = axes; %#ok<LAXES>
        end
        axes(ax); %#ok<LAXES>
    imagesc(x, y, probMap);
        axis image; set(ax, 'YDir', 'normal');
        colormap(ax, cmap);
        % Color limits heuristics
        
        clim(ax, [0, 1]);
        
        colorbar('peer', ax);
        if modeBatch
            ttl = sprintf('Probability map (batch %d pairs, agg=%s)', Npairs, aggregate);
        else
            ttl = 'Probability map from ellipses (outermost assignment)';
        end
        title(ax, ttl);
        xlabel(ax, 'X'); ylabel(ax, 'Y');
        
        hold(ax, 'on');
        pos = [25 25; 
            25 100; 
            215 140; 
            215 66; 
            120 25; 
            120 140; 
            90 82.5; 
            180 82.5];
        scatter(ax, pos(:,2), pos(:,1), 50, 'g', 'filled', 'DisplayName', 'Sensors');
        rectangle(ax, 'Position',[0 0 165 240], 'EdgeColor','k', 'LineStyle','--', 'LineWidth',1.5)
        if doContour
            if modeBatch
                pairsToDraw = contourPairs;
                if isempty(pairsToDraw)
                    pairsToDraw = 1:min(Npairs, 3); % limit clutter
                end
                for n = pairsToDraw
                    % Recompute S for pair n only for contour drawing
                    Sd = sqrt((X - F1(n,1)).^2 + (Y - F1(n,2)).^2) + ...
                         sqrt((X - F2(n,1)).^2 + (Y - F2(n,2)).^2);
                    contour(ax, X, Y, Sd, 2*aVec, '-', 'LineWidth', lw, 'LineColor', [0 0 0]);
                end
            else
                % Ssingle for pair 1
                Sd = sqrt((X - F1(1,1)).^2 + (Y - F1(1,2)).^2) + ...
                     sqrt((X - F2(1,1)).^2 + (Y - F2(1,2)).^2);
                h = contour(ax, X, Y, Sd, 2*aVec, 'k-', 'LineWidth', lw); %#ok<NASGU>
            end
        end
        if doMarks
            if modeBatch
                plot(ax, F1(:,1), F1(:,2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 3);
                plot(ax, F2(:,1), F2(:,2), 'bs', 'MarkerFaceColor', 'b', 'MarkerSize', 3);
            else
                plot(ax, [F1(1) F2(1)], [F1(2) F2(2)], 'k--', 'LineWidth', 1);
                plot(ax, F1(1), F1(2), 'ro', 'MarkerFaceColor', 'r');
                plot(ax, F2(1), F2(2), 'bs', 'MarkerFaceColor', 'b');
            end
        end
        % Optional bounding box (commented; enable if needed)
        % rectangle('Parent',ax,'Position',[0 0 165 240], 'EdgeColor','k', 'LineStyle','--', 'LineWidth',1.2);
        % Simplified legend omitted to reduce clutter in batch mode
        hold(ax, 'off');
    end

catch ME
    fprintf(2, '[probabilityImagescFromEllipses][ERROR] %s: %s\n', ME.identifier, ME.message);
    rethrow(ME);
end

end
