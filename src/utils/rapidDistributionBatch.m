function [X, Y, aggMap, perPair, meta] = rapidDistributionBatch(F1, F2, beta, varargin)
% RAPIDDISTRIBUTIONBATCH Vectorized RAPID-style distribution for many focus pairs.
%
% [X, Y, aggMap, perPair, meta] = rapidDistributionBatch(F1, F2, beta, ...)
%
% Inputs
%   F1, F2 : Nx2 arrays of focus coordinates (N pairs). Also accepts 1x2.
%   beta   : scalar or Nx1 vector. Defines ellipse boundary via Rg <= beta.
%
% Name-Value options
%   'GridStep' : grid resolution (default 1)
%   'Bounds'   : [xmin xmax ymin ymax]; if empty, auto from union of ellipses (default [])
%   'Pad'      : fractional padding for auto-bounds (default 0.1)
%   'Aggregate': 'max'|'sum'|'mean' aggregation across pairs (default 'max')
%   'Show'     : logical, imagesc plot (default true)
%   'Axes'     : axes handle (default [])
%   'Colormap' : colormap name (default 'jet')
%
% Outputs
%   X, Y     : meshgrid coordinates
%   aggMap   : aggregated map over pairs (H x W)
%   perPair  : per-pair maps (H x W x N)
%   meta     : struct with geometry and params

try
    tStart = tic;
    % ---- Validate inputs
    validateattributes(F1, {'numeric'}, {'ncols',2}, mfilename, 'F1');
    validateattributes(F2, {'numeric'}, {'ncols',2}, mfilename, 'F2');
    if isvector(F1) && numel(F1)==2 && isvector(F2) && numel(F2)==2
        F1 = F1(:).'; F2 = F2(:).';
    end
    if size(F1,1) ~= size(F2,1)
        error('rapidDistributionBatch:SizeMismatch', 'F1 and F2 must have same number of rows.');
    end
    N = size(F1,1);

    ip = inputParser; ip.FunctionName = mfilename;
    addParameter(ip, 'GridStep', 1, @(v)isnumeric(v)&&isscalar(v)&&v>0);
    addParameter(ip, 'Bounds', [], @(v)isnumeric(v) && (isempty(v) || numel(v)==4));
    addParameter(ip, 'Pad', 0.1, @(v)isnumeric(v)&&isscalar(v)&&v>=0);
    addParameter(ip, 'Aggregate', 'max', @(s)ischar(s)|| (isstring(s)&&isscalar(s)) );
    addParameter(ip, 'Show', true, @(v)islogical(v)&&isscalar(v));
    addParameter(ip, 'Axes', [], @(a)isempty(a) || isgraphics(a,'axes'));
    addParameter(ip, 'Colormap', 'jet');
    parse(ip, varargin{:});

    step    = ip.Results.GridStep;
    bounds  = ip.Results.Bounds;
    padFrac = ip.Results.Pad;
    agg     = char(ip.Results.Aggregate);
    % doShow  = ip.Results.Show; % plotting controlled via ip.Results.Show directly
    ax      = ip.Results.Axes;
    cmap    = ip.Results.Colormap;

    % Broadcast beta to N x 1
    if isscalar(beta), beta = repmat(beta, N, 1); else, beta = beta(:); end
    if numel(beta) ~= N
        error('rapidDistributionBatch:BetaSize', 'beta must be scalar or N-element vector.');
    end

    % Geometry per pair
    L = hypot(F2(:,1)-F1(:,1), F2(:,2)-F1(:,2)); % 2c
    if any(L<=0)
        error('rapidDistributionBatch:CoincidentFoci', 'All foci pairs must be distinct.');
    end

    % Auto bounds from union of ellipses Rg<=beta: 2a = beta*L
    if isempty(bounds)
        xmin= inf; ymin= inf; xmax= -inf; ymax= -inf;
        for n=1:N
            a = beta(n)*L(n)/2; c = L(n)/2; b = max(sqrt(max(a^2 - c^2,0)), 0);
            mx = (F1(n,1) + F2(n,1))/2; my = (F1(n,2) + F2(n,2))/2;
            th = atan2(F2(n,2)-F1(n,2), F2(n,1)-F1(n,1));
            R = [cos(th) -sin(th); sin(th) cos(th)];
            ext = [ a 0; -a 0; 0 b; 0 -b];
            pts = (ext * R.');
            pts(:,1) = pts(:,1)+mx; pts(:,2) = pts(:,2)+my;
            xmin = min(xmin, min(pts(:,1))); xmax = max(xmax, max(pts(:,1))); 
            ymin = min(ymin, min(pts(:,2))); ymax = max(ymax, max(pts(:,2)));
        end
        dx = xmax-xmin; dy = ymax-ymin;
        bounds = [xmin - padFrac*dx, xmax + padFrac*dx, ymin - padFrac*dy, ymax + padFrac*dy];
    end

    % Grid
    x = bounds(1):step:bounds(2); y = bounds(3):step:bounds(4);
    [X, Y] = meshgrid(x,y);

    % Vectorized Rg: (d1+d2)/L
    fx1 = reshape(F1(:,1),1,1,[]); fy1 = reshape(F1(:,2),1,1,[]);
    fx2 = reshape(F2(:,1),1,1,[]); fy2 = reshape(F2(:,2),1,1,[]);
    d1 = sqrt((X - fx1).^2 + (Y - fy1).^2);
    d2 = sqrt((X - fx2).^2 + (Y - fy2).^2);
    L3 = reshape(L,1,1,[]);
    Rg = (d1 + d2) ./ L3;  % H x W x N

    % RAPID: (beta - Rg)/(1-beta), zero outside (Rg>beta)
    B3 = reshape(beta,1,1,[]);
    perPair = -(B3 - Rg) ./ (1 - B3);
    perPair(Rg > B3) = 0;

    % Aggregate
    switch lower(agg)
        case 'max',  aggMap = max(perPair,[],3);
        case 'sum',  aggMap = sum(perPair,3);
        case 'mean'
            nz = sum(perPair>0,3);
            aggMap = sum(perPair,3) ./ max(nz,1);
        otherwise, error('rapidDistributionBatch:Aggregate','Unknown: %s', agg);
    end

    % Meta
    meta = struct('F1',F1,'F2',F2,'beta',beta,'L',L,'bounds',bounds,'step',step, ...
                  'aggregate',agg,'elapsedSec',toc(tStart));

    % Plot
    if ip.Results.Show
        if isempty(ax) || ~isvalid(ax), figure; ax = axes; end
        imagesc(ax, x, y, aggMap); axis(ax,'image'); set(ax,'YDir','normal');
        colormap(ax, cmap); colorbar(ax);
        title(ax, sprintf('RAPID agg (%s) over %d pairs', agg, N)); xlabel(ax,'X'); ylabel(ax,'Y');
        hold(ax,'on');
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
        hold(ax,'off');
    end

catch ME
    fprintf(2,'[rapidDistributionBatch][ERROR] %s: %s\n', ME.identifier, ME.message);
    rethrow(ME);
end

end
