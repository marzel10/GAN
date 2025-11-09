function [X, Y, aggMap, perPair, meta] = gaussianDistributionBatch(F1, F2, DI, sigma, varargin)
% GAUSSIANDISTRIBUTIONBATCH Vectorized Gaussian along perpendicular distance to i-j line, for many pairs.
%
% [X, Y, aggMap, perPair, meta] = gaussianDistributionBatch(F1, F2, DI, sigma, maxRg, ...)
%
% Inputs
%   F1, F2 : Nx2 or 1x2 focus points (pairs)
%   DI     : scalar or Nx1 mean distance (in same units as grid)
%   sigma  : scalar or Nx1 standard deviation
%   maxRg  : scalar or Nx1 Rg threshold to zero-out values outside ellipse (Rg > maxRg)
%
% Name-Value options
%   'GridStep' : default 1
%   'Bounds'   : [xmin xmax ymin ymax] (default [])
%   'Pad'      : 0.1
%   'Aggregate': 'max'|'sum'|'mean' (default 'max')
%   'Show'     : true
%   'Axes'     : []
%   'Colormap' : 'jet'
%   'Enable_Rg_mask' : true (default) - whether to zero-out values outside 
% Outputs
%   X, Y     : meshgrid
%   aggMap   : aggregated map
%   perPair  : H x W x N values
%   meta     : struct

try
    tStart = tic;
    validateattributes(F1, {'numeric'}, {'ncols',2});
    validateattributes(F2, {'numeric'}, {'ncols',2});
    if isvector(F1) && numel(F1)==2 && isvector(F2) && numel(F2)==2
        F1 = F1(:).'; F2 = F2(:).';
    end
    if size(F1,1) ~= size(F2,1)
        error('gaussianDistributionBatch:SizeMismatch','F1 and F2 must have same number of rows.');
    end
    N = size(F1,1);

    ip = inputParser; ip.FunctionName = mfilename;
    addParameter(ip,'GridStep',1,@(v)isnumeric(v)&&isscalar(v)&&v>0);
    addParameter(ip,'Bounds',[],@(v)isnumeric(v)&&(isempty(v)||numel(v)==4));
    addParameter(ip,'Pad',0.1,@(v)isnumeric(v)&&isscalar(v)&&v>=0);
    addParameter(ip,'Aggregate','max',@(s)ischar(s)||(isstring(s)&&isscalar(s)));
    addParameter(ip,'Show',true,@(v)islogical(v)&&isscalar(v));
    addParameter(ip,'Axes',[],@(a)isempty(a)||isgraphics(a,'axes'));
    addParameter(ip,'Colormap','jet');
    addParameter(ip,'Enable_Rg_mask',true,@(v)islogical(v)&&isscalar(v));
    parse(ip,varargin{:});

    step    = ip.Results.GridStep;
    bounds  = ip.Results.Bounds;
    padFrac = ip.Results.Pad;
    agg     = char(ip.Results.Aggregate);
    ax      = ip.Results.Axes; cmap = ip.Results.Colormap;
    enableRgMask = ip.Results.Enable_Rg_mask;

    % Broadcast parameters
    toN = @(v) (isscalar(v) .* ones(N,1) .* v) + (~isscalar(v) .* v(:));
    DI    = toN(DI);
    sigma = toN(sigma);
    

    % Geometry
    L = hypot(F2(:,1)-F1(:,1), F2(:,2)-F1(:,2));
    v = 1000;
    s = 9.995*10^-4 * v*10^3; % in mm
    maxRg = L/s;
    if any(L<=0), error('gaussianDistributionBatch:CoincidentFoci','Distinct foci required.'); end

    % Bounds: we use union of boxes around segments extended by ~4*sigma in normal direction
    if isempty(bounds)
        xmin= inf; ymin= inf; xmax= -inf; ymax= -inf;
        for n=1:N
            mx = (F1(n,1)+F2(n,1))/2; my=(F1(n,2)+F2(n,2))/2;
            th = atan2(F2(n,2)-F1(n,2), F2(n,1)-F1(n,1));
            % Use an approximate box: along major ~L/2, along minor ~4*sigma(n)
            a = L(n)/2; b = 4*sigma(n);
            R = [cos(th) -sin(th); sin(th) cos(th)];
            ext = [ a 0; -a 0; 0 b; 0 -b];
            pts = (ext * R.'); pts(:,1)=pts(:,1)+mx; pts(:,2)=pts(:,2)+my;
            xmin = min(xmin, min(pts(:,1))); xmax = max(xmax, max(pts(:,1)));
            ymin = min(ymin, min(pts(:,2))); ymax = max(ymax, max(pts(:,2)));
        end
        dx=xmax-xmin; dy=ymax-ymin;
        bounds = [xmin - padFrac*dx, xmax + padFrac*dx, ymin - padFrac*dy, ymax + padFrac*dy];
    end

    % Grid
    x = bounds(1):step:bounds(2); y=bounds(3):step:bounds(4);
    [X, Y] = meshgrid(x,y);

    % Perpendicular distance to i-j line (vectorized)
    fx1 = reshape(F1(:,1),1,1,[]); fy1 = reshape(F1(:,2),1,1,[]);
    fx2 = reshape(F2(:,1),1,1,[]); fy2 = reshape(F2(:,2),1,1,[]);
    vjx = fx2 - fx1; vjy = fy2 - fy1; % 1x1xN
    vnorm = sqrt(vjx.^2 + vjy.^2);    % 1x1xN
    vix = X - fx1; viy = Y - fy1;     % HxWxN
    cross2d = vix .* vjy - viy .* vjx; % HxWxN
    D = abs(cross2d) ./ vnorm;        % perpendicular distance

    % Rg for masking
    Rg = (sqrt((X - fx1).^2 + (Y - fy1).^2) + sqrt((X - fx2).^2 + (Y - fy2).^2)) ./ vnorm; % HxWxN but vnorm=|F1F2|, not L; scale ok
    % Note: vnorm = |F1F2| = L; same dimension

    % Gaussian per pair
    DI3 = reshape(DI,1,1,[]); SG3 = reshape(sigma,1,1,[]); MR3 = reshape(maxRg,1,1,[]);
    perPair = (1./(SG3.*sqrt(2*pi))) .* exp(-0.5*((D - DI3)./SG3).^2);
    disp(['Gaussian perPair range: [', num2str(min(perPair(:))), ', ', num2str(max(perPair(:))), ']']);
    if enableRgMask
        perPair(Rg > MR3) = 0;
    end

    % Aggregate
    switch lower(agg)
        case 'max',  aggMap = max(perPair,[],3);
        case 'sum',  aggMap = sum(perPair,3);
        case 'mean'
            nz = sum(perPair>0,3);
            aggMap = sum(perPair,3) ./ max(nz,1);
        otherwise, error('gaussianDistributionBatch:Aggregate','Unknown: %s', agg);
    end

    meta = struct('F1',F1,'F2',F2,'DI',DI,'sigma',sigma,'maxRg',maxRg,'bounds',bounds,'step',step, ...
                  'aggregate',agg,'elapsedSec',toc(tStart));

    if ip.Results.Show
        if isempty(ax) || ~isvalid(ax), figure; ax = axes; end
        imagesc(ax, x, y, aggMap); axis(ax,'image'); set(ax,'YDir','normal');
        colormap(ax, cmap); colorbar(ax);
        title(ax, sprintf('Gaussian agg (%s) over %d pairs', agg, N)); xlabel(ax,'X'); ylabel(ax,'Y');
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
    fprintf(2,'[gaussianDistributionBatch][ERROR] %s: %s\n', ME.identifier, ME.message);
    rethrow(ME);
end

end
