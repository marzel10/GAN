function [series, statesVec] = probe_probability_series(latent_space, paths, statesVec, probePoints, varargin)
% probe_probability_series  Sample summed probability at points over states.
%
% Inputs:
%   latent_space - matrix [numPaths x numStates] of sHI values
%   paths        - vector of path indices (as used by SP_table rows)
%   statesVec    - vector of states to evaluate (e.g., 1:numCycles)
%   probePoints  - Nx2 array of [x y] positions in meters
%
% Name-Value options (forwarded to probability_map_pairs):
%   'Beta'              - default 1.3
%   'GridSize'          - default [400 400]
%   'NormalizePerPair'  - default true
%   'SumNormalize'      - default true
%
% Outputs:
%   series     - [numProbes x numStates] probability values at probes
%   statesVec  - states vector returned for convenience

% Parse options
p = inputParser;
p.addParameter('Beta', 1.3, @(x) isnumeric(x) && isscalar(x) && x>1);
p.addParameter('GridSize', [400 400], @(v) isnumeric(v) && numel(v)==2 && all(v>0));
p.addParameter('NormalizePerPair', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('SumNormalize', true, @(x) islogical(x) || isnumeric(x));
% Test/override options
p.addParameter('Test', false, @(x) islogical(x) || isnumeric(x));
p.addParameter('TestPattern', 'linear-dec', @(s) ischar(s) || isstring(s));
p.addParameter('sHIOverride', [], @(v) isnumeric(v));
p.parse(varargin{:});
beta = p.Results.Beta;
gr = p.Results.GridSize;
normPer = logical(p.Results.NormalizePerPair);
normSum = logical(p.Results.SumNormalize);
doTest = logical(p.Results.Test);
testPattern = string(p.Results.TestPattern);
sHIOverride = p.Results.sHIOverride;

% Validate inputs
if isempty(statesVec)
    statesVec = 1:size(latent_space, 2);
end
if ~isvector(statesVec)
    error('statesVec must be a vector of state indices.');
end
statesVec = statesVec(:)';

if size(probePoints,2) ~= 2
    error('probePoints must be Nx2 array of [x y] in meters.');
end

numProbes = size(probePoints,1);
numStates = numel(statesVec);
series = nan(numProbes, numStates);

% Prepare optional override/test series
perPairOverride = [];    % size: [numel(paths) x numStates]
perStateOverride = [];   % size: [1 x numStates]

if ~isempty(sHIOverride)
    if isscalar(sHIOverride)
        perStateOverride = repmat(sHIOverride, 1, numStates);
    elseif isvector(sHIOverride)
        if numel(sHIOverride) ~= numStates
            error('sHIOverride vector length (%d) must equal numStates (%d).', numel(sHIOverride), numStates);
        end
        perStateOverride = sHIOverride(:)';
    else
        if size(sHIOverride,1) ~= numel(paths) || size(sHIOverride,2) ~= numStates
            error('sHIOverride matrix must be [numPaths x numStates] = [%d x %d].', numel(paths), numStates);
        end
        perPairOverride = sHIOverride;
    end
elseif doTest
    switch lower(testPattern)
        case {'linear-dec','linear'}
            perStateOverride = linspace(1, 0, numStates);
        case 'linear-inc'
            perStateOverride = linspace(0, 1, numStates);
        case 'cosine'
            t = linspace(0, 1, numStates);
            perStateOverride = 0.5*(1+cos(pi*t)); % 1->0 smooth
        case 'step'
            perStateOverride = ones(1, numStates);
            if numStates >= 2
                perStateOverride(round(numStates/2):end) = 0;
            end
        otherwise
            warning('Unknown TestPattern=%s. Using linear-dec.', testPattern);
            perStateOverride = linspace(1, 0, numStates);
    end
end

% Loop over states, compute map, sample at probe points
for si = 1:numStates
    s = statesVec(si);
    % Build sHI vector per pair for this state
    if ~isempty(perPairOverride)
        sHI_path = perPairOverride(:, si);
    elseif ~isempty(perStateOverride)
        sHI_path = repmat(perStateOverride(si), numel(paths), 1);
    else
        if size(latent_space,1) == 1 && numel(paths) > 1
            % FC mode with single path latent: replicate scalar across selected paths
            sHI_path = repmat(latent_space(1, s), numel(paths), 1);
        else
            sHI_path = latent_space(paths, s)';
        end
    end
    
    % Compute summed probability map without plotting
    [P, xRange, yRange] = probability_map_pairs(paths, 'sHI', sHI_path, ...
        'Beta', beta, 'GridSize', gr, 'NormalizePerPair', normPer, 'SumNormalize', normSum, ...
        'ShowPanel', false, 'Plot', false);

    % Sample at each probe via nearest grid cell
    for pi = 1:numProbes
        x = probePoints(pi,1); y = probePoints(pi,2);
        % Clip to grid bounds
        if x < xRange(1) || x > xRange(end) || y < yRange(1) || y > yRange(end)
            series(pi, si) = NaN; %#ok<AGROW>
            continue;
        end
        [~, ix] = min(abs(xRange - x));
        [~, iy] = min(abs(yRange - y));
        series(pi, si) = P(iy, ix);
    end
end

end
