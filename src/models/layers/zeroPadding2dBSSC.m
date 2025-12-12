classdef zeroPadding2dBSSC < nnet.layer.Layer
    % Zero-pad along spatial axes for BSSC dlarray
    % Padding = [top bottom left right]
    properties
        Padding (1,4) double {mustBeNonnegative} = [0 0 0 0]
        PaddingValue (1,1) double = 0
    end
    methods
        function layer = zeroPadding2dBSSC(pad, name, padVal)
            if nargin >= 2, layer.Name = name; end
            if nargin >= 1, layer.Padding = pad; end
            if nargin >= 3, layer.PaddingValue = padVal; end
        end

        function Z = predict(layer, X)
            Z = iPadBSSC(X, layer.Padding, layer.PaddingValue);
        end

        function Z = forward(layer, X)
            Z = iPadBSSC(X, layer.Padding, layer.PaddingValue);
        end
    end
end

function Z = iPadBSSC(X, p, v)
% X is dlarray with format 'BSSC' (Batch, TimeRows, SensorCols, Channels)
% p = [top bottom left right]; v = scalar fill value
% During initialize, some dims may be unknown (NaN). If so, no-op.
sz = size(X);
if any(isnan(sz))
    Z = X;  % defer padding until real sizes are known
    return
end

B = sz(1); S1 = sz(2); S2 = sz(3); C = sz(4);

% Create zeros on same device/type as X
u = extractdata(X);  % preserves CPU/GPU and class
mk = @(dims) dlarray(zeros(dims,'like',u), 'BSSC');

Z = X;
% top
if p(1) > 0
    top = mk([B p(1) S2 C]);
    if v ~= 0, top = top + v; end
    Z = cat(2, top, Z);
end
% bottom
if p(2) > 0
    bottom = mk([B p(2) S2 C]);
    if v ~= 0, bottom = bottom + v; end
    Z = cat(2, Z, bottom);
end
% left/right use current S1 after top/bottom
S1p = size(Z,2);
if p(3) > 0
    left = mk([B S1p p(3) C]);
    if v ~= 0, left = left + v; end
    Z = cat(3, left, Z);
end
if p(4) > 0
    right = mk([B S1p p(4) C]);
    if v ~= 0, right = right + v; end
    Z = cat(3, Z, right);
end
end