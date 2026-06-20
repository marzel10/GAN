classdef amplitudeScaleLayer < nnet.layer.Layer
    properties (Learnable)
        Alpha   % scalar scale factor
    end
    methods
        function layer = amplitudeScaleLayer(name, initValue)
            layer.Name = name;
            if nargin < 2, initValue = 0.025; end
            layer.Alpha = dlarray(single(initValue));
        end
        function Z = predict(layer,X)
            Z = X .* layer.Alpha;
        end
        function Z = forward(layer,X)
            Z = X .* layer.Alpha;
        end
    end
end