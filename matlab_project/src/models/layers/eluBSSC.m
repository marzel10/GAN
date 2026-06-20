classdef eluBSSC < nnet.layer.Layer
    properties (Learnable)
        Alpha;
    end
    methods
        function layer = eluBSSC(name, alpha)
            layer.Name = name;
            if nargin > 1, layer.Alpha = single(alpha); end
        end
        function Z = predict(layer, X)
            a = layer.Alpha;          % scalar, no labels
            Xneg = min(X, 0);
            Z = max(X, 0) + (exp(Xneg) - 1) .* a;
        end
        function Z = forward(layer, X)
            a = layer.Alpha;
            Xneg = min(X, 0);
            Z = max(X, 0) + (exp(Xneg) - 1) .* a;
        end
    end
end