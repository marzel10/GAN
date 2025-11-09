classdef removeMeanLayer < nnet.layer.Layer
    % Subtract mean over time for each sample/channel: SSCB dims [S T C B] => mean over T(=1st dim here is S)
    methods
        function layer = removeMeanLayer(name)
            layer.Name = name;
        end
        function Z = predict(layer, X)
            % X is SSCB (your project uses SSCB). Time is dim 1.
            m = mean(X, 1);                 % 1 x 2 x 6 x B
            Z = X - m;                      % broadcast subtract
        end
        function Z = forward(layer, X)
            Z = predict(layer, X);
        end
    end
end