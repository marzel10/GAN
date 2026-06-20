classdef kSparseLayer < nnet.layer.Layer
    properties
        K   % number of active neurons
    end

    methods
        function layer = kSparseLayer(k, name)
            layer.Name = name;
            layer.Description = "k-sparse layer with k=" + k;
            layer.K = k;
        end

        function Z = predict(layer, X)
            % X size: [features batch]
            Z = X;
            [~, idx] = sort(abs(X), 1, 'descend');
            for i = 1:size(X,2)
                Z(idx(layer.K+1:end, i), i) = 0;
            end
        end

        function dLdX = backward(layer, X, ~, dLdZ, ~)
            % Gradient passes only through non-zero units
            mask = X ~= 0;
            dLdX = dLdZ .* mask;
        end
    end
end
