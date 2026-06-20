classdef Scaled_tanh < nnet.layer.Layer
    % Scaled_tanh   Custom scaled tanh activation layer
    %
    % This layer applies the scaled tanh activation function:
    %   f(x) = a * tanh(b * x)
    % where 'a' and 'b' are learnable parameters.
    %
    % Example:
    %   layer = Scaled_tanh('ScaledTanhLayer', 1, 1);
    
    properties (Learnable)
        % Scale parameter 'a'
        ScaleA

        % Scale parameter 'b'
        ScaleB
    end

    methods
        function layer = Scaled_tanh(name, initialA)
            % Constructor for the Scaled_tanh layer
            layer.Name = name;
            layer.Description = "Scaled tanh activation layer";

            % Initialize learnable parameters
            layer.ScaleA = initialA;
        end

        function Z = predict(layer, X)
            % Forward pass: apply scaled tanh activation
            Z = layer.ScaleA * tanh(X/layer.ScaleA);
        end

        function Z = forward(layer, X)
            % Forward pass for training
            Z = layer.ScaleA * tanh(X/layer.ScaleA);
        end
    end
end