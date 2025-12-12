classdef Polynomial_activation < nnet.layer.Layer
   
    properties (Learnable)
        % Scale parameter 'a'
        ScaleA

        % Scale parameter 'b'
        ScaleB
    end

    methods
        function layer = Polynomial_activation(name)
            % Constructor for the Scaled_tanh layer
            layer.Name = name;
            layer.Description = "Custom polynomial activation layer";

            % Initialize learnable parameters
            layer.ScaleA = 1.0;
            layer.ScaleB = 0.1;
        end

        function Z = predict(layer, X)
            Z = layer.ScaleA * X + layer.ScaleB * X.*abs(X);
        end

        function Z = forward(layer, X)
            % Forward pass for training
            Z = layer.ScaleA * X + layer.ScaleB * X.*abs(X);
        end
    end
end