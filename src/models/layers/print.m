classdef print < nnet.layer.Layer & nnet.layer.Acceleratable
    properties
        State
    end

    properties(Learnable)
        Weights
        Bias
    end

    methods
        function layer = print(name)
            % Constructor needs description
            layer.Name = name;
            layer.Description = "Print layer for debugging";
            
            % Initialize weights and bias
            layer.Weights = rand(1,1);
            layer.Bias = rand(1,1);
        end

        function Y = predict(layer,X)
            % Missing semicolons will cause unwanted output
            Y = X;
            disp(layer.Name+" The size is:");
            disp(size(X));
        end

        function Y = forward(layer,X)
            disp("Using forward method of print layer");
            % Missing semicolons will cause unwanted output
            Y = X;
            disp(layer.Name+" The size is:");
            disp(size(X));
        end
    end
end