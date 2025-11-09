classdef SharedSwish < nnet.layer.Layer

    properties (Learnable)
        Beta; % Parameter for Swish activation
    end

    methods
        function layer = SharedSwish(name, num_in)
            % layer = SharedSwish(name) creates a Swish activation layer
            % with the specified name.
            arguments
                name (1,1) string
                num_in (1,1) double {mustBeInteger,mustBePositive} = 28
            end
            layer.Name = char(name);
            layer.Description = "Swish activation layer";
            layer.NumInputs = num_in;
            layer.NumOutputs = num_in;
            layer.Beta = single(1);  % initialize learnable scale
        end

        function varargout = predict(layer, varargin)
            % Forward input data through the layer at prediction time and
            % output the result.
            % varargin{i}: dlarray BSSC [B 4000 2 6]
            n = numel(varargin);
            varargout = cell(1,n);
            for i = 1:n
                X = varargin{i};  % BSSC
                X = X ./ (1 + exp(-layer.Beta .*X)); % Swish activation
                varargout{i} = dlarray(X);
            end
        end
    end

end