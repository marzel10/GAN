classdef SharedConv < nnet.layer.Layer ...
                             & nnet.layer.Formattable
    % SharedConvEncoder
    % Applies the same conv/pool/conv stack to each of N inputs.
    % Inputs/Outputs use SSCB (inputs) and SCB (flattened vectors).
    properties
        NumStreams (1,1) double {mustBeInteger,mustBePositive} = 28
        % Hyperparameters
        Kernel1 (1,2) double = [50 2]
        Stride1 (1,2) double = [20 1]
    end

    properties (Learnable)
        % conv1: [H W InC OutC]
        W1
        B1
        
    end

    % properties (SetAccess=private)
    %     InputNames
    %     OutputNames
    % end

    methods
        function layer = SharedConv(name, numStreams, kernel, in_chan, out_chan, stride)
            arguments
                name (1,1) string
                numStreams (1,1) double {mustBeInteger,mustBePositive} = 28
                kernel (1,2) double = [50 2]
                in_chan (1,1) double {mustBeInteger,mustBePositive} = 6
                out_chan (1,1) double {mustBeInteger,mustBePositive} = 6
                stride (1,2) double = [20 1]
            end
            layer.Name = char(name);
            layer.NumStreams = numStreams;
            layer.NumInputs = numStreams;
            layer.NumOutputs = numStreams;
            layer.Kernel1 = kernel;
            layer.Stride1 = stride;
            % Define multi-input/multi-output names
            % layer.InputNames  = cellfun(@(i) sprintf("in%d",i), 1:numStreams, 'UniformOutput', false);
            % layer.OutputNames = cellfun(@(i) sprintf("out%d",i), 1:numStreams, 'UniformOutput', false);

            % Initialize learnables (Xavier/Glorot uniform)
            % conv1: [kH kW InC OutC]
            k1 = layer.Kernel1;
            inC  = in_chan;     % input_z
            out1 = out_chan; 

            layer.W1 = SharedConv.initXavier([k1(1) k1(2) inC  out1]);
            layer.B1 = zeros([1 1 out1],'single');

        end

        function varargout = predict(layer, varargin)
            % varargin{i}: dlarray BSSC [B 4000 2 6]
            n = layer.NumStreams;
            varargout = cell(1,n);
            for i = 1:n
                X = varargin{i};  % BSSC
                Y = dlconv(X, layer.W1, layer.B1, ...
                           'Stride', layer.Stride1, 'Padding','same');
                varargout{i} = Y;  % BSSC
            end
        end

        function varargout = forward(layer, varargin)
            % Same as predict for training
            [varargout{1:layer.NumStreams}] = predict(layer, varargin{:});
        end
    end 

    methods (Static, Access = private)
        function W = initXavier(sz)
            fanIn  = prod(sz(1:3));
            fanOut = sz(1)*sz(2)*sz(4);
            limit = sqrt(6 / single(fanIn + fanOut));
            W = (rand(sz,'single')*2*limit) - limit;
        end

    end

end