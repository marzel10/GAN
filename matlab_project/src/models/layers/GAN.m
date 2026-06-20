classdef GAN < nnet.layer.Layer ...
        & nnet.layer.Acceleratable

    properties
        adj
        type
        conv_mode
    end

    properties(Learnable)
        Weights
        Bias

    end

    methods

        function layer = GAN(name,inputsize,outputsize,adj,type,varargin)


            p = inputParser;
            addParameter(p, 'conv_mode', false);
            parse(p, varargin{:});
            layer.conv_mode = p.Results.conv_mode;
            layer.Weights = rand(outputsize,inputsize);
            layer.Bias = rand(1,outputsize);  % Row vector for broadcasting

            layer.Name = name;
            layer.Description = "Graph Attention network, changing" + ...
                " representations from size " + inputsize + " to " + outputsize;
            layer.adj = adj;
            layer.type = type;  % can be 0 which is default or 1 with is latent space

            % Set number of outputs based on type
            if type == 1
                layer.NumOutputs = 2;  % For latent space: graph output + latent output
                layer.OutputNames = {'graph_out', 'latent_out'};
            else
                layer.NumOutputs = 1;  % Default: single output
                layer.OutputNames = {'out'};
            end

            if isempty(layer.adj)
                warning('The adjacency matrix is empty')
            end
        end

        function varargout = predict(layer,X)
            % X can be [feature_dim, num_nodes, batch_size] or [feature_dim, num_nodes, 1, batch_size]
            % Handle concatenation output from CNN layers
           
            inputSize = size(X);
            A_norm = layer.adj;
            % Debug: print sizes
            % disp(layer.Name)
            % disp("The X size is ")
            % disp(size(X))
            
            % Handle different input dimensions
            if length(inputSize) == 4
                % Input is [feat_dim, num_nodes, 1, batch_size] - squeeze the singleton dimension
                %X = squeeze(X); % This removes singleton dimensions
                % After squeeze: should be [feat_dim, num_nodes, batch_size] or [num_nodes, batch_size] if feat_dim=1
                inputSize = size(X);
            end
            
            % Ensure we have at least 3 dimensions [feat_dim, num_nodes, batch_size]
            if length(inputSize) == 2
                % Add batch dimension if missing: [feature_dim, num_nodes] -> [feature_dim, num_nodes, 1]
                X = reshape(X, inputSize(1), inputSize(2), 1);
            elseif length(inputSize) == 1
                % Handle 1D case: [feature_dim] -> [feature_dim, 1, 1]
                X = reshape(X, inputSize(1), 1, 1);
            end
          
            % Get dimensions after reshaping
            if layer.conv_mode
                [~,feat_dim, num_nodes, batch_size] = size(X);
            else
                [feat_dim, num_nodes, batch_size] = size(X);
            end
            
            % Debug: print sizes
            % disp(layer.Name)
            % disp("The X size is ")
            % disp(size(X))

            % Reshape for graph operations: [num_nodes, feat_dim, batch_size]
            if layer.conv_mode
                X = reshape(X, [],feat_dim, num_nodes, batch_size);
                %X = permute(X, [3, 2, 4, 1]); % [num_nodes, feat_dim, batch_size, ?]
            else
                X = permute(X, [2, 1, 3]); % [num_nodes, feat_dim, batch_size]
            end
            
            
            % Apply linear transformation to each node's features
            % layer.Weights should be [output_feat_dim, input_feat_dim]
            Y = zeros(num_nodes, size(layer.Weights,1), batch_size,'like',X);
            
            for b = 1:batch_size
                % Debug: Check actual dimensions
                % Debug output can be switched off by commenting out or removing these lines:
                % fprintf('Weights size: %s\n', mat2str(size(layer.Weights)));
                % fprintf('X(:,:,b) size: %s\n', mat2str(size(X(:,:,b))));
                % fprintf('X(:,:,b)'' size: %s\n', mat2str(size(X(:,:,b)')));
                
                % For each example in the batch, apply: Weights * X'
                % X(:,:,b) is [num_nodes, feat_dim], X(:,:,b)' is [feat_dim, num_nodes]
                % Weights * X(:,:,b)' gives [output_dim, num_nodes]
                
                temp = layer.Weights * X(:,:,b)'; % [output_dim, num_nodes]
               

                Y(:,:,b) = temp' + layer.Bias; % [num_nodes, output_dim]
                
                % Apply adjacency matrix to aggregate features from neighbors
                % layer.adj should be [num_nodes, num_nodes] (node-to-node attention)
                
                Y(:,:,b) = A_norm * Y(:,:,b); % [num_nodes, output_dim]
            end
            
            % Reshape back to expected format: [output_feat_dim, num_nodes, batch_size]
            Y = permute(Y, [2, 1, 3]); % [output_dim, num_nodes, batch_size]

            % Return outputs based on layer type
            if layer.type == 1
                % Return both graph output and latent space output as separate arguments
                varargout{1} = Y; % graph output
                varargout{2} = Y; % latent space output (same for now, can be modified)
            else
                % Single output
                varargout{1} = Y;
            end
        end

        function varargout = forward(layer,X)
            
            % X can be [feature_dim, num_nodes, batch_size] or [feature_dim, num_nodes, 1, batch_size]
            % Handle concatenation output from CNN layers
            
            inputSize = size(X);
            A_norm = layer.adj;

            % Debug: print sizes
            % disp(layer.Name)
            % disp("The X size is ")
            % disp(size(X))
            
            % Handle different input dimensions
            if length(inputSize) == 4
                % Input is [feat_dim, num_nodes, 1, batch_size] - squeeze the singleton dimension
                %X = squeeze(X); % This removes singleton dimensions
                % After squeeze: should be [feat_dim, num_nodes, batch_size] or [num_nodes, batch_size] if feat_dim=1
                inputSize = size(X);
                batch_size = inputSize(4);
            else
                batch_size = 1;
            end
            
            % Ensure we have at least 3 dimensions [feat_dim, num_nodes, batch_size]
            if length(inputSize) == 2
                % Add batch dimension if missing: [feature_dim, num_nodes] -> [feature_dim, num_nodes, 1]
                X = reshape(X, inputSize(1), inputSize(2), 1);
            elseif length(inputSize) == 1
                % Handle 1D case: [feature_dim] -> [feature_dim, 1, 1]
                X = reshape(X, inputSize(1), 1, 1);
            end

             % Get dimensions after reshaping
            if layer.conv_mode
              
                
                num_nodes =28;
                
                X = reshape(X,1, [], 28, batch_size);
                
                %X = permute(X, [3, 2, 4, 1]); % [num_nodes, feat_dim, batch_size, ?]
            else
                [feat_dim, num_nodes, batch_size] = size(X);
                X = permute(X, [2, 1, 3]); % [num_nodes, feat_dim, batch_size]
            end
            
            % Debug: print sizes
            % disp(layer.Name)
            % disp("The X size is ")
            % disp(size(X))

            
            % Apply linear transformation to each node's features
            % layer.Weights should be [output_feat_dim, input_feat_dim]
            Y = zeros(num_nodes, size(layer.Weights,1), batch_size,'like',X);
            
            for b = 1:batch_size
                % Debug: Check actual dimensions
                % Debug output can be switched off by commenting out or removing these lines:
                % fprintf('Weights size: %s\n', mat2str(size(layer.Weights)));
                % fprintf('X(:,:,b) size: %s\n', mat2str(size(X(:,:,b))));
                % fprintf('X(:,:,b)'' size: %s\n', mat2str(size(X(:,:,b)')));
                
                % For each example in the batch, apply: Weights * X'
                % X(:,:,b) is [num_nodes, feat_dim], X(:,:,b)' is [feat_dim, num_nodes]
                % Weights * X(:,:,b)' gives [output_dim, num_nodes]
                temp = layer.Weights * X(:,:,b)'; % [output_dim, num_nodes]
                
                Y(:,:,b) = temp' + layer.Bias; % [num_nodes, output_dim]
                
                % Apply adjacency matrix to aggregate features from neighbors
                % layer.adj should be [num_nodes, num_nodes] (node-to-node attention)
                
                Y(:,:,b) = A_norm * Y(:,:,b); % [num_nodes, output_dim]
            end
            
            % Reshape back to expected format: [output_feat_dim, num_nodes, batch_size]
            Y = permute(Y, [2, 1, 3]); % [output_dim, num_nodes, batch_size]

            % Return outputs based on layer type
            if layer.type == 1
                % Return both graph output and latent space output as separate arguments
                varargout{1} = Y; % graph output
                varargout{2} = Y; % latent space output (same for now, can be modified)
            else
                % Single output
                varargout{1} = Y;
            end
        end
    end
end




