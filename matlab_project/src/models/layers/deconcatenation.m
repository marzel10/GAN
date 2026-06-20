classdef deconcatenation < nnet.layer.Layer
    
    properties
        % Layer properties go here if needed
    end
    
    methods
        function layer = deconcatenation(name, num_outputs)
            % Create a deconcatenation layer
            layer.Name = name;
            layer.Description = "Deconcatenation layer with " + num_outputs + " outputs";
            
            % Set the number of inputs and outputs
            layer.NumInputs = 1;
            layer.NumOutputs = num_outputs;
        end
        
        function varargout = predict(layer, X)
            % Forward pass for prediction
            
            % for debugging: print size of input
            % fprintf('deconcatenation: size(X) = %s\n', mat2str(size(X)));
            nodes_number = size(X,2);
            varargout = cell(1, nodes_number);

            % Ensure output is 500x4 for each slice
            for i = 1:nodes_number 
                varargout{i} = squeeze(X(:,i,:));  % This will keep the 4 channels
            end

            % Debug: print first output size
            % if ~isempty(varargout{1})
            %     fprintf('deconcatenation: first output size = %s\n', mat2str(size(varargout{1})));
            % else
            %     fprintf('deconcatenation: WARNING - first output is empty!\n');
            % end
        end

        function varargout = forward(layer, X)            
            % for debugging: print size of input
            % fprintf('deconcatenation: size(X) = %s\n', mat2str(size(X)));
            nodes_number = size(X,2);
            varargout = cell(1, nodes_number);

            % Ensure output is 500x4 for each slice
            for i = 1:nodes_number 
                varargout{i} = squeeze(X(:,i,:));  % This will keep the 4 channels
            end

            % Debug: print first output size
            % if ~isempty(varargout{1})
            %     fprintf('deconcatenation: first output size = %s\n', mat2str(size(varargout{1})));
            % else
            %     fprintf('deconcatenation: WARNING - first output is empty!\n');
            % end
        end
   
   
        
    end
end
