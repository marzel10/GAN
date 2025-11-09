classdef change_format < nnet.layer.Layer ...
        & nnet.layer.Acceleratable ...
        & nnet.layer.Formattable ...

        properties
            format
            size
        end

        methods

            function layer = change_format(name, format, size)
                layer.Name = name;
                layer.format = format;
                layer.size = size;
            end

            function Y = predict(layer, X)
                % Ensure numeric data (remove dlarray formatting if present)
                if isa(X, 'dlarray')
                    X = extractdata(X);
                end

                switch layer.format
                    case "BSSC"
                        % BSSC format: [Batch, Spatial1, Spatial2, Channel]
                        X = reshape(X, [], layer.size(1),layer.size(2),layer.size(3));
                    
                    case "SCB"
                        % SCB format: [Spatial, Channel, Batch]
                        X = reshape(X, layer.size(1),layer.size(2), []);
                    otherwise
                        error("change_format:forward:InvalidFormat", ...
                            "Unsupported format '%s'. Use 'BSSC' or 'SCB'.", layer.format);
                end
                
                % Change format using dlarray
                Y = dlarray(X, layer.format);
            end

            function Y = forward(layer, X)
                % Ensure numeric data (remove dlarray formatting if present)
                if isa(X, 'dlarray')
                    X = extractdata(X);
                end

                switch layer.format
                    case "BSSC"
                        % BSSC format: [Batch, Spatial1, Spatial2, Channel]
                        X = reshape(X, [], layer.size(1),layer.size(2),layer.size(3));
                    
                    case "SCB"
                        % SCB format: [ Spatial, Channel, Batch]
                        X = reshape(X, layer.size(1),layer.size(2), []);
                    otherwise
                        error("change_format:forward:InvalidFormat", ...
                            "Unsupported format '%s'. Use 'BSSC' or 'SCB'.", layer.format);
                end
                
                % Change format using dlarray
                Y = dlarray(X, layer.format);
            end

        end

end