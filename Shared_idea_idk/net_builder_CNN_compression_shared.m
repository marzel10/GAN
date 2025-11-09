classdef net_builder_CNN_compression_shared

    methods(Static)

        function layer = helperConcatenateVectors(name, num_inputs)
            % Custom concatenation for flattened vectors
            disp('Creating concatenation layer');
            layer = functionLayer(@(varargin) dlarray(cat(2, varargin{:}),"SCB"), ...
                "Formattable", true, ...
                "NumInputs", num_inputs, ...
                "Name", name);
        end

        function layer = helperchange_format(name,format,size)
            layer = change_format(name,format,size);
        end
      

        function layer = helperDeconcatenation(name,output_number)
            layer = deconcatenation(name,output_number);
        end

        function layer = helperGAN(name,adj,input_size,output_size,type)
            layer = GAN(name,input_size,output_size,adj,type);
        end

        function layer = helperprint(name)
            layer = print(name);
        end

        function net = build_net_with_28_inputs(latent_size)
            
            net = dlnetwork;

            num_in = 28;
            input_x = 4000;
            input_y = 2;
            input_z = 6;

            kernel_pool1 = [5 1];
            stride_pool1 = [5 1];

            single_input_size = 20*1*6;  % Size of one flattened input
            for i=1:num_in

                input = inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i));       
                net = addLayers(net,input);

                pooling = maxPooling2dLayer(kernel_pool1,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool1); % Shared max pooling
                net = addLayers(net, pooling);

                flatten = [flattenLayer("Name",sprintf("flatten_%d",i)), net_builder.helperchange_format(sprintf("change_format_%d",i),"SCB",[single_input_size 1])];
                net = addLayers(net, flatten);


            end

            conv1 = SharedConv("conv1", num_in, [50 2], 6, 6, [20 1]); % Shared conv layer
            net = addLayers(net, conv1);

            net = addLayers(net, SharedSwish("swish_encoder", num_in));

            conv2 = SharedConv("conv2", num_in, [4 2], 6, 6, [2 1]); % Shared conv layer
            net = addLayers(net, conv2);

            conv3 = SharedConv("conv3", num_in, [1 2], 6, 6, [1 2]); % Shared conv layer
            net = addLayers(net, conv3);


            adjacency_matrix = attention_matrix.build_attention();
            
            % More gradual dimension reduction for better learning
            hidden_size = 50;    % Increased to handle larger input
            
            % Encoder path with attention and normalization
            tempNet = [
                net_builder.helperConcatenateVectors("concat", num_in)  % Custom vector concatenation
                
                % Pre-attention normalization
                %batchNormalizationLayer("Name","bn_pre_attention")
                
                % Graph attention layers with residual connections
                net_builder.helperGAN("G1", adjacency_matrix,single_input_size,hidden_size,0)
                %batchNormalizationLayer("Name","bn_g1")
                reluLayer("Name","reluG1")
                dropoutLayer(0.1,"Name","dropout_g1")  % Very light dropout
                
                net_builder.helperGAN("G2",adjacency_matrix,hidden_size,latent_size,1)
                
                ];
            net = addLayers(net,tempNet);

            % Decoder path with attention and skip connections
            tempNet = [
                reluLayer("Name","reluG2")
                
                % Decoder with symmetric architecture
                net_builder.helperGAN("G3",adjacency_matrix,latent_size,hidden_size,0)
                %batchNormalizationLayer("Name","bn_g3")
                reluLayer("Name","reluG3")
                dropoutLayer(0.1,"Name","dropout_g3")  % Light dropout
                
                net_builder.helperGAN("G4",adjacency_matrix,hidden_size,single_input_size,0)
                %batchNormalizationLayer("Name","bn_g4")
                reluLayer("Name","reluG4")
                
                net_builder.helperDeconcatenation("d1",num_in)];
            net = addLayers(net,tempNet);



            for i=1:num_in
                tempNet = net_builder.helperchange_format(sprintf("reshape_%d",i),"BSSC",[20,1,6]);
                net = addLayers(net,tempNet);

                net = addLayers(net, maxUnpooling2dLayer('Name', sprintf("maxunpool_1_%d",i))); % Shared max unpooling

            end

            transpconv1 = SharedtranspConv("transpconv1", num_in, [1 2], 6, 6, [1 2]); % Shared transp conv layer
            net = addLayers(net, transpconv1);

            transpconv2 = SharedtranspConv("transpconv2", num_in, [4 2], 6, 6, [2 1]); % Shared transp conv layer
            net = addLayers(net, transpconv2);

            net = addLayers(net, SharedSwish( "swish_decoder", num_in));

            transpconv3 = SharedtranspConv("transpconv3", num_in, [50 2], 6, 6, [20 1]); % Shared transp conv layer
            net = addLayers(net, transpconv3);

             % clean up helper variable
            clear tempNet;

            for i=1:num_in
                
                net = connectLayers(net,sprintf("input_%d",i),sprintf("conv1/in%d",i));
                net = connectLayers(net,sprintf("conv1/out%d",i),sprintf("swish_encoder/in%d",i));
                net = connectLayers(net,sprintf("swish_encoder/out%d",i),sprintf("conv2/in%d",i));
                net = connectLayers(net,sprintf("conv2/out%d",i),sprintf("maxpoolForUnpool_1_%d/in",i));
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/out",i),sprintf("conv3/in%d",i));
                net = connectLayers(net,sprintf("conv3/out%d",i),sprintf("flatten_%d/in",i));
                % Connect flatten to concatenation layer
                net = connectLayers(net,sprintf("change_format_%d",i),sprintf("concat/in%d",i));
                % Connect deconcatenation output to reshape layer
                net = connectLayers(net,sprintf("d1/out%d",i),sprintf("reshape_%d",i));
                net = connectLayers(net,sprintf("reshape_%d/out",i),sprintf("transpconv1/in%d",i));
                net = connectLayers(net,sprintf("transpconv1/out%d",i),sprintf("maxunpool_1_%d/in",i));
                net = connectLayers(net,sprintf("maxunpool_1_%d/out",i),sprintf("transpconv2/in%d",i));
                net = connectLayers(net,sprintf("transpconv2/out%d",i),sprintf("swish_decoder/in%d",i));
                net = connectLayers(net,sprintf("swish_decoder/out%d",i),sprintf("transpconv3/in%d",i));

                net = connectLayers(net, sprintf("maxpoolForUnpool_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net, sprintf("maxpoolForUnpool_1_%d/size",i),sprintf("maxunpool_1_%d/size",i));

            end

            net = connectLayers(net, "G2/graph_out", "reluG2");

            % Initialize network with improved weight initialization
            net = initialize(net);
            
            % % Apply Xavier/Glorot initialization for better gradient flow
            % net = net_builder.improveWeightInitialization(net);
        end
        
        function net = improveWeightInitialization(net)
            % Complete Xavier/Glorot initialization for stable training
            learnables = net.Learnables;
            
            for i = 1:height(learnables)
                layerName = learnables.Layer{i};
                paramName = learnables.Parameter{i};
                
                % Get the actual layer from the network
                try
                    layer = net.Layers(strcmp({net.Layers.Name}, layerName));
                    if isempty(layer)
                        continue;
                    end
                    layer = layer(1);
                catch
                    continue;
                end
                
                % Initialize weights with Xavier for conv layers
                if (isa(layer, 'nnet.cnn.layer.Convolution2DLayer') || ...
                    isa(layer, 'nnet.cnn.layer.TransposedConvolution2DLayer')) && ...
                    strcmp(paramName, 'Weights')
                    
                    currentWeights = learnables.Value{i};
                    weightSize = size(currentWeights);
                    
                    % Calculate fan_in and fan_out for conv layers
                    % Conv2D: [H, W, InputChannels, OutputChannels]
                    if length(weightSize) == 4
                        fan_in = weightSize(1) * weightSize(2) * weightSize(3);  % H*W*InputCh
                        fan_out = weightSize(1) * weightSize(2) * weightSize(4); % H*W*OutputCh
                    else
                        % Fallback for other dimensions
                        fan_in = prod(weightSize(1:end-1));
                        fan_out = prod([weightSize(1:end-2), weightSize(end)]);
                    end
                    
                    % Xavier uniform initialization: U(-a, a) where a = sqrt(6/(fan_in+fan_out))
                    limit = sqrt(6 / (fan_in + fan_out));
                    newWeights = (rand(weightSize, 'single') * 2 * limit) - limit;
                    learnables.Value{i} = dlarray(newWeights);
                    
                    fprintf('Xavier init: %s - fan_in=%d, fan_out=%d, limit=%.4f, std=%.4f\n', ...
                        layerName, fan_in, fan_out, limit, std(newWeights(:)));
                        
                elseif strcmp(paramName, 'Bias')
                    % Initialize biases to zero (standard practice)
                    currentBias = learnables.Value{i};
                    newBias = zeros(size(currentBias), 'single');
                    learnables.Value{i} = dlarray(newBias);
                    
                    fprintf('Zero bias init: %s\n', layerName);
                    
                elseif contains(layerName, {'G1', 'G2', 'G3', 'G4'}) && strcmp(paramName, 'Weights')
                    % Initialize custom GAN layer weights with Xavier
                    currentWeights = learnables.Value{i};
                    weightSize = size(currentWeights);
                    
                    % For GAN layers: assume fully connected style [input_size, output_size]
                    if length(weightSize) == 2
                        fan_in = weightSize(1);
                        fan_out = weightSize(2);
                        limit = sqrt(6 / (fan_in + fan_out));
                        newWeights = (rand(weightSize, 'single') * 2 * limit) - limit;
                        learnables.Value{i} = dlarray(newWeights);
                        
                        fprintf('Xavier init (GAN): %s - fan_in=%d, fan_out=%d, limit=%.4f\n', ...
                            layerName, fan_in, fan_out, limit);
                    else
                        fprintf('Skipping GAN layer %s (unexpected weight shape: %s)\n', ...
                            layerName, mat2str(weightSize));
                    end
                end
            end
            
            % Apply the updated learnables back to the network
            net.Learnables = learnables;
            
            fprintf('Weight initialization complete. Total parameters initialized: %d\n', height(learnables));
        end
        
        function verifyInitialization(net)
            % Quick verification of weight initialization statistics
            learnables = net.Learnables;
            weightLayers = learnables(strcmp(learnables.Parameter, 'Weights'), :);
            
            fprintf('\n=== Weight Initialization Verification ===\n');
            for i = 1:height(weightLayers)
                weights = extractdata(weightLayers.Value{i});
                layerName = weightLayers.Layer{i};
                
                fprintf('%s: mean=%.6f, std=%.6f, min=%.4f, max=%.4f\n', ...
                    layerName, mean(weights(:)), std(weights(:)), ...
                    min(weights(:)), max(weights(:)));
            end
            fprintf('==========================================\n\n');
        end
    end
end
