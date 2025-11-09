classdef AE_for_each_path_separate_E9

    methods(Static)

       
        function layer = helperchange_format(name,format,size)
            layer = change_format(name,format,size);
        end
        

        function layer = helperprint(name)
            layer = print(name);
        end

        function layer = helper_amplitudeScaleLayer(name, initValue)
            layer = amplitudeScaleLayer(name, initValue);
        end

        function [net, title] = build_net_with_28_inputs()
            
            net = dlnetwork;

            num_in = 28;
            input_x = 4000;
            input_y = 2;
            input_z = 6;

            kernel_conv1 = [25,2];
            kernel_conv2 = [9,2];
            kernel_conv3 = [5,2];
            channels_conv1 = 8;
            group_size1 = 4;
            group_size2 = 3;
            % OLD: stride1=[20 1]
            stride1        = [10 1];    % 4000 -> 400
            channels_conv2 = 6;
            stride2        = [2 1];     % 400 -> 200
            kernel_pool2   = [4 1];
            stride_pool2   = [5 1];     % 200 -> 40
            channels_conv3 = 2;         % keep latent size small
            stride3        = [1 2];     % reduce sensor axis only


            latentT = input_x / stride1(1) / stride2(1) / stride_pool2(1) / stride3(1);                     % 4000/10/2/4
            single_input_size = latentT*1*channels_conv3;  % 50*1*12 = 600
            title = sprintf('28_AE_latent_%d_E9', single_input_size);
            for i=1:num_in

                tempNet = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same","Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    %batchNormalizationLayer("Name",sprintf("bn__enc_1_%d",i))
                    leakyReluLayer("Name",sprintf("leaky_relu_enc_1_%d",i))
                    %additionLayer(2,"Name",sprintf("add_res_1_%d",i))
                    %maxPooling2dLayer(kernel_pool1,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool1)
                    
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv2,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride2)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_2_enc_%d",i))
                    %batchNormalizationLayer("Name",sprintf("bn_2_enc_%d",i))
                    leakyReluLayer("Name",sprintf("leaky_relu_2_enc_%d",i))
                    %reluLayer("Name",sprintf("Relu_2_%d",i))
                    %dropoutLayer(dropout_rate,"Name",sprintf("dropout_%d",i))
                    %additionLayer(2,"Name",sprintf("add_res_2_%d",i))
                    maxPooling2dLayer(kernel_pool2,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool2)
                    convolution2dLayer(kernel_conv3, channels_conv3, "Name", sprintf("conv_3_%d", i), "Padding", "same","Stride", stride3)
                    leakyReluLayer("Name", sprintf("leaky_relu_3_%d", i))
                    
                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer([3 1], channels_conv3, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", [2 1])
                    leakyReluLayer("Name", sprintf("leaky_relu_dil1_%d", i))
                    convolution2dLayer([3 1], channels_conv3, "Name", sprintf("conv_dil2_%d", i), "Padding", "same", "DilationFactor", [4 1])
                    leakyReluLayer("Name", sprintf("leaky_relu_dil2_%d", i))

                   
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    %AE_for_each_path_separate.helperprint(sprintf("latent_print_%d",i))
                    AE_for_each_path_separate.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[single_input_size,1])

                    % Use custom helper for reshape operation
                    AE_for_each_path_separate.helperchange_format(sprintf("reshape_%d",i),"BSSC",[latentT,1,channels_conv3])  % Reshape to [50 1 channels_conv3]
                    % Improved decoder with normalization
                    %maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))
                    
                    % First transposed conv block
                    transposedConv2dLayer(kernel_conv3,channels_conv2,"Name",sprintf("transposed-conv_1_%d",i),"Cropping","same","Stride", stride3)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_dec_1_%d",i))
                    %batchNormalizationLayer("Name",sprintf("bn_dec_1_%d",i))
                    leakyReluLayer("Name",sprintf("leaky_relu_dec_1_%d",i))
                    %reluLayer("Name",sprintf("Relu_3_%d",i))

                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))

                    % Second transposed conv block
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping","same","Stride", stride2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_2_%d",i))
                    %batchNormalizationLayer("Name",sprintf("bn_dec_2_%d",i))
                    leakyReluLayer("Name",sprintf("leaky_relu_dec_2_%d",i))
                    
                    additionLayer(2,"Name",sprintf("add_res_dec_2_%d",i))
                    
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_3_%d",i),"Cropping","same","Stride", stride1)
                    tanhLayer("Name",sprintf("tanh_bound_%d",i))
                    AE_for_each_path_separate.helper_amplitudeScaleLayer(sprintf("amp_scale_%d",i),0.025)
                    %batchNormalizationLayer("Name",sprintf("bn_dec_2_%d",i))
                    % Final activation could be tanh for bounded output
                    %AE_for_each_path_separate.helperprint(sprintf("after_last_trn_%d",i))
                    %tanhLayer("Name",sprintf("final_activation_%d",i))
                    ];       
                net = addLayers(net,tempNet);

                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/size",i),sprintf("maxunpool_1_%d/size",i));
                % Skip connection: from output of first activation to add_skip (needs matching size after upsampling)
                projName = sprintf("skip_proj_%d",i);
                projLayer = convolution2dLayer([1 1],channels_conv1,"Name",projName,"Padding","same");
                net = addLayers(net, projLayer);
                net = connectLayers(net,sprintf("leaky_relu_enc_1_%d",i),projName);
                net = connectLayers(net,projName,sprintf("add_res_dec_2_%d/in2",i));
            end
           

            % clean up helper variable
            clear tempNet;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
            
            % Apply Xavier/Glorot initialization for better gradient flow
            %net = net_builder.improveWeightInitialization(net);
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
