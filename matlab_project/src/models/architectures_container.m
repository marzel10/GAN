classdef architectures_container
    methods(Static)
%% Compressed network
        
        function net = buildOptimizedNetwork_compressed(params)
            
            net = dlnetwork;
            
            % Input dimensions
            input_x = 4000;
            input_y = 2;
            input_z = 6;
            num_in = 28;
            desired_latent_size = 80;

            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size2,2];

            channels_conv1 = params.channels1;
            channels_conv2 = params.channels2;
        
            % Define possible group sizes (must be divisors of numChannels)
            possibleGroupSizes1 = 1:channels_conv1; % All integers from 1 to numChannels
            validGroupSizes1 = possibleGroupSizes1(mod(channels_conv1, possibleGroupSizes1) == 0);

            group_size1 = validGroupSizes1(ceil(length(validGroupSizes1)/2)); % Choose the middle valid size for consistency

            possibleGroupSizes2 = 1:channels_conv2; % All integers from 1 to numChannels
            validGroupSizes2 = possibleGroupSizes2(mod(channels_conv2, possibleGroupSizes2) == 0);

            group_size2 = validGroupSizes2(ceil(length(validGroupSizes2)/2)); % Choose the middle valid size for consistency

            stride1        = [params.stride1 1];    
            
            if stride1(1) > 10
                stride2 = [ceil(params.kernel_size2/4) 2];
            else
                stride2 = [ceil(params.kernel_size2/2) 2];
            end
            
            while mod(ceil(input_x/stride1(1)), stride2(1) ) ~= 0
                stride2(1) = stride2(1) - 1;
            end
            
            kernel_pool2   = [ceil(kernel_conv2(1)/2) 1];
            stride_pool2   = [kernel_pool2(1) 1];     

            DilationFactor = [params.dilation_factor 1];
            dilatation_kernel = [3 1];

            crop1 = 'same';
            crop2 = 'same';
            
            
            latentT = ceil(ceil(ceil(input_x / stride1(1)) / stride2(1)) / stride_pool2(1));                     % 4000/10/2/4
            single_input_size = latentT*1*channels_conv2;  % 50*1*12 = 600
            disp(" I am about to go into the loop\n");
            for i=1:num_in

                % Build encoder layers and append optional latent-adjust layers outside the array literal
                Encoder = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same","Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_1_%d",i))
            
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv2,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride2)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_2_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_2_%d",i))

                    % Max pooling with indices for unpooling
                    maxPooling2dLayer(kernel_pool2,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool2)
                    
                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer(dilatation_kernel, channels_conv2, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", DilationFactor)
                    architectures_container.helperELU( sprintf("elu_dil1_%d", i))
                    
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    architectures_container.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[single_input_size,1])
                ];
                net = addLayers(net,Encoder);

                if single_input_size ~= desired_latent_size
                    latent_fix_NN = [
                        fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_adjust_%d", i))
                        architectures_container.helperELU( sprintf("elu_latent_adjust_%d", i))
                        fullyConnectedLayer(single_input_size, "Name", sprintf("fc_latent_revert_%d", i))
                        architectures_container.helper_scaled_tanh(sprintf("tanh_latent_revert_%d", i), 6.0)
                    ];
                    net = addLayers(net, latent_fix_NN);
                    net = connectLayers(net, sprintf("add_channel_dim_%d", i), sprintf("fc_latent_adjust_%d", i));
                end

                Decoder = [

                    % Use custom helper for reshape operation
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[latentT,1,channels_conv2])  % Reshape to [50 1 channels_conv3]

                    % Decoder-side symmetric dilated block (size-preserving)
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil_dec1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor)
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dil_dec1_%d", i), 6.0)
                    
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))

                    % Reverting second convolution
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping",crop2,"Stride", stride2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_2_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_2_%d", i), 6.0)
                    %architectures_container.helper_removeMeanLayer(sprintf("remove_mean_dec_2_%d",i))

                    % Reverting first convolution
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_3_%d",i),"Cropping",crop1,"Stride", stride1,"BiasInitializer","zeros","BiasLearnRateFactor",0)
                    architectures_container.helper_scaled_tanh(sprintf("tanh_bound_%d", i), 6.0)
                    %architectures_container.helper_amplitudeScaleLayer(sprintf("amp_scale_%d",i),0.025)
                    
                ];       
                net = addLayers(net,Decoder);
                if single_input_size ~= desired_latent_size
                    net = connectLayers(net, sprintf("tanh_latent_revert_%d", i), sprintf("reshape_%d", i));
                else
                    net = connectLayers(net, sprintf("add_channel_dim_%d", i), sprintf("reshape_%d", i));
                end
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/size",i),sprintf("maxunpool_1_%d/size",i));
                
                
            end
            
            % clean up helper variable
            clear Encoder Decoder;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
        end
%% Compressed downsampled network
        function net = buildOptimizedNetwork_compressed_downsampled(params)
            
            net = dlnetwork;
            
            % Input dimensions
            input_x = 800;
            input_y = 2;
            input_z = 6;
            num_in = 28;
            desired_latent_size = 80;

            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size2,2];

            channels_conv1 = params.channels1;
            channels_conv2 = params.channels2;
        
            % Define possible group sizes (must be divisors of numChannels)
            possibleGroupSizes1 = 1:channels_conv1; % All integers from 1 to numChannels
            validGroupSizes1 = possibleGroupSizes1(mod(channels_conv1, possibleGroupSizes1) == 0);

            group_size1 = validGroupSizes1(ceil(length(validGroupSizes1)/2)); % Choose the middle valid size for consistency

            possibleGroupSizes2 = 1:channels_conv2; % All integers from 1 to numChannels
            validGroupSizes2 = possibleGroupSizes2(mod(channels_conv2, possibleGroupSizes2) == 0);

            group_size2 = validGroupSizes2(ceil(length(validGroupSizes2)/2)); % Choose the middle valid size for consistency

            stride1        = [params.stride1 1];    
            
            if stride1(1) > 10
                stride2 = [ceil(params.kernel_size2/4) 2];
            else
                stride2 = [ceil(params.kernel_size2/2) 2];
            end
            
            while mod(ceil(input_x/stride1(1)), stride2(1) ) ~= 0
                stride2(1) = stride2(1) - 1;
            end
            
            kernel_pool2   = [ceil(kernel_conv2(1)/2) 1];
            stride_pool2   = [kernel_pool2(1) 1];     

            DilationFactor = [params.dilation_factor 1];
            dilatation_kernel = [3 1];

            crop1 = 'same';
            crop2 = 'same';
            
            
            latentT = ceil(ceil(ceil(input_x / stride1(1)) / stride2(1)) / stride_pool2(1));                     % 4000/10/2/4
            single_input_size = latentT*1*channels_conv2;  % 50*1*12 = 600
            disp(" I am about to go into the loop\n");
            for i=1:num_in

                % Build encoder layers and append optional latent-adjust layers outside the array literal
                Encoder = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same","Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_1_%d",i))
            
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv2,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride2)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_2_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_2_%d",i))

                    % Max pooling with indices for unpooling
                    maxPooling2dLayer(kernel_pool2,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool2)
                    
                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer(dilatation_kernel, channels_conv2, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", DilationFactor)
                    architectures_container.helperELU( sprintf("elu_dil1_%d", i))
                    
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    architectures_container.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[single_input_size,1])
                ];
                net = addLayers(net,Encoder);

                if single_input_size ~= desired_latent_size
                    latent_fix_NN = [
                        fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_adjust_%d", i))
                        architectures_container.helperELU( sprintf("elu_latent_adjust_%d", i))
                        fullyConnectedLayer(single_input_size, "Name", sprintf("fc_latent_revert_%d", i))
                        architectures_container.helper_scaled_tanh(sprintf("tanh_latent_revert_%d", i), 6.0)
                    ];
                    net = addLayers(net, latent_fix_NN);
                    net = connectLayers(net, sprintf("add_channel_dim_%d", i), sprintf("fc_latent_adjust_%d", i));
                end

                Decoder = [

                    % Use custom helper for reshape operation
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[latentT,1,channels_conv2])  % Reshape to [50 1 channels_conv3]

                    % Decoder-side symmetric dilated block (size-preserving)
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil_dec1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor)
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dil_dec1_%d", i), 6.0)
                    
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))

                    % Reverting second convolution
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping",crop2,"Stride", stride2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_2_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_2_%d", i), 6.0)
                    %architectures_container.helper_removeMeanLayer(sprintf("remove_mean_dec_2_%d",i))

                    % Reverting first convolution
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_3_%d",i),"Cropping",crop1,"Stride", stride1,"BiasInitializer","zeros","BiasLearnRateFactor",0)
                    architectures_container.helper_scaled_tanh(sprintf("tanh_bound_%d", i), 6.0)
                    %architectures_container.helper_amplitudeScaleLayer(sprintf("amp_scale_%d",i),0.025)
                    
                ];       
                net = addLayers(net,Decoder);
                if single_input_size ~= desired_latent_size
                    net = connectLayers(net, sprintf("tanh_latent_revert_%d", i), sprintf("reshape_%d", i));
                else
                    net = connectLayers(net, sprintf("add_channel_dim_%d", i), sprintf("reshape_%d", i));
                end
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/size",i),sprintf("maxunpool_1_%d/size",i));
                
                
            end
            
            % clean up helper variable
            clear Encoder Decoder;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
        end
%% Compressed single frequency network
        function net = buildOptimizedNetwork_compressed_sin_freq(params)
            
            net = dlnetwork;
            
            % Input dimensions
            input_x = 4000;
            input_y = 2;
            input_z = 1;
            num_in = 28;
            desired_latent_size = 15;

            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size1,2];
            kernel_conv3 = [params.kernel_size2,2];
            kernel_conv4 = [params.kernel_size2,2];
            kernel_conv5 = [params.kernel_size3,2];

            channels_conv1 = params.channels1;
            channels_conv2 = params.channels2;
        
            % Define possible group sizes (must be divisors of numChannels)
            possibleGroupSizes1 = 1:channels_conv1; % All integers from 1 to numChannels
            validGroupSizes1 = possibleGroupSizes1(mod(channels_conv1, possibleGroupSizes1) == 0);

            group_size1 = validGroupSizes1(ceil(length(validGroupSizes1)/2)); % Choose the middle valid size for consistency

            possibleGroupSizes2 = 1:channels_conv2; % All integers from 1 to numChannels
            validGroupSizes2 = possibleGroupSizes2(mod(channels_conv2, possibleGroupSizes2) == 0);

            group_size2 = validGroupSizes2(ceil(length(validGroupSizes2)/2)); % Choose the middle valid size for consistency

            % first frequency is  50 kHz, sampling rate is 2e6 Hz, therefore 40 samples per period, stride 20 seems reasonable
            stride1        = [5 1];  
            stride1_2      = [4 1];  
            stride2        = [4 2];
            
            fc_layer_size = input_x / (stride1(1)^3 * stride1_2(1) * stride2(1)); % there will be 4 downsampling operations and one chnaging the dim of the second axis
            disp(['Fully connected layer size: ', num2str(fc_layer_size)]);
            % if stride1(1) > 10
            %     stride2 = [ceil(params.kernel_size2/4) 2];
            % else
            %     stride2 = [ceil(params.kernel_size2/2) 2];
            % end
            
            % while mod(ceil(input_x/stride1(1)), stride2(1) ) ~= 0
            %     stride2(1) = stride2(1) - 1;
            % end
            
            %kernel_pool2   = [kernel_conv2(1) 1];
            %stride_pool2   = [kernel_pool2(1) 1];     

            DilationFactor = [params.dilation_factor 1];
            dilatation_kernel = [3 1];

            crop1 = 'same';
            crop2 = 'same';
            
            %T0 = 4;
            %upsampling_stride = [4 1];
            %upsampling_kernel = upsampling_stride;
            
            %latentT = ceil(ceil(ceil(input_x / stride1(1)) / stride2(1)) / stride_pool2(1));                     % 4000/10/2/4
            %nr_upsampling = ceil(log2(latentT / T0)/2); % each upsampling layer upsamples by 4 (2^2)
            %single_input_size = latentT*1*channels_conv2;  % 50*1*12 = 600
            disp(" I am about to go into the loop\n");
            for i=1:num_in

                % Build encoder layers and append optional latent-adjust layers outside the array literal
                Encoder = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same","Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_1_%d",i))
            
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_2_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_2_%d",i))

                    % Third conv block with normalization and res connections
                    convolution2dLayer(kernel_conv3,channels_conv1,"Name",sprintf("conv_3_%d",i),"Padding","same", "Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_3_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_3_%d",i))

                    % Fourth conv block with normalization and res connections
                    convolution2dLayer(kernel_conv4,channels_conv1,"Name",sprintf("conv_4_%d",i),"Padding","same", "Stride", stride1_2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_4_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_4_%d",i))

                    % Fifth conv block that will reduce the second dimension to 1
                    convolution2dLayer(kernel_conv5,channels_conv2,"Name",sprintf("conv_5_%d",i),"Padding","same", "Stride", stride2)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_5_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_5_%d",i))

                    % Max pooling with indices for unpooling
                    %maxPooling2dLayer(kernel_pool2,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool2)
                    % Convolutional downsampling instead of maxpooling
                    % convolution2dLayer(kernel_pool2,channels_conv2,"Name",sprintf("conv_downsample_%d",i),"Padding","same","Stride", stride_pool2)


                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer(dilatation_kernel, channels_conv2, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", DilationFactor)
                    architectures_container.helperELU( sprintf("elu_dil1_%d", i))
                    
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    % use global1daveragepooling instead of flatten
                    %globalAveragePooling2dLayer("Name",sprintf("global_avg_pool_%d",i))
                    %architectures_container.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[channels_conv2,1])
                    fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_adjust_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_adjust_%d", i))
                ];
                net = addLayers(net,Encoder);

                % if fc_layer_size*channels_conv2 ~= desired_latent_size
                %     latent_fix_NN = [
                %         fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_adjust_%d", i))
                %         architectures_container.helperELU( sprintf("elu_latent_adjust_%d", i))
                %         fullyConnectedLayer(fc_layer_size*channels_conv2, "Name", sprintf("fc_latent_revert_%d", i))
                %         architectures_container.helper_scaled_tanh(sprintf("tanh_latent_revert_%d", i), 6.0)
                %     ];
                %     net = addLayers(net, latent_fix_NN);
                %     net = connectLayers(net, sprintf("add_channel_dim_%d", i), sprintf("fc_latent_adjust_%d", i));
                %     reshape_layer = architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[fc_layer_size,1,channels_conv2]);  % Reshape to [50 1 channels_conv3]
                %     net = addLayers(net, reshape_layer);
                % else
                %     reshape_layer = architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[1,1,channels_conv2]);  % Reshape to [50 1 channels_conv3]
                %     net = addLayers(net, reshape_layer);
                % end
                

                % % Progressive temporal upsampling (each layer upsamples by 4 in time)
                % lastUpsampleName = sprintf("reshape_%d", i);
                % if nr_upsampling > 0
                %     for j = 1:nr_upsampling
                %         layerName = sprintf("conv_upsample%d_%d", j, i);
                %         up_samp = transposedConv2dLayer(upsampling_kernel, channels_conv2, ...
                %             "Name", layerName, "Cropping", "same", "Stride", upsampling_stride);
                %         net = addLayers(net, up_samp);
                %         net = connectLayers(net, lastUpsampleName, layerName);
                %         lastUpsampleName = layerName;
                %     end
                % end
                % % Store name for later connection to first decoder layer
                % upsamplingOutputName = lastUpsampleName;
                
                % Decoder-side layers (moved inside loop; removed stray end)
                Decoder = [

                    fullyConnectedLayer(fc_layer_size*channels_conv2, "Name", sprintf("fc_latent_revert_%d", i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_latent_revert_%d", i), 6.0)
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[fc_layer_size,1,channels_conv2]);

                    % Decoder-side symmetric dilated block (size-preserving)
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil_dec1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor)
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dil_dec1_%d", i), 6.0)
                    
                    % Convolutional upsampling instead of maxunpooling
                    % transposedConv2dLayer(kernel_pool2,channels_conv2,"Name",sprintf("conv_upsample_%d",i),"Cropping","same","Stride", stride_pool2)

                    % Reverting fifth convolution
                    transposedConv2dLayer(kernel_conv5,channels_conv1,"Name",sprintf("transposed-conv_5_%d",i),"Cropping",crop2,"Stride", stride2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_2_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_2_%d", i), 6.0)

                    % Reverting fourth convolution
                    transposedConv2dLayer(kernel_conv4,channels_conv1,"Name",sprintf("transposed-conv_4_%d",i),"Cropping",crop2,"Stride", stride1_2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_3_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_3_%d", i), 6.0)

                    % Reverting third convolution
                    transposedConv2dLayer(kernel_conv3,channels_conv1,"Name",sprintf("transposed-conv_3_%d",i),"Cropping",crop2,"Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_4_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_4_%d", i), 6.0)

                    % Reverting second convolution
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping",crop2,"Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_5_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_5_%d", i), 6.0)

                    % Reverting first convolution
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_1_%d",i),"Cropping",crop1,"Stride", stride1,"BiasInitializer","zeros","BiasLearnRateFactor",0)
                    architectures_container.helper_scaled_tanh(sprintf("tanh_bound_%d", i), 6.0)
                ];
                net = addLayers(net,Decoder);
                net = connectLayers(net, sprintf("elu_latent_adjust_%d", i), sprintf("fc_latent_revert_%d", i));
                % Connect upsampling output to decoder start
                % net = connectLayers(net, upsamplingOutputName, sprintf("conv_dil_dec1_%d", i));
                
                % Correct latent revert connection condition
                % if channels_conv2 ~= desired_latent_size
                %     net = connectLayers(net, sprintf("tanh_latent_revert_%d", i), sprintf("reshape_%d", i));
                % else
                %     net = connectLayers(net, sprintf("add_channel_dim_%d", i), sprintf("reshape_%d", i));
                % end
                
            end
            
            % clean up helper variable
            clear Encoder Decoder;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
        end
%% Compressed single frequency downsampled network
        function net = buildOptimizedNetwork_compressed_downsampled_sin_freq_paper(params)

            if ~isfield(params,'dropout_rate_enc'), params.dropout_rate_enc = 0; end
            if ~isfield(params,'dropout_rate_dec'), params.dropout_rate_dec = 0; end
            if ~isfield(params,'desired_latent_size'), params.desired_latent_size = 15; end
            drop_rate_encoder = params.dropout_rate_enc;
            drop_rate_decoder = params.dropout_rate_dec;
            
            net = dlnetwork;
            
            % Input dimensions
            input_x = 800;
            input_y = 2;
            input_z = 1;
            num_in = 1;
            desired_latent_size =  params.desired_latent_size;

            % first frequency is  50 kHz, sampling rate is 2e6 Hz, therefore 40 samples per period, stride 20 seems reasonable
            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size2,2];
            kernel_conv3 = [params.kernel_size3,2];
            kernel_conv4 = [params.kernel_size4,2];
            kernel_conv5 = [params.kernel_size5,2];

            channels_conv1 = params.channels1;
            channels_conv2 = params.channels2;
            channels_conv3 = params.channels3;
            channels_conv4 = params.channels4;
            channels_conv5 = params.channels5;
        
            pooling_kernel = [2 1];
            pooling_stride = [2 1];

            pooling_kernel2 = [2 2];
            pooling_stride2 = [2 2];        
            
            fc_layer_size = input_x / (pooling_kernel(1)^4 * pooling_kernel2(1)); % there will be 4 downsampling operations and one chnaging the dim of the second axis
            disp(['Fully connected layer size: ', num2str(fc_layer_size)]);

            disp(" I am about to go into the loop\n");
            for i=1:num_in

                net = addLayers(net,convolution2dLayer([1 1], channels_conv1, "Name", sprintf("res_conv1_%d", i), "Padding", "same", "Stride", pooling_stride));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv2_%d", i), "Padding", "same", "Stride", pooling_stride.^2));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv3, "Name", sprintf("res_conv3_%d", i), "Padding", "same", "Stride", pooling_stride.^3));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv4, "Name", sprintf("res_conv4_%d", i), "Padding", "same", "Stride", pooling_stride.^4));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv5, "Name", sprintf("res_conv5_%d", i), "Padding", "same", "Stride", pooling_stride.^5));

                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv1_2_%d", i), "Padding", "same", "Stride", pooling_stride));
               
                
                net = addLayers(net,convolution2dLayer([1 1], channels_conv3, "Name", sprintf("res_conv2_3_%d", i), "Padding", "same", "Stride", pooling_stride));
               

                net = addLayers(net,convolution2dLayer([1 1], channels_conv4, "Name", sprintf("res_conv3_4_%d", i), "Padding", "same", "Stride", pooling_stride));

                net = addLayers(net,convolution2dLayer([1 1], channels_conv5, "Name", sprintf("res_conv4_5_%d", i), "Padding", "same", "Stride", pooling_stride2));

                % Build encoder layers and append optional latent-adjust layers outside the array literal
                Encoder = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same")
                    %batchNormalizationLayer("Name",sprintf("bn_enc_1_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_enc_1_%d",i))
                    dropoutLayer(drop_rate_encoder,"Name",sprintf("dropout_enc_1_%d",i))
                    maxPooling2dLayer(pooling_kernel,"Stride",pooling_stride,"Padding","same","Name",sprintf("maxpool_enc_1_%d",i),'HasUnpoolingOutputs',true)
            
                    additionLayer(2,"Name",sprintf("res_add_1_%d",i))

                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv2,"Name",sprintf("conv_2_%d",i),"Padding","same")
                    %batchNormalizationLayer("Name",sprintf("bn_enc_2_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_enc_2_%d",i))
                    dropoutLayer(drop_rate_encoder,"Name",sprintf("dropout_enc_2_%d",i))
                    maxPooling2dLayer(pooling_kernel,"Stride",pooling_stride,"Padding","same","Name",sprintf("maxpool_enc_2_%d",i),'HasUnpoolingOutputs',true)

                    additionLayer(3,"Name",sprintf("res_add_2_%d",i))

                    % Third conv block with normalization and res connections
                    convolution2dLayer(kernel_conv3,channels_conv3,"Name",sprintf("conv_3_%d",i),"Padding","same")
                    %batchNormalizationLayer("Name",sprintf("bn_enc_3_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_enc_3_%d",i))
                    dropoutLayer(drop_rate_encoder,"Name",sprintf("dropout_enc_3_%d",i))
                    maxPooling2dLayer(pooling_kernel,"Stride",pooling_stride,"Padding","same","Name",sprintf("maxpool_enc_3_%d",i),'HasUnpoolingOutputs',true)

                    additionLayer(3,"Name",sprintf("res_add_3_%d",i))

                    % Fourth conv block with normalization and res connections
                    convolution2dLayer(kernel_conv4,channels_conv4,"Name",sprintf("conv_4_%d",i),"Padding","same")
                    %batchNormalizationLayer("Name",sprintf("bn_enc_4_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_enc_4_%d",i))
                    dropoutLayer(drop_rate_encoder,"Name",sprintf("dropout_enc_4_%d",i))
                    maxPooling2dLayer(pooling_kernel,"Stride",pooling_stride,"Padding","same","Name",sprintf("maxpool_enc_4_%d",i),'HasUnpoolingOutputs',true)

                    additionLayer(3,"Name",sprintf("res_add_4_%d",i))

                    % Fifth conv block that will reduce the second dimension to 1
                    convolution2dLayer(kernel_conv5,channels_conv5,"Name",sprintf("conv_5_%d",i),"Padding","same")
                    %batchNormalizationLayer("Name",sprintf("bn_enc_5_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_enc_5_%d",i))
                    dropoutLayer(drop_rate_encoder,"Name",sprintf("dropout_enc_5_%d",i))
                    maxPooling2dLayer(pooling_kernel2,"Stride",pooling_stride2,"Padding","same","Name",sprintf("maxpool_enc_5_%d",i),'HasUnpoolingOutputs',true)

                    additionLayer(3,"Name",sprintf("res_add_5_%d",i))

                    % Temporal dilated block in bottleneck (no size change)
                    %convolution2dLayer(dilatation_kernel, channels_conv2, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", DilationFactor)
                    %architectures_container.helperELU( sprintf("elu_dil1_%d", i))
                    
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    % use global1daveragepooling instead of flatten
                    %globalAveragePooling2dLayer("Name",sprintf("global_avg_pool_%d",i))
                    %architectures_container.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[channels_conv2,1])
                    fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_adjust_%d", i))
                    %architectures_container.helperELU( sprintf("elu_latent_adjust_%d", i))
                ];
                net = addLayers(net,Encoder);
             
                net = connectLayers(net, sprintf("input_%d",i), sprintf("res_conv1_%d", i));
                net = connectLayers(net, sprintf("input_%d",i), sprintf("res_conv2_%d", i));
                net = connectLayers(net, sprintf("input_%d",i), sprintf("res_conv3_%d", i));
                net = connectLayers(net, sprintf("input_%d",i), sprintf("res_conv4_%d", i));
                net = connectLayers(net, sprintf("input_%d",i), sprintf("res_conv5_%d", i));

                net = connectLayers(net, sprintf("res_conv1_%d", i), sprintf("res_add_1_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv2_%d", i), sprintf("res_add_2_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv3_%d", i), sprintf("res_add_3_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv4_%d", i), sprintf("res_add_4_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv5_%d", i), sprintf("res_add_5_%d/in2", i));

                net = connectLayers(net, sprintf("maxpool_enc_1_%d/out", i), sprintf("res_conv1_2_%d", i));
                net = connectLayers(net, sprintf("maxpool_enc_2_%d/out", i), sprintf("res_conv2_3_%d", i));
                net = connectLayers(net, sprintf("maxpool_enc_3_%d/out", i), sprintf("res_conv3_4_%d", i));
                net = connectLayers(net, sprintf("maxpool_enc_4_%d/out", i), sprintf("res_conv4_5_%d", i));

                net = connectLayers(net, sprintf("res_conv1_2_%d", i), sprintf("res_add_2_%d/in3", i));
                net = connectLayers(net, sprintf("res_conv2_3_%d", i), sprintf("res_add_3_%d/in3", i));
                net = connectLayers(net, sprintf("res_conv3_4_%d", i), sprintf("res_add_4_%d/in3", i));
                net = connectLayers(net, sprintf("res_conv4_5_%d", i), sprintf("res_add_5_%d/in3", i));




                
                % Decoder-side layers (moved inside loop; removed stray end)
                Decoder = [

                    fullyConnectedLayer(fc_layer_size*channels_conv5, "Name", sprintf("fc_latent_revert_%d", i))
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[fc_layer_size,1,channels_conv5]);

                    maxUnpooling2dLayer("Name",sprintf("maxunpool_5_%d",i))

                    % reverse fifth convolution
                    transposedConv2dLayer(kernel_conv5,channels_conv4,"Name",sprintf("transposed-conv_5_%d",i),"Cropping","same")
                    %batchNormalizationLayer("Name",sprintf("bn_dec_5_%d",i))
                    tanhLayer("Name",sprintf("tanh_dec_5_%d", i))
                    dropoutLayer(drop_rate_decoder,"Name",sprintf("dropout_dec_5_%d",i))

                    % reverse fourth convolution
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_4_%d",i))
                    transposedConv2dLayer(kernel_conv4,channels_conv3,"Name",sprintf("transposed-conv_4_%d",i),"Cropping","same")
                    %batchNormalizationLayer("Name",sprintf("bn_dec_4_%d",i))
                    tanhLayer("Name",sprintf("tanh_dec_4_%d", i))
                    dropoutLayer(drop_rate_decoder,"Name",sprintf("dropout_dec_4_%d",i))

                    % reverse third convolution
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_3_%d",i))
                    transposedConv2dLayer(kernel_conv3,channels_conv2,"Name",sprintf("transposed-conv_3_%d",i),"Cropping","same")
                    %batchNormalizationLayer("Name",sprintf("bn_dec_3_%d",i))
                    tanhLayer("Name",sprintf("tanh_dec_3_%d", i))
                    dropoutLayer(drop_rate_decoder,"Name",sprintf("dropout_dec_3_%d",i))

                    % reverse second convolution
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_2_%d",i))
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping","same")
                    %batchNormalizationLayer("Name",sprintf("bn_dec_2_%d",i))
                    tanhLayer("Name",sprintf("tanh_dec_2_%d", i))
                    dropoutLayer(drop_rate_decoder,"Name",sprintf("dropout_dec_2_%d",i))

                    % reverse first convolution
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_1_%d",i),"Cropping","same","BiasInitializer","zeros","BiasLearnRateFactor",0)
                    % REMOVED tanh to allow full amplitude range
                    % tanhLayer("Name",sprintf("tanh_bound_%d", i))
                ];
                 
                net = addLayers(net,Decoder);
                net = connectLayers(net, sprintf("fc_latent_adjust_%d", i), sprintf("fc_latent_revert_%d", i));

                % %connect pooling and unpooling layers
                net = connectLayers(net,sprintf("maxpool_enc_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,sprintf("maxpool_enc_1_%d/size",i),sprintf("maxunpool_1_%d/size",i));

                net = connectLayers(net,sprintf("maxpool_enc_2_%d/indices",i),sprintf("maxunpool_2_%d/indices",i));
                net = connectLayers(net,sprintf("maxpool_enc_2_%d/size",i),sprintf("maxunpool_2_%d/size",i));   

                net = connectLayers(net,sprintf("maxpool_enc_3_%d/indices",i),sprintf("maxunpool_3_%d/indices",i));
                net = connectLayers(net,sprintf("maxpool_enc_3_%d/size",i),sprintf("maxunpool_3_%d/size",i));

                net = connectLayers(net,sprintf("maxpool_enc_4_%d/indices",i),sprintf("maxunpool_4_%d/indices",i));
                net = connectLayers(net,sprintf("maxpool_enc_4_%d/size",i),sprintf("maxunpool_4_%d/size",i));

                net = connectLayers(net,sprintf("maxpool_enc_5_%d/indices",i),sprintf("maxunpool_5_%d/indices",i));
                net = connectLayers(net,sprintf("maxpool_enc_5_%d/size",i),sprintf("maxunpool_5_%d/size",i));
               
            end
            
            % clean up helper variable
            clear Encoder Decoder;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
        end


        function net = buildOptimizedNetwork_compressed_downsampled_sin_freq(params)
            
            net = dlnetwork;
            
            % Input dimensions
            input_x = 800;
            input_y = 2;
            input_z = 1;
            if isfield(params, 'num_inputs')
                num_in = params.num_inputs;
            else
                num_in = 1;
            end
            desired_latent_size = 15;

            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size1,2];
            kernel_conv3 = [params.kernel_size2,2];
            kernel_conv4 = [params.kernel_size3,2];
        

            channels_conv1 = params.channels1;
            channels_conv2 = params.channels2;
        
            % Define possible group sizes (must be divisors of numChannels)
            possibleGroupSizes1 = 1:channels_conv1; % All integers from 1 to numChannels
            validGroupSizes1 = possibleGroupSizes1(mod(channels_conv1, possibleGroupSizes1) == 0);

            group_size1 = validGroupSizes1(ceil(length(validGroupSizes1)/2)); % Choose the middle valid size for consistency

            possibleGroupSizes2 = 1:channels_conv2; % All integers from 1 to numChannels
            validGroupSizes2 = possibleGroupSizes2(mod(channels_conv2, possibleGroupSizes2) == 0);

            group_size2 = validGroupSizes2(ceil(length(validGroupSizes2)/2)); % Choose the middle valid size for consistency

            % first frequency is  50 kHz, sampling rate is 2e6 Hz, therefore 40 samples per period, stride 20 seems reasonable
            stride1        = [5 1];  
            stride1_2      = [4 1];  
            stride2        = [4 2];
            
            fc_layer_size = input_x / (stride1(1)^2 * stride1_2(1) * stride2(1)); % there will be 4 downsampling operations and one chnaging the dim of the second axis
            disp(['Fully connected layer size: ', num2str(fc_layer_size)]);
            
            DilationFactor = [params.dilation_factor 1];
            dilatation_kernel = [3 1];

            crop1 = 'same';
            crop2 = 'same';
            
           
            disp(" I am about to go into the loop\n");
            for i=1:num_in

                % Build encoder layers and append optional latent-adjust layers outside the array literal
                Encoder = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same","Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_1_%d",i))
            
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_2_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_2_%d",i))

                    % Third conv block with normalization and res connections
                    convolution2dLayer(kernel_conv3,channels_conv1,"Name",sprintf("conv_3_%d",i),"Padding","same", "Stride", stride1_2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_3_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_3_%d",i))

                    % Fourth conv block with normalization and res connections
                    convolution2dLayer(kernel_conv4,channels_conv2,"Name",sprintf("conv_4_%d",i),"Padding","same", "Stride", stride2)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_4_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_4_%d",i))

                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer(dilatation_kernel, channels_conv2, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", DilationFactor)
                    architectures_container.helperELU( sprintf("elu_dil1_%d", i))
                    
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    % use global1daveragepooling instead of flatten
                    %globalAveragePooling2dLayer("Name",sprintf("global_avg_pool_%d",i))
                    %architectures_container.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[channels_conv2,1])
                    fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_adjust_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_adjust_%d", i))
                ];
                net = addLayers(net,Encoder);
             
                
                % Decoder-side layers (moved inside loop; removed stray end)
                Decoder = [

                    fullyConnectedLayer(fc_layer_size*channels_conv2, "Name", sprintf("fc_latent_revert_%d", i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_latent_revert_%d", i), 6.0)
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[fc_layer_size,1,channels_conv2]);

                    % Decoder-side symmetric dilated block (size-preserving)
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil_dec1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor)
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dil_dec1_%d", i), 6.0)
                    
                    % Convolutional upsampling instead of maxunpooling
                    % transposedConv2dLayer(kernel_pool2,channels_conv2,"Name",sprintf("conv_upsample_%d",i),"Cropping","same","Stride", stride_pool2)

                    % Reverting fourth convolution
                    transposedConv2dLayer(kernel_conv4,channels_conv1,"Name",sprintf("transposed-conv_4_%d",i),"Cropping",crop2,"Stride", stride2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_3_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_3_%d", i), 6.0)

                    % Reverting third convolution
                    transposedConv2dLayer(kernel_conv3,channels_conv1,"Name",sprintf("transposed-conv_3_%d",i),"Cropping",crop2,"Stride", stride1_2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_4_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_4_%d", i), 6.0)

                    % Reverting second convolution
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping",crop2,"Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_5_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_5_%d", i), 6.0)

                    % Reverting first convolution
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_1_%d",i),"Cropping",crop1,"Stride", stride1,"BiasInitializer","zeros","BiasLearnRateFactor",0)
                    architectures_container.helper_scaled_tanh(sprintf("tanh_bound_%d", i), 6.0)
                ];
                net = addLayers(net,Decoder);
                net = connectLayers(net, sprintf("elu_latent_adjust_%d", i), sprintf("fc_latent_revert_%d", i));
               
            end
            
            % clean up helper variable
            clear Encoder Decoder;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
        end


        function net = buildOptimizedNetwork_compressed_downsampled_sin_freq1(params)
    
            net = dlnetwork;
            
            % Input dimensions
            input_x = 800;
            input_y = 2;
            input_z = 1;
            num_in = 28;
            desired_latent_size = 15;

            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size1,2];
            kernel_conv3 = [params.kernel_size2,2];
            kernel_conv4 = [params.kernel_size3,2];

            channels_conv1 = params.channels1;
            channels_conv2 = params.channels2;
            
            % Define possible group sizes (must be divisors of numChannels)
            possibleGroupSizes1 = 1:channels_conv1;
            validGroupSizes1 = possibleGroupSizes1(mod(channels_conv1, possibleGroupSizes1) == 0);
            group_size1 = validGroupSizes1(ceil(length(validGroupSizes1)/2));

            possibleGroupSizes2 = 1:channels_conv2;
            validGroupSizes2 = possibleGroupSizes2(mod(channels_conv2, possibleGroupSizes2) == 0);
            group_size2 = validGroupSizes2(ceil(length(validGroupSizes2)/2));

            stride1        = [5 1];  
            stride1_2      = [4 1];  
            stride2        = [4 2];
            
            fc_layer_size = input_x / (stride1(1)^2 * stride1_2(1) * stride2(1));
            disp(['Fully connected layer size: ', num2str(fc_layer_size)]);
            
            DilationFactor = [params.dilation_factor 1];
            dilatation_kernel = [3 1];

            crop1 = 'same';
            crop2 = 'same';
            
            disp(" I am about to go into the loop\n");
            for i=1:num_in

                % Build encoder layers
                Encoder = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    
                    % First conv block - Use leakyReLU instead of ELU for better gradient flow
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i), ...
                        "Padding","same","Stride", stride1, ...
                        "WeightsInitializer","he")  % He initialization for ReLU-like activations
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_enc_1_%d",i))  % LeakyReLU instead of ELU
            
                    % Second conv block
                    convolution2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("conv_2_%d",i), ...
                        "Padding","same", "Stride", stride1, ...
                        "WeightsInitializer","he")
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_2_enc_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_enc_2_%d",i))

                    % Third conv block
                    convolution2dLayer(kernel_conv3,channels_conv1,"Name",sprintf("conv_3_%d",i), ...
                        "Padding","same", "Stride", stride1_2, ...
                        "WeightsInitializer","he")
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_3_enc_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_enc_3_%d",i))

                    % Fourth conv block
                    convolution2dLayer(kernel_conv4,channels_conv2,"Name",sprintf("conv_4_%d",i), ...
                        "Padding","same", "Stride", stride2, ...
                        "WeightsInitializer","he")
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_4_enc_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_enc_4_%d",i))

                    % Temporal dilated block in bottleneck
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor, ...
                        "WeightsInitializer","he")
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_dil1_%d",i))
                    
                    % Flatten to vector
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    
                    % Latent space with smaller learning rate factor for stability
                    fullyConnectedLayer(desired_latent_size, ...
                        "Name", sprintf("fc_latent_adjust_%d", i), ...
                        "WeightsInitializer","he", ...
                        "WeightLearnRateFactor", 0.5)  % Reduce learning rate for bottleneck
                ];
                net = addLayers(net,Encoder);
                
                % Decoder-side layers
                Decoder = [
                    % Expand from latent space
                    fullyConnectedLayer(fc_layer_size*channels_conv2, ...
                        "Name", sprintf("fc_latent_revert_%d", i), ...
                        "WeightsInitializer","he", ...
                        "WeightLearnRateFactor", 0.5)  % Match encoder bottleneck
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_latent_revert_%d",i))
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[fc_layer_size,1,channels_conv2])

                    % Decoder-side symmetric dilated block
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil_dec1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor, ...
                        "WeightsInitializer","he")
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_dil_dec1_%d",i))
                    
                    % Reverting fourth convolution
                    transposedConv2dLayer(kernel_conv4,channels_conv1, ...
                        "Name",sprintf("transposed-conv_4_%d",i), ...
                        "Cropping",crop2,"Stride", stride2, ...
                        "WeightsInitializer","he")
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_3_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_dec_3_%d",i))

                    % Reverting third convolution
                    transposedConv2dLayer(kernel_conv3,channels_conv1, ...
                        "Name",sprintf("transposed-conv_3_%d",i), ...
                        "Cropping",crop2,"Stride", stride1_2, ...
                        "WeightsInitializer","he")
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_4_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_dec_4_%d",i))

                    % Reverting second convolution
                    transposedConv2dLayer(kernel_conv2,channels_conv1, ...
                        "Name",sprintf("transposed-conv_2_%d",i), ...
                        "Cropping",crop2,"Stride", stride1, ...
                        "WeightsInitializer","he")
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_5_%d",i))
                    leakyReluLayer(0.2,"Name",sprintf("lrelu_dec_5_%d",i))

                    % Final layer - CRITICAL FIXES HERE
                    transposedConv2dLayer(kernel_conv1,input_z, ...
                        "Name",sprintf("transposed-conv_1_%d",i), ...
                        "Cropping",crop1,"Stride", stride1, ...
                        "WeightsInitializer","he")  % Removed bias restrictions
                    % Use tanh only if your output needs to be bounded to [-1, 1]
                    % Otherwise remove this activation entirely
                    tanhLayer("Name",sprintf("tanh_output_%d",i))
                ];
                net = addLayers(net,Decoder);
                net = connectLayers(net, sprintf("fc_latent_adjust_%d", i), sprintf("fc_latent_revert_%d", i));
            
            end
            
            % clean up helper variable
            clear Encoder Decoder;
            
            % Initialize network
            net = initialize(net);
        end

        function net = buildOptimizedNetwork_compressed_downsampled_sin_freq_2(params)
    
            net = dlnetwork;
            
            % Input dimensions
            input_x = 800;
            input_y = 2;
            input_z = 1;
            num_in = 28;
            desired_latent_size = 15;

            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size1,2];
            kernel_conv3 = [params.kernel_size2,2];
            kernel_conv4 = [params.kernel_size3,2];

            channels_conv1 = params.channels1;
            channels_conv2 = params.channels2;
            
            % Define possible group sizes (must be divisors of numChannels)
            possibleGroupSizes1 = 1:channels_conv1;
            validGroupSizes1 = possibleGroupSizes1(mod(channels_conv1, possibleGroupSizes1) == 0);
            group_size1 = validGroupSizes1(ceil(length(validGroupSizes1)/2));

            possibleGroupSizes2 = 1:channels_conv2;
            validGroupSizes2 = possibleGroupSizes2(mod(channels_conv2, possibleGroupSizes2) == 0);
            group_size2 = validGroupSizes2(ceil(length(validGroupSizes2)/2));

            stride1        = [5 1];  
            stride1_2      = [4 1];  
            stride2        = [4 2];
            
            fc_layer_size = input_x / (stride1(1)^2 * stride1_2(1) * stride2(1));
            disp(['Fully connected layer size: ', num2str(fc_layer_size)]);
            
            DilationFactor = [params.dilation_factor 1];
            dilatation_kernel = [3 1];

            crop1 = 'same';
            crop2 = 'same';
            
            % IMPROVEMENT 1: Add dropout rate parameter
            dropout_rate = 0.1;  % Light dropout for regularization
            
            disp(" I am about to go into the loop\n");
            for i=1:num_in

                % Build encoder layers
                Encoder = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    
                    % IMPROVEMENT 2: Mix ELU and LeakyReLU for better gradient flow
                    % First conv block - ELU for smooth gradients
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i), ...
                        "Padding","same","Stride", stride1, ...
                        "WeightsInitializer","glorot")  % Glorot for ELU
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_1_%d",i))
                    dropoutLayer(dropout_rate,"Name",sprintf("dropout_enc_1_%d",i))
            
                    % Second conv block
                    convolution2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("conv_2_%d",i), ...
                        "Padding","same", "Stride", stride1, ...
                        "WeightsInitializer","glorot")
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_2_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_2_%d",i))
                    dropoutLayer(dropout_rate,"Name",sprintf("dropout_enc_2_%d",i))

                    % Third conv block
                    convolution2dLayer(kernel_conv3,channels_conv1,"Name",sprintf("conv_3_%d",i), ...
                        "Padding","same", "Stride", stride1_2, ...
                        "WeightsInitializer","glorot")
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_3_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_3_%d",i))
                    dropoutLayer(dropout_rate,"Name",sprintf("dropout_enc_3_%d",i))

                    % Fourth conv block
                    convolution2dLayer(kernel_conv4,channels_conv2,"Name",sprintf("conv_4_%d",i), ...
                        "Padding","same", "Stride", stride2, ...
                        "WeightsInitializer","glorot")
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_4_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_4_%d",i))
                    dropoutLayer(dropout_rate,"Name",sprintf("dropout_enc_4_%d",i))

                    % Temporal dilated block in bottleneck
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor, ...
                        "WeightsInitializer","glorot")
                    architectures_container.helperELU(sprintf("elu_dil1_%d",i))
                    
                    % Flatten to vector
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    
                    % IMPROVEMENT 3: Latent space with careful initialization and no scaling
                    fullyConnectedLayer(desired_latent_size, ...
                        "Name", sprintf("fc_latent_adjust_%d", i), ...
                        "WeightsInitializer","narrow-normal")  % Smaller initial weights
                    % No activation here to allow full latent space exploration
                ];
                net = addLayers(net,Encoder);
                
                % Decoder-side layers
                Decoder = [
                    % IMPROVEMENT 4: Expand from latent space with careful scaling
                    fullyConnectedLayer(fc_layer_size*channels_conv2, ...
                        "Name", sprintf("fc_latent_revert_%d", i), ...
                        "WeightsInitializer","narrow-normal")
                    architectures_container.helperELU(sprintf("elu_latent_revert_%d",i))
                    dropoutLayer(dropout_rate,"Name",sprintf("dropout_latent_%d",i))
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[fc_layer_size,1,channels_conv2])

                    % Decoder-side symmetric dilated block
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil_dec1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor, ...
                        "WeightsInitializer","glorot")
                    architectures_container.helperELU(sprintf("elu_dil_dec1_%d",i))
                    dropoutLayer(dropout_rate,"Name",sprintf("dropout_dil_%d",i))
                    
                    % IMPROVEMENT 5: Use scaled tanh in decoder for bounded outputs
                    % Reverting fourth convolution
                    transposedConv2dLayer(kernel_conv4,channels_conv1, ...
                        "Name",sprintf("transposed-conv_4_%d",i), ...
                        "Cropping",crop2,"Stride", stride2, ...
                        "WeightsInitializer","glorot")
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_3_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_3_%d", i), 4.0)
                    dropoutLayer(dropout_rate,"Name",sprintf("dropout_dec_3_%d",i))

                    % Reverting third convolution
                    transposedConv2dLayer(kernel_conv3,channels_conv1, ...
                        "Name",sprintf("transposed-conv_3_%d",i), ...
                        "Cropping",crop2,"Stride", stride1_2, ...
                        "WeightsInitializer","glorot")
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_4_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_4_%d", i), 4.0)
                    dropoutLayer(dropout_rate,"Name",sprintf("dropout_dec_4_%d",i))

                    % Reverting second convolution
                    transposedConv2dLayer(kernel_conv2,channels_conv1, ...
                        "Name",sprintf("transposed-conv_2_%d",i), ...
                        "Cropping",crop2,"Stride", stride1, ...
                        "WeightsInitializer","glorot")
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_5_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_5_%d", i), 4.0)
                    dropoutLayer(dropout_rate,"Name",sprintf("dropout_dec_5_%d",i))

                    % IMPROVEMENT 6: Final layer with proper initialization
                    transposedConv2dLayer(kernel_conv1,input_z, ...
                        "Name",sprintf("transposed-conv_1_%d",i), ...
                        "Cropping",crop1,"Stride", stride1, ...
                        "WeightsInitializer","narrow-normal", ...
                        "BiasInitializer","zeros")
                    % Final bounded activation
                    architectures_container.helper_scaled_tanh(sprintf("tanh_output_%d", i), 6.0)
                ];
                net = addLayers(net,Decoder);
                net = connectLayers(net, sprintf("fc_latent_adjust_%d", i), sprintf("fc_latent_revert_%d", i));
            
            end
            
            % clean up helper variable
            clear Encoder Decoder;
            
            % Initialize network
            net = initialize(net);
        end
            

        function net = buildOptimizedNetwork_compressed_downsampled_sin_freq_RES(params)
            % --- Defaults ---
            if ~isfield(params,'input_x'), params.input_x = 800; end
            if ~isfield(params,'input_y'), params.input_y = 2; end
            if ~isfield(params,'input_z'), params.input_z = 1; end
            if ~isfield(params,'num_in'), params.num_in = 1; end
            if ~isfield(params,'desired_latent_size'), params.desired_latent_size = 15; end

            if ~isfield(params,'kernel_size1'), params.kernel_size1 = 3; end
            if ~isfield(params,'kernel_size2'), params.kernel_size2 = 3; end
            if ~isfield(params,'kernel_size3'), params.kernel_size3 = 3; end

            if ~isfield(params,'channels1'), params.channels1 = 10; end
            if ~isfield(params,'channels2'), params.channels2 = 20; end

            if ~isfield(params,'dilation_factor'), params.dilation_factor = 2; end

            input_x = params.input_x;
            input_y = params.input_y;
            input_z = params.input_z;
            num_in = params.num_in;
            desired_latent_size = params.desired_latent_size;

            % Make channels at least modest
            channels_conv1 = max(params.channels1, 10);
            channels_conv2 = max(params.channels2, 20);
            net = dlnetwork;


            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size1,2];
            kernel_conv3 = [params.kernel_size2,2];
            kernel_conv4 = [params.kernel_size3,2];
        

           
            % Define possible group sizes (must be divisors of numChannels)
            possibleGroupSizes1 = 1:channels_conv1; % All integers from 1 to numChannels
            validGroupSizes1 = possibleGroupSizes1(mod(channels_conv1, possibleGroupSizes1) == 0);

            group_size1 = validGroupSizes1(ceil(length(validGroupSizes1)/2)); % Choose the middle valid size for consistency

            possibleGroupSizes2 = 1:channels_conv2; % All integers from 1 to numChannels
            validGroupSizes2 = possibleGroupSizes2(mod(channels_conv2, possibleGroupSizes2) == 0);

            group_size2 = validGroupSizes2(ceil(length(validGroupSizes2)/2)); % Choose the middle valid size for consistency


            group_size1 = 1;
            group_size2 = 1;
            % first frequency is  50 kHz, sampling rate is 2e6 Hz, therefore 40 samples per period, stride 20 seems reasonable
            stride1        = [2 1];  
            stride1_2      = [2 1];  
            stride2        = [4 2];
            
            fc_layer_size = input_x / (stride1(1)^2 * stride1_2(1) * stride2(1)); % there will be 4 downsampling operations and one chnaging the dim of the second axis
            disp(['Fully connected layer size: ', num2str(fc_layer_size)]);
            
            DilationFactor = [params.dilation_factor 1];
            dilatation_kernel = [3 1];

            crop1 = 'same';
            crop2 = 'same';
            
           
            disp(" I am about to go into the loop\n");
            for i=1:num_in
                in = inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i));
                net = addLayers(net,in);

                
                net = addLayers(net,convolution2dLayer([1 1], channels_conv1, "Name", sprintf("res_conv1_%d", i), "Padding", "same", "Stride", stride1));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv1, "Name", sprintf("res_conv2_%d", i), "Padding", "same", "Stride", stride1.^3));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv3_%d", i), "Padding", "same", "Stride", (stride1.^3) .* stride2));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv4_%d", i), "Padding", "same", "Stride", (stride1.^3) .* stride2));

                net = addLayers(net,convolution2dLayer([1 1], channels_conv1, "Name", sprintf("res_conv1_2_%d", i), "Padding", "same", "Stride", stride1.^2));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv1_3_%d", i), "Padding", "same", "Stride", (stride1.^2) .* stride2));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv1_4_%d", i), "Padding", "same", "Stride", (stride1.^2) .* stride2));
                
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv2_3_%d", i), "Padding", "same", "Stride", stride2));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv2_4_%d", i), "Padding", "same", "Stride", stride2));

                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv3_4_%d", i), "Padding", "same"));
                % Build encoder layers and append optional latent-adjust layers outside the array literal
                Encoder = [
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same","Stride", stride1, "WeightsInitializer","he", 'BiasInitializer','zeros')  % He initialization for ReLU-like activations
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_1_%d",i))
                    additionLayer(2,"Name",sprintf("res_add_0_%d",i))
            
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride1, "WeightsInitializer","he", 'BiasInitializer','zeros')
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_2_enc_%d",i))

                    % max pooling layer instead of conv layer for downsampling
                    maxPooling2dLayer(stride1,"Name",sprintf("maxpool_1_%d",i),"Padding","same","Stride", stride1,"HasUnpoolingOutputs",true)

                    additionLayer(3,"Name",sprintf("res_add_1_%d",i))

                    % Fourth conv block with normalization and res connections
                    convolution2dLayer(kernel_conv4,channels_conv2,"Name",sprintf("conv_4_%d",i),"Padding","same", "Stride", stride2, "WeightsInitializer","he", 'BiasInitializer','zeros')
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_4_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_4_%d",i))
                    additionLayer(4,"Name",sprintf("res_add_2_%d",i))

                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer(dilatation_kernel, channels_conv2, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", DilationFactor, "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU( sprintf("elu_dil1_%d", i))
                    additionLayer(5,"Name",sprintf("res_add_3_%d",i))
                    
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    % use global1daveragepooling instead of flatten
                    %globalAveragePooling2dLayer("Name",sprintf("global_avg_pool_%d",i))
                    %architectures_container.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[channels_conv2,1])
                    fullyConnectedLayer(floor(fc_layer_size*channels_conv2/2), "Name", sprintf("fc_latent_pre_adjust_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_pre_adjust_%d", i))
                    fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_adjust_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_adjust_%d", i))
                ];
                net = addLayers(net,Encoder);
                net = connectLayers(net, sprintf("input_%d", i), sprintf("conv_1_%d", i));
                
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv1_%d", i));
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv2_%d", i));
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv3_%d", i));
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv4_%d", i));

                net = connectLayers(net, sprintf("res_conv1_%d", i), sprintf("res_add_0_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv2_%d", i), sprintf("res_add_1_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv3_%d", i), sprintf("res_add_2_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv4_%d", i), sprintf("res_add_3_%d/in2", i));

                net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("res_conv1_2_%d", i));
                net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("res_conv1_3_%d", i));
                net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("res_conv1_4_%d", i));

                net = connectLayers(net, sprintf("res_conv1_2_%d", i), sprintf("res_add_1_%d/in3", i));
                net = connectLayers(net, sprintf("res_conv1_3_%d", i), sprintf("res_add_2_%d/in3", i));
                net = connectLayers(net, sprintf("res_conv1_4_%d", i), sprintf("res_add_3_%d/in3", i));

                net = connectLayers(net, sprintf("maxpool_1_%d/out",i), sprintf("res_conv2_3_%d", i));
                net = connectLayers(net, sprintf("maxpool_1_%d/out",i), sprintf("res_conv2_4_%d", i));

                net = connectLayers(net, sprintf("res_conv2_3_%d", i), sprintf("res_add_2_%d/in4", i));
                net = connectLayers(net, sprintf("res_conv2_4_%d", i), sprintf("res_add_3_%d/in4", i));

                net = connectLayers(net, sprintf("elu_enc_4_%d", i), sprintf("res_conv3_4_%d", i));

                net = connectLayers(net, sprintf("res_conv3_4_%d", i), sprintf("res_add_3_%d/in5", i));

                
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv2, "Name", sprintf("res_b_1_%d", i)));
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_b_2_%d", i),'Cropping',"same","Stride", stride2 .* stride1));
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_b_3_%d", i),'Cropping',"same","Stride", stride1.^2 .* stride2));
                net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_b_4_%d", i),'Cropping',"same","Stride", stride1.^3 .* stride2));

                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_tc1_2_%d", i), "Cropping", "same", "Stride", stride2 .* stride1));
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_tc1_3_%d", i), "Cropping", "same", "Stride", stride1.^2 .* stride2));
                net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_tc1_4_%d", i), "Cropping", "same", "Stride", stride1.^3 .* stride2));

                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_tc2_3_%d", i), "Cropping", "same", "Stride", stride1));
                net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_tc2_4_%d", i), "Cropping", "same", "Stride", stride1.^2));

                net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_tc3_4_%d", i), "Cropping", "same", "Stride", stride1));

                
                % Decoder-side layers (moved inside loop; removed stray end)
                Decoder = [
                    fullyConnectedLayer(floor(fc_layer_size*channels_conv2/2), "Name", sprintf("fc_latent_pre_revert_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_pre_revert_%d", i))
                    fullyConnectedLayer(fc_layer_size*channels_conv2, "Name", sprintf("fc_latent_revert_%d", i))
                    architectures_container.helperELU( sprintf("elu_dec_1_%d", i))
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[fc_layer_size,1,channels_conv2]);

                    % Decoder-side symmetric dilated block (size-preserving)
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil_dec1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_dil_dec1_%d",i))
                    additionLayer(2,"Name",sprintf("res_add_dec1_%d",i))
                    
                    % Convolutional upsampling instead of maxunpooling
                    % transposedConv2dLayer(kernel_pool2,channels_conv2,"Name",sprintf("conv_upsample_%d",i),"Cropping","same","Stride", stride_pool2)

                    % Reverting fourth convolution
                    transposedConv2dLayer(kernel_conv4,channels_conv1,"Name",sprintf("transposed-conv_4_%d",i),"Cropping",crop2,"Stride", stride2, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_3_%d",i))
                    %additionLayer(2,"Name",sprintf("enc_dec_%d",i))
                    % Unpooling layer to revert maxpooling
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))
                    additionLayer(3,"Name",sprintf("res_add_dec2_%d",i))
                    

                    % Reverting second convolution
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping",crop2,"Stride", stride1, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_5_%d",i))
                    architectures_container.helperELU(sprintf("elu_dec_5_%d", i))
                    additionLayer(4,"Name",sprintf("res_add_dec3_%d",i))

                    % Reverting first convolution
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_1_%d",i),"Cropping",crop1,"Stride", stride1, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    additionLayer(5,"Name",sprintf("res_add_dec4_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_6_%d", i), 1.0)

                ];
                net = addLayers(net,Decoder);
                net = connectLayers(net, sprintf("elu_latent_adjust_%d", i), sprintf("fc_latent_pre_revert_%d", i));
                net = connectLayers(net, sprintf("maxpool_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,   sprintf("maxpool_1_%d/size",i), sprintf("maxunpool_1_%d/size",i));

                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_1_%d", i));
                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_2_%d", i));
                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_3_%d", i));
                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_4_%d", i));

                net = connectLayers(net, sprintf("res_b_1_%d", i), sprintf("res_add_dec1_%d/in2",i));
                net = connectLayers(net, sprintf("res_b_2_%d", i), sprintf("res_add_dec2_%d/in2",i));
                net = connectLayers(net, sprintf("res_b_3_%d", i), sprintf("res_add_dec3_%d/in2",i));
                net = connectLayers(net, sprintf("res_b_4_%d", i), sprintf("res_add_dec4_%d/in2",i));

                net = connectLayers(net, sprintf("elu_dil_dec1_%d",i), sprintf("res_tc1_2_%d", i));
                net = connectLayers(net, sprintf("elu_dil_dec1_%d",i), sprintf("res_tc1_3_%d", i));
                net = connectLayers(net, sprintf("elu_dil_dec1_%d",i), sprintf("res_tc1_4_%d", i));

                net = connectLayers(net, sprintf("res_tc1_2_%d", i), sprintf("res_add_dec2_%d/in3", i));
                net = connectLayers(net, sprintf("res_tc1_3_%d", i), sprintf("res_add_dec3_%d/in3", i));
                net = connectLayers(net, sprintf("res_tc1_4_%d", i), sprintf("res_add_dec4_%d/in3", i));

                net = connectLayers(net, sprintf("maxunpool_1_%d",i), sprintf("res_tc2_3_%d", i));
                net = connectLayers(net, sprintf("maxunpool_1_%d",i), sprintf("res_tc2_4_%d", i));

                net = connectLayers(net, sprintf("res_tc2_3_%d", i), sprintf("res_add_dec3_%d/in4", i));
                net = connectLayers(net, sprintf("res_tc2_4_%d", i), sprintf("res_add_dec4_%d/in4", i));

                net = connectLayers(net, sprintf("elu_dec_5_%d", i), sprintf("res_tc3_4_%d", i));

                net = connectLayers(net, sprintf("res_tc3_4_%d", i), sprintf("res_add_dec4_%d/in5", i));

                %net = connectLayers(net,sprintf("maxpool_1_%d/out",i),sprintf("enc_dec_%d/in2", i));


               
            end
            
            % clean up helper variable
            clear Encoder Decoder;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
        end

        function net = deep_fully_connected_network(params)
            % --- Defaults ---
            if ~isfield(params,'input_size'), params.input_size = 4000; end
            if ~isfield(params,'num_in'), params.num_in = 1; end
            if ~isfield(params,'desired_latent_size'), params.desired_latent_size = 15; end
            if ~isfield(params,'drop_rate'), params.drop_rate = 0.1; end
            if ~isfield(params,'k_sparse'), params.k_sparse = 10; end
            

            input_size = params.input_size;
            num_in = params.num_in;
            desired_latent_size = params.desired_latent_size;
            hidden_layer_size1 = params.hidden_layer_size1;
            hidden_layer_size2 = params.hidden_layer_size2;
            hidden_layer_size3 = params.hidden_layer_size3;
            hidden_layer_size4 = params.hidden_layer_size4;
            drop_rate = params.drop_rate;

            net = dlnetwork;

            for i=1:num_in
                in = inputLayer([NaN, input_size],"BC", "Name",sprintf("input_%d",i));
                net = addLayers(net,in);

                encoder = [
                    fullyConnectedLayer(hidden_layer_size1, "Name", sprintf("fc_1_%d", i), "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_1_%d", i))
                    batchNormalizationLayer("Name",sprintf("bn_1_%d",i))
                    dropoutLayer(drop_rate,"Name",sprintf("dropout_1_%d",i))

                    fullyConnectedLayer(hidden_layer_size2, "Name", sprintf("fc_2_%d", i), "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_2_%d", i))
                    batchNormalizationLayer("Name",sprintf("bn_2_%d",i))
                    dropoutLayer(drop_rate,"Name",sprintf("dropout_2_%d",i))

                    fullyConnectedLayer(hidden_layer_size3, "Name", sprintf("fc_3_%d", i), "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_3_%d", i))
                    batchNormalizationLayer("Name",sprintf("bn_3_%d",i))
                    dropoutLayer(drop_rate,"Name",sprintf("dropout_3_%d",i))

                    fullyConnectedLayer(hidden_layer_size4, "Name", sprintf("fc_4_%d", i), "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_4_%d", i))
                    batchNormalizationLayer("Name",sprintf("bn_4_%d",i))
                    dropoutLayer(drop_rate,"Name",sprintf("dropout_4_%d",i))

                    fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_%d", i), "WeightsInitializer","narrow-normal")
                    kSparseLayer(params.k_sparse, sprintf("k_sparse_%d", i))
                ];
                

                decoder = [
                    fullyConnectedLayer(hidden_layer_size4, "Name", sprintf("fc_dec_4_%d", i), "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_dec_4_%d", i))
                    batchNormalizationLayer("Name",sprintf("bn_dec_4_%d",i))
                    dropoutLayer(drop_rate,"Name",sprintf("dropout_dec_4_%d",i))

                    fullyConnectedLayer(hidden_layer_size3, "Name", sprintf("fc_dec_3_%d", i), "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_dec_3_%d", i))
                    batchNormalizationLayer("Name",sprintf("bn_dec_3_%d",i))
                    dropoutLayer(drop_rate,"Name",sprintf("dropout_dec_3_%d",i))

                    fullyConnectedLayer(hidden_layer_size2, "Name", sprintf("fc_dec_2_%d", i), "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_dec_2_%d", i))
                    batchNormalizationLayer("Name",sprintf("bn_dec_2_%d",i))
                    dropoutLayer(drop_rate,"Name",sprintf("dropout_dec_2_%d",i))

                    fullyConnectedLayer(hidden_layer_size1, "Name", sprintf("fc_dec_1_%d", i), "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_dec_1_%d", i))
                    batchNormalizationLayer("Name",sprintf("bn_dec_1_%d",i))
                    dropoutLayer(drop_rate,"Name",sprintf("dropout_dec_1_%d",i))

                    fullyConnectedLayer(input_size, "Name", sprintf("fc_output_%d", i), "WeightsInitializer","narrow-normal")
                ];
                net = addLayers(net,encoder);
                net = addLayers(net,decoder);

                net = connectLayers(net, sprintf("input_%d", i), sprintf("fc_1_%d", i));
                net = connectLayers(net, sprintf("k_sparse_%d", i), sprintf("fc_dec_4_%d", i));

            end

            % Initialize network
            net = initialize(net);
        end


        function net = buildOptimizedNetwork_compressed_downsampled_sin_freq_less_RES_bn_dr(params)
            % --- Defaults ---
            if ~isfield(params,'input_x'), params.input_x = 800; end
            if ~isfield(params,'input_y'), params.input_y = 2; end
            if ~isfield(params,'input_z'), params.input_z = 1; end
            if ~isfield(params,'num_in'), params.num_in = 1; end
            if ~isfield(params,'desired_latent_size'), params.desired_latent_size = 15; end

            if ~isfield(params,'kernel_size1'), params.kernel_size1 = 3; end
            if ~isfield(params,'kernel_size2'), params.kernel_size2 = 3; end
            if ~isfield(params,'kernel_size3'), params.kernel_size3 = 3; end

            if ~isfield(params,'channels1'), params.channels1 = 10; end
            if ~isfield(params,'channels2'), params.channels2 = 20; end

            if ~isfield(params,'dilation_factor'), params.dilation_factor = 2; end

            if ~isfield(params,'dropout_rate_enc'), params.dropout_rate_enc = 0.2; end
            if ~isfield(params,'dropout_rate_dec'), params.dropout_rate_dec = 0.2; end

            if ~isfield(params,'benchmark_mode'), params.benchmark_mode = true; end

            input_x = params.input_x;
            input_y = params.input_y;
            input_z = params.input_z;
            num_in = params.num_in;
            desired_latent_size = params.desired_latent_size;

            % Make channels at least modest
            channels_conv1 = max(params.channels1, 10);
            channels_conv2 = max(params.channels2, 20);
            net = dlnetwork;


            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size1,2];
            kernel_conv3 = [params.kernel_size2,2];
            kernel_conv4 = [params.kernel_size3,2];
            dropout_rate_enc = params.dropout_rate_enc;
            dropout_rate_dec = params.dropout_rate_dec;
            
            % first frequency is  50 kHz, sampling rate is 2e6 Hz, therefore 40 samples per period, stride 20 seems reasonable
            stride1        = [2 1];  
            stride1_2      = [2 1]; 
            if input_y > 1
                stride2        = [4 2];
            else
                stride2        = [4 1];
            end
           
            
            fc_layer_size = input_x / (stride1(1)^2 * stride1_2(1) * stride2(1)); % there will be 4 downsampling operations and one chnaging the dim of the second axis
            disp(['Fully connected layer size: ', num2str(fc_layer_size)]);
            
            DilationFactor = [params.dilation_factor 1];
            dilatation_kernel = [3 1];

            crop1 = 'same';
            crop2 = 'same';
            
           
            disp(" I am about to go into the loop\n");
            for i=1:num_in

                if input_y > 1
                    in = inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i));
                else
                    in = inputLayer([NaN, input_x, 1, input_z],"BSSC", "Name",sprintf("input_%d",i));
                    disp("Using single-channel input format");
                end
                net = addLayers(net,in);

                
                net = addLayers(net,convolution2dLayer([1 1], channels_conv1, "Name", sprintf("res_conv1_%d", i), "Padding", "same", "Stride", stride1));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv1, "Name", sprintf("res_conv2_%d", i), "Padding", "same", "Stride", stride1.^3));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv3_%d", i), "Padding", "same", "Stride", (stride1.^3) .* stride2));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv4_%d", i), "Padding", "same", "Stride", (stride1.^3) .* stride2));

                net = addLayers(net,convolution2dLayer([1 1], channels_conv1, "Name", sprintf("res_conv1_2_%d", i), "Padding", "same", "Stride", stride1.^2));
                % net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv1_3_%d", i), "Padding", "same", "Stride", (stride1.^2) .* stride2));
                % net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv1_4_%d", i), "Padding", "same", "Stride", (stride1.^2) .* stride2));
                
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv2_3_%d", i), "Padding", "same", "Stride", stride2));
                % net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv2_4_%d", i), "Padding", "same", "Stride", stride2));

                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv3_4_%d", i), "Padding", "same"));
                % Build encoder layers and append optional latent-adjust layers outside the array literal
                Encoder = [
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same","Stride", stride1, "WeightsInitializer","he", 'BiasInitializer','zeros')  % He initialization for ReLU-like activations
                    batchNormalizationLayer("Name",sprintf("bn_1_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_1_%d",i))
                    dropoutLayer(dropout_rate_enc,"Name",sprintf("dropout_enc_1_%d",i))
                    additionLayer(2,"Name",sprintf("res_add_0_%d",i))
            
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride1, "WeightsInitializer","he", 'BiasInitializer','zeros')
                    batchNormalizationLayer("Name",sprintf("bn_2_enc_%d",i))

                    % max pooling layer instead of conv layer for downsampling
                    maxPooling2dLayer(stride1,"Name",sprintf("maxpool_1_%d",i),"Padding","same","Stride", stride1,"HasUnpoolingOutputs",true)
                    dropoutLayer(dropout_rate_enc,"Name",sprintf("dropout_enc_2_%d",i))
                    additionLayer(3,"Name",sprintf("res_add_1_%d",i))

                    % Fourth conv block with normalization and res connections
                    convolution2dLayer(kernel_conv4,channels_conv2,"Name",sprintf("conv_4_%d",i),"Padding","same", "Stride", stride2, "WeightsInitializer","he", 'BiasInitializer','zeros')
                    batchNormalizationLayer("Name",sprintf("bn_4_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_4_%d",i))
                    dropoutLayer(dropout_rate_enc,"Name",sprintf("dropout_enc_4_%d",i))
                    additionLayer(3,"Name",sprintf("res_add_2_%d",i))

                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer(dilatation_kernel, channels_conv2, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", DilationFactor, "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU( sprintf("elu_dil1_%d", i))
                    dropoutLayer(dropout_rate_enc,"Name",sprintf("dropout_dil1_%d",i))
                    additionLayer(3,"Name",sprintf("res_add_3_%d",i))
                    
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    % use global1daveragepooling instead of flatten
                    %globalAveragePooling2dLayer("Name",sprintf("global_avg_pool_%d",i))
                    %architectures_container.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[channels_conv2,1])
                    fullyConnectedLayer(floor(fc_layer_size*channels_conv2/2), "Name", sprintf("fc_latent_pre_adjust_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_pre_adjust_%d", i))
                    dropoutLayer(dropout_rate_enc,"Name",sprintf("dropout_latent_pre_%d",i))
                    fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_adjust_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_adjust_%d", i))
                ];
                net = addLayers(net,Encoder);
                net = connectLayers(net, sprintf("input_%d", i), sprintf("conv_1_%d", i));
                
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv1_%d", i));
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv2_%d", i));
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv3_%d", i));
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv4_%d", i));

                net = connectLayers(net, sprintf("res_conv1_%d", i), sprintf("res_add_0_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv2_%d", i), sprintf("res_add_1_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv3_%d", i), sprintf("res_add_2_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv4_%d", i), sprintf("res_add_3_%d/in2", i));

                net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("res_conv1_2_%d", i));
                % net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("res_conv1_3_%d", i));
                % net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("res_conv1_4_%d", i));

                net = connectLayers(net, sprintf("res_conv1_2_%d", i), sprintf("res_add_1_%d/in3", i));
                % net = connectLayers(net, sprintf("res_conv1_3_%d", i), sprintf("res_add_2_%d/in3", i));
                % net = connectLayers(net, sprintf("res_conv1_4_%d", i), sprintf("res_add_3_%d/in3", i));

                net = connectLayers(net, sprintf("maxpool_1_%d/out",i), sprintf("res_conv2_3_%d", i));
                %net = connectLayers(net, sprintf("maxpool_1_%d/out",i), sprintf("res_conv2_4_%d", i));

                net = connectLayers(net, sprintf("res_conv2_3_%d", i), sprintf("res_add_2_%d/in3", i));
                %net = connectLayers(net, sprintf("res_conv2_4_%d", i), sprintf("res_add_3_%d/in4", i));

                net = connectLayers(net, sprintf("elu_enc_4_%d", i), sprintf("res_conv3_4_%d", i));

                net = connectLayers(net, sprintf("res_conv3_4_%d", i), sprintf("res_add_3_%d/in3", i));

                
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv2, "Name", sprintf("res_b_1_%d", i)));
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_b_2_%d", i),'Cropping',"same","Stride", stride2 .* stride1));
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_b_3_%d", i),'Cropping',"same","Stride", stride1.^2 .* stride2));
                net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_b_4_%d", i),'Cropping',"same","Stride", stride1.^3 .* stride2));

                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_tc1_2_%d", i), "Cropping", "same", "Stride", stride2 .* stride1));
                % net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_tc1_3_%d", i), "Cropping", "same", "Stride", stride1.^2 .* stride2));
                % net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_tc1_4_%d", i), "Cropping", "same", "Stride", stride1.^3 .* stride2));

                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_tc2_3_%d", i), "Cropping", "same", "Stride", stride1));
                % net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_tc2_4_%d", i), "Cropping", "same", "Stride", stride1.^2));

                net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_tc3_4_%d", i), "Cropping", "same", "Stride", stride1));

                
                % Decoder-side layers (moved inside loop; removed stray end)
                Decoder = [
                    fullyConnectedLayer(floor(fc_layer_size*channels_conv2/2), "Name", sprintf("fc_latent_pre_revert_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_pre_revert_%d", i))
                    dropoutLayer(dropout_rate_dec,"Name",sprintf("dropout_latent_pre_revert_%d",i))
                    fullyConnectedLayer(fc_layer_size*channels_conv2, "Name", sprintf("fc_latent_revert_%d", i))
                    architectures_container.helperELU( sprintf("elu_dec_1_%d", i))
                    dropoutLayer(dropout_rate_dec,"Name",sprintf("dropout_dec_1_%d",i))
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[fc_layer_size,1,channels_conv2]);

                    % Decoder-side symmetric dilated block (size-preserving)
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil_dec1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_dil_dec1_%d",i))
                    dropoutLayer(dropout_rate_dec,"Name",sprintf("dropout_dil_dec1_%d",i))
                    additionLayer(2,"Name",sprintf("res_add_dec1_%d",i))
                    
                    % Convolutional upsampling instead of maxunpooling
                    % transposedConv2dLayer(kernel_pool2,channels_conv2,"Name",sprintf("conv_upsample_%d",i),"Cropping","same","Stride", stride_pool2)

                    % Reverting fourth convolution
                    transposedConv2dLayer(kernel_conv4,channels_conv1,"Name",sprintf("transposed-conv_4_%d",i),"Cropping",crop2,"Stride", stride2, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    batchNormalizationLayer("Name",sprintf("bn_dec_3_%d",i))
                    %additionLayer(2,"Name",sprintf("enc_dec_%d",i))
                    % Unpooling layer to revert maxpooling
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))
                    dropoutLayer(dropout_rate_dec,"Name",sprintf("dropout_dec_3_%d",i))
                    additionLayer(3,"Name",sprintf("res_add_dec2_%d",i))
                    

                    % Reverting second convolution
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping",crop2,"Stride", stride1, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    batchNormalizationLayer("Name",sprintf("bn_dec_5_%d",i))
                    architectures_container.helperELU(sprintf("elu_dec_5_%d", i))
                    dropoutLayer(dropout_rate_dec,"Name",sprintf("dropout_dec_5_%d",i))
                    additionLayer(3,"Name",sprintf("res_add_dec3_%d",i))

                    % Reverting first convolution
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_1_%d",i),"Cropping",crop1,"Stride", stride1, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    additionLayer(3,"Name",sprintf("res_add_dec4_%d",i))
                    architectures_container.helper_scaled_tanh(sprintf("tanh_dec_6_%d", i),10)

                ];
                net = addLayers(net,Decoder);
                net = connectLayers(net, sprintf("elu_latent_adjust_%d", i), sprintf("fc_latent_pre_revert_%d", i));
                net = connectLayers(net, sprintf("maxpool_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,   sprintf("maxpool_1_%d/size",i), sprintf("maxunpool_1_%d/size",i));

                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_1_%d", i));
                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_2_%d", i));
                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_3_%d", i));
                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_4_%d", i));

                net = connectLayers(net, sprintf("res_b_1_%d", i), sprintf("res_add_dec1_%d/in2",i));
                net = connectLayers(net, sprintf("res_b_2_%d", i), sprintf("res_add_dec2_%d/in2",i));
                net = connectLayers(net, sprintf("res_b_3_%d", i), sprintf("res_add_dec3_%d/in2",i));
                net = connectLayers(net, sprintf("res_b_4_%d", i), sprintf("res_add_dec4_%d/in2",i));

                net = connectLayers(net, sprintf("elu_dil_dec1_%d",i), sprintf("res_tc1_2_%d", i));
                % net = connectLayers(net, sprintf("elu_dil_dec1_%d",i), sprintf("res_tc1_3_%d", i));
                % net = connectLayers(net, sprintf("elu_dil_dec1_%d",i), sprintf("res_tc1_4_%d", i));

                net = connectLayers(net, sprintf("res_tc1_2_%d", i), sprintf("res_add_dec2_%d/in3", i));
                % net = connectLayers(net, sprintf("res_tc1_3_%d", i), sprintf("res_add_dec3_%d/in3", i));
                % net = connectLayers(net, sprintf("res_tc1_4_%d", i), sprintf("res_add_dec4_%d/in3", i));

                net = connectLayers(net, sprintf("maxunpool_1_%d",i), sprintf("res_tc2_3_%d", i));
                % net = connectLayers(net, sprintf("maxunpool_1_%d",i), sprintf("res_tc2_4_%d", i));

                net = connectLayers(net, sprintf("res_tc2_3_%d", i), sprintf("res_add_dec3_%d/in3", i));
                % net = connectLayers(net, sprintf("res_tc2_4_%d", i), sprintf("res_add_dec4_%d/in4", i));

                net = connectLayers(net, sprintf("elu_dec_5_%d", i), sprintf("res_tc3_4_%d", i));

                net = connectLayers(net, sprintf("res_tc3_4_%d", i), sprintf("res_add_dec4_%d/in3", i));

                %net = connectLayers(net,sprintf("maxpool_1_%d/out",i),sprintf("enc_dec_%d/in2", i));


               
            end
            
            % clean up helper variable
            clear Encoder Decoder;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
        end

        function net = buildOptimizedNetwork_compressed_downsampled_sin_freq_RES_bn_dr(params)
            % --- Defaults ---
            if ~isfield(params,'input_x'), params.input_x = 800; end
            if ~isfield(params,'input_y'), params.input_y = 2; end
            if ~isfield(params,'input_z'), params.input_z = 1; end
            if ~isfield(params,'num_in'), params.num_in = 1; end
            if ~isfield(params,'desired_latent_size'), params.desired_latent_size = 15; end

            if ~isfield(params,'kernel_size1'), params.kernel_size1 = 3; end
            if ~isfield(params,'kernel_size2'), params.kernel_size2 = 3; end
            if ~isfield(params,'kernel_size3'), params.kernel_size3 = 3; end

            if ~isfield(params,'channels1'), params.channels1 = 10; end
            if ~isfield(params,'channels2'), params.channels2 = 20; end

            if ~isfield(params,'dilation_factor'), params.dilation_factor = 2; end

            if ~isfield(params,'dropout_rate_enc'), params.dropout_rate_enc = 0.2; end
            if ~isfield(params,'dropout_rate_dec'), params.dropout_rate_dec = 0.2; end

            input_x = params.input_x;
            input_y = params.input_y;
            input_z = params.input_z;
            num_in = params.num_in;
            desired_latent_size = params.desired_latent_size;

            % Make channels at least modest
            channels_conv1 = max(params.channels1, 10);
            channels_conv2 = max(params.channels2, 20);
            net = dlnetwork;


            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size1,2];
            kernel_conv3 = [params.kernel_size2,2];
            kernel_conv4 = [params.kernel_size3,2];
            dropout_rate_enc = params.dropout_rate_enc;
            dropout_rate_dec = params.dropout_rate_dec;
            
            % first frequency is  50 kHz, sampling rate is 2e6 Hz, therefore 40 samples per period, stride 20 seems reasonable
            stride1        = [2 1];  
            stride1_2      = [2 1];  
            stride2        = [4 2];
            
            fc_layer_size = input_x / (stride1(1)^2 * stride1_2(1) * stride2(1)); % there will be 4 downsampling operations and one chnaging the dim of the second axis
            disp(['Fully connected layer size: ', num2str(fc_layer_size)]);
            
            DilationFactor = [params.dilation_factor 1];
            dilatation_kernel = [3 1];

            crop1 = 'same';
            crop2 = 'same';
            
           
            disp(" I am about to go into the loop\n");
            for i=1:num_in
                in = inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i));
                net = addLayers(net,in);

                
                net = addLayers(net,convolution2dLayer([1 1], channels_conv1, "Name", sprintf("res_conv1_%d", i), "Padding", "same", "Stride", stride1));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv1, "Name", sprintf("res_conv2_%d", i), "Padding", "same", "Stride", stride1.^3));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv3_%d", i), "Padding", "same", "Stride", (stride1.^3) .* stride2));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv4_%d", i), "Padding", "same", "Stride", (stride1.^3) .* stride2));

                net = addLayers(net,convolution2dLayer([1 1], channels_conv1, "Name", sprintf("res_conv1_2_%d", i), "Padding", "same", "Stride", stride1.^2));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv1_3_%d", i), "Padding", "same", "Stride", (stride1.^2) .* stride2));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv1_4_%d", i), "Padding", "same", "Stride", (stride1.^2) .* stride2));
                
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv2_3_%d", i), "Padding", "same", "Stride", stride2));
                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv2_4_%d", i), "Padding", "same", "Stride", stride2));

                net = addLayers(net,convolution2dLayer([1 1], channels_conv2, "Name", sprintf("res_conv3_4_%d", i), "Padding", "same"));
                % Build encoder layers and append optional latent-adjust layers outside the array literal
                Encoder = [
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same","Stride", stride1, "WeightsInitializer","he", 'BiasInitializer','zeros')  % He initialization for ReLU-like activations
                    batchNormalizationLayer("Name",sprintf("bn_1_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_1_%d",i))
                    dropoutLayer(dropout_rate_enc,"Name",sprintf("dropout_enc_1_%d",i))
                    additionLayer(2,"Name",sprintf("res_add_0_%d",i))
            
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride1, "WeightsInitializer","he", 'BiasInitializer','zeros')
                    batchNormalizationLayer("Name",sprintf("bn_2_enc_%d",i))

                    % max pooling layer instead of conv layer for downsampling
                    maxPooling2dLayer(stride1,"Name",sprintf("maxpool_1_%d",i),"Padding","same","Stride", stride1,"HasUnpoolingOutputs",true)
                    dropoutLayer(dropout_rate_enc,"Name",sprintf("dropout_enc_2_%d",i))
                    additionLayer(3,"Name",sprintf("res_add_1_%d",i))

                    % Fourth conv block with normalization and res connections
                    convolution2dLayer(kernel_conv4,channels_conv2,"Name",sprintf("conv_4_%d",i),"Padding","same", "Stride", stride2, "WeightsInitializer","he", 'BiasInitializer','zeros')
                    batchNormalizationLayer("Name",sprintf("bn_4_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_4_%d",i))
                    dropoutLayer(dropout_rate_enc,"Name",sprintf("dropout_enc_4_%d",i))
                    additionLayer(4,"Name",sprintf("res_add_2_%d",i))

                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer(dilatation_kernel, channels_conv2, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", DilationFactor, "WeightsInitializer","he", 'BiasInitializer','zeros')
                    architectures_container.helperELU( sprintf("elu_dil1_%d", i))
                    dropoutLayer(dropout_rate_enc,"Name",sprintf("dropout_dil1_%d",i))
                    additionLayer(5,"Name",sprintf("res_add_3_%d",i))
                    
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    % use global1daveragepooling instead of flatten
                    %globalAveragePooling2dLayer("Name",sprintf("global_avg_pool_%d",i))
                    %architectures_container.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[channels_conv2,1])
                    fullyConnectedLayer(floor(fc_layer_size*channels_conv2/2), "Name", sprintf("fc_latent_pre_adjust_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_pre_adjust_%d", i))
                    dropoutLayer(dropout_rate_enc,"Name",sprintf("dropout_latent_pre_%d",i))
                    fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_adjust_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_adjust_%d", i))
                ];
                net = addLayers(net,Encoder);
                net = connectLayers(net, sprintf("input_%d", i), sprintf("conv_1_%d", i));
                
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv1_%d", i));
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv2_%d", i));
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv3_%d", i));
                net = connectLayers(net, sprintf("input_%d", i), sprintf("res_conv4_%d", i));

                net = connectLayers(net, sprintf("res_conv1_%d", i), sprintf("res_add_0_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv2_%d", i), sprintf("res_add_1_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv3_%d", i), sprintf("res_add_2_%d/in2", i));
                net = connectLayers(net, sprintf("res_conv4_%d", i), sprintf("res_add_3_%d/in2", i));

                net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("res_conv1_2_%d", i));
                net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("res_conv1_3_%d", i));
                net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("res_conv1_4_%d", i));

                net = connectLayers(net, sprintf("res_conv1_2_%d", i), sprintf("res_add_1_%d/in3", i));
                net = connectLayers(net, sprintf("res_conv1_3_%d", i), sprintf("res_add_2_%d/in3", i));
                net = connectLayers(net, sprintf("res_conv1_4_%d", i), sprintf("res_add_3_%d/in3", i));

                net = connectLayers(net, sprintf("maxpool_1_%d/out",i), sprintf("res_conv2_3_%d", i));
                net = connectLayers(net, sprintf("maxpool_1_%d/out",i), sprintf("res_conv2_4_%d", i));

                net = connectLayers(net, sprintf("res_conv2_3_%d", i), sprintf("res_add_2_%d/in4", i));
                net = connectLayers(net, sprintf("res_conv2_4_%d", i), sprintf("res_add_3_%d/in4", i));

                net = connectLayers(net, sprintf("elu_enc_4_%d", i), sprintf("res_conv3_4_%d", i));

                net = connectLayers(net, sprintf("res_conv3_4_%d", i), sprintf("res_add_3_%d/in5", i));

                
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv2, "Name", sprintf("res_b_1_%d", i)));
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_b_2_%d", i),'Cropping',"same","Stride", stride2 .* stride1));
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_b_3_%d", i),'Cropping',"same","Stride", stride1.^2 .* stride2));
                net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_b_4_%d", i),'Cropping',"same","Stride", stride1.^3 .* stride2));

                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_tc1_2_%d", i), "Cropping", "same", "Stride", stride2 .* stride1));
                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_tc1_3_%d", i), "Cropping", "same", "Stride", stride1.^2 .* stride2));
                net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_tc1_4_%d", i), "Cropping", "same", "Stride", stride1.^3 .* stride2));

                net = addLayers(net,transposedConv2dLayer([1 1], channels_conv1, "Name", sprintf("res_tc2_3_%d", i), "Cropping", "same", "Stride", stride1));
                net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_tc2_4_%d", i), "Cropping", "same", "Stride", stride1.^2));

                net = addLayers(net,transposedConv2dLayer([1 1], input_z, "Name", sprintf("res_tc3_4_%d", i), "Cropping", "same", "Stride", stride1));

                
                % Decoder-side layers (moved inside loop; removed stray end)
                Decoder = [
                    fullyConnectedLayer(floor(fc_layer_size*channels_conv2/2), "Name", sprintf("fc_latent_pre_revert_%d", i))
                    architectures_container.helperELU( sprintf("elu_latent_pre_revert_%d", i))
                    dropoutLayer(dropout_rate_dec,"Name",sprintf("dropout_latent_pre_revert_%d",i))
                    fullyConnectedLayer(fc_layer_size*channels_conv2, "Name", sprintf("fc_latent_revert_%d", i))
                    architectures_container.helperELU( sprintf("elu_dec_1_%d", i))
                    dropoutLayer(dropout_rate_dec,"Name",sprintf("dropout_dec_1_%d",i))
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[fc_layer_size,1,channels_conv2]);

                    % Decoder-side symmetric dilated block (size-preserving)
                    convolution2dLayer(dilatation_kernel, channels_conv2, ...
                        "Name", sprintf("conv_dil_dec1_%d", i), ...
                        "Padding", "same", "DilationFactor", DilationFactor, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    architectures_container.helperELU(sprintf("elu_dil_dec1_%d",i))
                    dropoutLayer(dropout_rate_dec,"Name",sprintf("dropout_dil_dec1_%d",i))
                    additionLayer(2,"Name",sprintf("res_add_dec1_%d",i))
                    
                    % Convolutional upsampling instead of maxunpooling
                    % transposedConv2dLayer(kernel_pool2,channels_conv2,"Name",sprintf("conv_upsample_%d",i),"Cropping","same","Stride", stride_pool2)

                    % Reverting fourth convolution
                    transposedConv2dLayer(kernel_conv4,channels_conv1,"Name",sprintf("transposed-conv_4_%d",i),"Cropping",crop2,"Stride", stride2, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    batchNormalizationLayer("Name",sprintf("bn_dec_3_%d",i))
                    %additionLayer(2,"Name",sprintf("enc_dec_%d",i))
                    % Unpooling layer to revert maxpooling
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))
                    dropoutLayer(dropout_rate_dec,"Name",sprintf("dropout_dec_3_%d",i))
                    additionLayer(3,"Name",sprintf("res_add_dec2_%d",i))
                    

                    % Reverting second convolution
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping",crop2,"Stride", stride1, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    batchNormalizationLayer("Name",sprintf("bn_dec_5_%d",i))
                    architectures_container.helperELU(sprintf("elu_dec_5_%d", i))
                    dropoutLayer(dropout_rate_dec,"Name",sprintf("dropout_dec_5_%d",i))
                    additionLayer(4,"Name",sprintf("res_add_dec3_%d",i))

                    % Reverting first convolution
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_1_%d",i),"Cropping",crop1,"Stride", stride1, 'WeightsInitializer','he', 'BiasInitializer','zeros')
                    additionLayer(5,"Name",sprintf("res_add_dec4_%d",i))
                    % REMOVED Scaled Tanh to allow full amplitude range learning
                    % architectures_container.helper_scaled_tanh(sprintf("tanh_dec_6_%d", i),10)

                ];
                net = addLayers(net,Decoder);
                net = connectLayers(net, sprintf("elu_latent_adjust_%d", i), sprintf("fc_latent_pre_revert_%d", i));
                net = connectLayers(net, sprintf("maxpool_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,   sprintf("maxpool_1_%d/size",i), sprintf("maxunpool_1_%d/size",i));

                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_1_%d", i));
                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_2_%d", i));
                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_3_%d", i));
                net = connectLayers(net, sprintf("reshape_%d",i), sprintf("res_b_4_%d", i));

                net = connectLayers(net, sprintf("res_b_1_%d", i), sprintf("res_add_dec1_%d/in2",i));
                net = connectLayers(net, sprintf("res_b_2_%d", i), sprintf("res_add_dec2_%d/in2",i));
                net = connectLayers(net, sprintf("res_b_3_%d", i), sprintf("res_add_dec3_%d/in2",i));
                net = connectLayers(net, sprintf("res_b_4_%d", i), sprintf("res_add_dec4_%d/in2",i));

                net = connectLayers(net, sprintf("elu_dil_dec1_%d",i), sprintf("res_tc1_2_%d", i));
                net = connectLayers(net, sprintf("elu_dil_dec1_%d",i), sprintf("res_tc1_3_%d", i));
                net = connectLayers(net, sprintf("elu_dil_dec1_%d",i), sprintf("res_tc1_4_%d", i));

                net = connectLayers(net, sprintf("res_tc1_2_%d", i), sprintf("res_add_dec2_%d/in3", i));
                net = connectLayers(net, sprintf("res_tc1_3_%d", i), sprintf("res_add_dec3_%d/in3", i));
                net = connectLayers(net, sprintf("res_tc1_4_%d", i), sprintf("res_add_dec4_%d/in3", i));

                net = connectLayers(net, sprintf("maxunpool_1_%d",i), sprintf("res_tc2_3_%d", i));
                net = connectLayers(net, sprintf("maxunpool_1_%d",i), sprintf("res_tc2_4_%d", i));

                net = connectLayers(net, sprintf("res_tc2_3_%d", i), sprintf("res_add_dec3_%d/in4", i));
                net = connectLayers(net, sprintf("res_tc2_4_%d", i), sprintf("res_add_dec4_%d/in4", i));

                net = connectLayers(net, sprintf("elu_dec_5_%d", i), sprintf("res_tc3_4_%d", i));

                net = connectLayers(net, sprintf("res_tc3_4_%d", i), sprintf("res_add_dec4_%d/in5", i));

                %net = connectLayers(net,sprintf("maxpool_1_%d/out",i),sprintf("enc_dec_%d/in2", i));


               
            end
            
            % clean up helper variable
            clear Encoder Decoder;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
        end

        function net = buildOptimizedNetwork_improved(params)
        % Improved architecture based on analysis
        % Key improvements:
        % 1. Increased latent dimension (15 -> 64)
        % 2. Reduced aggressive downsampling (32x -> 8x)
        % 3. Simplified residual connections (U-Net style)
        % 4. Multi-scale encoder for frequency awareness
        % 5. Enhanced decoder capacity
        % 6. Better activation strategy
        
        % --- Defaults ---
        if ~isfield(params,'input_x'), params.input_x = 800; end
        if ~isfield(params,'input_y'), params.input_y = 2; end
        if ~isfield(params,'input_z'), params.input_z = 1; end
        if ~isfield(params,'num_in'), params.num_in = 1; end
        if ~isfield(params,'desired_latent_size'), params.desired_latent_size = 64; end % INCREASED from 15

        if ~isfield(params,'kernel_size1'), params.kernel_size1 = 3; end
        if ~isfield(params,'kernel_size2'), params.kernel_size2 = 3; end
        if ~isfield(params,'kernel_size3'), params.kernel_size3 = 3; end

        if ~isfield(params,'channels1'), params.channels1 = 16; end % Increased from 10
        if ~isfield(params,'channels2'), params.channels2 = 32; end % Increased from 20
        if ~isfield(params,'channels3'), params.channels3 = 64; end % New deeper layer

        if ~isfield(params,'dilation_factor'), params.dilation_factor = 2; end

        input_x = params.input_x;
        input_y = params.input_y;
        input_z = params.input_z;
        num_in = params.num_in;
        desired_latent_size = params.desired_latent_size;

        % Channel sizes
        channels_conv1 = max(params.channels1, 16);
        channels_conv2 = max(params.channels2, 32);
        channels_conv3 = max(params.channels3, 64);
        
        net = dlnetwork;

        % Kernel sizes
        kernel_conv1 = [params.kernel_size1, 2];
        kernel_conv2 = [params.kernel_size2, 2];
        kernel_conv3 = [params.kernel_size3, 2];
        
        % IMPROVED: Less aggressive downsampling
        % Old: stride1=[2,1], stride1_2=[2,1], stride2=[4,2] -> total 32x downsampling
        % New: stride1=[2,1], stride2=[2,2], stride3=[2,1] -> total 8x downsampling
        stride1 = [2, 1];  % First downsampling
        stride2 = [2, 2];  % Second downsampling (changed from [4,2])
        stride3 = [2, 1];  % Third downsampling (new)
        
        % Calculate fully connected layer size
        total_downsample = stride1(1) * stride2(1) * stride3(1); % 2*2*2 = 8
        fc_layer_size = input_x / total_downsample;
        disp(['Fully connected layer size: ', num2str(fc_layer_size)]);
        disp(['Total downsampling factor: ', num2str(total_downsample), 'x']);
        disp(['Latent dimension: ', num2str(desired_latent_size)]);
        
        DilationFactor = [params.dilation_factor, 1];
        dilatation_kernel = [3, 1];

        % Group normalization sizes
        group_size1 = 4;  % Fixed reasonable group size
        group_size2 = 8;
        group_size3 = 8;
        
        disp("Building improved network architecture...");
        
        for i = 1:num_in
            % Input layer
            in = inputLayer([NaN, input_x, input_y, input_z], "BSSC", "Name", sprintf("input_%d", i));
            net = addLayers(net, in);

            % ================================================================
            % ENCODER with Multi-Scale Feature Extraction
            % ================================================================
            
            % --- Block 1: Multi-scale first conv ---
            % High-frequency path (small kernel)
            net = addLayers(net, [
                convolution2dLayer([3, 1], channels_conv1/3, "Name", sprintf("conv_1a_hf_%d", i), ...
                    "Padding", "same", "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
            ]);
            
            % Mid-frequency path (medium kernel)
            net = addLayers(net, [
                convolution2dLayer([7, 1], channels_conv1/3, "Name", sprintf("conv_1b_mf_%d", i), ...
                    "Padding", "same", "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
            ]);
            
            % Low-frequency path (large kernel)
            net = addLayers(net, [
                convolution2dLayer([15, 1], channels_conv1/3, "Name", sprintf("conv_1c_lf_%d", i), ...
                    "Padding", "same", "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
            ]);
            
            % Concatenate multi-scale features
            net = addLayers(net, [
                depthConcatenationLayer(3, "Name", sprintf("concat_multiscale_1_%d", i))
                groupNormalizationLayer(group_size1, "Name", sprintf("gn_1_enc_%d", i))
                architectures_container.helperELU(sprintf("elu_enc_1_%d", i))
            ]);
            
            % Connect multi-scale paths
            net = connectLayers(net, sprintf("input_%d", i), sprintf("conv_1a_hf_%d", i));
            net = connectLayers(net, sprintf("input_%d", i), sprintf("conv_1b_mf_%d", i));
            net = connectLayers(net, sprintf("input_%d", i), sprintf("conv_1c_lf_%d", i));
            net = connectLayers(net, sprintf("conv_1a_hf_%d", i), sprintf("concat_multiscale_1_%d/in1", i));
            net = connectLayers(net, sprintf("conv_1b_mf_%d", i), sprintf("concat_multiscale_1_%d/in2", i));
            net = connectLayers(net, sprintf("conv_1c_lf_%d", i), sprintf("concat_multiscale_1_%d/in3", i));
            
            % --- Block 2: First downsampling with residual ---
            net = addLayers(net, [
                convolution2dLayer(kernel_conv1, channels_conv1, "Name", sprintf("conv_2_%d", i), ...
                    "Padding", "same", "Stride", stride1, "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                groupNormalizationLayer(group_size1, "Name", sprintf("gn_2_enc_%d", i))
                architectures_container.helperELU(sprintf("elu_enc_2_%d", i))
            ]);
            
            % Residual connection for block 2
            net = addLayers(net, [
                convolution2dLayer([1, 1], channels_conv1, "Name", sprintf("res_conv_2_%d", i), ...
                    "Padding", "same", "Stride", stride1)
                additionLayer(2, "Name", sprintf("res_add_2_%d", i))
            ]);
            
            net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("conv_2_%d", i));
            net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("res_conv_2_%d", i));
            net = connectLayers(net, sprintf("elu_enc_2_%d", i), sprintf("res_add_2_%d/in2", i));
            
            % --- Block 3: Second downsampling ---
            net = addLayers(net, [
                convolution2dLayer(kernel_conv2, channels_conv2, "Name", sprintf("conv_3_%d", i), ...
                    "Padding", "same", "Stride", stride2, "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                groupNormalizationLayer(group_size2, "Name", sprintf("gn_3_enc_%d", i))
                architectures_container.helperELU(sprintf("elu_enc_3_%d", i))
            ]);
            
            % Residual connection for block 3
            net = addLayers(net, [
                convolution2dLayer([1, 1], channels_conv2, "Name", sprintf("res_conv_3_%d", i), ...
                    "Padding", "same", "Stride", stride2)
                additionLayer(2, "Name", sprintf("res_add_3_%d", i))
            ]);
            
            net = connectLayers(net, sprintf("res_add_2_%d", i), sprintf("conv_3_%d", i));
            net = connectLayers(net, sprintf("res_add_2_%d", i), sprintf("res_conv_3_%d", i));
        
            net = connectLayers(net, sprintf("elu_enc_3_%d", i), sprintf("res_add_3_%d/in2", i));
            
            % --- Block 4: Third downsampling (bottleneck approach) ---
            net = addLayers(net, [
                convolution2dLayer(kernel_conv3, channels_conv3, "Name", sprintf("conv_4_%d", i), ...
                    "Padding", "same", "Stride", stride3, "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                groupNormalizationLayer(group_size3, "Name", sprintf("gn_4_enc_%d", i))
                architectures_container.helperELU(sprintf("elu_enc_4_%d", i))
            ]);
            
            net = connectLayers(net, sprintf("res_add_3_%d", i), sprintf("conv_4_%d", i));
            
            % --- Bottleneck: Dilated convolution ---
            net = addLayers(net, [
                convolution2dLayer(dilatation_kernel, channels_conv3, "Name", sprintf("conv_dil_%d", i), ...
                    "Padding", "same", "DilationFactor", DilationFactor, ...
                    "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                architectures_container.helperELU(sprintf("elu_dil_%d", i))
                additionLayer(2, "Name", sprintf("res_add_dil_%d", i))
            ]);
            
            net = connectLayers(net, sprintf("elu_enc_4_%d", i), sprintf("conv_dil_%d", i));
            net = connectLayers(net, sprintf("elu_enc_4_%d", i), sprintf("res_add_dil_%d/in2", i));
            
            % --- Latent representation ---
            net = addLayers(net, [
                flattenLayer("Name", sprintf("flatten_%d", i))
                fullyConnectedLayer(desired_latent_size * 2, "Name", sprintf("fc_latent_pre_%d", i))
                architectures_container.helperELU(sprintf("elu_latent_pre_%d", i))
                fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_%d", i))
                architectures_container.helperELU(sprintf("elu_latent_%d", i))
            ]);
            
            net = connectLayers(net, sprintf("res_add_dil_%d", i), sprintf("flatten_%d", i));
            
            % ================================================================
            % DECODER with Enhanced Capacity
            % ================================================================
            
            % --- Unflatten from latent ---
            net = addLayers(net, [
                fullyConnectedLayer(desired_latent_size * 2, "Name", sprintf("fc_unflatten_pre_%d", i))
                architectures_container.helperELU(sprintf("elu_unflatten_pre_%d", i))
                fullyConnectedLayer(fc_layer_size * channels_conv3, "Name", sprintf("fc_unflatten_%d", i))
                architectures_container.helperELU(sprintf("elu_unflatten_%d", i))
                architectures_container.helperchange_format(sprintf("reshape_%d", i), "BSSC", [fc_layer_size, 1, channels_conv3])
            ]);
            
            net = connectLayers(net, sprintf("elu_latent_%d", i), sprintf("fc_unflatten_pre_%d", i));
            
            % --- Decoder Block 1: Upsample from bottleneck ---
            net = addLayers(net, [
                convolution2dLayer(dilatation_kernel, channels_conv3, "Name", sprintf("conv_dec_dil_%d", i), ...
                    "Padding", "same", "DilationFactor", DilationFactor, ...
                    "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                architectures_container.helperELU(sprintf("elu_dec_dil_%d", i))
            ]);
            
            net = connectLayers(net, sprintf("reshape_%d", i), sprintf("conv_dec_dil_%d", i));
            
            % --- Decoder Block 2: First upsampling (with skip connection) ---
            % Upsample + 2 conv layers for better capacity
            net = addLayers(net, [
                transposedConv2dLayer(kernel_conv3, channels_conv2, "Name", sprintf("upconv_4_%d", i), ...
                    "Cropping", "same", "Stride", stride3, 'WeightsInitializer', 'he', 'BiasInitializer', 'zeros')
                convolution2dLayer([3, 1], channels_conv2, "Name", sprintf("refine_4a_%d", i), ...
                    "Padding", "same", "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                groupNormalizationLayer(group_size2, "Name", sprintf("gn_dec_4a_%d", i))
                architectures_container.helperELU(sprintf("elu_dec_4a_%d", i))
                convolution2dLayer([3, 1], channels_conv2, "Name", sprintf("refine_4b_%d", i), ...
                    "Padding", "same", "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                groupNormalizationLayer(group_size2, "Name", sprintf("gn_dec_4b_%d", i))
                architectures_container.helperELU(sprintf("elu_dec_4b_%d", i))
            ]);
            
            % U-Net skip connection from encoder block 3
            net = addLayers(net, [
                depthConcatenationLayer(2, "Name", sprintf("skip_concat_3_%d", i))
            ]);
            
            net = connectLayers(net, sprintf("elu_dec_dil_%d", i), sprintf("upconv_4_%d", i));
            net = connectLayers(net, sprintf("elu_dec_4b_%d", i), sprintf("skip_concat_3_%d/in1", i));
            net = connectLayers(net, sprintf("res_add_3_%d", i), sprintf("skip_concat_3_%d/in2", i));
            
            % --- Decoder Block 3: Second upsampling ---
            net = addLayers(net, [
                transposedConv2dLayer(kernel_conv2, channels_conv1, "Name", sprintf("upconv_3_%d", i), ...
                    "Cropping", "same", "Stride", stride2, 'WeightsInitializer', 'he', 'BiasInitializer', 'zeros')
                convolution2dLayer([3, 1], channels_conv1, "Name", sprintf("refine_3a_%d", i), ...
                    "Padding", "same", "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                groupNormalizationLayer(group_size1, "Name", sprintf("gn_dec_3a_%d", i))
                architectures_container.helperELU(sprintf("elu_dec_3a_%d", i))
                convolution2dLayer([3, 1], channels_conv1, "Name", sprintf("refine_3b_%d", i), ...
                    "Padding", "same", "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                groupNormalizationLayer(group_size1, "Name", sprintf("gn_dec_3b_%d", i))
                architectures_container.helperELU(sprintf("elu_dec_3b_%d", i))
            ]);
            
            % U-Net skip connection from encoder block 2
            net = addLayers(net, [
                depthConcatenationLayer(2, "Name", sprintf("skip_concat_2_%d", i))
            ]);
            
            net = connectLayers(net, sprintf("skip_concat_3_%d", i), sprintf("upconv_3_%d", i));
            net = connectLayers(net, sprintf("elu_dec_3b_%d", i), sprintf("skip_concat_2_%d/in1", i));
            net = connectLayers(net, sprintf("res_add_2_%d", i), sprintf("skip_concat_2_%d/in2", i));
            
            % --- Decoder Block 4: Final upsampling ---
            net = addLayers(net, [
                transposedConv2dLayer(kernel_conv1, channels_conv1, "Name", sprintf("upconv_2_%d", i), ...
                    "Cropping", "same", "Stride", stride1, 'WeightsInitializer', 'he', 'BiasInitializer', 'zeros')
                convolution2dLayer([3, 1], channels_conv1, "Name", sprintf("refine_2a_%d", i), ...
                    "Padding", "same", "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                groupNormalizationLayer(group_size1, "Name", sprintf("gn_dec_2a_%d", i))
                architectures_container.helperELU(sprintf("elu_dec_2a_%d", i))
                convolution2dLayer([3, 1], channels_conv1, "Name", sprintf("refine_2b_%d", i), ...
                    "Padding", "same", "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                groupNormalizationLayer(group_size1, "Name", sprintf("gn_dec_2b_%d", i))
                architectures_container.helperELU(sprintf("elu_dec_2b_%d", i))
            ]);
            
            % U-Net skip connection from encoder block 1 (multi-scale features)
            net = addLayers(net, [
                depthConcatenationLayer(2, "Name", sprintf("skip_concat_1_%d", i))
            ]);
            
            net = connectLayers(net, sprintf("skip_concat_2_%d", i), sprintf("upconv_2_%d", i));
            net = connectLayers(net, sprintf("elu_dec_2b_%d", i), sprintf("skip_concat_1_%d/in1", i));
            net = connectLayers(net, sprintf("elu_enc_1_%d", i), sprintf("skip_concat_1_%d/in2", i));
            
            % --- Final output layer ---
            net = addLayers(net, [
                convolution2dLayer([3, 1], input_z, "Name", sprintf("output_conv_%d", i), ...
                    "Padding", "same", "WeightsInitializer", "he", 'BiasInitializer', 'zeros')
                % NO final activation - let network learn full range
            ]);
            
            net = connectLayers(net, sprintf("skip_concat_1_%d", i), sprintf("output_conv_%d", i));
        end
        
        disp("Network architecture built successfully!");
        disp("Key improvements implemented:");
        disp("  1. Latent dimension increased to 64");
        disp("  2. Downsampling reduced from 32x to 8x");
        disp("  3. U-Net skip connections for detail preservation");
        disp("  4. Multi-scale encoder for frequency awareness");
        disp("  5. Enhanced decoder with refinement layers");
        disp("  6. No final activation for unrestricted output range");
        
        % Initialize network
        net = initialize(net);
    end

    
        

%% Original network
        function net = buildOptimizedNetwork(params)
            disp('Building network with optimized hyperparameters...\n\n\n');
            % Build network with optimized hyperparameters
            
            net = dlnetwork;
            
            % Input dimensions
            %input_x = 4000;
            input_x = 800;
            input_y = 2;
            input_z = 6;
            num_in = 28;
            desired_latent_size = 80;

            kernel_conv1 = [params.kernel_size1,2];
            kernel_conv2 = [params.kernel_size2,2];
            kernel_conv3 = [params.kernel_size3,2];

            channels_conv1 = params.channels1;
            channels_conv2 = params.channels2;
            channels_conv3 = params.channels3;         % keep latent size small

            % Define possible group sizes (must be divisors of numChannels)
            possibleGroupSizes1 = 1:channels_conv1; % All integers from 1 to numChannels
            validGroupSizes1 = possibleGroupSizes1(mod(channels_conv1, possibleGroupSizes1) == 0);

            % Randomly select a valid group size
            group_size1 = validGroupSizes1(ceil(length(validGroupSizes1)/4)); % Choose the middle valid size for consistency

            possibleGroupSizes2 = 1:channels_conv2; % All integers from 1 to numChannels
            validGroupSizes2 = possibleGroupSizes2(mod(channels_conv2, possibleGroupSizes2) == 0);

            group_size2 = validGroupSizes2(ceil(length(validGroupSizes2)/4)); % Choose the middle valid size for consistency

            stride1        = [params.stride1, 1];    % 4000 -> 400
            stride2        = [params.stride2 1];     % 400 -> 200
            stride3        = [1 2];     % reduce sensor axis only
            
            kernel_pool2   = [ceil(kernel_conv2(1)/2) 1];
            stride_pool2   = [ceil(kernel_conv2(1)/2) 1];     % 200 -> 40

            crop1 = 'same';
            crop2 = 'same';          

            latentT = ceil(ceil(ceil(input_x / stride1(1)) / stride2(1)) / stride_pool2(1));                     % 4000/10/2/4
            single_input_size = latentT*1*channels_conv3;  % 50*1*12 = 600
            disp(" I am about to go into the loop\n");
            for i=1:num_in

                Encoder = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same","Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_1_%d",i))
            
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv2,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride2)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_2_enc_%d",i))
                    architectures_container.helperELU(sprintf("elu_enc_2_%d",i))
                    maxPooling2dLayer(kernel_pool2,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool2)
                    
                    % Third conv block (no size change)
                    convolution2dLayer(kernel_conv3, channels_conv3, "Name", sprintf("conv_3_%d", i), "Padding", "same","Stride", stride3)
                    architectures_container.helperELU( sprintf("elu_3_%d", i))
                    
                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer([3 1], channels_conv3, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", [2 1])
                    architectures_container.helperELU( sprintf("elu_dil1_%d", i))
                    convolution2dLayer([3 1], channels_conv3, "Name", sprintf("conv_dil2_%d", i), "Padding", "same", "DilationFactor", [4 1])
                    architectures_container.helperELU( sprintf("elu_dil2_%d", i))

                    
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    architectures_container.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[single_input_size,1])
                ];

                Predecoder = [];
                if single_input_size ~= desired_latent_size
                    Encoder = [ 
                        Encoder
                        fullyConnectedLayer(desired_latent_size, "Name", sprintf("fc_latent_adjust_%d", i))
                        architectures_container.helperELU( sprintf("elu_latent_adjust_%d", i))
                    ];
                    Predecoder = [
                        fullyConnectedLayer(single_input_size, "Name", sprintf("fc_latent_revert_%d", i))
                        architectures_container.helper_scaled_tanh(sprintf("tanh_latent_revert_%d", i), 6.0)
                    ];
                end


                Decoder = [
                    Predecoder
                    % Use custom helper for reshape operation
                    architectures_container.helperchange_format(sprintf("reshape_%d",i),"BSSC",[latentT,1,channels_conv3])  % Reshape to [50 1 channels_conv3]
                    
                    % First transposed conv block
                    transposedConv2dLayer(kernel_conv3,channels_conv2,"Name",sprintf("transposed-conv_1_%d",i),"Cropping","same","Stride", stride3)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_dec_1_%d",i))
                    architectures_container.helperELU(sprintf("elu_dec_1_%d",i))
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))

                    % Second transposed conv block
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping",crop2,"Stride", stride2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_2_%d",i))
                    architectures_container.helper_removeMeanLayer(sprintf("remove_mean_dec_2_%d",i))


                    
                    % Final conv to get back to original number of channels
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_3_%d",i),"Cropping",crop1,"Stride", stride1,"BiasInitializer","zeros","BiasLearnRateFactor",0)
                    architectures_container.helper_scaled_tanh(sprintf("tanh_bound_%d", i), 6.0)
                    architectures_container.helper_amplitudeScaleLayer(sprintf("amp_scale_%d",i),0.025)
                    
                    ];       
                net = addLayers(net,Encoder);
                net = addLayers(net,Decoder);

                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/size",i),sprintf("maxunpool_1_%d/size",i));
                
                
            end
            
            % clean up helper variable
            clear tempNet;
       
            % Initialize network with improved weight initialization
            net = initialize(net);
        end
%% Helper functions to create layers
        function layer = helperchange_format(name,format,size)
            layer = change_format(name,format,size);
        end
        

        function layer = helperprint(name)
            layer = print(name);
        end

        function layer = helper_amplitudeScaleLayer(name, initValue)
            layer = amplitudeScaleLayer(name, initValue);
        end

        function layer = helperELU(name)
            layer = eluBSSC(name, 1); % alpha=1
        end

        function layer = helper_removeMeanLayer(name)
            layer = removeMeanLayer(name);
        end

        function layer = helper_scaled_tanh(name, scale)
            layer = Scaled_tanh(name, scale);
        end

        function layer = helper_polynomial(name)
            layer = Polynomial_activation(name);
        end
    end
end