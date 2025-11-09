classdef AE_for_each_path_separate_E11

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

        function layer = helperELU(name)
            layer = eluBSSC(name, 1); % alpha=1
        end

        function [net, title] = build_net_with_28_inputs()
            
            net = dlnetwork;

            num_in = 28;
            input_x = 4000;
            input_y = 2;
            input_z = 6;

            % Encoder branch kernels (inception-like)
            kernel_conv1_1 = [25,2];
            kernel_conv1_2 = [50,2];
            kernel_conv1_3 = [10,2];

            % Next blocks
            kernel_conv2 = [9,2];
            kernel_conv3 = [5,2];

            % Channels per branch and totals
            channels_conv1_1 = 5;
            channels_conv1_2 = 5;
            channels_conv1_3 = 5;
            channels_conv1_tot = channels_conv1_1 + channels_conv1_2 + channels_conv1_3; % 15

            % GroupNorm groups must divide channel count
            group_size1 = 5; % 15/5 = 3 per group (OK)
            group_size2 = 3; % 6/3 = 2 per group (OK)

            % Strides
            stride1 = [10 1];  % 4000 -> 400
            channels_conv2 = 6;
            stride2 = [2 1];   % 400 -> 200
            kernel_pool2 = [4 1];
            stride_pool2 = [5 1]; % 200 -> 40
            channels_conv3 = 2;   % bottleneck channels (keeps latent small)
            stride3 = [1 2];      % reduce sensor axis only (time unchanged)


            latentT = input_x / stride1(1) / stride2(1) / stride_pool2(1) / stride3(1);                     % 4000/10/2/4
            single_input_size = latentT*1*channels_conv3;  % 50*1*12 = 600
            title = sprintf('28_AE_latent_%d_E11', single_input_size);
            for i=1:num_in
                in = inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                net = addLayers(net, in);

                % Branch A: small kernel
                c1_1 = convolution2dLayer(kernel_conv1_1, channels_conv1_1,"Name",sprintf("conv1_1_%d",i),"Padding","same","Stride",stride1);
                % Branch B: medium kernel
                c1_2 = convolution2dLayer(kernel_conv1_2, channels_conv1_2,"Name",sprintf("conv1_2_%d",i),"Padding","same","Stride",stride1);
                % Branch C: large kernel
                c1_3 = convolution2dLayer(kernel_conv1_3, channels_conv1_3,"Name",sprintf("conv1_3_%d",i),"Padding","same","Stride",stride1);

                net = addLayers(net, c1_1);
                net = addLayers(net, c1_2);
                net = addLayers(net, c1_3);

                tempNet = [
                    
    
                    % Depth concat of the three branches (along channel dim)
                    depthConcatenationLayer(3,"Name",sprintf("concat_enc1_%d",i))



                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    %batchNormalizationLayer("Name",sprintf("bn__enc_1_%d",i))
                    AE_for_each_path_separate.helperELU(sprintf("elu_enc_1_%d",i))
                    %additionLayer(2,"Name",sprintf("add_res_1_%d",i))
                    %maxPooling2dLayer(kernel_pool1,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool1)
                    
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv2,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride2)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_2_enc_%d",i))
                    %batchNormalizationLayer("Name",sprintf("bn_2_enc_%d",i))
                    AE_for_each_path_separate.helperELU(sprintf("elu_enc_2_%d",i))
                    %reluLayer("Name",sprintf("Relu_2_%d",i))
                    %dropoutLayer(dropout_rate,"Name",sprintf("dropout_%d",i))
                    %additionLayer(2,"Name",sprintf("add_res_2_%d",i))
                    maxPooling2dLayer(kernel_pool2,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool2)
                    convolution2dLayer(kernel_conv3, channels_conv3, "Name", sprintf("conv_3_%d", i), "Padding", "same","Stride", stride3)
                    AE_for_each_path_separate.helperELU( sprintf("elu_3_%d", i))
                    
                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer([3 1], channels_conv3, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", [2 1])
                    AE_for_each_path_separate.helperELU( sprintf("elu_dil1_%d", i))
                    convolution2dLayer([3 1], channels_conv3, "Name", sprintf("conv_dil2_%d", i), "Padding", "same", "DilationFactor", [4 1])
                    AE_for_each_path_separate.helperELU( sprintf("elu_dil2_%d", i))

                   
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
                    AE_for_each_path_separate.helperELU(sprintf("elu_dec_1_%d",i))
                    %reluLayer("Name",sprintf("Relu_3_%d",i))

                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))

                    % Second transposed conv block
                    transposedConv2dLayer(kernel_conv2,channels_conv1_tot,"Name",sprintf("transposed-conv_2_%d",i),"Cropping","same","Stride", stride2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_2_%d",i))
                    %batchNormalizationLayer("Name",sprintf("bn_dec_2_%d",i))
                    AE_for_each_path_separate.helperELU(sprintf("leaky_relu_dec_2_%d",i))
                    
                    additionLayer(2,"Name",sprintf("add_res_dec_2_%d",i))
                    ];       
                net = addLayers(net,tempNet);

                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/size",i),sprintf("maxunpool_1_%d/size",i));

                % Connect input to each branch
                net = connectLayers(net,sprintf("input_%d",i),sprintf("conv1_1_%d",i));
                net = connectLayers(net,sprintf("input_%d",i),sprintf("conv1_2_%d",i));
                net = connectLayers(net,sprintf("input_%d",i),sprintf("conv1_3_%d",i));
                % Connect branch outputs to concat
                net = connectLayers(net,sprintf("conv1_1_%d",i), sprintf("concat_enc1_%d/in1",i));
                net = connectLayers(net,sprintf("conv1_2_%d",i),sprintf("concat_enc1_%d/in2",i));
                net = connectLayers(net,sprintf("conv1_3_%d",i),sprintf("concat_enc1_%d/in3",i));

                 % Skip connection: from output of first activation to add_skip (needs matching size after upsampling)
                projName = sprintf("skip_proj_%d",i);
                projLayer = convolution2dLayer([1 1],channels_conv1_tot,"Name",projName,"Padding","same");
                net = addLayers(net, projLayer);
                net = connectLayers(net,sprintf("elu_enc_1_%d",i),projName);
                net = connectLayers(net,projName,sprintf("add_res_dec_2_%d/in2",i));

                % ===== Inverse inception at final scale (time 4000) =====
                % 3 parallel transposed-convs mirroring the encoder kernels
                t3a = transposedConv2dLayer(kernel_conv1_1, 2,"Name",sprintf("tconv3_1_%d",i),"Cropping","same","Stride",stride1);
                t3b = transposedConv2dLayer(kernel_conv1_2, 2,"Name",sprintf("tconv3_2_%d",i),"Cropping","same","Stride",stride1);
                t3c = transposedConv2dLayer(kernel_conv1_3, 2,"Name",sprintf("tconv3_3_%d",i),"Cropping","same","Stride",stride1);
                net = addLayers(net, t3a);
                net = addLayers(net, t3b);
                net = addLayers(net, t3c);

                % Fan-out from residual sum to the three branches
                net = connectLayers(net,sprintf("add_res_dec_2_%d",i),sprintf("tconv3_1_%d",i));
                net = connectLayers(net,sprintf("add_res_dec_2_%d",i),sprintf("tconv3_2_%d",i));
                net = connectLayers(net,sprintf("add_res_dec_2_%d",i),sprintf("tconv3_3_%d",i));

                % Concatenate back to 6 channels (2+2+2 = 6), then bound and scale
                net = addLayers(net, depthConcatenationLayer(3,"Name",sprintf("concat_dec3_%d",i)));
                net = connectLayers(net,sprintf("tconv3_1_%d",i),sprintf("concat_dec3_%d/in1",i));
                net = connectLayers(net,sprintf("tconv3_2_%d",i),sprintf("concat_dec3_%d/in2",i));
                net = connectLayers(net,sprintf("tconv3_3_%d",i),sprintf("concat_dec3_%d/in3",i));

                % Final activation and learnable global scaler (optional)
                net = addLayers(net, [
                    tanhLayer("Name",sprintf("tanh_bound_%d",i))
                    AE_for_each_path_separate.helper_amplitudeScaleLayer(sprintf("amp_scale_%d",i),0.02)
                ]);
                net = connectLayers(net,sprintf("concat_dec3_%d",i),sprintf("tanh_bound_%d",i));
            end
           

            % clean up helper variable
            clear tempNet;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
        end
        
    end
end