classdef AE_for_each_path_separate

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

        function layer = helper_removeMeanLayer(name)
            layer = removeMeanLayer(name);
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
            title = sprintf('28_AE_latent_%d_E15', single_input_size);
            for i=1:num_in

                tempNet = [
                    inputLayer([NaN, input_x, input_y, input_z],"BSSC", "Name",sprintf("input_%d",i))
                    
                    
                    % First conv block with normalization and res connections
                    convolution2dLayer(kernel_conv1,channels_conv1,"Name",sprintf("conv_1_%d",i),"Padding","same","Stride", stride1)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_1_enc_%d",i))
                    AE_for_each_path_separate.helperELU(sprintf("elu_enc_1_%d",i))
            
                    % Second conv block with normalization and res connections
                    convolution2dLayer(kernel_conv2,channels_conv2,"Name",sprintf("conv_2_%d",i),"Padding","same", "Stride", stride2)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_2_enc_%d",i))
                    AE_for_each_path_separate.helperELU(sprintf("elu_enc_2_%d",i))
                    maxPooling2dLayer(kernel_pool2,"Name",sprintf("maxpoolForUnpool_1_%d",i),"HasUnpoolingOutputs",true,"Padding","same","Stride",stride_pool2)
                    
                    % Third conv block (no size change)
                    convolution2dLayer(kernel_conv3, channels_conv3, "Name", sprintf("conv_3_%d", i), "Padding", "same","Stride", stride3)
                    AE_for_each_path_separate.helperELU( sprintf("elu_3_%d", i))
                    
                    % Temporal dilated block in bottleneck (no size change)
                    convolution2dLayer([3 1], channels_conv3, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", [2 1])
                    AE_for_each_path_separate.helperELU( sprintf("elu_dil1_%d", i))
                    convolution2dLayer([3 1], channels_conv3, "Name", sprintf("conv_dil2_%d", i), "Padding", "same", "DilationFactor", [4 1])
                    AE_for_each_path_separate.helperELU( sprintf("elu_dil2_%d", i))

                   
                    % Flatten to vector for GAN input
                    flattenLayer("Name",sprintf("flatten_%d",i))
                    AE_for_each_path_separate.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[single_input_size,1])

                    % Use custom helper for reshape operation
                    AE_for_each_path_separate.helperchange_format(sprintf("reshape_%d",i),"BSSC",[latentT,1,channels_conv3])  % Reshape to [50 1 channels_conv3]
                    
                    % First transposed conv block
                    transposedConv2dLayer(kernel_conv3,channels_conv2,"Name",sprintf("transposed-conv_1_%d",i),"Cropping","same","Stride", stride3)
                    groupNormalizationLayer(group_size2,"Name",sprintf("gn_dec_1_%d",i))
                    AE_for_each_path_separate.helperELU(sprintf("elu_dec_1_%d",i))
                    maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))

                    % Second transposed conv block
                    transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping","same","Stride", stride2)
                    groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_2_%d",i))
                    AE_for_each_path_separate.helper_removeMeanLayer(sprintf("remove_mean_dec_2_%d",i))


                  
                    % Final conv to get back to original number of channels
                    transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_3_%d",i),"Cropping","same","Stride", stride1,"BiasInitializer","zeros","BiasLearnRateFactor",0)
                    tanhLayer("Name",sprintf("tanh_bound_%d",i))
                    AE_for_each_path_separate.helper_amplitudeScaleLayer(sprintf("amp_scale_%d",i),0.025)
                  
                    ];       
                net = addLayers(net,tempNet);

                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/indices",i),sprintf("maxunpool_1_%d/indices",i));
                net = connectLayers(net,sprintf("maxpoolForUnpool_1_%d/size",i),sprintf("maxunpool_1_%d/size",i));
                
                
            end
           

            % clean up helper variable
            clear tempNet;

            
            % Initialize network with improved weight initialization
            net = initialize(net);
        end
        
    end
end