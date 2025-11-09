% filepath: c:\Users\Maria\Documents\Honours Programme\PZT_data\PZT_L103\GAN\net_visualization.m
% ...existing code...
clearvars; clc;

% Add project paths (same as training)
projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src', 'models'), '-begin');
addpath(fullfile(projectRoot, 'src', 'models', 'layers'), '-begin');
addpath(fullfile(projectRoot, 'src', 'utils'), '-begin');

% Sanity: verify custom layers are found
assert(~isempty(which('amplitudeScaleLayer')), 'amplitudeScaleLayer not on path');
assert(~isempty(which('eluBSSC')), 'eluBSSC not on path');
assert(~isempty(which('change_format')), 'change_format not on path');

function net = buildOptimizedNetwork(params)
    disp('Building network with optimized hyperparameters...\n\n\n');
    % Build network with optimized hyperparameters
    
    net = dlnetwork;
    
    % Input dimensions
    input_x = 4000;
    input_y = 2;
    input_z = 6;
    num_in = 28;

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
    stride2        = [params.stride2 2];     % 400 -> 200
    stride3        = [1 2];     % reduce sensor axis only

    p1 = stride1(1) - mod(input_x, stride1(1));
    p2 = stride2(1) - mod(ceil(input_x / stride1(1)), stride2(1));
    

    kernel_pool2   = [ceil(kernel_conv2(1)/2) 1];
    stride_pool2   = [ceil(kernel_conv2(1)/2) 1];     % 200 -> 40


    if p1 == stride1(1)
        crop1 = 'same';
        disp('\n Using same padding for crop1\n');
    else
        in = ceil(ceil(input_x / stride1(1)) / stride2(1));
        t_d_crop = (in-1)*stride1(1)+kernel_conv1(1)-stride1(1)*in+p1;
        l_r_crop = stride1(2)+kernel_conv1(2)-2;
        crop1 = [floor(t_d_crop/2), ceil(t_d_crop/2), floor(l_r_crop/2), ceil(l_r_crop/2)];
    end
    if p2 == stride2(1)
        crop2 = 'same';
        disp('\n Using same padding for crop2\n');
    else
        in = ceil(ceil(ceil(input_x / stride1(1)) / stride2(1)) / stride_pool2(1));
        t_d_crop = (in-1)*stride2(1)+kernel_conv2(1)-stride2(1)*in+p2;
        %l_r_crop = stride2(2)+kernel_conv2(2)-2;
        l_r_crop = 0;
        crop2 = [floor(t_d_crop/2), ceil(t_d_crop/2), floor(l_r_crop/2), ceil(l_r_crop/2)];
    end
    

    latentT = ceil(ceil(ceil(input_x / stride1(1)) / stride2(1)) / stride_pool2(1));                     % 4000/10/2/4
    single_input_size = latentT*1*channels_conv2;  % 50*1*12 = 600
    disp(" I am about to go into the loop\n");
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
            %convolution2dLayer(kernel_conv3, channels_conv3, "Name", sprintf("conv_3_%d", i), "Padding", "same","Stride", stride3)
            %AE_for_each_path_separate.helperELU( sprintf("elu_3_%d", i))
            
            % Temporal dilated block in bottleneck (no size change)
            convolution2dLayer([3 1], channels_conv2, "Name", sprintf("conv_dil1_%d", i), "Padding", "same", "DilationFactor", [2 1])
            AE_for_each_path_separate.helperELU( sprintf("elu_dil1_%d", i))
            % convolution2dLayer([3 1], channels_conv2, "Name", sprintf("conv_dil2_%d", i), "Padding", "same", "DilationFactor", [4 1])
            % AE_for_each_path_separate.helperELU( sprintf("elu_dil2_%d", i))

            
            % Flatten to vector for GAN input
            flattenLayer("Name",sprintf("flatten_%d",i))
            AE_for_each_path_separate.helperchange_format(sprintf("add_channel_dim_%d",i),"SCB",[single_input_size,1])

            % Use custom helper for reshape operation
            AE_for_each_path_separate.helperchange_format(sprintf("reshape_%d",i),"BSSC",[latentT,1,channels_conv2])  % Reshape to [50 1 channels_conv3]
            
            % First transposed conv block
            %transposedConv2dLayer(kernel_conv3,channels_conv2,"Name",sprintf("transposed-conv_1_%d",i),"Cropping","same","Stride", stride3)
            %groupNormalizationLayer(group_size2,"Name",sprintf("gn_dec_1_%d",i))
            %AE_for_each_path_separate.helperELU(sprintf("elu_dec_1_%d",i))
            maxUnpooling2dLayer("Name",sprintf("maxunpool_1_%d",i))

            % Second transposed conv block
            transposedConv2dLayer(kernel_conv2,channels_conv1,"Name",sprintf("transposed-conv_2_%d",i),"Cropping",crop2,"Stride", stride2)
            groupNormalizationLayer(group_size1,"Name",sprintf("gn_dec_2_%d",i))
            AE_for_each_path_separate.helper_removeMeanLayer(sprintf("remove_mean_dec_2_%d",i))


            
            % Final conv to get back to original number of channels
            transposedConv2dLayer(kernel_conv1,input_z,"Name",sprintf("transposed-conv_3_%d",i),"Cropping",crop1,"Stride", stride1,"BiasInitializer","zeros","BiasLearnRateFactor",0)
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
 
params.channels1 = 13;
params.channels2 = 9;
params.channels3 = 6;
params.kernel_size1 = 29;
params.kernel_size2 = 11;
params.kernel_size3 = 6;
params.stride1 = 19;
params.stride2 = 6;
net = buildOptimizedNetwork(params);

% Load the saved network
%S = load(fullfile(projectRoot, 'results', 'net_only_2025-10-09_17-34-06.mat'));  % adjust file if needed
%net=S.trained_net;
disp(class(net));
% Open in Designer via layerGraph (more robust for dlnetworks)
lgraph = layerGraph(net);
deepNetworkDesigner(lgraph);