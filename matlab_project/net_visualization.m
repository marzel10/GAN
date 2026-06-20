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
          

 
% params.channels1 = 8;
% params.channels2 = 4;
% params.channels3 = 4;
% params.kernel_size1 = 22;
% params.kernel_size2 = 14;
% params.kernel_size3 = 7;
% params.stride1 = 20;
% params.stride2 = 6;

params.channels1 = 265;
params.channels2 = 128;
params.channels3 = 64;
params.channels4 = 16;
params.channels5 = 1;

params.kernel_size1 = 40;% first frequency is  50 kHz, sampling rate is 2e6 Hz, therefore 40 samples per period, kernel 10 seems reasonable

 params.channels1 =  62;
 params.channels2 =  23;
 params.kernel_size1 = 19;
 params.kernel_size2 = 14;
 params.kernel_size3 = 13;
 params.batch_size =  4;
 params.dilation_factor = 8;
 params.desired_latent_size = 15;
 params.input_y = 1;


%net = architectures_container.buildOptimizedNetwork_compressed_downsampled_sin_freq_paper(params);

%net = architectures_container.buildOptimizedNetwork_compressed_downsampled_sin_freq_less_RES_bn_dr(params);
%param.no_fields =  true;
%net1 = architectures_container.deep_fully_connected_network(param);
net1 = load('C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Fully Connected Network\freq_4\path_1\bayesian_optimization\path_1_freq_4_net_loss_3.529762.mat').trained_net;
net2 = load('C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Fully Connected Network\freq_4\path_2\bayesian_optimization\2p4f_h1_1197_h2_923_h3_415_h4_38_dr_0.05_bs_16_valLoss_5.346821.mat').net;
% Cell array to hold networks
networks = {};
for i =1:2
    if i==1
        net = net1;
    else
        net = net2;
    end
    networks{i} = net;
end
for net_id = 1:length(networks)
    net1 = networks{net_id};
    disp(['Analyzing Network ' num2str(net_id)]);
    for i = 1:length(net1.Layers)
        if isprop(net1.Layers(i), 'WeightLearnRateFactor')
            disp(net1.Layers(i).Name);
            disp(net1.Layers(i).WeightLearnRateFactor);
        else
            disp(['Layer ' net1.Layers(i).Name ' does not have LearnRateFactor property.']);
        end
    
    end

end

GAN_params.input_size = 15;
GAN_params.latent_size = 1;
[~, GAN_params.adjacency_matrix] = attention_matrix.build_attention();

fc_params.input_size = 15;
fc_params.latent_size = 1;

fcnet = tiny_architectures_container.tiny_fully_connected_network(fc_params);
gannet = tiny_architectures_container.tiny_graph_network(GAN_params);

deepNetworkDesigner(fcnet);
deepNetworkDesigner(gannet);
% Load the saved network
%S = load(fullfile(projectRoot, 'results', 'net_only_2025-10-09_17-34-06.mat'));  % adjust file if needed
%net=S.trained_net;
disp(class(net1));
%deepNetworkDesigner(combined_net);