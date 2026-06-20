GAN_params.input_size = 15;
GAN_params.latent_size = 1;
[~, GAN_params.adjacency_matrix] = attention_matrix.build_attention();

fc_params.input_size = 15;
fc_params.latent_size = 1;

% net_name = 'C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Fully Connected Network\freq_4\path_2\bayesian_optimization\2p4f_h1_1197_h2_923_h3_415_h4_38_dr_0.05_bs_16_valLoss_5.346821.mat';
% fcnet = tiny_architectures_container.tiny_fully_connected_network(fc_params);
% new_net = tiny_architectures_container.network_connecter(net_name, fcnet, 0);
% AE_name = 'C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Fully Connected Network\freq_4\path_1\bayesian_optimization\path_1_freq_4_net_loss_3.317124.mat';
% tiny_net = tiny_architectures_container.tiny_fully_connected_network(fc_params);
% net = tiny_architectures_container.network_connecter(AE_name, tiny_net, AE_learning_rate);
% disp(net.OutputNames);
% disp(net.InputNames);
%gannet = tiny_architectures_container.tiny_graph_network(GAN_params);

% for i = 1:length(new_net.Layers)
%     if strcmp(new_net.Layers(i).Name,'tiny_fc_output')
%         continue; % Skip the tiny_fc_output layer
%     end
%     if isprop(new_net.Layers(i), 'WeightLearnRateFactor') 
%         disp(new_net.Layers(i).Name);
%         disp(new_net.Layers(i).WeightLearnRateFactor);
%         new_net.Layers(i).WeightLearnRateFactor = 0;
%         new_net.Layers(i).BiasLearnRateFactor = 0;
%         new_net.Layers(i).WeightL2Factor = 0;
%         new_net.Layers(i).BiasL2Factor = 0;
%     elseif isprop(new_net.Layers(i), 'OffsetLearnRateFactor') 
%         new_net.Layers(i).OffsetLearnRateFactor = 0;
%         new_net.Layers(i).ScaleLearnRateFactor = 0;
%         new_net.Layers(i).OffsetL2Factor = 0;
%         new_net.Layers(i).ScaleL2Factor = 0;
%     end

     
% end
%deepNetworkDesigner(fcnet);
%deepNetworkDesigner(gannet);
% net = load('C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\f4_FC_Net_with_AE\fc_path_1_net_loss_0.970759.mat').trained_net;
% deepNetworkDesigner(net);

net = tiny_architectures_container.AE_GAN_connector('C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Fully Connected Network\freq_4\path_11\bayesian_optimization', GAN_params, 0.1);
deepNetworkDesigner(net);