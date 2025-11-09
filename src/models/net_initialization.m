% Add all necessary paths relative to project root (LOCAL FIRST for precedence!)
projectRoot='\Users\Maria\Documents\Honours Programme\PZT_data\PZT_L103\GAN';
addpath(fullfile(projectRoot, 'src', 'models','layers'), '-begin');  % HIGH PRIORITY - Local classes
addpath(fullfile(projectRoot, 'src', 'data'), '-begin');    % HIGH
addpath(fullfile(projectRoot, 'src', 'utils'), '-begin');   % HIGH PRIORITY - Local utilities

close all;

% Call the network building function
ls = 1; % Latent size (adjust as needed)
%net = net_builder_CNN_compression.build_net_with_28_inputs(ls); % Adjust latent size as needed
[net, title] = AE_for_each_path_separate.build_net_with_28_inputs(); % Adjust latent size as needed
% save(sprintf('temp_net_%d_improved.mat', ls), 'net'); % Save the network to a .mat file
%save(sprintf('temp_net_%d_CNN_compression.mat', ls), 'net'); % Save the network to a .mat file
save(sprintf('%s.mat', title), 'net'); % Save the network to a .mat file
disp("------------------------------------------------")
disp("Network built successfully. Go to fixed_training.m to start training...")
disp("------------------------------------------------")

