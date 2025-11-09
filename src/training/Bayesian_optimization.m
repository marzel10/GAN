% Bayesian Hyperparameter Optimization for Autoencoder
% This script optimizes the architecture and training parameters

%% Setup
close all; clear; clc;
rng(42); % For reproducibility
%% MATLAB Path Setup - Run this first!
% Get the directory where THIS script is located
scriptDir = fileparts(mfilename('fullpath'));
fprintf('Script location: %s\n', scriptDir);

% Navigate to project root (2 levels up from src/training)
projectRoot = fullfile(scriptDir, '..', '..');
cd(projectRoot);
fprintf('Changed MATLAB working directory to: %s\n', pwd);

% Remove any conflicting paths from old PZT folder (to avoid conflicts)
if contains(path, 'C:\Users\Maria\Documents\Honours Programme\PZT')
    rmpath('C:\Users\Maria\Documents\Honours Programme\PZT');
    fprintf('🗑️  Removed old PZT folder from path to avoid conflicts\n');
end

% Add all necessary paths relative to project root (LOCAL FIRST for precedence!)
addpath(fullfile(projectRoot, 'src', 'models'), '-begin');  % HIGH PRIORITY - Local classes
addpath(fullfile(projectRoot, 'src', 'models','layers'), '-begin');  % HIGH PRIORITY - Local classes
addpath(fullfile(projectRoot, 'src', 'data'), '-begin');    % HIGH PRIORITY - Local datastore
addpath(fullfile(projectRoot, 'src', 'utils'), '-begin');   % HIGH PRIORITY - Local utilities  
addpath('C:\Users\Maria\Documents\Honours Programme\PZT', '-end');  % LOW PRIORITY - Original data files

fprintf('✅ All paths configured successfully\n\n');
%% Load your data (modify paths as needed)
% Create single datastore that handles all 28 inputs and targets
num_in = 28;
b_size = 4;
overlap_size = 2;

Cycle1 = load(sprintf("data\\Cycle_%d.mat", 103)).Cycle1;  % Load the Cycle1 datastore
Cycle2 = load(sprintf("data\\Cycle_%d.mat", 104)).Cycle1;  % Load the Cycle2 datastore
Cycle3 = load(sprintf("data\\Cycle_%d.mat", 105)).Cycle1;  % Load the Cycle3 datastore
Cycle4 = load(sprintf("data\\Cycle_%d.mat", 109)).Cycle1;  % Load the Cycle4 datastore


% 148 instances of data for the training and 28 for validation
envelope = true; % Whether to use envelope of signals or raw signals
inputDs1 = CyclemultiInputDatastore_separate(Cycle1, num_in, b_size, overlap_size,32,envelope);
inputDs2 = CyclemultiInputDatastore_separate(Cycle2, num_in, b_size, overlap_size,58,envelope);
inputDs3 = CyclemultiInputDatastore_separate(Cycle3, num_in, b_size, overlap_size,30,envelope);
inputDs4 = CyclemultiInputDatastore_separate(Cycle4, num_in, b_size, overlap_size,28,envelope);
inputDs_train = combine(inputDs1,  inputDs2,ReadOrder="sequential");  % Combine datastores
inputDs_val = combine(inputDs4, inputDs3,ReadOrder="sequential");

trainData = inputDs_train;
valData = inputDs_val;
%% Define Hyperparameters to Optimize

% Architecture parameters
channels1 = optimizableVariable('channels1', [8, 32], 'Type', 'integer');
channels2 = optimizableVariable('channels2', [4, 16], 'Type', 'integer');
channels3 = optimizableVariable('channels3', [2, 6], 'Type', 'integer');

kernel_size1 = optimizableVariable('kernel_size1', [15, 35], 'Type', 'integer');
kernel_size2 = optimizableVariable('kernel_size2', [5, 15], 'Type', 'integer');
kernel_size3 = optimizableVariable('kernel_size3', [3, 7], 'Type', 'integer');

stride1 = optimizableVariable('stride1', [10, 20], 'Type', 'integer');  % Increased min to reduce latent size
stride2 = optimizableVariable('stride2', [5, 10], 'Type', 'integer');

% Training parameters
learning_rate = optimizableVariable('learning_rate', [1e-4, 1e-2], 'Transform', 'log');
batch_size = 4;
% dropout_rate = optimizableVariable('dropout_rate', [0.0, 0.3]);
l2_regularization = optimizableVariable('l2_regularization', [1e-6, 1e-3], 'Transform', 'log');

% Dilation rates
% dilation1 = optimizableVariable('dilation1', [1, 4], 'Type', 'integer');
% dilation2 = optimizableVariable('dilation2', [2, 8], 'Type', 'integer');

%% Bayesian Optimization Settings

% Combine all optimizable variables
optimVars_full = [channels1, channels2, channels3,  ...
             kernel_size1, kernel_size2, kernel_size3, stride1, stride2, ...
             learning_rate,  l2_regularization];

optimVars_compressed = [channels1, channels2,   ...
            kernel_size1, kernel_size2,  stride1, stride2, ...
            learning_rate,  l2_regularization];

optimVars = optimVars_full;  % Use compressed version to enforce latent size constraint

% Bayesian optimization options
bayesOpts = struct(...
    'MaxObjectiveEvaluations', 1, ...  % Adjust based on your time budget
    'MaxTime', 24*3600, ...              % 24 hours max
    'IsObjectiveDeterministic', false, ...
    'UseParallel', false, ...             % Use parallel if available
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'Verbose', 1, ...
    'OutputFcn', @boCheckpointFn);


% Convert struct fields to name-value pairs
bayesOptArgs = struct2cell(bayesOpts);
bayesOptNames = fieldnames(bayesOpts);

disp(bayesOptNames);
disp(bayesOptArgs);

nameValuePairs = [bayesOptNames, bayesOptArgs];




%% Define Objective Function

objectiveFcn = @(params) trainAndEvaluateAE(params, trainData, valData);

%% Run Bayesian Optimization

fprintf('Starting Bayesian Optimization...\n');
results = bayesopt(objectiveFcn, optimVars, ...
    'MaxObjectiveEvaluations', bayesOpts.MaxObjectiveEvaluations, ...
    'MaxTime', bayesOpts.MaxTime, ...
    'IsObjectiveDeterministic', bayesOpts.IsObjectiveDeterministic, ...
    'UseParallel', bayesOpts.UseParallel, ...
    'AcquisitionFunctionName', bayesOpts.AcquisitionFunctionName, ...
    'OutputFcn', bayesOpts.OutputFcn);

%% Save Results

outDir = fullfile(projectRoot,'results');
if ~exist(outDir,'dir'), mkdir(outDir); end
finalFile = fullfile(outDir,'bayesian_optimization_results.mat');
save(finalFile, 'results', '-v7.3');   % save once, not inside OutputFcn
fprintf('Saved BO results: %s\n', finalFile);
fprintf('\nOptimization complete!\n');
fprintf('Best validation loss: %.6f\n', results.MinObjective);
disp('Best hyperparameters:');
disp(results.XAtMinObjective);

%% Visualize Results

outDir = fullfile(projectRoot,'results');
if ~exist(outDir,'dir'), mkdir(outDir); end

obj = results.ObjectiveTrace(:);
iters = 1:numel(obj);
bestObs = cummin(obj);  % best-so-far (observed)

h = figure('Position',[100 100 1200 400],'Color','w');

subplot(1,3,1);
plot(iters, obj, 'b-o','MarkerSize',3); grid on;
xlabel('Iteration'); ylabel('Validation Loss');
title('Optimization Progress');

subplot(1,3,2);
plot(iters, bestObs, 'r-','LineWidth',1.5); grid on;
xlabel('Iteration'); ylabel('Best Validation Loss');
title('Best-So-Far (Observed)');

subplot(1,3,3);
if ~isempty(obj)
    histogram(obj, 20); grid on;
else
    text(0.5,0.5,'No objective values','HorizontalAlignment','center'); axis off;
end
xlabel('Validation Loss'); ylabel('Frequency');
title('Loss Distribution');

exportgraphics(h, fullfile(outDir,'optimization_results.png'), 'Resolution',150);
close(h);

%% Remove all the models that are not the best one to save space
modelFiles = dir(fullfile(pwd, 'net_*.mat'));
bestParams = results.XAtMinObjective;
bestModelFile = sprintf('net_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d.mat', ...
    bestParams.channels1, bestParams.channels2, bestParams.channels3, ...
    bestParams.kernel_size1, bestParams.kernel_size2, bestParams.kernel_size3, ...
    bestParams.stride1, bestParams.stride2, ...
    bestParams.learning_rate, bestParams.l2_regularization);
for k = 1:length(modelFiles)
    if ~strcmp(modelFiles(k).name, bestModelFile)
        delete(fullfile(modelFiles(k).folder, modelFiles(k).name));
    end
end
%% (Optional) Train Final Model with Best Hyperparameters

% fprintf('\nTraining final model with best parameters...\n');
% bestParams = results.XAtMinObjective;
% finalNet = buildOptimizedNetwork(bestParams);

% % Train with more epochs
% finalLoss = trainAndEvaluateAE(bestParams, trainData, valData, 'FinalTraining', true);
% fprintf('Final validation loss: %.6f\n', finalLoss);

% save('best_autoencoder_model.mat', 'finalNet', 'bestParams');

%% ========== HELPER FUNCTIONS ==========

function loss = trainAndEvaluateAE(params, trainData, valData, varargin)
    % Train and evaluate autoencoder with given hyperparameters
    
    p = inputParser;
    addParameter(p, 'FinalTraining', false);
    parse(p, varargin{:});
    isFinal = p.Results.FinalTraining;
    
    try
        % Check if latent size constraint is satisfied
        input_x = 4000;
        
        latentT = floor(input_x / (params.stride1 * params.stride2 * ceil(params.kernel_size2/2) * 1)); % considering pooling
        latent_size = latentT * 1 * params.channels3;
        
        % If latent size exceeds 100, penalize heavily
        if latent_size > 100
            loss_latent= 10 * latent_size;  % Heavy penalty
            fprintf('Latent size %d exceeds limit (s1=%d, s2=%d, p=%d, ch3=%d). Penalized.\n', ...
                    latent_size, params.stride1, params.stride2, ceil(params.kernel_size2/2), params.channels3);
        else
            loss_latent=0;    
        end
        
        disp('Building and training network with parameters:');
        % Build network with current parameters
        net = buildOptimizedNetwork_compressed(params);
        
        % Training options
        if isFinal
            maxEpochs = 30;
        else
            maxEpochs = 10;  % Shorter for optimization
        end
        
        batch_size=4;  % Fixed batch size for simplicity
        options = trainingOptions('adam', ...
            'InitialLearnRate', params.learning_rate, ...
            'MiniBatchSize', batch_size, ...
            'MaxEpochs', maxEpochs, ...
            'L2Regularization', params.l2_regularization, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropPeriod', 20, ...
            'LearnRateDropFactor', 0.5, ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', valData, ...
            'ValidationFrequency', 50, ...
            'Verbose', true, ...
            'Plots', 'none');
        
        lossFn = @(varargin) amplitude_aware_loss(varargin{:});
        % Train network
        [net, info] = trainnet(trainData, net, lossFn, options);
        save(sprintf('net_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d.mat', params.channels1, params.channels2, params.channels3, kernel_size1, kernel_size2, kernel_size3, stride1, stride2, ...
             learning_rate,  l2_regularization), 'net','-v7.3');
        valLoss = min(info.ValidationHistory.Loss);
    
        % Penalize for complexity (optional)
        complexityPenalty = 0.0001 * (params.channels1 + params.channels2);
        loss = valLoss + complexityPenalty + loss_latent;
        
    catch ME
        % Return high loss if training fails
        fprintf('Training failed: %s\n', ME.message);
        fprintf('%s\n', getReport(ME,'extended')); % full stack + causes
        loss = 1e6;
    end
end
function net = buildOptimizedNetwork_compressed(params)
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
    stride2        = [params.stride2 1];     % 400 -> 200
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
        l_r_crop = stride2(2)+kernel_conv2(2)-2;
        crop2 = [floor(t_d_crop/2), ceil(t_d_crop/2), floor(l_r_crop/2), ceil(l_r_crop/2)];
    end
    
    

    latentT = ceil(ceil(ceil(input_x / stride1(1)) / stride2(1)) / stride_pool2(1));                     % 4000/10/2/4
    single_input_size = latentT*1*channels_conv3;  % 50*1*12 = 600
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
             
     

% Custom composite loss (L1 + transient weight)
function L = amplitude_aware_loss(varargin)
    numOut = numel(varargin)/2;
    accum = 0;
    for k=1:numOut
        Yk = varargin{1,k};
        Tk = varargin{1,k+numOut};
        if iscell(Yk), Yk=Yk{1}; end
        if iscell(Tk), Tk=Tk{1}; end
        % L1 + MSE mix
        l1 = mean(abs(Yk-Tk),'all');
        mse = mean((Yk-Tk).^2,'all');
        % Emphasize higher-amplitude (post scaling) > 0.01
        mask = abs(Tk) > 0.01;
        if any(mask,'all')
            high = mean(abs(Yk(mask)-Tk(mask)));
            accum = accum + (0.5*mse + 0.3*l1 + 0.2*high);
        else
            accum = accum + (0.6*mse + 0.4*l1);
        end
    end
    L = accum/numOut;
end

function stop = boCheckpointFn(boResults, state)
% Lightweight checkpoint to avoid saving the full BayesianOptimization object.
stop = false;
if ~strcmp(state,'iteration'), return; end
try
    outDir = fullfile(pwd,'results');
    if ~exist(outDir,'dir'), mkdir(outDir); end

    payload.iteration      = boResults.NumObjectiveEvaluations;
    payload.MinObjective   = boResults.MinObjective;
    payload.ObservedMin    = boResults.MinObjective; % alias
    payload.XTrace         = boResults.XTrace;
    payload.ObjectiveTrace = boResults.ObjectiveTrace;

    tmp = fullfile(outDir, sprintf('bo_checkpoint_tmp_%03d.mat', payload.iteration));
    dst = fullfile(outDir, 'bo_checkpoint.mat');

    save(tmp, '-struct', 'payload', '-v7');  % small struct only
    movefile(tmp, dst, 'f');                 % atomic replace
catch ME
    warning(ME.identifier, '%s', ME.message);
end
end