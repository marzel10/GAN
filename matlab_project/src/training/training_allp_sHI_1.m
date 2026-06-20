%% MATLAB Path Setup - Run this first!
% Get the directory where THIS script is located
scriptDir = fileparts(mfilename('fullpath'));
fprintf('Script location: %s\n', scriptDir);

% Navigate to project root (2 levels up from src/training)
projectRoot = fullfile(scriptDir, '..', '..');
cd(projectRoot);
fprintf('Changed MATLAB working directory to: %s\n', pwd);

% Remove any conflicting paths from old PZT folder (to avoid conflicts)
if contains(matlabpath, 'C:\Users\Maria\Documents\Honours Programme\PZT')
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

%% Verify which class files MATLAB will use

fprintf('🔍 Verifying class locations:\n');
ganLocation = which('GAN');
deconLocation = which('deconcatenation');
datastoreLocation = which('CyclemultiInputDatastore');

if contains(ganLocation, 'GAN\src\models')
    fprintf('✅ GAN class: %s (LOCAL - GOOD!)\n', ganLocation);
else
    fprintf('⚠️  GAN class: %s (EXTERNAL - might be wrong!)\n', ganLocation);
end

if contains(deconLocation, 'GAN\src\models')
    fprintf('✅ deconcatenation class: %s (LOCAL - GOOD!)\n', deconLocation);
else
    fprintf('⚠️  deconcatenation class: %s (EXTERNAL - might be wrong!)\n', deconLocation);
end

if contains(datastoreLocation, 'GAN\src\data')
    fprintf('✅ CyclemultiInputDatastore3 class: %s (LOCAL - GOOD!)\n', datastoreLocation);
else
    fprintf('⚠️  CyclemultiInputDatastore3 class: %s (EXTERNAL - might be wrong!)\n', datastoreLocation);
end
fprintf('✅ All paths configured successfully\n\n');


%% Custom Loss Function

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

% Custom composite loss (L1 + transient weight)
function L = mse_l1_loss(varargin)
    numOut = numel(varargin)/2;
    accum = 0;
    for k=1:numOut
        Yk = varargin{1,k};
        Tk = varargin{1,k+numOut};
        if iscell(Yk), Yk=Yk{1}; end
        if iscell(Tk), Tk=Tk{1}; end

        % Penalize higher amplitudes more
        weights =  abs(Tk).^2+0.1; % Simple linear weighting based on target amplitude
        weights(abs(Tk) < 0.1) = 0.1; % High weight for low amplitudes

        % L1 + MSE mix
        l1 = mean(weights .* abs(Yk-Tk),'all');
        mse = mean(weights .* (Yk-Tk).^2,'all');
        accum = accum + (0.6*mse + 0.4*l1);
    end
    L = accum/numOut;
end

function L = multiOutputMSE_monotonicity(latent_size, a, b, c, conv_mode, varargin)      
    if a>0
        MSE = multiOutputMSE(varargin{:});
    else
        MSE = 0;
    end
    if latent_size == 1 && b>0
        if conv_mode
            mono = monotonicity_conv(varargin{:});
        else
            mono = monotonicity1(varargin{:});
        end
    else
        mono = 0;
    end
    if latent_size == 1 && c>0
        if conv_mode
            proxy_loss = global_proxy_labels_loss_conv(varargin{:});
        else
            proxy_loss = global_proxy_labels_loss(varargin{:});
        end
    else
        proxy_loss = 0;
    end

    L = a* MSE + b*mono + c*proxy_loss;
    
end

function L = proxy_labels_loss(varargin)
    
    predicted_latent = varargin{1};  % latent is stored as a second output (the size is 1xbatchsize)
    weibull_target = varargin{2}(1,:); % First target is the proxy label (scalar) (the size of varargin{3} is 4000xbatchsize but only first row is not zero, so you take that row for every batch)
    weibull_target = repmat(weibull_target, size(predicted_latent,2),1); % Replicate to match predicted_latent size
    weibull_target = reshape(weibull_target, size(predicted_latent)); % Transpose to match predicted_latent size
 
   
    L = dlarray(mean((predicted_latent - weibull_target).^2,'all'));

end

function L = global_proxy_labels_loss(varargin)
    
    predicted_latent = varargin{1};  % latent is stored as a second output (the size is 1xbatchsize)
  
    global_health = sum(predicted_latent.^2,2); % Sum over path dimension to get global health indicator

    weibull_target = varargin{2}(1,:); % First target is the proxy label (scalar) (the size of varargin{3} is 4000xbatchsize but only first row is not zero, so you take that row for every batch)
 
    %weibull_target = repmat(weibull_target, size(predicted_latent,2),1); % Replicate to match predicted_latent size
    weibull_target = reshape(weibull_target, size(global_health)); % Transpose to match predicted_latent size
    
   
    L = dlarray(mean((global_health - weibull_target).^2,'all'));

end

function L = global_proxy_labels_loss_conv(varargin)
    
    predicted_latent = varargin{1};  % size: 1x1x28xbatchsize
  
    global_health = sum(predicted_latent.^2,2); % Sum over path dimension to get global health indicator (size 1x1x1xbatchsize)
   disp(size(global_health));
    weibull_target = varargin{2}(1,:); 
 
    weibull_target = reshape(weibull_target, size(global_health)); % Transpose to match predicted_latent size
   
    L = dlarray(mean((global_health - weibull_target).^2,'all'));

end



function L = monotonicity(varargin)
    relu_version = true; % Set to true to use ReLU-based monotonicity loss
    predicted_latent = varargin{1};  % Second output (the size is 1xbatchsize)
    r = 9.5; % How strong the increase should be penalized
    if relu_version
        % ReLU-based monotonicity loss
        diff_per_path = predicted_latent(1, :, 2:end) -  predicted_latent(1, :, 1:end-1); % Enforce a minimum increase of 1 unit per cycle
        diff_per_path = relu(diff_per_path);  % Differentiable alternative to max(0, ...)
        health_per_state = sum(diff_per_path.^2,2);
        weibull_target = varargin{2}(1,:);
        n = size(health_per_state,3);
        violations= mean((health_per_state-weibull_target(1,1:n)).^2,'all');
    
        % violations = health_per_state(1,1,n+1)-health_per_state(1,1,1)+0.01*n; % Total violation over the path dimension
        % violations = relu(violations); % Only consider positive violations
      
    else
        % Squared difference-based monotonicity loss
        diff = predicted_latent(1, :, 2:end) -  predicted_latent(1, :, 1:end-1) + r;
        diff = diff.* diff; % Square the differences to penalize larger violations more
        violations = sum(diff, 2)./r^2; % Sum over the path dimmension
      
        
    end
    L = dlarray(mean(violations, 'all'));
end

function L = monotonicity1(varargin)
    relu_mode = false; % Set to true to use ReLU-based monotonicity loss
    predicted_latent = varargin{1};  % Second output (the size is 1x28xbatchsize)
    weibull_target = varargin{2}(1,:); % Extract it just so see which sample is from the earlier cycle (size 1xbatchsize)   
    signs = zeros(1,28,size(weibull_target,2)-1);
    for i = 2:size(weibull_target,2)
        if weibull_target(1,i) < weibull_target(1,i-1)
            signs(1,:,i-1) = 1;
        else
            signs(1,:,i-1) = -1;

        end
    end

    diff_per_path = signs .* (predicted_latent(1, :, 2:end) -  predicted_latent(1, :, 1:end-1)); % size 1x28x(batchsize-1)
    if relu_mode
        diff_per_path = relu(diff_per_path);  % Differentiable alternative to max(0, ...)
    else
        diff_per_path = diff_per_path .* diff_per_path; % Square the differences to penalize larger violations more
        
        diff_per_path = sum(diff_per_path,2); % Take square root to bring back to original scale
    end
    
    
    L = dlarray(mean(diff_per_path,'all'));
end

function L = monotonicity_conv(varargin)
    predicted_latent = varargin{1};  % Second output (the size is 1x28xbatchsize)
    disp(size(predicted_latent));
    weibull_target = varargin{2}(1,:); % Extract it just so see which sample is from the earlier cycle (size 1xbatchsize)
        
    relu_mode = false; % Set to true to use ReLU-based monotonicity loss
    signs = zeros(1,28,size(weibull_target,2)-1,1);
    for i = 2:size(weibull_target,2)
        if weibull_target(1,i) < weibull_target(1,i-1)
            signs(1,:,i-1) = 1;
        else
            signs(1,:,i-1) = -1;

        end
    end

    diff_per_path = signs .* (predicted_latent(1, :, 2:end,1) -  predicted_latent(1, :, 1:end-1,1)); % size 1x28x(batchsize-1)
    disp(size(diff_per_path));
    if relu_mode
        diff_per_path = relu(diff_per_path);  % Differentiable alternative to max(0, ...)
    else
        diff_per_path = diff_per_path .* diff_per_path; % Square the differences to penalize larger violations more
        diff_per_path = sum(diff_per_path,3); % Take square root to bring back to original scale
    end
    
    L = dlarray(mean(diff_per_path,'all'));
end

function L = multiOutputMSE(varargin)    
    numOut = numel(varargin)/2-1; 
    accum = 0; 
   
    for k = 1:numOut 

        Yk = varargin{1,k}; 
        Tk = varargin{1,k+numOut+2}; 
        % Unwrap if elements are cells (can happen with some datastores/trainnet packings)
        if iscell(Yk), Yk = Yk{1}; end
        if iscell(Tk), Tk = Tk{1}; end

        accum = accum + mean((Yk - Tk).^2,'all'); 
    end 
    L = accum / numOut; 
end

%% Data Preparation

rng(42); % For reproducibility
% Create single datastore that handles all 28 inputs and targets

b_size = 4;
overlap_size = 3;
% 148 instances of data for the training and 28 for validation
freq = 4; % frequency index to extract
fc = false; % whether to use fully connected network
net_folder = 'C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Sparse_Fully_Connected_AE\freq_4\bayesian_optimization';
conv_mode = true; % whether to use convolutional format (SSC) or not (SCB)


cacheRoot = fullfile(projectRoot, 'data', 'latent_cache');
if ~exist(cacheRoot,'dir'), mkdir(cacheRoot); end
cf103 = fullfile(cacheRoot, sprintf('latent_cache_panel_%d_freq_%d_wls.mat', 103, freq));
cf104 = fullfile(cacheRoot, sprintf('latent_cache_panel_%d_freq_%d_wls.mat', 104, freq));
cf105 = fullfile(cacheRoot, sprintf('latent_cache_panel_%d_freq_%d_wls.mat', 105, freq));
cf109 = fullfile(cacheRoot, sprintf('latent_cache_panel_%d_freq_%d_wls.mat', 109, freq));

% fc=false means we use GAN; datastore expects GAN_mode=true to output C=28
ganMode = ~fc;
inputDs1 = latent_space_datastore(103, b_size, overlap_size, freq, ganMode, net_folder, 'cacheFile', cf103, 'enable_shuffle', true, 'convolutional_mode', conv_mode);
inputDs2 = latent_space_datastore(104, b_size, overlap_size, freq, ganMode, net_folder, 'cacheFile', cf104, 'enable_shuffle', true, 'convolutional_mode', conv_mode);
inputDs3 = latent_space_datastore(105, b_size, overlap_size, freq, ganMode, net_folder, 'cacheFile', cf105, 'enable_shuffle', true, 'convolutional_mode', conv_mode);
inputDs4 = latent_space_datastore(109, b_size, overlap_size, freq, ganMode, net_folder, 'cacheFile', cf109, 'enable_shuffle', true, 'convolutional_mode', conv_mode);

% test size of the input data
disp("Training data size:");

inputDs_train = combine(inputDs1,  inputDs2, inputDs3, ReadOrder="sequential");
inputDs_val = inputDs4;
fprintf('Data preparation complete. Training observations: %d, Validation observations: %d\n', ...
    numpartitions(inputDs_train), numpartitions(inputDs_val));


%% Network Training





if conv_mode
    format_input ="SSCB";
    format_targets = "SSCB"; 
    metric_fc = {@monotonicity_conv, @global_proxy_labels_loss_conv};
else
    format_input ="SCB";
    format_targets = "SCB"; 
    metric_fc = {@monotonicity, @global_proxy_labels_loss};
end
options = trainingOptions("adam", ...
    MaxEpochs=100, ...  % More epochs
    MiniBatchSize=b_size, ...  % Smaller batch size for stability
    InitialLearnRate= 0.0010145, ... % Lower learning rate
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=15, ...
    LearnRateDropFactor=0.5, ...
    GradientThresholdMethod="l2norm", ...
    Metrics=metric_fc, ...
    GradientThreshold=10, ...
    Verbose=true, ...
    VerboseFrequency=10, ...
    ValidationPatience=100, ...  % More patience
    Plots="training-progress", ...
    InputDataFormats=format_input, ...
    TargetDataFormats=format_targets, ...
    Shuffle="every-epoch",...
    ValidationData=inputDs_val,...
    ValidationFrequency=10);

GAN_params.input_size = 15;
GAN_params.latent_size = 1;
GAN_params.adjacency_matrix = attention_matrix.build_attention();


[a, b, c] = deal(0, 0, 1); % Weights for MSE, monotonicity, proxy loss
% Select scalable composite loss (append scaling args) REMEBER TO CHANGE METADATA THEN
lossFcn = @(varargin) multiOutputMSE_monotonicity(GAN_params.latent_size,a,b,c,conv_mode,varargin{:}); % Use standard MSE for stability
% lossFcn = @(varargin) peak_preserving_noise_suppressing_loss(varargin{:});

if conv_mode
    %net = tiny_architectures_container.path_convolution();
    GAN_params_conv.input_size = 15;
    GAN_params_conv.latent_size = 1;
    GAN_params_conv.adjacency_matrix = attention_matrix.build_attention();
    GAN_params_conv.conv_mode = true;
    net = tiny_architectures_container.path_convolution_and_GNN(GAN_params_conv);
else
    net = tiny_architectures_container.tiny_graph_network(GAN_params);
end 

disp("Number of network outputs and inputs:");
disp(length(net.OutputNames)+length(net.InputNames));
net_name = "GAN_Net_without_mono";

fprintf('Network loaded successfully. Starting training...\n');

%An 'epoch' represents one pass through the entire training dataset, 
% while an 'iteration' corresponds to one update of the model's parameters using a mini-batch of data during training.

% Train the network
[trained_net, training_info] = trainnet(inputDs_train, net, lossFcn, options);

disp(training_info.TrainingHistory);
%% Save Results

% Create results directory if it doesn't exist
resultsDir = fullfile(projectRoot, 'results',sprintf('f%d_%s_with_AE', freq, net_name));
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
    fprintf('Created results directory: %s\n', resultsDir);
end


% Save complete model with timestamp and metadata
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

% Save with comprehensive metadata
model_metadata = struct();
model_metadata.timestamp = timestamp;
model_metadata.frequency = freq;
model_metadata.batch_size = b_size;
model_metadata.max_epochs = options.MaxEpochs;
model_metadata.initial_lr = options.InitialLearnRate;
model_metadata.optimizer = "adam";
model_metadata.loss_function = "mse_l1_loss";
model_metadata.data_cycles = [103, 104, 105, 109];
model_metadata.loss_weights = struct('MSE', a, 'Monotonicity', b, 'Proxy', c);
model_metadata.final_loss = training_info.TrainingHistory.Loss(end);
model_metadata.final_val_loss = training_info.ValidationHistory.Loss(end);

modelName = sprintf("%s_loss_%f.mat", net_name, model_metadata.final_val_loss);


modelPath = fullfile(resultsDir, modelName);


save(modelPath, "trained_net", "training_info", "model_metadata", '-v7.3');
fprintf('✅ Complete model saved to: %s\n', modelPath);



