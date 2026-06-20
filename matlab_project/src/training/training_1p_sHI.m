%% MATLAB Path Setup - Run this first!
% Get the directory where THIS script is located
scriptDir = fileparts(mfilename('fullpath'));
fprintf('Script location: %s\n', scriptDir);

% Navigate to project root (2 levels up from src/training)
projectRoot = fullfile(scriptDir, '..', '..');
cd(projectRoot);
fprintf('Changed MATLAB working directory to: %s\n', pwd);

% Add all necessary paths relative to project root (LOCAL FIRST for precedence!)
addpath(fullfile(projectRoot, 'src', 'models'), '-begin');  % HIGH PRIORITY - Local classes
addpath(fullfile(projectRoot, 'src', 'models','layers'), '-begin');  % HIGH PRIORITY - Local classes
addpath(fullfile(projectRoot, 'src', 'data'), '-begin');    % HIGH PRIORITY - Local datastore
addpath(fullfile(projectRoot, 'src', 'utils'), '-begin');   % HIGH PRIORITY - Local utilities  

fprintf('✅ All paths configured successfully\n\n');

%% Custom Loss Functions
function L = multiOutputMSE_monotonicity(latent_size, a, b, c, varargin)    

    if a>0
        MSE = multiOutputMSE(varargin{:});
    else
        MSE = 0;
    end

    if latent_size == 1 && b>0
        mono = monotonicity(varargin{:});
    else
        mono = 0;
    end

    if latent_size == 1 && c>0
        proxy_loss = proxy_labels_loss(varargin{:});
    else
        proxy_loss = 0;
    end

    L = a* MSE + b*mono + c*proxy_loss;
    
end

function L = proxy_labels_loss(varargin)
    predicted_latent = varargin{2};  % latent is stored as a second output (the size is 1xbatchsize)

    % First target is the proxy label (scalar) (the size of varargin{3} is 4000xbatchsize but only first row is not zero, so you take that row for every batch)
    weibull_target = 100* varargin{3}(1,:); 
    L = dlarray(mean((predicted_latent - weibull_target).^2,'all'));

end

function L = monotonicity(varargin)
   
    predicted_latent = varargin{2};  % Second output (the size is 1xbatchsize)
    batchSize = size(predicted_latent, 2);

    % If batch has < 2 elements, no pairwise differences can be computed
    if batchSize < 2
        L = dlarray(0);   % no monotonicity penalty for single-sample batches
        return;
    end

    diff = (predicted_latent( :, 2:end) -  predicted_latent( :,1:end-1)); % Enforce a minimum increase of 0.01 units per cycle
    diff = diff + 0.1;

    diff = diff.*diff;

    L = dlarray(mean(diff, 'all'));

end

function L = multiOutputMSE(varargin)    
  
    Y_pred = varargin{1};    
    Y_target = varargin{4};

    % % clear the figure
    % clf;
    % hold on;
    % plot(Y_target(:,1),color='b');
    % plot(Y_pred(:,1),color='r');
    % legend('Target','Predicted');
    % hold off;
   
    batch_L = sqrt(sum((Y_pred - Y_target).^2, 1)); % Element-wise squared error for each output and batch sample

    L = mean(batch_L); 

end

rng(42); % For reproducibility

%% Data Preparation
num_in = 1;         % Number of input paths (for this program must be 1)
b_size = 27;        % Batch size
overlap_size = floor(b_size/2);   % How much overlap between batches (overlap_size < batch_size)
freq = 4;           % Signal frequency index
path_idx = 2;       % Path index 

% Load the PZT data of each pannel (and every state, frequency, sensor-actuator path)
States103 = load(sprintf("data\\States_%d.mat", 103)).States;   % 32 states, 6 frequencies, 28 paths
States104 = load(sprintf("data\\States_%d.mat", 104)).States;   % 58 states, 6 frequencies, 28 paths
States105 = load(sprintf("data\\States_%d.mat", 105)).States;   % 30 states, 6 frequencies, 28 paths
States109 = load(sprintf("data\\States_%d.mat", 109)).States;   % 28 states, 6 frequencies, 28 paths

% Create the datastores for the chosen path and frequency, the last state is excluded as it was after failure 
inputDs1 = one_path_sHI_sin_freq_fc_datastore(States103, num_in, b_size, overlap_size, 32-1, freq, 'paths', path_idx); 
inputDs2 = one_path_sHI_sin_freq_fc_datastore(States104, num_in, b_size, overlap_size, 58-1, freq, 'paths', path_idx);
inputDs3 = one_path_sHI_sin_freq_fc_datastore(States105, num_in, b_size, overlap_size, 30-1, freq, 'paths', path_idx);
inputDs4 = one_path_sHI_sin_freq_fc_datastore(States109, num_in, b_size, overlap_size, 28-1, freq, 'paths', path_idx);

inputDs_train = combine(inputDs1,  inputDs2, inputDs3, ReadOrder="sequential");
%inputDs_train = inputDs2; % TEMPORARY - use only Cycle 2 for training
inputDs_val = inputDs4;

fprintf('Data preparation complete. Training observations: %d, Validation observations: %d\n', ...
    numpartitions(inputDs_train), numpartitions(inputDs_val));

% Test the datastore
fprintf('Testing datastore...\n');
if hasdata(inputDs4)
    testData = read(inputDs4);
    fprintf('Data table has %d columns (should be %d for %d inputs + %d targets)\n', ...
        width(testData), 2*num_in+1, num_in, num_in+1);
    
    % Check the size of the first input
    firstInput = testData{1,1};  % Get first column, first row
    fprintf('First input size: %s (should be 1x4000)\n', mat2str(size(firstInput)));
    reset(inputDs4);  % Reset for training
end
fprintf('Datastore test complete.\n');

%% Network Training

format_input =repmat("BC", 1, num_in);
format_targets = repmat("BC", 1, num_in+1); % the output of the autoencoder and the 1 dimmensional latent space (sHI)

options = trainingOptions("adam", ...
    MaxEpochs=100, ...  
    MiniBatchSize=b_size, ...  
    InitialLearnRate= 0.01, ... 
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=50, ...
    LearnRateDropFactor=0.5, ...
    GradientThresholdMethod="l2norm", ...
    Metrics={@monotonicity, @multiOutputMSE}, ...
    GradientThreshold=10, ...
    Verbose=true, ...
    VerboseFrequency=10, ...
    ValidationPatience=30, ... 
    Plots="training-progress", ...
    InputDataFormats=format_input, ...
    TargetDataFormats=format_targets, ...
    Shuffle="every-epoch",...
    ValidationData=inputDs_val,...
    ValidationFrequency=10);

% Net configuration
fc_params.input_size = 15;
fc_params.latent_size = 1;
% Net transforming the latent space of the big autoencoder into one value (sHI)
tiny_net = tiny_architectures_container.tiny_fully_connected_network(fc_params);

% Path to the folder containing optimized AEs for each path index 
AE_folder= 'C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Sparse_Fully_Connected_AE\freq_4\bayesian_optimization'; 
% find the path to the file with the AE for the specified path_idx (search for every file with path_indx in the name)
pattern = sprintf('%dp*.mat', path_idx);        
files   = dir(fullfile(AE_folder, pattern));
if length(files) > 1
    error('Multiple AE model files found for path index %d in folder %s. Please ensure only one file matches the pattern.', path_idx, AE_folder);
elseif isempty(files)
    error('No AE model file found for path index %d in folder %s. Please ensure a file matches the pattern.', path_idx, AE_folder);
else
    AE_name = fullfile(AE_folder, files(1).name); % Take the first match (there should be only one file per path index)
end

AE_learning_rate = 0.0; % Learning rate for the autoencoder weights and biases
net = tiny_architectures_container.network_connecter(AE_name, tiny_net, AE_learning_rate);

disp("Number of network outputs and inputs:");
disp(length(net.OutputNames)+length(net.InputNames));
net_name = "FC_Net_with_ae_training";

fprintf('Network loaded successfully. Starting training...\n');

% Loss function setup 
[a, b, c] = deal(0.0, 0.0, 1.0); % Weights for MSE, monotonicity, proxy loss
lossFcn = @(varargin) multiOutputMSE_monotonicity(fc_params.latent_size,a,b,c,varargin{:}); 

% Train the network
[trained_net, training_info] = trainnet(inputDs_train, net, lossFcn, options);

%% Save Results
% Create results directory if it doesn't exist
resultsDir = fullfile(projectRoot, 'results',sprintf('f%d_%s_with_AE', freq, net_name));
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
    fprintf('Created results directory: %s\n', resultsDir);
end


% Save complete model with timestamp and metadata
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

% Save with metadata
model_metadata = struct();
model_metadata.timestamp = timestamp;
model_metadata.num_inputs = num_in;
model_metadata.path_index = path_idx;
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

modelName = sprintf("fc_path_%d_net_loss_%f.mat", path_idx, model_metadata.final_val_loss);

modelPath = fullfile(resultsDir, modelName);

save(modelPath, "trained_net", "training_info", "model_metadata", '-v7.3');
fprintf('✅ Complete model saved to: %s\n', modelPath);



