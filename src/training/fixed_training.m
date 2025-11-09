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

%% Custom Loss Functions


function L = multiOutputMSE_monotonicity(latent_size, a, b, c, varargin)    
    numOut = numel(varargin)/2; 
    accum = 0; 

    % Expand varargin when forwarding to helper losses
    mono = monotonicity(varargin{:});
    MSE = multiOutputMSE(varargin{:});
    if latent_size == 1
        proxy_loss = proxy_labels_loss(varargin{:});
    else
        proxy_loss = 0;
    end

    L = a* MSE + b*mono + c*proxy_loss;
    
end

function L = proxy_labels_loss(varargin)
    numOut = numel(varargin)/2; 
    latent = varargin{1,1};  % First argument is the prediction to check
    weibull_target = varargin{1, numOut+1}(:,1); % First target is the proxy label (scalar)

    L = dlarray(mean((latent - weibull_target).^2,'all'));

end

function L = monotonicity(varargin)
    
    latent = varargin{1,1};  % First argument is the prediction to check
    % Unwrap if trainnet passed a nested cell for the first output
    if iscell(latent)
        latent = latent{1};
    end
    % Use only differentiable operations
    % Example: penalize negative differences between consecutive elements
    diff = latent( :, :,2:end) - latent( :, :,1:end-1);
    violations = (diff-10).^2;  % Differentiable alternative to max(0, ...)
    L = dlarray(mean(violations, 'all'));
end

function L = multiOutputMSE(varargin)    
    numOut = numel(varargin)/2-1; 
    accum = 0; 
   
    for k = 1:numOut 

        Yk = varargin{1,k+1}; 
        Tk = varargin{1,k+numOut+1}; 
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
num_in = 28;
b_size = 4;
overlap_size = 2;

Cycle1 = load(sprintf("data\\Cycle_%d.mat", 103)).Cycle1;  % Load the Cycle1 datastore
Cycle2 = load(sprintf("data\\Cycle_%d.mat", 104)).Cycle1;  % Load the Cycle2 datastore
Cycle3 = load(sprintf("data\\Cycle_%d.mat", 105)).Cycle1;  % Load the Cycle3 datastore
Cycle4 = load(sprintf("data\\Cycle_%d.mat", 109)).Cycle1;  % Load the Cycle4 datastore


% 148 instances of data for the training and 28 for validation
inputDs1 = CyclemultiInputDatastore(Cycle1, num_in, b_size, overlap_size,32);
inputDs2 = CyclemultiInputDatastore(Cycle2, num_in, b_size, overlap_size,58);
inputDs3 = CyclemultiInputDatastore(Cycle3, num_in, b_size, overlap_size,30);
inputDs4 = CyclemultiInputDatastore(Cycle4, num_in, b_size, overlap_size,28);
inputDs_train = combine(inputDs1,  inputDs2,ReadOrder="sequential");  % Combine datastores
inputDs_val = combine(inputDs4, inputDs3,ReadOrder="sequential");

% Test the datastore
fprintf('Testing datastore...\n');
if hasdata(inputDs1)
    testData = read(inputDs1);
    fprintf('Data table has %d columns (should be %d for %d inputs + %d targets)\n', ...
        width(testData), 2*num_in, num_in, num_in);
    
    disp(size(testData));  % Display size of the table
    % Check the size of the first input
    firstInput = testData{1,1};  % Get first column, first row
    fprintf('First input size: %s (should be 4000×2×6)\n', mat2str(size(firstInput)));
    disp(firstInput(1,1:100,:,1));  % Display a slice of the first input for verification
    reset(inputDs1);  % Reset for training
end

%% Network Training

options = trainingOptions("adam", ...  % Adam optimizer for better convergence
    MaxEpochs=10, ...  % Increased epochs for the improved network
    MiniBatchSize=b_size, ...  % Smaller batch size for stability
    InitialLearnRate=0.001, ...  % Conservative learning rate
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=3, ...
    LearnRateDropFactor=0.5, ...
    GradientThreshold=1, ...  % Gradient clipping to prevent explosion
    Verbose=true, ...
    Plots="training-progress", ...
    Metrics={@monotonicity, @multiOutputMSE, @proxy_labels_loss}, ...
    InputDataFormats=repmat("BSSC", 1, 28), ...  % 28 inputs
    TargetDataFormats=repmat("BSSC", 1, 29), ...  % 29 targets (including G2/latent_out)
    Shuffle="never",...
    ValidationData=inputDs_val,...
    ValidationFrequency=5,...
    ValidationPatience=5);
    
latent_size = 1; % Size of the latent output (1 for scalar Weibull proxy)
[a, b, c] = deal(1, 1, 0); % Weights for MSE, monotonicity, proxy loss
% Select scalable composite loss (append scaling args)
lossFcn = @(varargin) multiOutputMSE_monotonicity(latent_size, a, b, c, varargin{:});


model_type = 'CNN_compression_with_shared_weights';
% model_type = 'CNN_compression';
% ds_type = 'CyclemultiInputDatastore_envelope';
ds_type = 'CyclemultiInputDatastore';
load(sprintf('temp_net_%d_CNN_compression_shared.mat', latent_size), 'net'); % Load the pre-built network
fprintf('Network loaded successfully. Starting training...\n');

%An 'epoch' represents one pass through the entire training dataset, 
% while an 'iteration' corresponds to one update of the model's parameters using a mini-batch of data during training.

% Train the network
[trained_net, training_info] = trainnet(inputDs_train, net, lossFcn, options);

if isnan(training_info.TrainingHistory.multiOutputMSE(end))
    error('Training resulted in NaN loss. Please check the training configuration and data.');
end
%% Save Results
% Create results directory if it doesn't exist
resultsDir = fullfile(projectRoot, 'results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
    fprintf('Created results directory: %s\n', resultsDir);
end

% Save training info (you already had this)
save("train_info.mat","training_info")

% Save complete model with timestamp and metadata
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
modelName = sprintf("trained_GAN_28inputs_%s_%d_%d_%d_%d.mat", timestamp, a, b, c, latent_size);
modelPath = fullfile(resultsDir, modelName);

% Save with comprehensive metadata
model_metadata = struct();
model_metadata.timestamp = timestamp;
model_metadata.num_inputs = num_in;
model_metadata.batch_size = b_size;
model_metadata.max_epochs = options.MaxEpochs;
model_metadata.initial_lr = options.InitialLearnRate;
model_metadata.optimizer = "adam";
model_metadata.loss_function = "multiOutputMSE_monotonicity";
model_metadata.data_cycles = [103, 104, 105, 109];
model_metadata.latent_size = latent_size;
model_metadata.loss_weights = struct('MSE', a, 'Monotonicity', b, 'Proxy', c);
model_metadata.model_type = model_type;
model_metadata.ds_type = ds_type;
model_metadata.final_loss = training_info.TrainingHistory.multiOutputMSE(end);
if isfield(training_info, 'ValidationHistory') && ~isempty(training_info.ValidationHistory.multiOutputMSE)
    model_metadata.final_val_loss = training_info.ValidationHistory.multiOutputMSE(end);
end

save(modelPath, "trained_net", "training_info", "model_metadata", '-v7.3');
fprintf('✅ Complete model saved to: %s\n', modelPath);

% Also save a lightweight version (just the network) for quick loading
lightPath = fullfile(resultsDir, sprintf("net_only_%s.mat", timestamp));
save(lightPath, "trained_net");
fprintf('✅ Network-only version saved to: %s\n', lightPath);

