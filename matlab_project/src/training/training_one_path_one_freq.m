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



function L = total_loss(varargin)    
    
    % Expand varargin when forwarding to helper losses
    MSE = multiOutputMSE(varargin{:});

    L = MSE;
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

function L = multiOutputMSE(varargin)    
    numOut = numel(varargin)/2; 

    accum = 0; 
   
    for k = 1:numOut 

        Yk = varargin{1,k}; 
        Tk = varargin{1,k+numOut}; 
        % Unwrap if elements are cells (can happen with some datastores/trainnet packings)
        if iscell(Yk), Yk = Yk{1}; end
        if iscell(Tk), Tk = Tk{1}; end

        accum = accum + mean((Yk - Tk).^2,'all'); 
    end 
    L = accum / numOut; 
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

function L = peak_preserving_noise_suppressing_loss(varargin)
    numOut = numel(varargin)/2;
    accum = 0;
    for k=1:numOut
        Yk = varargin{1,k};
        Tk = varargin{1,k+numOut};
        if iscell(Yk), Yk=Yk{1}; end
        if iscell(Tk), Tk=Tk{1}; end

        % Define what's a "peak" vs "noise" in your data
        peak_threshold = 2.0;  % Adjust based on your data
        
        % Separate masks
        is_peak = abs(Tk) > peak_threshold;
        is_noise = abs(Tk) <= peak_threshold;
        
        % Peak reconstruction loss (heavily weighted)
        peak_error = ((Yk - Tk) .* is_peak).^2;
        peak_loss = sum(peak_error, 'all') / (sum(is_peak, 'all') + 1e-6);
        
        % Noise suppression loss (penalize predicted noise)
        noise_pred = Yk .* is_noise;
        noise_true = Tk .* is_noise;
        noise_loss = mean((noise_pred - noise_true).^2, 'all');
        
        % Additional: penalize high predictions where truth is low
        false_peak_penalty = sum((abs(Yk) .* is_noise - abs(Tk) .* is_noise).^2, 'all') / (sum(is_noise, 'all') + 1e-6);
        
        % Combine with heavy emphasis on peaks
        accum = accum + (3.0*peak_loss + 1.0*noise_loss + 2.0*false_peak_penalty);
    end
    L = accum/numOut;
end



%% Data Preparation

rng(42); % For reproducibility
% Create single datastore that handles all 28 inputs and targets
num_in = 1;
b_size = 16;
overlap_size = 2;
% 148 instances of data for the training and 28 for validation
envelope = false; % Whether to use envelope of signals or raw signals
freq = 5; % frequency index to extract
path_idx = 1; % path index to extract
benchmark = false; % whether to use benchmark datastore (no random cropping)
fc = true; % whether to use fully connected network
downsampled = false;
if downsampled
    States103 = load(sprintf("data\\States_downsampled_%d.mat", 103)).States_downsampled;  % Load the Cycle1 datastore
    States104 = load(sprintf("data\\States_downsampled_%d.mat", 104)).States_downsampled;  % Load the Cycle2 datastore
    States105 = load(sprintf("data\\States_downsampled_%d.mat", 105)).States_downsampled;  % Load the Cycle3 datastore
    States109 = load(sprintf("data\\States_downsampled_%d.mat", 109)).States_downsampled;  % Load the Cycle4 datastore
else
    States103 = load(sprintf("data\\States_%d.mat", 103)).States;  % Load the Cycle1 datastore
    States104 = load(sprintf("data\\States_%d.mat", 104)).States;  % Load the Cycle2 datastore
    States105 = load(sprintf("data\\States_%d.mat", 105)).States;  % Load the Cycle3 datastore
    States109 = load(sprintf("data\\States_%d.mat", 109)).States;  % Load the Cycle4 datastore
end
if fc
    inputDs1 = CyclemultiInputDatastore_separate_sin_freq_fc(States103, num_in, b_size, overlap_size,32,envelope,freq, 'paths', path_idx); % Provide paths only for first datastore
    inputDs2 = CyclemultiInputDatastore_separate_sin_freq_fc(States104, num_in, b_size, overlap_size,58,envelope,freq, 'paths', path_idx);
    inputDs3 = CyclemultiInputDatastore_separate_sin_freq_fc(States105, num_in, b_size, overlap_size,30,envelope,freq, 'paths', path_idx);
    inputDs4 = CyclemultiInputDatastore_separate_sin_freq_fc(States109, num_in, b_size, overlap_size,28,envelope,freq, 'paths', path_idx);
else
    inputDs1 = CyclemultiInputDatastore_separate_sin_freq(States103, num_in, b_size, overlap_size,32,envelope,freq, benchmark, 'paths', path_idx); % Provide paths only for first datastore
    inputDs2 = CyclemultiInputDatastore_separate_sin_freq(States104, num_in, b_size, overlap_size,58,envelope,freq, benchmark, 'paths', path_idx);
    inputDs3 = CyclemultiInputDatastore_separate_sin_freq(States105, num_in, b_size, overlap_size,30,envelope,freq, benchmark, 'paths', path_idx);
    inputDs4 = CyclemultiInputDatastore_separate_sin_freq(States109, num_in, b_size, overlap_size,28,envelope,freq, benchmark, 'paths', path_idx);
end
inputDs = combine(inputDs1,  inputDs2, inputDs3, inputDs4, ReadOrder="sequential");  % Combine datastores'


total_obs = inputDs1.NumObservations+inputDs2.NumObservations+inputDs3.NumObservations+inputDs4.NumObservations;
train_obs_idx = randperm(total_obs, round(0.80*total_obs));
val_obs_idx = setdiff(1:total_obs, train_obs_idx);

inputDs_train = subset(inputDs, train_obs_idx);
inputDs_val = subset(inputDs, val_obs_idx);
fprintf('Data preparation complete. Training observations: %d, Validation observations: %d\n', ...
    numpartitions(inputDs_train), numpartitions(inputDs_val));

% Test the datastore
fprintf('Testing datastore...\n');
if hasdata(inputDs4)
    testData = read(inputDs4);
    fprintf('Data table has %d columns (should be %d for %d inputs + %d targets)\n', ...
        width(testData), 2*num_in, num_in, num_in);
    
    disp(size(testData));  % Display size of the table
    % Check the size of the first input
    firstInput = testData{1,1};  % Get first column, first row
    fprintf('First input size: %s (should be 4000×2×6)\n', mat2str(size(firstInput)));
    
    disp(firstInput(1,1:100,:,1));  % Display a slice of the first input for verification
    reset(inputDs2);  % Reset for training
end

%% Network Training

if fc 
    format =repmat("BC", 1, num_in),
else
    format =repmat("BSSC", 1, num_in),
end

options = trainingOptions("adam", ...  % Adam optimizer for better convergence
    MaxEpochs=50, ...  % Increased epochs for the improved network
    MiniBatchSize=b_size, ...  % Smaller batch size for stability
    InitialLearnRate=0.001, ...  % Conservative learning rate
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=15, ...
    LearnRateDropFactor=0.5, ...
    GradientThresholdMethod="l2norm", ...
    GradientThreshold=10, ...  % Gradient clipping to prevent explosion
    Verbose=true, ...
    VerboseFrequency=10, ...
    ValidationPatience=5, ...  % Early stopping if no improvement
    Plots="training-progress", ...
    InputDataFormats=format, ...  % 28 inputs
    TargetDataFormats=format, ...  % 28 targets (including G2/latent_out)
    Shuffle="every-epoch",...
    ValidationData=inputDs_val,...
    ValidationFrequency=10);

options = trainingOptions("adam", ...
    MaxEpochs=100, ...  % More epochs
    MiniBatchSize=b_size, ...  % Smaller batch size for stability
    InitialLearnRate= 0.0010145, ... % Lower learning rate
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=15, ...
    LearnRateDropFactor=0.5, ...
    GradientThresholdMethod="l2norm", ...
    GradientThreshold=10, ...
    Verbose=true, ...
    VerboseFrequency=10, ...
    ValidationPatience=30, ...  % More patience
    Plots="training-progress", ...
    InputDataFormats=format, ...
    TargetDataFormats=format, ...
    Shuffle="every-epoch",...
    ValidationData=inputDs_val,...
    ValidationFrequency=10);


% Select scalable composite loss (append scaling args) REMEBER TO CHANGE METADATA THEN
lossFcn = @(varargin) multiOutputMSE(varargin{:}); % Use standard MSE for stability
% lossFcn = @(varargin) peak_preserving_noise_suppressing_loss(varargin{:});




fc_params.no_params =true;
fc_params.input_size = 4000;
fc_params.hidden_layer_size1 = 1961;%1024;
fc_params.hidden_layer_size2 = 585;%512;
fc_params.hidden_layer_size3 = 374;%256;
fc_params.hidden_layer_size4 = 59;%128;
fc_params.drop_rate = 0.48;
fc_params.desired_latent_size = 15;

net = architectures_container.deep_fully_connected_network(fc_params);
net_name = "Fully Connected Network";

fprintf('Network loaded successfully. Starting training...\n');

%An 'epoch' represents one pass through the entire training dataset, 
% while an 'iteration' corresponds to one update of the model's parameters using a mini-batch of data during training.

% Train the network
[trained_net, training_info] = trainnet(inputDs_train, net, lossFcn, options);

disp(training_info.TrainingHistory);
%% Save Results

% Create results directory if it doesn't exist
resultsDir = fullfile(projectRoot, 'results',sprintf('1p1f_%s', net_name));
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
    fprintf('Created results directory: %s\n', resultsDir);
end


% Save complete model with timestamp and metadata
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');

% Save with comprehensive metadata
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
model_metadata.latent_size = latent_size;
model_metadata.final_loss = training_info.TrainingHistory.Loss(end);
model_metadata.final_val_loss = training_info.ValidationHistory.Loss(end);

modelName = sprintf("path_%d_freq_%d_net_loss_%f.mat", path_idx, freq, model_metadata.final_val_loss);


modelPath = fullfile(resultsDir, modelName);


save(modelPath, "trained_net", "training_info", "model_metadata", '-v7.3');
fprintf('✅ Complete model saved to: %s\n', modelPath);



