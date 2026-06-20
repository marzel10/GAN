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
    numOut = numel(varargin)/2-1;
    predicted_latent = varargin{numOut+1};  % latent is stored as a second output (the size is 1xbatchsize)
    weibull_target = varargin{numOut+2}(1,:); % First target is the proxy label (scalar) (the size of varargin{3} is 4000xbatchsize but only first row is not zero, so you take that row for every batch)
    weibull_target = repmat(weibull_target, numOut, 1); % Replicate to match predicted_latent size
    disp(weibull_target);
    disp(predicted_latent);
    L = dlarray(mean((predicted_latent - weibull_target).^2,'all'));

end

function L = monotonicity(varargin)
    numOut = numel(varargin)/2-1;
    
    % disp("sisetot"+size(varargin))
    % disp("size1"+ size(varargin{1}))
    % disp("size2"+size(varargin{2}))
    % disp("size3"+size(varargin{3}))
    % disp("size4"+size(varargin{4}))
    % disp("size5"+size(varargin{5}))
    % disp("size6"+size(varargin{6}))
    % disp(varargin{3})
    % disp(varargin{4}(1:10,:))
    

    predicted_latent = varargin{numOut+1};  % Second output (the size is 1xbatchsize)

    diff = predicted_latent( :, 2:end) -  predicted_latent( :,1:end-1) + 0.01; % Enforce a minimum increase of 1 unit per cycle
    violations = relu(diff);  % Differentiable alternative to max(0, ...)
    L = dlarray(mean(violations, 'all'));
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
num_in = 28;
b_size = 4;
overlap_size = 3;
% 148 instances of data for the training and 28 for validation
envelope = false; % Whether to use envelope of signals or raw signals
freq = 4; % frequency index to extract
fc = false; % whether to use fully connected network


States103 = load(sprintf("data\\States_%d.mat", 103)).States;  % Load the Cycle1 datastore
States104 = load(sprintf("data\\States_%d.mat", 104)).States;  % Load the Cycle2 datastore
States105 = load(sprintf("data\\States_%d.mat", 105)).States;  % Load the Cycle3 datastore
States109 = load(sprintf("data\\States_%d.mat", 109)).States;  % Load the Cycle4 datastore

inputDs1 = one_path_sHI_sin_freq_fc_datastore(States103, num_in, b_size, overlap_size, 32-1, freq); % Provide paths only for first datastore
inputDs2 = one_path_sHI_sin_freq_fc_datastore(States104, num_in, b_size, overlap_size, 58-1, freq);
inputDs3 = one_path_sHI_sin_freq_fc_datastore(States105, num_in, b_size, overlap_size, 30-1, freq);
inputDs4 = one_path_sHI_sin_freq_fc_datastore(States109, num_in, b_size, overlap_size, 28-1, freq);

inputDs_train = combine(inputDs1,  inputDs2, inputDs3, ReadOrder="sequential");
inputDs_val = inputDs4;
fprintf('Data preparation complete. Training observations: %d, Validation observations: %d\n', ...
    numpartitions(inputDs_train), numpartitions(inputDs_val));

% Test the datastore
fprintf('Testing datastore...\n');
if hasdata(inputDs4)
    testData = read(inputDs4);
    fprintf('Data table has %d columns (should be %d for %d inputs + %d targets)\n', ...
        width(testData), 2*num_in+1, num_in, num_in+1);
    
    disp(size(testData));  % Display size of the table
    % Check the size of the first input
    firstInput = testData{1,1};  % Get first column, first row
    fprintf('First input size: %s (should be 1x4000)\n', mat2str(size(firstInput)));
    reset(inputDs4);  % Reset for training
end
fprintf('Datastore test complete.\n');
%% Network Training


format_input =repmat("BC", 1, num_in);
format_targets = repmat("BC", 1, num_in+1); % Last target is scalar (proxy label)


options = trainingOptions("adam", ...
    MaxEpochs=100, ...  % More epochs
    MiniBatchSize=b_size, ...  % Smaller batch size for stability
    InitialLearnRate= 0.0010145, ... % Lower learning rate
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=15, ...
    LearnRateDropFactor=0.5, ...
    GradientThresholdMethod="l2norm", ...
    Metrics={@monotonicity, @proxy_labels_loss}, ...
    GradientThreshold=10, ...
    Verbose=true, ...
    VerboseFrequency=10, ...
    ValidationPatience=30, ...  % More patience
    Plots="training-progress", ...
    InputDataFormats=format_input, ...
    TargetDataFormats=format_targets, ...
    Shuffle="never",...
    ValidationData=inputDs_val,...
    ValidationFrequency=10    );

GAN_params.input_size = 15;
GAN_params.latent_size = 1;
GAN_params.adjacency_matrix = attention_matrix.build_attention();
AE_learning_rate = 0;
[a, b, c] = deal(0, 1, 1); % Weights for MSE, monotonicity, proxy loss
% Select scalable composite loss (append scaling args) REMEBER TO CHANGE METADATA THEN
lossFcn = @(varargin) multiOutputMSE_monotonicity(GAN_params.latent_size,a,b,c,varargin{:}); % Use standard MSE for stability
% lossFcn = @(varargin) peak_preserving_noise_suppressing_loss(varargin{:});

net = tiny_architectures_container.AE_GAN_connector('C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Sparse_Fully_Connected_AE_waste\freq_4\bayesian_optimization', GAN_params, 0.1);
disp("Number of network outputs and inputs:");
disp(length(net.OutputNames)+length(net.InputNames));
net_name = "GAN_Net";

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
model_metadata.num_inputs = num_in;
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

modelName = sprintf("fc_all_path_net_loss_%f.mat", model_metadata.final_val_loss);


modelPath = fullfile(resultsDir, modelName);


save(modelPath, "trained_net", "training_info", "model_metadata", '-v7.3');
fprintf('✅ Complete model saved to: %s\n', modelPath);



