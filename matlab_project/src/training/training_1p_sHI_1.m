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

        % New: always add a tiny spread term on the main output
    %spread = variance_penalty(varargin{1});  % first network output
    spread = 0;
    L = a*MSE + b*mono + c*proxy_loss + 1e-3*spread; 
    
end

function L = proxy_labels_loss(varargin)
    predicted_latent = varargin{1};  % latent (size 1 x B)
    weibull_target = varargin{2};    % target (size 1 x B)
    % Basic MSE proxy loss
    L = dlarray(mean((predicted_latent - weibull_target).^2,'all'));
end

function L = variance_penalty(predicted_latent)
    if ndims(predicted_latent) > 2
        y = squeeze(predicted_latent);
    else
        y = predicted_latent;
    end
    yvec = y(:);
    mu = mean(yvec);
    v = mean((yvec - mu).^2);
    L = dlarray(1 ./ (v + 1e-6));   % smaller when variance is larger
end

function L = monotonicity(varargin)
    relu_version = true; % Set to true to use ReLU-based monotonicity loss
    predicted_latent = varargin{1};  % Second output (the size is 1xbatchsize)
    r = 0.9; % How strong the increase should be penalized
    
    if relu_version
        % ReLU-based monotonicity loss
       diff = predicted_latent( :, 2:end) -  predicted_latent(:, 1:end-1) + 0.001; % Enforce a minimum increase of 1 unit per cycle
      
       violations = relu(diff);  % Differentiable alternative to max(0, ...)
    else
        % Squared difference-based monotonicity loss
        diff = predicted_latent( :, 2:end) -  predicted_latent(:, 1:end-1) + r;
        diff = diff.* diff; % Square the differences to penalize larger violations more
        violations = diff;
    end
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

b_size = 27;
overlap_size = 26;
% 148 instances of data for the training and 28 for validation
freq = 4; % frequency index to extract
fc = true; % whether to use fully connected network
net_folder = 'C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Sparse_Fully_Connected_AE\freq_4\bayesian_optimization';

for path_idx = 28:28
    % fc=false means we use GAN; datastore expects GAN_mode=true to output C=28
    ganMode = ~fc;
    inputDs1 = latent_space_datastore(103, b_size, overlap_size, freq, ganMode, net_folder, 'path_id', path_idx);
    inputDs2 = latent_space_datastore(104, b_size, overlap_size, freq, ganMode, net_folder, 'path_id', path_idx);
    inputDs3 = latent_space_datastore(105, b_size, overlap_size, freq, ganMode, net_folder, 'path_id', path_idx);
    inputDs4 = latent_space_datastore(109, b_size, overlap_size, freq, ganMode, net_folder, 'path_id', path_idx);

    inputDs_train = combine(inputDs1,  inputDs2, inputDs3, ReadOrder="sequential");
    inputDs_val = inputDs4;

    fprintf('Data preparation complete. Training observations: %d, Validation observations: %d\n', ...
        numpartitions(inputDs_train), numpartitions(inputDs_val));


    %% Network Training


    format_input ="CB";
    format_targets = "CB";


    options = trainingOptions("adam", ...
        MaxEpochs=1000, ...  % More epochs
        MiniBatchSize=b_size, ...  % Smaller batch size for stability
        InitialLearnRate= 0.0010145, ... % Lower learning rate
        LearnRateSchedule="piecewise", ...
        LearnRateDropPeriod=100, ...
        LearnRateDropFactor=0.8, ...
        GradientThresholdMethod="l2norm", ...
        Metrics={@monotonicity}, ...
        GradientThreshold=10, ...
        Verbose=true, ...
        VerboseFrequency=10, ...
        ValidationPatience=150, ...  % More patience
        Plots="training-progress", ...
        InputDataFormats=format_input, ...
        TargetDataFormats=format_targets, ...
        Shuffle="every-epoch",...
        ValidationData=inputDs_val,...
        ValidationFrequency=10);

    fc_params.input_size = 15;
    fc_params.latent_size = 1;
    AE_learning_rate = 0;
    [a, b, c] = deal(0, 1, 0); % Weights for MSE, monotonicity, proxy loss
    % Select scalable composite loss (append scaling args) REMEBER TO CHANGE METADATA THEN
    lossFcn = @(varargin) multiOutputMSE_monotonicity(fc_params.latent_size,a,b,c,varargin{:}); % Use standard MSE for stability
    % lossFcn = @(varargin) peak_preserving_noise_suppressing_loss(varargin{:});

    net = tiny_architectures_container.tiny_fully_connected_network(fc_params);
    disp("Number of network outputs and inputs:");
    disp(length(net.OutputNames)+length(net.InputNames));
    net_name = "FC_Net_stupid_loss";

    % Sanity check: print learnable parameter stats after init
    try
        L = net.Learnables;
        fprintf('\nInitial learnables summary (mean/std/nnz):\n');
        for i = 1:size(L,1)
            v = gather(extractdata(L.Value{i}));
            fprintf('  %-30s  %-20s  mean=%+0.4g  std=%0.4g  nnz=%d\n', ...
                L.Layer(i), L.Parameter(i), mean(v(:),'omitnan'), std(v(:),[],'omitnan'), nnz(v));
        end
        fprintf('\n');
    catch ME
        warning(ME.identifier, 'Could not print learnables summary: %s', ME.message);
    end

    
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
end



