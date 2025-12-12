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

%% plotTrainingProgress function
function hFig = plotTrainingProgress(training_info, savePath, titleStr)
% Recreates a training-progress style plot (offline) from trainnet info.
% - training_info: struct returned by trainnet
% - savePath: full path (png) to save the figure (optional)
% - titleStr: figure title (optional)

if nargin < 2 || isempty(savePath), savePath = ''; end
if nargin < 3, titleStr = 'Training Progress'; end

% Extract histories (robust to field names)
TH = []; VH = [];
if isfield(training_info,'TrainingHistory'), TH = training_info.TrainingHistory; end
if isfield(training_info,'ValidationHistory'), VH = training_info.ValidationHistory; end

% Helper to fetch a variable by any of these names
getVar = @(T,names) getFirstMatch(T, names);

iterT = getVar(TH, {'Iteration','Iter'});
epochT = getVar(TH, {'Epoch'});
lr     = getVar(TH, {'LearnRate','LearningRate','BaseLearnRate'});
lossT  = getVar(TH, {'TrainingLoss','Loss','Objective'});
accT   = getVar(TH, {'TrainingAccuracy','Accuracy'});
f1T    = getVar(TH, {'TrainingFScore','FScore','F1'});

iterV = []; lossV = []; accV = []; f1V = [];
if ~isempty(VH)
    iterV = getVar(VH, {'Iteration','Iter','ValidationIteration'});
    lossV = getVar(VH, {'ValidationLoss','Loss'});
    accV  = getVar(VH, {'ValidationAccuracy','Accuracy'});
    f1V   = getVar(VH, {'ValidationFScore','FScore','F1'});
end

% Build the figure (up to 3 rows like training-progress)
rows = 0;
hasAcc = ~isempty(accT) || ~isempty(accV);
hasF1  = ~isempty(f1T)  || ~isempty(f1V);
hasLoss = ~isempty(lossT);
if hasAcc, rows = rows+1; end
if hasF1,  rows = rows+1; end
if hasLoss,rows = rows+1; end
if rows==0, error('plotTrainingProgress:NoMetrics','No metrics found in training_info.'); end
if rows<3 && ~isempty(lr), rows = rows+1; end  % add LR panel if room

hFig = figure('Name',titleStr,'NumberTitle','off','Color','w','Position',[100 100 900 650]);
tlo = tiledlayout(hFig, rows, 1, 'TileSpacing','compact');
title(tlo, titleStr);

row = 0;

% 1) Accuracy
if hasAcc
    row = row+1; ax = nexttile(tlo,row);
    if ~isempty(accT), plot(ax, iterT, accT, 'b-', 'DisplayName','Training'); hold(ax,'on'); end
    if ~isempty(accV), plot(ax, iterV, accV, 'r-', 'DisplayName','Validation'); end
    grid(ax,'on'); xlabel(ax,'Iteration'); ylabel(ax,'Accuracy (%)'); legend(ax,'Location','southeast'); title(ax,'Accuracy');
end

% 2) F-score
if hasF1
    row = row+1; ax = nexttile(tlo,row);
    if ~isempty(f1T), plot(ax, iterT, f1T, 'b-', 'DisplayName','Training'); hold(ax,'on'); end
    if ~isempty(f1V), plot(ax, iterV, f1V, 'r-', 'DisplayName','Validation'); end
    grid(ax,'on'); xlabel(ax,'Iteration'); ylabel(ax,'FScore'); legend(ax,'Location','southeast'); title(ax,'FScore');
end

% 3) Loss (semilogy like the UI)
if hasLoss
    row = row+1; ax = nexttile(tlo,row);
    if ~isempty(lossT), semilogy(ax, iterT, lossT, 'b-', 'DisplayName','Training'); hold(ax,'on'); end
    if ~isempty(lossV), semilogy(ax, iterV, lossV, 'r-', 'DisplayName','Validation'); end
    grid(ax,'on'); xlabel(ax,'Iteration'); ylabel(ax,'Loss'); legend(ax,'Location','northeast'); title(ax,'Loss');
end

% 4) Learning rate (if present and room)
if row < rows && ~isempty(lr)
    ax = nexttile(tlo,row+1);
    plot(ax, iterT, lr, 'k-'); grid(ax,'on'); xlabel(ax,'Iteration'); ylabel(ax,'Learn Rate'); title(ax,'Learning Rate Schedule');
end

% Save if requested
if ~isempty(savePath)
    [~,~,ext] = fileparts(savePath);
    if isempty(ext), savePath = [savePath '.png']; end
    exportgraphics(hFig, savePath, 'Resolution',150);
end
end

function v = getFirstMatch(T, names)
v = [];
if isempty(T), return; end
if istable(T)
    for k = 1:numel(names)
        if ismember(names{k}, T.Properties.VariableNames)
            v = T.(names{k}); v = v(:); return;
        end
    end
elseif isstruct(T)
    for k = 1:numel(names)
        if isfield(T, names{k})
            v = T.(names{k}); v = v(:); return;
        end
    end
end
end
%% Custom Loss Functions


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

% CRITICAL: Add latent regularization to force diversity
function L = LossWithLatentRegularization(varargin)
    disp('size of varargin:');
    disp(size(varargin));
    numOut = numel(varargin)/2;
    accum = 0;

    % 1. Reconstruction loss
    for i = 1:numOut
        Yk = varargin{1,i};
        Tk = varargin{1,i+numOut};
        if iscell(Yk), Yk = Yk{1}; end
        if iscell(Tk), Tk = Tk{1}; end
        accum = accum + mean((Yk - Tk).^2,'all');
    end
    
    % % 2. CRITICAL: Penalize low variance in latent space
    % % This forces the network to USE the latent dimensions
    % latent_mean = mean(Z, 4);  % Mean across batch
    % latent_var = mean((Z - latent_mean).^2, 4);  % Variance per dimension
    
    % % Penalize if variance is too low (collapsed latent space)
    % min_variance = 0.1;  % Target minimum variance
    % variance_penalty = mean(max(0, min_variance - latent_var), 'all');
    
    % % 3. High-frequency loss (your signals have high-freq content)
    % X_diff = diff(X, 1, 2);
    % Y_diff = diff(Y, 1, 2);
    % gradient_loss = mean((Y_diff - X_diff).^2, 'all');
    
    % Combine losses
    L = accum/numOut;% + 0.5 * gradient_loss + 2.0 * variance_penalty;
    
end

%% Data Preparation

rng(42); % For reproducibility
% Create single datastore that handles all 28 inputs and targets
num_in = 1;
b_size = 4;
overlap_size = 2;

States103 = load(sprintf("data\\States_downsampled_%d.mat", 103)).States_downsampled;  % Load the Cycle1 datastore
States104 = load(sprintf("data\\States_downsampled_%d.mat", 104)).States_downsampled;  % Load the Cycle2 datastore
States105 = load(sprintf("data\\States_downsampled_%d.mat", 105)).States_downsampled;  % Load the Cycle3 datastore
States109 = load(sprintf("data\\States_downsampled_%d.mat", 109)).States_downsampled;  % Load the Cycle4 datastore

% 148 instances of data for the training and 28 for validation
envelope = false; % Whether to use envelope of signals or raw signals
freq = 4; % frequency index to extract
path_idx = 1; % path index to extract
benchmark = false; % whether to use benchmark datastore (no random cropping)
inputDs1 = CyclemultiInputDatastore_separate_sin_freq(States103, num_in, b_size, overlap_size,32,envelope,freq, benchmark, 'paths', path_idx); % Provide paths only for first datastore
inputDs2 = CyclemultiInputDatastore_separate_sin_freq(States104, num_in, b_size, overlap_size,58,envelope,freq, benchmark, 'paths', path_idx);
inputDs3 = CyclemultiInputDatastore_separate_sin_freq(States105, num_in, b_size, overlap_size,30,envelope,freq, benchmark, 'paths', path_idx);
inputDs4 = CyclemultiInputDatastore_separate_sin_freq(States109, num_in, b_size, overlap_size,28,envelope,freq, benchmark, 'paths', path_idx);
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

options = trainingOptions("adam", ...  % Adam optimizer for better convergence
    MaxEpochs=50, ...  % Increased epochs for the improved network
    MiniBatchSize=b_size, ...  % Smaller batch size for stability
    InitialLearnRate=0.001, ...  % Conservative learning rate
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=20, ...
    LearnRateDropFactor=0.5, ...
    GradientThresholdMethod="l2norm", ...
    GradientThreshold=10, ...  % Gradient clipping to prevent explosion
    Verbose=true, ...
    VerboseFrequency=10, ...
    ValidationPatience=5, ...  % Early stopping if no improvement
    Plots="training-progress", ...
    InputDataFormats=repmat("BSSC", 1, num_in), ...  % 28 inputs
    TargetDataFormats=repmat("BSSC", 1, num_in), ...  % 28 targets (including G2/latent_out)
    Shuffle="every-epoch",...
    ValidationData=inputDs_val,...
    ValidationFrequency=10);

options = trainingOptions("adam", ...
    MaxEpochs=100, ...  % More epochs
    MiniBatchSize=4, ...  % Smaller batch size for stability
    InitialLearnRate=0.001, ... % Lower learning rate
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=10, ...
    LearnRateDropFactor=0.8, ...
    GradientThresholdMethod="l2norm", ...
    GradientThreshold=10, ...
    Verbose=true, ...
    VerboseFrequency=10, ...
    ValidationPatience=15, ...  % More patience
    Plots="training-progress", ...
    InputDataFormats=repmat("BSSC", 1, num_in), ...
    TargetDataFormats=repmat("BSSC", 1, num_in), ...
    Shuffle="every-epoch",...
    ValidationData=inputDs_val,...
    ValidationFrequency=10);


% Select scalable composite loss (append scaling args) REMEBER TO CHANGE METADATA THEN
lossFcn = @(varargin) peak_preserving_noise_suppressing_loss(varargin{:}); % Use standard MSE for stability
% lossFcn = @(varargin) peak_preserving_noise_suppressing_loss(varargin{:});


params.channels1 = 265;
params.channels2 = 128;
params.channels3 = 64;
params.channels4 = 16;
params.channels5 = 1;

params.kernel_size4 = 11;
params.kernel_size5 = 9;

params.kernel_size1 = 40;% first frequency is  50 kHz, sampling rate is 2e6 Hz, therefore 40 samples per period, kernel 10 seems reasonable
param.dropout_encoder = 0.2
param.dropout_decoder = 0.2

latent_size = 15

params.kernel_size1 = 13;
params.kernel_size2 = 11;
params.kernel_size3 = 11;

params.channels1 = 9;
params.channels2 = 22;
params.dilation_factor = 6;
params.num_inputs = num_in;

params.channels1 =  48;
params.channels2 =  27;
params.kernel_size1 = 18;
params.kernel_size2 = 11;
params.kernel_size3 = 15;
params.batch_size =  5;
params.dilation_factor = 4;
params.desired_latent_size = latent_size;

params.channels1 =  62;
params.channels2 =  23;
params.kernel_size1 = 19;
params.kernel_size2 = 14;
params.kernel_size3 = 13;
params.batch_size =  4;
params.dilation_factor = 8;
param.dropout_encoder = 0.030436 ;
param.dropout_decoder = 0.07946;
params.desired_latent_size = latent_size;
params.benchmark_mode = benchmark;
params.input_y = 1;



net = architectures_container.buildOptimizedNetwork_compressed_downsampled_sin_freq_less_RES_bn_dr(params);

fprintf('Network loaded successfully. Starting training...\n');

%An 'epoch' represents one pass through the entire training dataset, 
% while an 'iteration' corresponds to one update of the model's parameters using a mini-batch of data during training.

% Train the network
[trained_net, training_info] = trainnet(inputDs_train, net, lossFcn, options);

disp(training_info.TrainingHistory);
%% Save Results

% Create results directory if it doesn't exist
resultsDir = fullfile(projectRoot, 'results','one_path_one_freq_downsampled');
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
if isfield(training_info, 'ValidationHistory') && ~isempty(training_info.ValidationHistory.Loss)
    model_metadata.final_val_loss = training_info.ValidationHistory.Loss(end);
end

modelName = sprintf("path_%d_freq_%d_net_loss_%f.mat", path_idx, freq, model_metadata.final_loss);
modelPath = fullfile(resultsDir, modelName);


save(modelPath, "trained_net", "training_info", "model_metadata", '-v7.3');
fprintf('✅ Complete model saved to: %s\n', modelPath);



