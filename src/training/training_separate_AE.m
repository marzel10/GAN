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
envelope = true; % Whether to use envelope of signals or raw signals
inputDs1 = CyclemultiInputDatastore_separate(Cycle1, num_in, b_size, overlap_size,32,envelope);
inputDs2 = CyclemultiInputDatastore_separate(Cycle2, num_in, b_size, overlap_size,58,envelope);
inputDs3 = CyclemultiInputDatastore_separate(Cycle3, num_in, b_size, overlap_size,30,envelope);
inputDs4 = CyclemultiInputDatastore_separate(Cycle4, num_in, b_size, overlap_size,28,envelope);
inputDs_train = combine(inputDs1,  inputDs2,ReadOrder="sequential");  % Combine datastores
inputDs_val = combine(inputDs4, inputDs3,ReadOrder="sequential");

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
errorf('Datastore test complete. Stop execution before training.');
%% Network Training

options = trainingOptions("adam", ...  % Adam optimizer for better convergence
    MaxEpochs=10, ...  % Increased epochs for the improved network
    MiniBatchSize=b_size, ...  % Smaller batch size for stability
    InitialLearnRate=5e-4, ...  % Conservative learning rate
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=3, ...
    LearnRateDropFactor=0.5, ...
    GradientThresholdMethod="l2norm", ...
    GradientThreshold=1, ...  % Gradient clipping to prevent explosion
    Verbose=true, ...
    VerboseFrequency=10, ...
    ValidationPatience=5, ...  % Early stopping if no improvement
    Plots="none", ... %Plots="training-progress", ...
    InputDataFormats=repmat("BSSC", 1, 28), ...  % 28 inputs
    TargetDataFormats=repmat("BSSC", 1, 28), ...  % 28 targets (including G2/latent_out)
    Shuffle="every-epoch",...
    ValidationData=inputDs_val,...
    ValidationFrequency=5);
    

% Select scalable composite loss (append scaling args) REMEBER TO CHANGE METADATA THEN
% lossFcn = @(varargin) total_loss(varargin{:});
lossFcn = @(varargin) amplitude_aware_loss(varargin{:});

latent_size = 80;
[a, b, c] = deal(1, 0, 0); % Weights for MSE, monotonicity, proxy loss (last 2 obviously 0)
model_type = '28_AE_separate_E15';
% ds_type = 'CyclemultiInputDatastore_envelope';
ds_type = 'CyclemultiInputDatastore';
load(sprintf('28_AE_latent_%d_E15.mat', latent_size), 'net'); % Load the pre-built network
fprintf('Network loaded successfully. Starting training...\n');

%An 'epoch' represents one pass through the entire training dataset, 
% while an 'iteration' corresponds to one update of the model's parameters using a mini-batch of data during training.

% Train the network
[trained_net, training_info] = trainnet(inputDs_train, net, lossFcn, options);

disp(training_info.TrainingHistory);
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
modelName = sprintf("trained_GAN_28inputs_%s_%d_%s.mat", timestamp, latent_size, model_type(end-1:end));
modelPath = fullfile(resultsDir, modelName);

% Save with comprehensive metadata
model_metadata = struct();
model_metadata.timestamp = timestamp;
model_metadata.num_inputs = num_in;
model_metadata.batch_size = b_size;
model_metadata.max_epochs = options.MaxEpochs;
model_metadata.initial_lr = options.InitialLearnRate;
model_metadata.optimizer = "adam";
model_metadata.loss_function = "amplitude_aware_loss";
model_metadata.data_cycles = [103, 104, 105, 109];
model_metadata.latent_size = latent_size;
model_metadata.loss_weights = struct('MSE', a, 'Monotonicity', b, 'Proxy', c);
model_metadata.model_type = model_type;
model_metadata.ds_type = ds_type;
model_metadata.final_loss = training_info.TrainingHistory.Loss(end);
if isfield(training_info, 'ValidationHistory') && ~isempty(training_info.ValidationHistory.Loss)
    model_metadata.final_val_loss = training_info.ValidationHistory.Loss(end);
end

save(modelPath, "trained_net", "training_info", "model_metadata", '-v7.3');
fprintf('✅ Complete model saved to: %s\n', modelPath);

% Also save a lightweight version (just the network) for quick loading
lightPath = fullfile(resultsDir, sprintf("net_only_%s.mat", timestamp));
save(lightPath, "trained_net");
fprintf('✅ Network-only version saved to: %s\n', lightPath);


% Create and save an offline training-progress plot
try
    figOut = fullfile(resultsDir, sprintf('training_progress_%s.png', timestamp));
    plotTrainingProgress(training_info, figOut, sprintf('Training Progress (%s)', model_type));
    fprintf('📈 Saved training-progress plot to: %s\n', figOut);
catch ME
    warning(ME.identifier, 'Could not create training-progress plot: %s', ME.message);
end

