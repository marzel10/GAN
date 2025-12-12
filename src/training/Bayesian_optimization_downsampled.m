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

% Start a small parallel pool
if isempty(gcp('nocreate')); parpool('local', 2); end
%% Define Data Configuration
dataCfg = struct( ...
    'num_in', 28, ...
    'b_size', 4, ... % Fixed batch size for simplicity
    'overlap_size', 2, ...
    'envelope', false, ... % Whether to use envelope of signals or raw signals
    'train_files', [103 104], ...
    'val_files', [109 105], ...
    'input_size', 800, ... % Size of each input signal
    'frequency', 1 ... % Frequency index to extract (only if you want to train single frequency)
    ); 

%% Define Hyperparameters to Optimize

% Architecture parameters
kernel_size1 = optimizableVariable('kernel_size1', [10, 15], 'Type', 'integer');
kernel_size2 = optimizableVariable('kernel_size2', [10, 15], 'Type', 'integer');
kernel_size3 = optimizableVariable('kernel_size3', [10, 15], 'Type', 'integer');

channels1 = optimizableVariable('channels1', [5, 10], 'Type', 'integer'); % 8 is number of samples in one sine of the higest freq 
channels2 = optimizableVariable('channels2', [15, 25], 'Type', 'integer');

%stride1 = optimizableVariable('stride1', [4, 20], 'Type', 'integer');  % Increased min to reduce latent size

dilation_factor = optimizableVariable('dilation_factor', [2, 8], 'Type', 'integer');  % New variable to control dilation

function tf = stride_const(X)
    % Constraint for bayesopt. X is a table with one row per candidate.
    input_size = 800;

    stride1 = X.stride1;          % vector (nCandidates x 1)
    k1      = X.kernel_size1;     % vector (nCandidates x 1)

    tf_1 = stride1 < (k1 + 1);                 % element-wise
    tf_2 = mod(input_size, stride1) == 0;      % element-wise

    tf = tf_1 & tf_2;                         % element-wise AND (vector)
end

% Additional hyperparametes which were not added because of the compressed network version
channels3 = optimizableVariable('channels3', [2, 6], 'Type', 'integer');
%kernel_size3 = optimizableVariable('kernel_size3', [3, 7], 'Type', 'integer');
dilation2 = optimizableVariable('dilation2', [2, 8], 'Type', 'integer');
stride2 = optimizableVariable('stride2', [2, 6], 'Type', 'integer');

% Training parameters
%learning_rate = optimizableVariable('learning_rate', [1e-4, 1e-2], 'Transform', 'log');
%dropout_rate = optimizableVariable('dropout_rate', [0.0, 0.3]);
%l2_regularization = optimizableVariable('l2_regularization', [1e-6, 1e-3], 'Transform', 'log');

%% Bayesian Optimization Settings

% Combine all optimizable variables
% optimVars_full = [channels1, channels2, channels3,  ...
%              kernel_size1, kernel_size2, kernel_size3, stride1];

optimVars_compressed = [channels1, channels2,   ...
            kernel_size1, kernel_size2, kernel_size3, dilation_factor];

optimVars = optimVars_compressed;  % Use compressed version to enforce latent size constraint

% Bayesian optimization options
bayesOpts = struct(...
    'MaxObjectiveEvaluations', 6, ...  % Adjust based on your time budget
    'MaxTime', 24*3600, ...              % 24 hours max
    'IsObjectiveDeterministic', false, ...
    'UseParallel', true, ...             % Use parallel if available
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

objectiveFcn = @(params) trainAndEvaluateAE(params, dataCfg);

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

outDir = fullfile(projectRoot,'results\downsampled_states_optim\');
if ~exist(outDir,'dir'), mkdir(outDir); end
finalFile = fullfile(outDir,'bayesian_optimization_results.mat');
save(finalFile, 'results', '-v7.3');   % save once, not inside OutputFcn
fprintf('Saved BO results: %s\n', finalFile);
fprintf('\nOptimization complete!\n');
fprintf('Best validation loss: %.6f\n', results.MinObjective);
disp('Best hyperparameters:');
disp(results.XAtMinObjective);

%% Visualize Results

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

%% Locate and rename (or copy) the best model based on BO results
fprintf('Locating best model file based on Bayesian Optimization results...\n');
bestParams = results.XAtMinObjective;

% Expected filename produced during training (includes dilation_factor)
if isfield(dataCfg, 'frequency')
    expectedName = sprintf('net_%d_%d_%d_%d_%d_%d_freq%d.mat', ...
        bestParams.channels1, bestParams.channels2, ...
        bestParams.kernel_size1, bestParams.kernel_size2, bestParams.kernel_size3, bestParams.dilation_factor, dataCfg.frequency);
    dstName = sprintf('net_best_%d_%d_%d_%d_%d_%d_freq%d.mat', ...
        bestParams.channels1, bestParams.channels2, ...
        bestParams.kernel_size1, bestParams.kernel_size2, bestParams.kernel_size3, bestParams.dilation_factor, dataCfg.frequency);
else
    expectedName = sprintf('net_%d_%d_%d_%d_%d_%d.mat', ...
        bestParams.channels1, bestParams.channels2, ...
        bestParams.kernel_size1, bestParams.kernel_size2, bestParams.kernel_size3, ...
         bestParams.dilation_factor);
    dstName = sprintf('net_best_%d_%d_%d_%d_%d_%d.mat', ...
        bestParams.channels1, bestParams.channels2, ...
        bestParams.kernel_size1, bestParams.kernel_size2, bestParams.kernel_size3, ...
        bestParams.dilation_factor);
end

srcFile = fullfile(outDir, expectedName);
dstFile = fullfile(outDir, dstName);
movefile(srcFile, dstFile, 'f');
fprintf('✅ Renamed best model to: %s\n', dstFile);

%% (Optional) Train Final Model with Best Hyperparameters

% fprintf('\nTraining final model with best parameters...\n');
% bestParams = results.XAtMinObjective;
% finalNet = buildOptimizedNetwork(bestParams);

% % Train with more epochs
% finalLoss = trainAndEvaluateAE(bestParams, trainData, valData, 'FinalTraining', true);
% fprintf('Final validation loss: %.6f\n', finalLoss);

% save('best_autoencoder_model.mat', 'finalNet', 'bestParams');

%% ========== HELPER FUNCTIONS ==========

function loss = trainAndEvaluateAE(params, dataCfg, varargin)

    % --- per‑worker log setup ---
    t = getCurrentTask();
    if isempty(t)
        wid = 'client';
    else
        wid = sprintf('w%02d', t.ID);   % worker ID → w01, w02, ...
    end
    logDir = fullfile(pwd,'results','logs');
    if ~exist(logDir,'dir'), mkdir(logDir); end
    logFile = fullfile(logDir, sprintf('bo_%s.log', wid));
    [fid,msg] = fopen(logFile,'a');
    if fid<0, warning('LogOpenFailed:%s', msg); end
    c = onCleanup(@() (fid>0 && fclose(fid)));  %#ok<NASGU>

    % helper to write one line with time+worker tag
    log = @(fmt,varargin) ...
        (fid>0) && fprintf(fid,'[%s] %s: %s\n', ...
            datestr(now,'HH:MM:SS'), wid, sprintf(fmt,varargin{:}));

    log('start objective; params=%s', jsonencode(params));
    % --- end log setup ---


    % Create datastores
    % det_size helper defined as a nested function because MATLAB does not support inline if/elseif syntax inside anonymous functions.
    mkds = @(id) CyclemultiInputDatastore_separate(...
        load(sprintf("data\\States_downsampled_%d.mat", id)).States_downsampled, ...
        dataCfg.num_in, dataCfg.b_size, dataCfg.overlap_size, det_size(id), dataCfg.envelope);

    mkds_sin_freq = @(id) CyclemultiInputDatastore_separate_sin_freq(...
        load(sprintf("data\\States_downsampled_%d.mat", id)).States_downsampled, ...
        dataCfg.num_in, dataCfg.b_size, dataCfg.overlap_size, det_size(id), dataCfg.envelope, dataCfg.frequency);

    function out = det_size(id)
        switch id
            case 103
                out = 32;
            case 104
                out = 58;
            case 105
                out = 30;
            case 109
                out = 28;
            otherwise
                error('det_size:UnknownID', 'Unknown id %d', id);
        end
    end
    log('Preparing datastores for training and validation...\n');
    
    if isfield(dataCfg, 'frequency')
        trainData = combine(mkds_sin_freq(dataCfg.train_files(1)), mkds_sin_freq(dataCfg.train_files(2)), ReadOrder="sequential");
        valData = combine(mkds_sin_freq(dataCfg.val_files(1)), mkds_sin_freq(dataCfg.val_files(2)), ReadOrder="sequential");
        single_freq = true;
    else
        trainData = combine(mkds(dataCfg.train_files(1)), mkds(dataCfg.train_files(2)), ReadOrder="sequential");
        valData = combine(mkds(dataCfg.val_files(1)), mkds(dataCfg.val_files(2)), ReadOrder="sequential");
        single_freq = false;
    end
    % Train and evaluate autoencoder with given hyperparameters
    p = inputParser;
    addParameter(p, 'FinalTraining', false);
    parse(p, varargin{:});
    isFinal = p.Results.FinalTraining;
    
    try        
        % Pick device per worker: 1 GPU worker, others CPU
        t = getCurrentTask(); w = []; if ~isempty(t), w = t.ID; end
        log('1 am here before gpu');
        useGPU = gpuDeviceCount > 0 && (isempty(w) || w == 1);
        log('2 am here after gpu');
        execEnv = ternary(useGPU,'gpu','cpu');

        disp('Building and training network with parameters:');
        % Build network with current parameters
        if single_freq
            net = architectures_container.buildOptimizedNetwork_compressed_downsampled_sin_freq(params);
        else
            net = architectures_container.buildOptimizedNetwork_compressed_downsampled(params);
        end 
        % Training options
        if isFinal
            maxEpochs = 30;
        else
            maxEpochs = 10;  % Shorter for optimization
        end
        log('Training network (Final=%d) on %s for %d epochs...\n', isFinal, execEnv, maxEpochs);
        options = trainingOptions('adam', ...
            'InitialLearnRate', 0.005, ...
            'MiniBatchSize', dataCfg.b_size, ...
            'MaxEpochs', maxEpochs, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropPeriod', 20, ...
            'LearnRateDropFactor', 0.5, ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', valData, ...
            'ValidationFrequency', 50, ...
            'ExecutionEnvironment', execEnv, ...
            'Verbose', true, ...
            'Plots', 'none');
        
        lossFn = @(varargin) mse_l1_loss(varargin{:});
        % Train network
        [net, info] = trainnet(trainData, net, lossFn, options);
        if single_freq
            save(fullfile('results\downsampled_states_optim\',sprintf('net_%d_%d_%d_%d_%d_%d_freq%d.mat', params.channels1, params.channels2, params.kernel_size1, params.kernel_size2, params.kernel_size3, params.dilation_factor, dataCfg.frequency)), 'net','-v7.3');
        else
            save(fullfile('results\downsampled_states_optim\',sprintf('net_%d_%d_%d_%d_%d_%d.mat', params.channels1, params.channels2, params.kernel_size1, params.kernel_size2, params.kernel_size3, params.dilation_factor)), 'net','-v7.3');
        end
        valLoss = min(info.ValidationHistory.Loss);
        clear net info;  % Free up memory
        % Penalize for complexity (optional)
        complexityPenalty = 0.0001 * (params.channels1 + params.channels2);
        loss = valLoss + complexityPenalty;
        if useGPU, gpuDevice([]); end  % Reset GPU
    catch ME
        % Return high loss if training fails
        log('Training failed: %s\n', ME.message);
        log('%s\n', getReport(ME,'extended')); % full stack + causes
        loss = 1e6;
    end
end

function y = ternary(c,a,b)
if c, y=a; else, y=b; end
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
        % L1 + MSE mix
        l1 = mean(abs(Yk-Tk),'all');
        mse = mean((Yk-Tk).^2,'all');
        accum = accum + (0.6*mse + 0.4*l1);
    end
    L = accum/numOut;
end

function stop = boCheckpointFn(boResults, state)
% Lightweight checkpoint to avoid saving the full BayesianOptimization object.
stop = false;
if ~strcmp(state,'iteration'), return; end
try
    outDir = fullfile(pwd,'results\downsampled_states_optim\');
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