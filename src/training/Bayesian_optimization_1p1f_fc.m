% Bayesian Hyperparameter Optimization for Autoencoder
% This script optimizes the architecture and training parameters

%% Setup

rng(42); % For reproducibility
start = tic;
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

for fr= 4:6
    % Start a small parallel pool
    if isempty(gcp('nocreate')); parpool('local', 2); end
    %% Define Data Configuration
    dataCfg = struct( ...
        'num_in', 1, ...
        'envelope', false, ... % Whether to use envelope of signals or raw signals
        'input_size', 4000, ... % Size of each input signal
        'frequency', fr, ... % Frequency index to extract (only if you want to train single frequency)
        'path_idx', 1, ...
        'batch_size', 16 ...
        ); 

    %% Define Hyperparameters to Optimize

    % Architecture parameters
    desired_latent_size = optimizableVariable('desired_latent_size', [10, 15], 'Type', 'integer');
    hl_s1 = optimizableVariable('hl_s1', [1000, 2000], 'Type', 'integer');
    hl_s2 = optimizableVariable('hl_s2', [500, 1000], 'Type', 'integer');
    hl_s3 = optimizableVariable('hl_s3', [100, 500], 'Type', 'integer');
    hl_s4 = optimizableVariable('hl_s4', [30, 100], 'Type', 'integer');
    drop_rate = optimizableVariable('drop_rate', [0.0, 0.5]);

    % Datastore parameters
    %batch_size = optimizableVariable('batch_size', [4, 32], 'Type', 'integer'); % Overlap size will be set to half of batch size
            
    % Training parameters
    % learning_rate = optimizableVariable('learning_rate', [1e-3, 1e-2], 'Transform', 'log');
    % lr_drop_factor = optimizableVariable('lr_drop_factor', [0.1, 0.5]);
    % lr_drop_period = optimizableVariable('lr_drop_period', [5, 20], 'Type', 'integer');

    %% Bayesian Optimization Settingsla
    optimVars = [desired_latent_size, hl_s1, hl_s2, hl_s3, hl_s4, drop_rate]; %, batch_size, learning_rate, lr_drop_factor, lr_drop_period];  % Use compressed version to enforce latent size constraint
    network_type = 'Fully Connected Network'; 
    % Bayesian optimization options
    bayesOpts = struct(...
        'MaxObjectiveEvaluations', 100, ...  % Adjust based on your time budget
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
    outDir = fullfile(projectRoot,sprintf('results\\1p1f_%s\\freq_%d\\path_%d\\bayesian_optimization', network_type, dataCfg.frequency, dataCfg.path_idx));
    if ~exist(outDir,'dir'), mkdir(outDir); end
    objectiveFcn = @(params) trainAndEvaluateAE(params, dataCfg, 'network_type', network_type,'outdir',outDir);
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

    % Append run metadata (freq, path, latent size) to a CSV log
    latentCsv = fullfile('results', 'best_latent_sizes.csv');
    row = table(dataCfg.frequency, dataCfg.path_idx, bestParams.desired_latent_size, bestParams.hl_s1, bestParams.hl_s2, bestParams.hl_s3, bestParams.hl_s4, bestParams.drop_rate, ...
        'VariableNames', {'frequency', 'path_idx', 'latent_size', 'hl_s1', 'hl_s2', 'hl_s3', 'hl_s4', 'drop_rate'});
    if isfile(latentCsv)
        writetable(row, latentCsv, 'WriteMode', 'append');
    else
        writetable(row, latentCsv);
    end
    fprintf('Appended best latent size to: %s\n', latentCsv);

    %% (Optional) Train Final Model with Best Hyperparameters

    % % Train with more epochs
    finalLoss = trainAndEvaluateAE(bestParams, dataCfg, 'FinalTraining', true, 'network_type', network_type, 'outdir', outDir);
    fprintf('Final validation loss: %.6f\n', finalLoss);
    endTime = toc(start);
    fprintf('Total optimization time: %.2f seconds\n', endTime);
end
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
   
    mkds_sin_freq = @(id) CyclemultiInputDatastore_separate_sin_freq_fc(...
        load(sprintf("data\\States_%d.mat", id)).States, ...
        dataCfg.num_in, dataCfg.batch_size, 2, det_size(id), dataCfg.envelope, dataCfg.frequency, 'paths', dataCfg.path_idx);
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
 
    inputDs1 = mkds_sin_freq(103);
    inputDs2 = mkds_sin_freq(104);
    inputDs3 = mkds_sin_freq(105);
    inputDs4 = mkds_sin_freq(109);
    inputDs = combine(inputDs1,  inputDs2, inputDs3, inputDs4, ReadOrder="sequential");  % Combine datastores'


    total_obs = inputDs1.NumObservations+inputDs2.NumObservations+inputDs3.NumObservations+inputDs4.NumObservations;
    train_obs_idx = randperm(total_obs, round(0.80*total_obs));
    val_obs_idx = setdiff(1:total_obs, train_obs_idx);

    trainData= subset(inputDs, train_obs_idx);
    valData = subset(inputDs, val_obs_idx);
    
   
    % Train and evaluate autoencoder with given hyperparameters
    p = inputParser;
    addParameter(p, 'FinalTraining', false);
    addParameter(p, 'network_type', 'compressed');
    addParameter(p, 'outdir', '');
    parse(p, varargin{:});
    outDir = p.Results.outdir;
    network_type = p.Results.network_type;
    if isempty(outDir)
        outDir = fullfile(pwd, 'results', sprintf('1p1f_%s', network_type), ...
            sprintf('freq_%d', dataCfg.frequency),sprintf('path_%d', dataCfg.path_idx), ...
            'bayesian_optimization');
    end
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    isFinal = p.Results.FinalTraining;
    
    
    try        
        % Pick device per worker: 1 GPU worker, others CPU
        t = getCurrentTask(); w = []; if ~isempty(t), w = t.ID; end
        useGPU = gpuDeviceCount > 0 && (isempty(w) || w == 1);
        execEnv = ternary(useGPU,'gpu','cpu');

        % Build network with current parameters
        params.num_inputs = dataCfg.num_in;
        net = architectures_container.deep_fully_connected_network(params);
        % Training options
        if isFinal
            maxEpochs = 100;
        else
            maxEpochs = 30;  % Shorter for optimization
        end
        log('Training network (Final=%d) on %s for %d epochs...\n', isFinal, execEnv, maxEpochs);
        options = trainingOptions('adam', ...
            'InitialLearnRate', 0.001, ... %params.learning_rate, ...
            'MiniBatchSize', dataCfg.batch_size, ... %params.batch_size, ...
            'MaxEpochs', maxEpochs, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropPeriod', 18, ... %params.lr_drop_period, ...
            'LearnRateDropFactor', 0.5, ... %params.lr_drop_factor, ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', valData, ...
            'ValidationFrequency', 20, ...
            'ValidationPatience', 5, ...
            'OutputNetwork', 'best-validation-loss', ...
            'ExecutionEnvironment', execEnv, ...
            'Verbose', true, ...
            'Plots', 'none');
        
        lossFn = @(varargin) peak_preserving_noise_suppressing_loss(varargin{:});
        % Train network
        [net, info] = trainnet(trainData, net, lossFn, options);

        if isFinal
         
        %    model_name = sprintf('%dp%df_h1_%d_h2_%d_h3_%d_h4_%d_dr_%.2f_bs_%d_lr_%f_lrdropf_%f_lrdropp_%d_valLoss_%f.mat', ...
        %         dataCfg.path_idx, dataCfg.frequency, params.hl_s1, params.hl_s2, params.hl_s3, params.hl_s4, ...
        %         params.drop_rate, params.batch_size, ...
        %         params.learning_rate, params.lr_drop_factor, params.lr_drop_period, ...
        %         min(info.ValidationHistory.Loss));
        model_name = sprintf('%dp%df_h1_%d_h2_%d_h3_%d_h4_%d_dr_%.2f_bs_%d_valLoss_%f.mat', ...
                dataCfg.path_idx, dataCfg.frequency, params.hl_s1, params.hl_s2, params.hl_s3, params.hl_s4, ...
                params.drop_rate, dataCfg.batch_size,min(info.ValidationHistory.Loss));
            save(fullfile(outDir,model_name), 'net','-v7.3');
        end 
        valLoss = min(info.ValidationHistory.Loss);
        clear net info;  % Free up memory
        % Penalize for complexity (optional)
      
        loss = valLoss; 
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

function stop = boCheckpointFn(boResults, state)
% Lightweight checkpoint to avoid saving the full BayesianOptimization object.
stop = false;
if ~strcmp(state,'iteration'), return; end
try
    outDir = fullfile(pwd,'results\1p1f_Fully Connected Network\bayesian_optimization');
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