% Bayesian Hyperparameter Optimization for Autoencoder
% This script optimizes the architecture and training parameters

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

rng(42); % For reproducibility
start = tic; % Start timer 
for p_i= 2:3 % Loop over paths (adjust as needed)

    % Start a small parallel pool
    if isempty(gcp('nocreate')); parpool('local', 2); end

    %% Define Data Configuration
    dataCfg = struct( ...
        'num_in', 1, ...
        'frequency', 4, ... % Frequency index to extract (only if you want to train single frequency)
        'path_idx', p_i ...
        ); 

    %% Define Hyperparameters to Optimize
    % Datastore parameters
    batch_size = optimizableVariable('batch_size', [4, 27], 'Type', 'integer'); % Overlap size will be set to 0
            
    % Training parameters
    learning_rate = optimizableVariable('learning_rate', [1e-3, 1e-2], 'Transform', 'log');
    lr_drop_factor = optimizableVariable('lr_drop_factor', [0.1, 0.5]);
    lr_drop_period = optimizableVariable('lr_drop_period', [5, 20], 'Type', 'integer');

    optimVars = [batch_size, learning_rate, lr_drop_factor, lr_drop_period];  % Use compressed version to enforce latent size constraint
    
    % Bayesian optimization options
    bayesOpts = struct(...
        'MaxObjectiveEvaluations', 50, ...  % Adjust based on your time budget
        'MaxTime', 24*3600, ...              % 24 hours max
        'IsObjectiveDeterministic', false, ...
        'UseParallel', true, ...             % Use parallel if available
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'Verbose', 1, ...
        'OutputFcn', @boCheckpointFn);

    % Convert struct fields to name-value pairs
    bayesOptArgs = struct2cell(bayesOpts);
    bayesOptNames = fieldnames(bayesOpts);
    nameValuePairs = [bayesOptNames, bayesOptArgs]; % Table of names and values for bayesopt

    %% Define output directory 
    network_type = 'AE_with_sHI_1p'; 
    outDir = fullfile(projectRoot,sprintf('results\\1p1f_%s\\freq_%d\\bayesian_optimization', network_type, dataCfg.frequency));
    if ~exist(outDir,'dir'), mkdir(outDir); end
    %% Define Objective Function
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
    resdir = fullfile(outDir,'bayesian_optim_res');
    if ~exist(resdir,'dir'), mkdir(resdir); end
    finalFile = fullfile(resdir,sprintf('bayesian_optimization_results_p%d.mat', dataCfg.path_idx));
    save(finalFile, 'results', '-v7.3');   % save once, not inside OutputFcn
    fprintf('Saved BO results: %s\n', finalFile);
    fprintf('\nOptimization complete!\n');
    fprintf('Best validation loss: %.6f\n', results.MinObjective);
    disp('Best hyperparameters:');
    disp(results.XAtMinObjective);

    %% Visualize Results - Optimization Progress, Best-So-Far, and Loss Distribution

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

    exportgraphics(h, fullfile(outDir,sprintf('optimization_results_%d.png', dataCfg.path_idx)), 'Resolution',150);
    close(h);


    %% Append run metadata (freq, path, latent size) to a CSV log
    bestParams = results.XAtMinObjective;
    latentCsv = fullfile(outDir, sprintf('best_latent_sizes.csv'));
    row = table(dataCfg.frequency, dataCfg.path_idx, bestParams.batch_size, bestParams.learning_rate, bestParams.lr_drop_factor, bestParams.lr_drop_period, ...
        'VariableNames', {'frequency', 'path_idx', 'batch_size', 'learning_rate', 'lr_drop_factor', 'lr_drop_period'});
    if isfile(latentCsv)
        writetable(row, latentCsv, 'WriteMode', 'append');
    else
        writetable(row, latentCsv);
    end
    fprintf('Appended best latent size to: %s\n', latentCsv);

    %% Train Final Model with Best Hyperparameters
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

    % Log to command window only (safe for parallel workers)
    log = @(fmt,varargin) fprintf('[%s] %s: %s\n', ...
        datestr(now,'HH:MM:SS'), wid, sprintf(fmt,varargin{:}));

    log('start objective; params=%s', jsonencode(params));
    % --- end log setup ---


    % Create datastores
    % det_size helper defined as a nested function because MATLAB does not support inline if/elseif syntax inside anonymous functions.
   
    mkds_sin_freq = @(id) one_path_sHI_sin_freq_fc_datastore(...
        load(sprintf("data\\States_%d.mat", id)).States, ...
        dataCfg.num_in, params.batch_size, 0, det_size(id), dataCfg.frequency, 'paths', dataCfg.path_idx);
    
    function out = det_size(id)
        switch id
            case 103
                out = 32-1;
            case 104
                out = 58-1;
            case 105
                out = 30-1;
            case 109
                out = 28-1;
            otherwise
                error('det_size:UnknownID', 'Unknown id %d', id);
        end
    end
 
    % Build training and validation datastores
    inputDs1 = mkds_sin_freq(103);
    inputDs2 = mkds_sin_freq(104);
    inputDs3 = mkds_sin_freq(105);
    inputDs4 = mkds_sin_freq(109);
  
    trainData= combine(inputDs1,  inputDs2, inputDs3, ReadOrder="sequential");
    valData = inputDs4;
    
   
    % Parse arguments and create output directory
    p = inputParser;
    addParameter(p, 'FinalTraining', false);
    addParameter(p, 'network_type', 'no_name');
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

        % Net configuration
        fc_params.input_size = 15;
        fc_params.latent_size = 1;
        % Net transforming the latent space of the big autoencoder into one value (sHI)
        tiny_net = tiny_architectures_container.tiny_fully_connected_network(fc_params);

        % Path to the folder containing optimized AEs for each path index 
        AE_folder= 'C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Sparse_Fully_Connected_AE\freq_4\bayesian_optimization'; 
        % find the path to the file with the AE for the specified path_idx (search for every file with path_indx in the name)
        pattern = sprintf('%dp*.mat', dataCfg.path_idx);        
        files   = dir(fullfile(AE_folder, pattern));
        if length(files) > 1
            error('Multiple AE model files found for path index %d in folder %s. Please ensure only one file matches the pattern.', dataCfg.path_idx, AE_folder);
        elseif isempty(files)
            error('No AE model file found for path index %d in folder %s. Please ensure a file matches the pattern.', dataCfg.path_idx, AE_folder);
        else
            AE_name = fullfile(AE_folder, files(1).name); % Take the first match (there should be only one file per path index)
        end

        AE_learning_rate = 0.05; % Learning rate for the autoencoder weights and biases
        net = tiny_architectures_container.network_connecter(AE_name, tiny_net, AE_learning_rate);
                
        % Training options
        [a, b, c] = deal(0.001, 1.0, 0.0); % Weights for MSE, monotonicity, proxy loss
        lossFcn = @(varargin) multiOutputMSE_monotonicity(fc_params.latent_size,a,b,c,varargin{:}); % Use standard MSE for stability
        
        if isFinal
            maxEpochs = 100;
        else
            maxEpochs = 50;  % Shorter for optimization
        end

        log('Training network (Final=%d) on %s for %d epochs...\n', isFinal, execEnv, maxEpochs);
        options = trainingOptions('adam', ...
            'InitialLearnRate', params.learning_rate, ... 
            'MiniBatchSize', params.batch_size, ...
            'MaxEpochs', maxEpochs, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropPeriod', params.lr_drop_period, ...
            'LearnRateDropFactor', params.lr_drop_factor, ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', valData, ...
            'ValidationFrequency', 20, ...
            'ValidationPatience', 5, ...
            'OutputNetwork', 'best-validation-loss', ...
            'ExecutionEnvironment', execEnv, ...
            'Verbose', true, ...
            'Plots', 'none');
       
        % Train network
        [net, info] = trainnet(trainData, net, lossFcn, options);

        if isFinal

            model_name = sprintf('%dp%df_b_%d_lr_%d_lrdf_%d_lrdp_%d_valLoss_%f.mat', ...
                dataCfg.path_idx, dataCfg.frequency, params.batch_size, params.learning_rate, params.lr_drop_factor, params.lr_drop_period, min(info.ValidationHistory.Loss));
            save(fullfile(outDir,model_name), 'net','-v7.3');

        end 

        valLoss = min(info.ValidationHistory.Loss);
        clear net info;  % Free up memory
        % Penalize for complexity (optional)
      
        loss = valLoss; 
        if useGPU, gpuDevice([]); end  % Reset GPU
    catch ME
        % Return high loss if training fails and persist error details
        log('Training failed: %s', ME.message);
        log('%s', getReport(ME,'extended')); % full stack + causes (worker-side)

        % Also write error report to a log file on disk so it is visible
        try
            errLogDir = fullfile(pwd,'results','logs');
            if ~exist(errLogDir,'dir'), mkdir(errLogDir); end

            tErr = getCurrentTask();
            if isempty(tErr)
                widErr = 'client';
            else
                widErr = sprintf('w%02d', tErr.ID);
            end

            ts = datestr(now,'yyyymmdd_HHMMSS_FFF');
            errFile = fullfile(errLogDir, sprintf('error_%s.txt', widErr));
            fidErr = fopen(errFile,'w');
            if fidErr>0
                fprintf(fidErr, 'Error on %s at %s\n\n', widErr, datestr(now));
                fprintf(fidErr, '%s\n', getReport(ME,'extended'));
                fclose(fidErr);
            end
        catch logErr %#ok<NASGU>
            % Swallow any logging errors to avoid interfering with BO
        end

        % Emit a warning on the client (if propagated) for quick visibility
        warning('trainAndEvaluateAE:Failure','%s', ME.message);

        loss = 1e6;
    end
end

function y = ternary(c,a,b)
if c, y=a; else, y=b; end
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
    predicted_latent = varargin{2};  % latent is stored as a second output (the size is 1xbatchsize)

    % First target is the proxy label (scalar) (the size of varargin{3} is 4000xbatchsize but only first row is not zero, so you take that row for every batch)
    weibull_target = varargin{3}(1,:); 
    L = dlarray(mean((predicted_latent - weibull_target).^2,'all'));

end

function L = monotonicity(varargin)
   
    predicted_latent = varargin{2};  % Second output (the size is 1xbatchsize)
    plot(predicted_latent( 1, 1:end-1),'o-');
    batchSize = size(predicted_latent, 2);

    % If batch has < 2 elements, no pairwise differences can be computed
    if batchSize < 2
        L = dlarray(0);   % no monotonicity penalty for single-sample batches
        return;
    end

    diff = (predicted_latent( :, 2:end) -  predicted_latent( :,1:end-1)); % Enforce a minimum increase of 0.01 units per cycle
    diff = diff + 0.1;

    %diff = relu(diff);  % Differentiable alternative to max(0, ...)
    diff = diff.*diff;

    L = dlarray(mean(diff, 'all'));

end

function L = multiOutputMSE(varargin)    
  
    Y_pred = varargin{1};    
    Y_target = varargin{4}; 

    L = mean((Y_pred - Y_target).^2,'all'); 

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