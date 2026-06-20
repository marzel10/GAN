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

function L = multiOutputMSE_monotonicity(latent_size, a, b, c, P, T, proxy)      
    if a>0
        MSE = multiOutputMSE(P,T);
    else
        MSE = 0;
    end
    if latent_size == 1 && b>0
        mono = monotonicity(P,T);
    else
        mono = 0;
    end
    if latent_size == 1 && c>0
        proxy_loss = proxy_labels_loss(P, proxy);
    else
        proxy_loss = 0;
    end

    L = a* MSE + b*mono + c*proxy_loss;
    
end

function L = proxy_labels_loss(P, proxy)
    predicted_latent = P;  % latent is stored as a second output (the size is 1xbatchsize)
    weibull_target = proxy; % Proxy scalar per sample

    L = dlarray(mean((predicted_latent - weibull_target).^2,'all'));

end

function L = monotonicity(P,T)
   
    predicted_latent = P;  % Second output (the size is 1xbatchsize)
    batchSize = size(predicted_latent, 2);
    

    % If batch has < 2 elements, no pairwise differences can be computed
    if batchSize < 2
        L = dlarray(0);   % no monotonicity penalty for single-sample batches
        return;
    end

    diff = (predicted_latent( :, 2:end) -  predicted_latent( :,1:end-1)); % Enforce a minimum increase of 0.01 units per cycle
    plot(diff);
    diff = diff + 0.01;
    %violations = relu(diff);  % Differentiable alternative to max(0, ...)
    
    violations = diff.*diff;
    L = dlarray(mean(violations, 'all'));
end

function L = multiOutputMSE(P,T)    

    Yk = P; 
    Tk = T; 

    % Unwrap if elements are cells (can happen with some datastores/trainnet packings)
    if iscell(Yk), Yk = Yk{1}; end
    if iscell(Tk), Tk = Tk{1}; end

    L = mean((Yk - Tk).^2,'all'); 
end

%% Data Preparation

rng(42); % For reproducibility
% Create single datastore that handles all 28 inputs and targets
num_in = 1;
b_size = 27;
overlap_size = 0;
% 148 instances of data for the training and 28 for validation
envelope = false; % Whether to use envelope of signals or raw signals
freq = 4; % frequency index to extract
path_idx = 2; % path index to extract
benchmark = false; % whether to use benchmark datastore (no random cropping)
fc = true; % whether to use fully connected network


States103 = load(sprintf("data\\States_%d.mat", 103)).States;  % Load the Cycle1 datastore
States104 = load(sprintf("data\\States_%d.mat", 104)).States;  % Load the Cycle2 datastore
States105 = load(sprintf("data\\States_%d.mat", 105)).States;  % Load the Cycle3 datastore
States109 = load(sprintf("data\\States_%d.mat", 109)).States;  % Load the Cycle4 datastore

inputDs1 = one_path_sHI_sin_freq_fc_datastore(States103, num_in, b_size, overlap_size, 32-1, freq, 'paths', path_idx); % Provide paths only for first datastore
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
    
    disp(size(testData));  % Display size of the table
    % Check the size of the first input
    firstInput = testData{1,1};  % Get first column, first row
    fprintf('First input size: %s (should be 1x4000)\n', mat2str(size(firstInput)));
    reset(inputDs4);  % Reset for training
end
fprintf('Datastore test complete.\n');
%% Network Training
AE_folder= 'C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Sparse_Fully_Connected_AE\freq_4\bayesian_optimization';
% find the path to path_idx AE (search for every file with path_indx in the name)
pattern = sprintf('%dp*.mat', path_idx);          % "28p*.mat"
files   = dir(fullfile(AE_folder, pattern));
AE_name = fullfile(AE_folder, files(1).name); % Take the first match
if isempty(AE_name)
    error('No AE model found for path index %d in folder %s', path_idx, AE_folder);
end
tiny_net = tiny_architectures_container.tiny_fully_connected_network(fc_params);
net = tiny_architectures_container.network_connecter(AE_name, tiny_net, AE_learning_rate);
disp("Number of network outputs and inputs:");
disp(length(net.OutputNames)+length(net.InputNames));
net_name = "FC_Net_with_ae_training";

fprintf('Network loaded successfully. Starting training...\n');

format_input =repmat("BC", 1, num_in);
format_targets = repmat("BC", 1, num_in+1);

% Assume net is a dlnetwork
mbq = minibatchqueue(inputDs_train, 3, ...
    "MiniBatchSize", b_size, ...
    "PartialMiniBatch", "return", ...
    "MiniBatchFcn", @preprocessMiniBatch);  % input, proxy, target


function [X, proxy, T] = preprocessMiniBatch(Xcol, proxyCol, Tcol)
    % incoming columns are B×F; transpose to F×B for "CB"
    X = dlarray(Xcol.', "CB");
    T = dlarray(Tcol.', "CB");
    proxy = dlarray(proxyCol.', "CB");  % scalar proxy per sample
end

numEpochs = 3; % keep small for testing
lossFig = figure('Name','Training Loss','NumberTitle','off');
lossAx = axes(lossFig);
trainLine = animatedline(lossAx,'Color',[0 0.45 0.74],'LineWidth',1.2);
valLine   = animatedline(lossAx,'Color',[0.85 0.33 0.1],'LineWidth',1.2);
grid(lossAx,'on'); xlabel(lossAx,'Iteration'); ylabel(lossAx,'Loss');
legend(lossAx,{'train','val'},'Location','northeast');
iterGlobal = 0;
gradientsAvg = [];
squaredGradientsAvg = [];
for epoch = 1:numEpochs
    reset(mbq);
    iter = 0;
    tic;
    while hasdata(mbq)
        iter = iter + 1;
        [XBatch, proxyBatch, TBatch] = next(mbq);

        [lossVal, gradients] = dlfeval(@modelGradients, net, XBatch, TBatch, proxyBatch);
        gradNorm = sqrt(sum(cellfun(@(g) sum(g.^2,'all'), gradients.ParameterGradients)));

        fprintf('Epoch %d Iter %d | Loss %.4f | GradNorm %.4f | t=%.2fs\n', ...
            epoch, iter, double(lossVal), double(gradNorm), toc);

        % Update parameters (Adam shown)
        [net, trailingAvg, trailingSqAvg] = adamupdate(net, gradients, ...
            trailingAvg, trailingSqAvg, iter, 0.001, 0.9, 0.999);

        iterGlobal = iterGlobal + 1;
        addpoints(trainLine, iterGlobal, double(lossVal));
        drawnow limitrate;

        if mod(iterGlobal, 50) == 0
            valLoss = double(evaluateLoss(net, inputDs_val)); % write this helper to run one pass
            addpoints(valLine, iterGlobal, valLoss);
        end
    end
end

function [lossVal, gradients] = modelGradients(net, X, T, proxy)
    Y = forward(net, X);
    [a, b, c] = deal(0.5, 1, 0); % Weights for MSE, monotonicity, proxy loss
    lossVal = multiOutputMSE_monotonicity(15, a, b, c, Y, T, proxy); % your loss
    gradients = dlgradient(lossVal, net.Learnables);
end

function valLoss = evaluateLoss(net, ds)
    reset(ds);
    total = 0; n = 0;
    while hasdata(ds)
        tbl = read(ds);
        X = dlarray(tbl{1,1}.', "CB");
        proxy = dlarray(tbl{1,2}.', "CB");
        T = dlarray(tbl{1,3}.', "CB");
        [a, b, c] = deal(0.5, 1, 0); % Weights for MSE, monotonicity, proxy loss
        l = multiOutputMSE_monotonicity(15, a, b, c, forward(net,X), T, proxy);
        total = total + double(l); n = n + 1;
    end
    valLoss = total / max(n,1);
    reset(ds);
end

%An 'epoch' represents one pass through the entire training dataset, 
% while an 'iteration' corresponds to one update of the model's parameters using a mini-batch of data during training.


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



