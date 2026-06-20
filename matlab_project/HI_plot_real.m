








%% Health Index (HI) / Prediction Plot Script
% Robust loader for trained multi-input GAN network and inference over a cycle.
% Fixes issues:
%  - Custom layer classes not on path (change_format, deconcatenation, GAN)
%  - Supplying full training-style table (with latent + targets) directly to predict
%  - MiniBatch/Overlap mismatch for inference

%% 1. Path Setup (mirror training script essentials)
scriptDir = fileparts(mfilename('fullpath'));
projectRoot = scriptDir; % this file appears at project root
addpath(fullfile(projectRoot,'src','models'));
addpath(fullfile(projectRoot,'src','models/layers'));
addpath(fullfile(projectRoot,'src','data'));
addpath(fullfile(projectRoot,'src','utils'));

reqClasses = { 'change_format','deconcatenation','GAN' };
for c = reqClasses
	pth = which(c{1});
	if isempty(pth)
		error('[HI_plot] Required class "%s" not found on path. Add src/models before loading network.', c{1});
	else
		fprintf('[HI_plot] Found class %s at %s\n', c{1}, pth);
	end
end


close all; 
fprintf('[HI_plot] Starting inference script...\n');

% Use explicit fprintf + input('>') to avoid Live Editor / console '?' issueHI_plot

fprintf('Enter the model file path:\n');
modelFile = strtrim(input('> ', 's'));

fprintf('Enter the panel number (e.g., 109):\n');
panel_number = strtrim(input('> ', 's'));
datapath = sprintf('data/States_%s.mat', panel_number); % default datapath based on panel number


% fprintf('Do you want to use one frequency only? (true/false):\n');
single_freq = true;
if single_freq
    % fprintf('Enter the frequency index to extract (1-6):\n');
    % frequencies = str2double(input('> ', 's'));
    frequencies = 4;
else
    frequencies = 1:6; % specify which frequencies to plot (1 to 6)
end
% Extract from the model file name the path 
% ib = strfind(modelFile,'fc_path_');
% ie = strfind(modelFile,'_net_loss');
% if isempty(ib) || isempty(ie) || ie <= ib
%     error('[HI_plot] Could not extract path index from model file name: %s', modelFile);
% end

% paths_to_plot = str2double(modelFile(ib+length('fc_path_'):ie-1));
paths_to_plot=2;
%% 4. Build Inference Datastore (MiniBatchSize=1, no overlap)
if ~isempty(regexp(datapath,'(?i)downsampled','once'))
    cycleStruct = load(datapath).States_downsampled; % single cycle
    disp('Using downsampled data for inference datastore.');
else
    cycleStruct = load(datapath).States; % single cycle
end
numCycles = size(cycleStruct, 2);
nInputs = 1; miniBatchSize = 1; overlap = 0;  % numCycles used only for latent formula

if single_freq
    rawDs = one_path_sHI_sin_freq_fc_datastore(cycleStruct, nInputs, miniBatchSize, overlap, numCycles, frequencies,'paths', paths_to_plot);
else
    rawDs = CyclemultiInputDatastore_separate(cycleStruct, nInputs, miniBatchSize, overlap, numCycles, envelope);
end
numCycles = rawDs.NumObservations;



% Basic validation / defaults
if isempty(modelFile)
    warning('[HI_plot] Empty modelFile. Set a valid path before continuing.');
end
if isempty(datapath)
    error('[HI_plot] datapath is required.');
end
if isempty(paths_to_plot)
    error('[HI_plot] paths_to_plot is required.');
end


%% 2. Load Trained Network
% Auto-select most recent valid trained GAN model file in results folder
resultsDir = fullfile(projectRoot,'results');
fprintf('[HI_plot] Using model file: %s\n', modelFile);
if ~isfile(modelFile)
    error('[HI_plot] Model file not found: %s', modelFile);
end

if isempty(modelFile)
    error('[HI_plot] No suitable model file containing trained_net found in %s.', resultsDir);
end
% --- Load and display model metadata safely --------------------------------
% Attempt to load model_metadata from the same model file as the network.
metaInfo = whos('-file', modelFile);
hasMeta  = ismember('model_metadata', {metaInfo.name});

if ~hasMeta
    warning('[HI_plot] model_metadata not found in %s. Skipping metadata display.', modelFile);
else
    Smeta = load(modelFile,'model_metadata');
    model_metadata = Smeta.model_metadata;
    if ~isstruct(model_metadata)
        warning('[HI_plot] model_metadata exists but is not a struct (class=%s).', class(model_metadata));
    else
        fprintf('[HI_plot] -------- Model Metadata --------\n');
        flds = fieldnames(model_metadata);
        for f = 1:numel(flds)
            key = flds{f};
            val = model_metadata.(key);
            if isnumeric(val)
                if isscalar(val)
                    fprintf('  %-15s : %g\n', key, val);
                else
                    if isvector(val)
                        fprintf('  %-15s : [%s]\n', key, strtrim(sprintf('%g ', val(:)')));
                    else
                        sz = size(val);
                        fprintf('  %-15s : numeric array size [%s]\n', key, strjoin(string(sz),'x'));
                    end
                end
            elseif isstring(val) || ischar(val)
                fprintf('  %-15s : %s\n', key, string(val));
            elseif isstruct(val)
                fprintf('  %-15s : MSE : %d Monotonicity: %d Proxy : %d\n', key, val.MSE, val.Monotonicity, val.Proxy);
            else
                fprintf('  %-15s : (%s)\n', key, class(val));
            end
        end
        % Explicit important fields (with existence checks)
        impFields = {'timestamp','num_inputs','batch_size','max_epochs','initial_lr', ...
            'optimizer','loss_function','data_cycles','latent_size','final_loss'};
        for k = 1:numel(impFields)
            if ~isfield(model_metadata, impFields{k})
                warning('[HI_plot] Missing expected field: %s', impFields{k});
            end
        end
        fprintf('[HI_plot] --------------------------------\n');
    end
end
if ~isfile(modelFile)
	error('[HI_plot] Model file not found: %s', modelFile);
end

S = load(modelFile, 'trained_net', 'net','finalNet');
if ~isfield(S,'trained_net') 
    if ~isfield(S,'net')
        if ~isfield(S,'finalNet')
            error('[HI_plot] Neither "trained_net" nor "net" nor "finalNet" found in %s', modelFile);
        else
            net=S.finalNet;
        end
    else
        warning('[HI_plot] Using field "net" instead of "trained_net" from %s', modelFile);
        net = S.net;
    end
else
    net = S.trained_net; % dlnetwork expected
end
fprintf('[HI_plot] Loaded network with %d learnables.\n', height(net.Learnables));

%% 3. Inspect Inputs
inNames = net.InputNames; nInputs = numel(inNames);
fprintf('[HI_plot] Network expects %d inputs.\n', nInputs);


X = zeros(1,28); % placeholder for previous prediction
% Previous input batch placeholder (cell array with correct length, empty contents)
Y = cell(1, nInputs);
latent_space = zeros(nInputs, rawDs.NumObservations);
plot_rec_cycle = [20];
for i = 1:rawDs.NumObservations

    % Read one batch to form a prototype; for multi-batch inference loop, reset after each read.
    [batchTbl, info] = read(rawDs); %#ok<NASGU>
    if width(batchTbl) ~= (2*nInputs+1)
        error('[HI_plot] Unexpected table width=%d (expected %d).', width(batchTbl), 2*nInputs+1);
    end

    % Extract only the input columns (first nInputs). Each entry: [B C]
    inputCells = cell(1, nInputs);
    for k = 1:nInputs
        arr = batchTbl{1,k}; % row 1, column k cell content        
        inputCells{k} = arr; % keep batch dimension
    end
    
    if isequal(Y, inputCells) 
        warning('[HI_plot] Cycle %d, InputCells is identical to previous cycle.', i);
    else
        Y = inputCells; % store for next comparison
    end
    %% 5. Forward Prediction
    try
        nOut = numel(net.OutputNames);
        YPred = cell(1,nOut);
        [YPred{:}] = predict(net, inputCells{:});
    catch ME
        fprintf(2,'[HI_plot][ERROR] Prediction failed: %s\n', ME.message);
        rethrow(ME);
    end
   
    latent_space(:,i) = squeeze(YPred{2}); % store latent space output

    if ismember(i,plot_rec_cycle)
        % Plot reconstructed signal 
        HI_reconstructed = squeeze(YPred{1});
        figure("Name",sprintf("Reconstructed HI at Cycle %d", i),"NumberTitle","off");
        hold on;
        plot(HI_reconstructed);
        plot(inputCells{1});
        legend('Reconstructed HI','Original Signal');
        title(sprintf('Reconstructed signal at Cycle %d', i));
        xlabel('Time Step');
        ylabel('Amplitude');
        hold off;
    end
end

%% 6. OPTIONAL: Visualize Latent Space

flatent = figure("Name","Latent Space","NumberTitle","off");
imagesc(latent_space);
colorbar;
title('Latent Space Over Cycles');
xlabel('Cycle Index');
ylabel('Input Index');

RUL = (1:numCycles)/numCycles; % Remaining Useful Life from 1 to 0
line_flatent = figure("Name","Latent Space Lines","NumberTitle","off");
hold on;
for i = 1:nInputs
    plot(RUL,latent_space(i,:), '-o', 'DisplayName', sprintf('Input %d', i));
end
title('Latent Space Over Cycles - Line Plot');
xlabel('Cycle Index');
ylabel('Latent Value');
legend('show');
hold off;


%% 7. OPTIONAL: Loop Over Entire Cycle (uncomment to process all observations)
% reset(rawDs);
% preds = {};
% while hasdata(rawDs)
%     [bt,~] = read(rawDs);
%     ic = cell(1,nInputs);
%     for k=1:nInputs
%         ic{k} = bt{1,k};
%     end
%     preds{end+1} = predict(net, ic{:}); %#ok<AGROW>
% end
% fprintf('[HI_plot] Processed %d mini-batches.\n', numel(preds));

fprintf('[HI_plot] Inference complete.\n');