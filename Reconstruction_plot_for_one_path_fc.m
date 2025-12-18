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



fprintf('[HI_plot] Starting inference script...\n');

% Use explicit fprintf + input('>') to avoid Live Editor / console '?' issueHI_plot
%% 4. Build Inference Datastore (MiniBatchSize=1, no overlap) and specify the network
fprintf('Enter the model file path:\n');
modelFile = strtrim(input('> ', 's'));

fprintf('Enter the data file path (e.g., data/States_109.mat), remember to give path to a dataset formated corresponding to the network you provided:\n');
datapath = strtrim(input('> ', 's'));

fprintf('Enter the frequency index to extract (1-6):\n');
frequencies = str2double(input('> ', 's'));

if ~isempty(regexp(datapath,'(?i)downsampled','once'))
    cycleStruct = load(datapath).States_downsampled; % single cycle
    disp('Using downsampled data for inference datastore.');
else
    cycleStruct = load(datapath).States; % single cycle
end

numCycles = size(cycleStruct, 2);
nInputs = 1; miniBatchSize = 1; overlap = 0; envelope = false;

fprintf('Enter the path :\n');
path_idx = input('> ');

rawDs = CyclemultiInputDatastore_separate_sin_freq_fc(cycleStruct, nInputs, miniBatchSize, overlap, numCycles, envelope, frequencies, 'paths', path_idx);

numCycles = rawDs.NumObservations;

fprintf('Enter the cycles to plot (e.g., [1 10 27] up to %d):\n', numCycles);
cycles_to_plot = input('> ');


% Basic validation / defaults
if isempty(modelFile)
    warning('[HI_plot] Empty modelFile. Set a valid path before continuing.');
end
if isempty(datapath)
    error('[HI_plot] datapath is required.');
end
if isempty(cycles_to_plot)
    cycles_to_plot = 1;
end
if isempty(path_idx)
    path_idx = 1;
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

X = zeros(1,nInputs); % placeholder for previous prediction
% Previous input batch placeholder (cell array with correct length, empty contents)
Y = cell(1, nInputs);

for i = 1:rawDs.NumObservations

    % Read one batch to form a prototype; for multi-batch inference loop, reset after each read.
    [batchTbl, info] = read(rawDs); 
    if width(batchTbl) ~= (2*nInputs)
        error('[HI_plot] Unexpected table width=%d (expected %d).', width(batchTbl), 2*nInputs);
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
    
    doPlot = true; % set true to enable plotting
    if ismember(i, cycles_to_plot)
        % Overlay all prediction outputs (excluding latent) on one axes
        % Create one figure per path (nInputs paths), each with 6 subplots (frequencies)
        numPreds = numel(YPred);
        % Heuristic: if there is an extra first output (latent), skip it
        if numPreds == nInputs + 1
            pathOutStart = 2;
        elseif numPreds == nInputs
            pathOutStart = 1;
        else
            warning('[HI_plot] Unexpected number of outputs (%d) for %d inputs. Attempting best-effort mapping.', numPreds, nInputs);
            pathOutStart = max(1, numPreds - nInputs + 1);
        end
        available_freq = [50,100,125,150,200,250]; % kHz
       
            
        try
        yk = YPred{1};
        catch
        warning('[HI_plot] Failed to access YPred');
        continue;
        end

        fh = figure('Name',sprintf('Cycle_%03d_Path_%02d', i, path_idx), 'NumberTitle','off');
        
        tiledlayout(fh, 1, 1, 'Padding','compact','TileSpacing','compact');
        
        nexttile;
        hold on; grid on;
        title(sprintf('Freq %d kHz Path %d', available_freq(frequencies), path_idx));
        xlabel('Time Index'); ylabel('Amplitude');
        
        try
            refSig = squeeze(extractdata(inputCells{1}(1,:)));
        catch
            refSig = squeeze(inputCells{1}(1,:));
        end
        hIn   = plot(refSig,'k--','LineWidth',1.1,'DisplayName','Input'); % drawn after -> on top
        try
            predSig = squeeze(extractdata(yk(1,:)));
        catch
            predSig = squeeze(yk(1,:));
        end
        % Plot prediction first, then input so input is on top
        hPred = plot(predSig,'b','LineWidth',0.9,'DisplayName','Pred');
        
        legend('Location','best','FontSize',7);
        sgtitle(sprintf('State %d - Path %d Frequency %d kHz', i, path_idx, available_freq(frequencies)));
        hold off;          
        
        
        
    end
    
    if isequal(X, YPred{1})
         warning('[HI_plot] Cycle %d, YPred is identical to previous cycle.', i);
    else
         X = YPred{1}; % store for next comparison
    end
    
end

fprintf('[HI_plot] Inference complete.\n');