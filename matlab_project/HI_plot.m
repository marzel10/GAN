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

fprintf('Enter the data file path (e.g., data/States_109.mat), remember to give path to a dataset formated corresponding to the network you provided:\n');
datapath = strtrim(input('> ', 's'));

fprintf('Do you want to use one frequency only? (true/false):\n');
single_freq = input('> ');
if single_freq
    fprintf('Enter the frequency index to extract (1-6):\n');
    frequencies = str2double(input('> ', 's'));
else
    frequencies = 1:6; % specify which frequencies to plot (1 to 6)
end
%% 4. Build Inference Datastore (MiniBatchSize=1, no overlap)
if ~isempty(regexp(datapath,'(?i)downsampled','once'))
    cycleStruct = load(datapath).States_downsampled; % single cycle
    disp('Using downsampled data for inference datastore.');
else
    cycleStruct = load(datapath).States; % single cycle
end
numCycles = size(cycleStruct, 2);
nInputs = 1; miniBatchSize = 1; overlap = 0; envelope = false; % numCycles used only for latent formula

if single_freq
    rawDs = CyclemultiInputDatastore_separate_sin_freq(cycleStruct, nInputs, miniBatchSize, overlap, numCycles, envelope, frequencies,'paths', paths_to_plot);
else
    rawDs = CyclemultiInputDatastore_separate(cycleStruct, nInputs, miniBatchSize, overlap, numCycles, envelope);
end
numCycles = rawDs.NumObservations;

fprintf('Enter the cycles to plot (e.g., [1 10 27] up to %d):\n', numCycles);
cycles_to_plot = input('> ');

fprintf('Enter the paths to plot (e.g., [2 15 28]):\n');
paths_to_plot = input('> ');

fprintf('Plot latent space? (true/false):\n');
plot_latent = input('> ');

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
if isempty(paths_to_plot)
    paths_to_plot = 1;
end
if isempty(plot_latent)
    plot_latent = false;
end

%% 2. Load Trained Network
% Auto-select most recent valid trained GAN model file in results folder
resultsDir = fullfile(projectRoot,'results');
fprintf('[HI_plot] Using model file: %s\n', modelFile);
if ~isfile(modelFile)
    error('[HI_plot] Model file not found: %s', modelFile);
end
whos('-file', modelFile); % display contents
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
if nInputs ~= 28
	warning('[HI_plot] Network expects %d inputs (not 28). Adjusting datastore construction accordingly.', nInputs);
end




%time = cycleStruct(1).Frequency(1).Pair_idx(1).Time;
%Fs = (time(2)-time(1))^-1; % sampling frequency
%L = length(time);
X = zeros(1,28); % placeholder for previous prediction
% Previous input batch placeholder (cell array with correct length, empty contents)
Y = cell(1, nInputs);
latent_space = zeros(nInputs, rawDs.NumObservations);

for i = 1:rawDs.NumObservations

    % Read one batch to form a prototype; for multi-batch inference loop, reset after each read.
    [batchTbl, info] = read(rawDs); %#ok<NASGU>
    if width(batchTbl) ~= (2*nInputs)
        error('[HI_plot] Unexpected table width=%d (expected %d).', width(batchTbl), 2*nInputs);
    end

    % Extract only the input columns (first nInputs). Each entry: [B T S C]
    inputCells = cell(1, nInputs);
    for k = 1:nInputs
        arr = batchTbl{1,k}; % row 1, column k cell content
        % if ndims(arr) ~= 4
        %     error('[HI_plot] Input %d has ndims=%d (expected 4: BxTxSxC).', k, ndims(arr));
        % end
        
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
        % for k = 1:nOut
        %     eval(sprintf('YPred%d = YPred{k};',k));
        % end
    catch ME
        fprintf(2,'[HI_plot][ERROR] Prediction failed: %s\n', ME.message);
        rethrow(ME);
    end
    %latent_space(:,i) = squeeze(YPred{1}); % store latent space output

    %cycles_to_plot = [1, 10, 27]; % specify which cycles to plot
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
        % Loop over each path
        for p = paths_to_plot
            outIdx = pathOutStart + (p-1);
            if outIdx > numPreds
            warning('[HI_plot] Output index %d exceeds available predictions (%d). Skipping path %d.', outIdx, numPreds, p);
            continue;
            end
            try
            yk = YPred{outIdx};
            catch
            warning('[HI_plot] Failed to access YPred{%d} for path %d.', outIdx, p);
            continue;
            end
            fh = figure('Name',sprintf('Cycle_%03d_Path_%02d', i, p), 'NumberTitle','off');
            if single_freq
                tiledlayout(fh, 1, 1, 'Padding','compact','TileSpacing','compact');
                freq = frequencies(1);
                nexttile;
                hold on; grid on;
                title(sprintf('Freq %d', freq));
                xlabel('Time Index'); ylabel('Amplitude');
                % Reference signal (input path p)
                try
                    refSig = squeeze(extractdata(inputCells{p}(1,:,1,freq)));
                catch
                    refSig = squeeze(inputCells{p}(1,:,1,freq));
                end
                hIn   = plot(refSig,'k--','LineWidth',1.1,'DisplayName','Input'); % drawn after -> on top
                try
                    predSig = squeeze(extractdata(yk(1,:,1,freq)));
                catch
                    predSig = squeeze(yk(1,:,1,freq));
                end
                % Plot prediction first, then input so input is on top
                hPred = plot(predSig,'b','LineWidth',0.9,'DisplayName','Pred');
                
                legend('Location','best','FontSize',7);
                sgtitle(sprintf('State %d - Path %d (Output %d)', i, p, outIdx));
                continue; % skip to next path
            else
                tiledlayout(fh, 2, 6, 'Padding','compact','TileSpacing','compact');
                for fi = 1:numel(frequencies)
                    freq = frequencies(fi);
                    nexttile;
                    hold on; grid on;
                    title(sprintf('Freq %d', freq));
                    xlabel('Time Index'); ylabel('Amplitude');
                    % Reference signal (input path p)
                    try
                        refSig = squeeze(extractdata(inputCells{p}(1,:,1,freq)));
                    catch
                        refSig = squeeze(inputCells{p}(1,:,1,freq));
                    end
                    hIn   = plot(refSig,'k--','LineWidth',1.1,'DisplayName','Input'); % drawn after -> on top
                    try
                        predSig = squeeze(extractdata(yk(1,:,1,freq)));
                    catch
                        predSig = squeeze(yk(1,:,1,freq));
                    end
                    % Plot prediction first, then input so input is on top
                    hPred = plot(predSig,'b','LineWidth',0.9,'DisplayName','Pred');
                    
                    if fi == 1
                        legend('Location','best','FontSize',7);
                    end
                    
                    % nexttile;
                    % hold on; grid on;
                    % title(sprintf('Freq %d', freq));
                    % xlabel('Time Index'); ylabel('Amplitude');

                    % % Predicted signal
                    % try
                    %     predSig = squeeze(extractdata(yk(1,:,1,freq)));
                    % catch
                    %     predSig = squeeze(yk(1,:,1,freq));
                    % end
                    % % Plot prediction first, then input so input is on top
                    % hPred = plot(predSig,'b','LineWidth',0.9,'DisplayName','Pred');
                    
                    % if fi == 1
                    %     legend('Location','best','FontSize',7);
                    % end
                end
            end
            sgtitle(sprintf('State %d - Path %d (Output %d)', i, p, outIdx));
        end
        
    end
    % (Optional) save each figure
    % outDir = fullfile(projectRoot,'results','HI_plots');
    % if ~exist(outDir,'dir'), mkdir(outDir); end
    % saveas(fh, fullfile(outDir, sprintf('cycle_%03d.png',i)));
    if isequal(YPred{2},YPred{3})
        disp('YPred2 and YPred3 are identical.');
    end
    if isequal(X, YPred{2})
         warning('[HI_plot] Cycle %d, YPred is identical to previous cycle.', i);
    else
         X = YPred{2}; % store for next comparison
        
    end
    
end

%% 6. OPTIONAL: Visualize Latent Space
if plot_latent 
    flatent = figure("Name","Latent Space","NumberTitle","off");
    imagesc(latent_space);
    colorbar;
    title('Latent Space Over Cycles');
    xlabel('Cycle Index');
    ylabel('Input Index');
end

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