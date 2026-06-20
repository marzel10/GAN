








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

fprintf('Enter which panel to use\n');
panel_number = str2double(input('> ', 's'));
fprintf('Is the network using Graph layer? (true/false):\n');
GAN_mode = input('> ');
if ~GAN_mode
    path_id = input('Enter the path (1-28):\n');
end

fprintf('Do you want to use one frequency only? (true/false):\n');
single_freq = input('> ');
if single_freq
    fprintf('Enter the frequency index to extract (1-6):\n');
    frequencies = str2double(input('> ', 's'));
else
    frequencies = 1:6; % specify which frequencies to plot (1 to 6)
end

fprintf('Enter the paths to plot (e.g., [2 15 28], 29 - all paths):\n');
paths_to_plot = input('> ');
if paths_to_plot == 29
    paths_to_plot = 1:28; % all paths
end
%% 4. Build Inference Datastore (MiniBatchSize=1, no overlap)
miniBatchSize = 1; overlap = 0;  % numCycles used only for latent formula
AE_net_folder = 'C:\Users\Maria\Documents\Honours Programme\Networks\GAN\results\1p1f_Sparse_Fully_Connected_AE\freq_4\bayesian_optimization';
cacheRoot = fullfile(projectRoot, 'data', 'latent_cache');
if ~exist(cacheRoot,'dir'), mkdir(cacheRoot); end

if single_freq
    if GAN_mode
        cf = fullfile(cacheRoot, sprintf('latent_cache_panel_%d_freq_%d.mat', panel_number, frequencies));
        rawDs = latent_space_datastore(panel_number, miniBatchSize, overlap, frequencies, true, AE_net_folder, 'cacheFile', cf);
    else
        % FC mode: single path latent per cycle (no spatial dims), CB input format
        rawDs = latent_space_datastore(panel_number, miniBatchSize, overlap, frequencies, false, AE_net_folder, 'path_id', path_id);
    end
else
    error('Only single frequency mode is supported in this script.');
end
numCycles = rawDs.NumObservations;



% Basic validation / defaults
if isempty(modelFile)
    warning('[HI_plot] Empty modelFile. Set a valid path before continuing.');
end

% if isempty(paths_to_plot)
%     error('[HI_plot] paths_to_plot is required.');
% end


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


% Preallocate latent space buffer: 28 paths for GAN, 1 for FC
if GAN_mode
    latent_space = zeros(28, rawDs.NumObservations);
else
    latent_space = zeros(1, rawDs.NumObservations);
end

for i = 1:rawDs.NumObservations

    % Read one batch to form a prototype; for multi-batch inference loop, reset after each read.
    [batchTbl, info] = read(rawDs); %#ok<NASGU>
    w = width(batchTbl);
    inputCells = cell(1, nInputs);
    if w == (nInputs + 1)
        % Case A: table provides each input as its own column
        for k = 1:nInputs
            raw = batchTbl{1,k};
            % GAN: SCB/SSCB, FC: CB
            if GAN_mode
                inputCells{k} = coerceToDlInput(raw, false, 'SCB');
            else
                inputCells{k} = coerceToDlInput(raw, false, 'CB');
            end
            sz = size(extractdata(inputCells{k}));
            fprintf('[HI_plot] Input %d -> dlarray size [%s], class=%s\n', k, strjoin(string(sz),'x'), class(extractdata(inputCells{k})));
        end
    elseif w == 2
        % Case B: single input column containing SxC with channels to split
        raw = batchTbl{1,1}; % S x C (or S x S2 x C)
        % If multiple inputs are expected, split along channel dimension
        if nInputs == 1
            if GAN_mode
                inputCells{1} = coerceToDlInput(raw, false, 'SCB');
            else
                % FC expects CB only, no spatial dims
                inputCells{1} = coerceToDlInput(raw, false, 'CB');
            end
            sz = size(extractdata(inputCells{1}));
            fprintf('[HI_plot] Input 1 -> dlarray size [%s], class=%s\n', strjoin(string(sz),'x'), class(extractdata(inputCells{1})));
        else
            % Determine channel dim position before adding batch: 2D->dim2, 3D->dim3
            if ndims(raw) == 2
                C = size(raw,2);
                if nInputs > C
                    warning('[HI_plot] Net expects %d inputs but only %d channels present; using %d.', nInputs, C, C);
                end
                for k = 1:min(nInputs,C)
                    if GAN_mode
                        inputCells{k} = coerceToDlInput(raw(:,k), false, 'SCB'); % S x 1 -> SCB
                    else
                        inputCells{k} = coerceToDlInput(raw(:,k), false, 'CB'); % 1D channel -> CB
                    end
                    sz = size(extractdata(inputCells{k}));
                    fprintf('[HI_plot] Split Input %d -> dlarray size [%s], class=%s\n', k, strjoin(string(sz),'x'), class(extractdata(inputCells{k})));
                end
            elseif ndims(raw) == 3
                C = size(raw,3);
                if nInputs > C
                    warning('[HI_plot] Net expects %d inputs but only %d channels present; using %d.', nInputs, C, C);
                end
                for k = 1:min(nInputs,C)
                    if GAN_mode
                        inputCells{k} = coerceToDlInput(raw(:,:,k), true, 'SSCB'); % S x S2 -> SSCB
                    else
                        % FC mode should not have spatial dims; collapse to channels
                        chVec = reshape(raw(:,:,k), [], 1);
                        inputCells{k} = coerceToDlInput(chVec, false, 'CB');
                    end
                    sz = size(extractdata(inputCells{k}));
                    fprintf('[HI_plot] Split Input %d -> dlarray size [%s], class=%s\n', k, strjoin(string(sz),'x'), class(extractdata(inputCells{k})));
                end
            else
                error('[HI_plot] Unsupported raw input rank=%d for channel-splitting.', ndims(raw));
            end
        end
    else
        error('[HI_plot] Unexpected table width=%d (expected %d or 2).', w, nInputs+1);
    end
   
    %% 5. Forward Prediction
    try
        nOut = numel(net.OutputNames);
        YPred = cell(1,nOut);
        [YPred{:}] = predict(net, inputCells{:});
    catch ME
        fprintf(2,'[HI_plot][ERROR] Prediction failed: %s\n', ME.message);
        % Extra diagnostics on failure
        for k = 1:numel(inputCells)
            if isa(inputCells{k}, 'dlarray')
                szk = size(extractdata(inputCells{k})); fmts = inputCells{k}.Formats;
                fprintf(2,'[HI_plot][DEBUG] input %d: dlarray size [%s], formats=%s\n', k, strjoin(string(szk),'x'), string(fmts));
            else
                fprintf(2,'[HI_plot][DEBUG] input %d: class=%s\n', k, class(inputCells{k}));
            end
        end
        rethrow(ME);
    end
   
    y = squeeze(extractdata(YPred{1}));
    if GAN_mode
        % Expect per-path latent over channels
        latent_space(:,i) = y;
    else
        % FC mode: single path latent scalar/vector
        if isvector(y)
            latent_space(1,i) = y(1);
        else
            latent_space(1,i) = y;
        end
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
if GAN_mode
    for idx = 1:numel(paths_to_plot)
        p = paths_to_plot(idx);
        plot(RUL, latent_space(p,:), 'DisplayName', sprintf('Path %d', p));
    end
else
    plot(RUL, latent_space(1,:), 'DisplayName', sprintf('Path %d', path_id));
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

%% Local utility: ensure inputs are dlarray with batch dim and correct formats
function dlX = coerceToDlInput(x, preferSSCB, fmt)
    if nargin < 2, preferSSCB = false; end
    if nargin < 3 || isempty(fmt), fmt = 'SCB'; end
    % Ensure underlying is numeric and full
    if iscell(x)
        if numel(x) == 1
            x = x{1};
        else
            error('[HI_plot] Invalid cell input with %d elements; expected scalar cell.', numel(x));
        end
    end

    if istable(x)
        error('[HI_plot] Table encountered where numeric array expected.');
    end

    if ~isnumeric(x) && ~islogical(x)
        error('[HI_plot] Input must be numeric/logical, got %s.', class(x));
    end

    if issparse(x)
        x = full(x);
    end

    % Cast to single to match typical training
    if ~isa(x,'single') && ~isa(x,'double') && ~islogical(x)
        x = single(x);
    end

    % Add batch dimension and format based on requested fmt
    switch upper(fmt)
        case 'CB' % channels x batch
            if isvector(x)
                x = reshape(x, [], 1);
            elseif ndims(x) == 2 && size(x,2) ~= 1
                % Assume columns are channels; keep B=1
                x = reshape(x, size(x,1), 1);
            elseif ndims(x) > 2
                x = reshape(x, [], 1);
            end
            dlX = dlarray(x, 'CB');
        case 'SCB'
            nd = ndims(x);
            if nd == 2
                x = reshape(x, size(x,1), size(x,2), 1);
            elseif nd == 1
                x = reshape(x, [], 1, 1);
            elseif nd > 3
                error('[HI_plot] SCB expects up to 2D + batch. Got rank=%d', nd);
            end
            dlX = dlarray(x, 'SCB');
        case 'SSCB'
            if preferSSCB
                if ndims(x) == 2
                    x = reshape(x, size(x,1), size(x,2), 1, 1);
                elseif ndims(x) == 3
                    x = reshape(x, size(x,1), size(x,2), size(x,3), 1);
                end
            else
                if ndims(x) == 2
                    x = reshape(x, size(x,1), size(x,2), 1, 1);
                end
            end
            dlX = dlarray(x, 'SSCB');
        otherwise
            error('[HI_plot] Unsupported format "%s"', fmt);
    end
end