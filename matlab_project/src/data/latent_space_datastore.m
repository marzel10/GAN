classdef latent_space_datastore < matlab.io.Datastore & ...
                                     matlab.io.datastore.MiniBatchable & ...
                                    matlab.io.datastore.Subsettable
    properties
        CurrentIndex
        MiniBatchSize
        Overlap_size
        NumCycles
        info
        frequency % frequency index to extract
        GAN_mode % whether we use GAN or fc network
        net_folder % folder where the networks are stored
        path_id % id of the path to use when only one input is desired (fc network)
        k
        lambda
        all_latent % cell array to store all latent space representations
        convolutional_mode % whether to output in convolutional format (SSC) or not (SCB)
    end

    properties (SetAccess = protected)
        NumObservations
    end

    methods
        function ds = latent_space_datastore(panel_id, miniBatchSize, overlap_size, frequency, GAN_mode, net_folder, varargin)
        
            ds.CurrentIndex = 1;
            ds.MiniBatchSize = miniBatchSize;  % Set mini batch size
            ds.Overlap_size = overlap_size;    % Set overlap size

            if panel_id == 103
                ds.NumCycles = 32-1;
            elseif panel_id == 104
                ds.NumCycles = 58-1;
            elseif panel_id == 105
                ds.NumCycles = 30-1;
            elseif panel_id == 109
                ds.NumCycles = 28-1;
            end
              
            % Validate overlap: must be in [0, MiniBatchSize-1]
            if overlap_size < 0
                error('Overlap size must be >= 0. Got %d.', overlap_size);
            end
            if overlap_size >= miniBatchSize
                error('Overlap size (%d) must be less than MiniBatchSize (%d).', overlap_size, miniBatchSize);
            end
            ds.NumObservations = ds.NumCycles; % One observation per cycle
            ds.frequency = frequency;
            ds.GAN_mode = GAN_mode;
            ds.net_folder = net_folder;
            

            p = inputParser;
            p.FunctionName = 'latent_space_datastore';
            addParameter(p, 'path_id', [], @(x) isnumeric(x)  && ~ds.GAN_mode);
            addParameter(p, 'cacheFile', [], @(x) ischar(x) || isstring(x));
            addParameter(p, 'enable_shuffle', false, @(x) islogical(x) || isnumeric(x));
            addParameter(p, 'convolutional_mode', false, @(x) islogical(x) || isnumeric(x) && ds.GAN_mode);
            parse(p, varargin{:});
            ds.path_id = p.Results.path_id;
            cacheFile = p.Results.cacheFile;
            enable_shuffle = p.Results.enable_shuffle;
            convolutional_mode = p.Results.convolutional_mode;
            ds.convolutional_mode = logical(convolutional_mode);
            if ~isempty(ds.path_id) &&  ds.GAN_mode
                error('When providing ''path_id'', GAN mode must be off. GAN_mode = %d',  ds.GAN_mode );
            end
            
            stepSize = ds.MiniBatchSize - ds.Overlap_size; % effective slide per batch
            if ds.NumCycles <= ds.MiniBatchSize
                expectedBatches = 1;
            else
                expectedBatches = ceil((ds.NumCycles - ds.MiniBatchSize) / stepSize) + 1;
            end
            ds.info = "cycles=" + num2str(ds.NumObservations) + ", batchSize=" + num2str(ds.MiniBatchSize) + ", overlap=" + num2str(ds.Overlap_size) + ", step=" + num2str(stepSize) + ", expectedBatches=" + num2str(expectedBatches);

            panel_data = load(sprintf("data\\States_%d.mat", panel_id)).States;

            % Try to load cached latents if provided
            if ~isempty(cacheFile) && isfile(cacheFile)
                S = load(cacheFile);
                if isfield(S, 'all_latent')
                    ds.all_latent = S.all_latent;
                    fprintf('Loaded cached latent space from %s\n', cacheFile);
                    if enable_shuffle
                        % Shuffle the latent representations
                        rng(42); % For reproducibility

                        if isempty(ds.path_id)
                            shuffled_indices = randperm(size(ds.all_latent, 1));
                            ds.all_latent = ds.all_latent(shuffled_indices, :);
                            
                        else
                            shuffled_indices = randperm(size(ds.all_latent, 1));
                            ds.all_latent = ds.all_latent(shuffled_indices);
                            
                        end
                    end
                else
                    warning('Cache file %s does not contain ''all_latent''. Recomputing...', cacheFile);
                end
            end

            % Build latents if not loaded from cache
            if isempty(ds.all_latent)
                if isempty(ds.path_id)
                    % Predict latent space for all paths
                    ds.all_latent = cell(ds.NumObservations, 28);

                    for pth=1:28
                        AE_datastore = CyclemultiInputDatastore_separate_sin_freq_fc(panel_data, 1, 1, 0, ds.NumCycles, false, frequency, 'paths', pth);
                        net_file = dir(fullfile(net_folder, sprintf('%dp*.mat', pth)));
                        net = load(fullfile(net_file.folder, net_file.name)).net;
                        for c=1:AE_datastore.NumObservations
                            [batch_data, ~] = read(AE_datastore);
                            ds.all_latent{c, pth} = predict(net, batch_data{1,1},Outputs='fc_latent_1');
                        end
                    end
                else
                    % Predict latent space for only one path
                    ds.all_latent = cell(ds.NumObservations, 1);
                    AE_datastore = CyclemultiInputDatastore_separate_sin_freq_fc(panel_data, 1, 1, 0, ds.NumCycles, false, frequency, 'paths', ds.path_id);
                    net_file = dir(fullfile(net_folder, sprintf('%dp*.mat', ds.path_id)));
                    disp(fullfile(net_file.folder, net_file.name))
                    net = load(fullfile(net_file.folder, net_file.name)).net;
                    for c=1:AE_datastore.NumObservations
                        [batch_data, ~] = read(AE_datastore);
                        ds.all_latent{c} = predict(net, batch_data{1,1},Outputs='fc_latent_1');
                    end
                end

                if enable_shuffle
                    % Shuffle the latent representations
                    rng(42); % For reproducibility
                    if isempty(ds.path_id)
                        shuffled_indices = randperm(size(ds.all_latent, 1));
                        ds.all_latent = ds.all_latent(shuffled_indices, :);
                    else
                        shuffled_indices = randperm(size(ds.all_latent, 1));
                        ds.all_latent = ds.all_latent(shuffled_indices);
                    end
                end

                % Save to cache if requested
                if ~isempty(cacheFile)
                    try
                        all_latent = ds.all_latent; %#ok<NASGU>
                        save(cacheFile, 'all_latent', '-v7.3');
                        fprintf('Saved latent space cache to %s\n', cacheFile);
                    catch ME
                        warning('Failed to save cache to %s: %s', cacheFile, ME.message);
                    end
                end

            end

            % Proxy labels parameters
            min_stress = -6.5; %kN
            max_stress = -65; %kN
            ultimate_strength = -104; %kN
            amp_stress = abs(max_stress - min_stress)/2; %kN
            mean_stress = (max_stress + min_stress)/2; %kN
            nor_amp_stress = amp_stress/ultimate_strength;
            nor_mean_stress = mean_stress/ultimate_strength;

            ds.k = -1.17*nor_mean_stress - 19.9*nor_amp_stress + 5.92;
            ds.lambda = 0.0661*nor_mean_stress - 1.05*nor_amp_stress + 0.754;


        end

        function tf = hasdata(ds)
            tf = ds.CurrentIndex <= ds.NumObservations;
        end

        function [data, info] = read(ds)
            info = ds.info;
            % Prepare per-sample cells for inputs (S x C) and targets (1 x 1)
            inputsBatch = cell(ds.MiniBatchSize, 1);
            targetsBatch = cell(ds.MiniBatchSize, 1);
            batchCount = 0;
            startIndex = ds.CurrentIndex; % Track to compute how many elements were consumed

            while batchCount < ds.MiniBatchSize && ds.hasdata()
                if ds.GAN_mode
                    if ds.convolutional_mode
                    % Reshape latents to the required format: SSC (latentLen x 1 x nr_paths)
                        latentsRowCells = cellfun(@(v) reshape(v, [], 1), ds.all_latent(ds.CurrentIndex, :), 'UniformOutput', false);
                        sampleSC = cat(3, latentsRowCells{:}); % latentLen x 1 x nr_paths (S x 1 x C)
                    else
                        % Build per-sample matrix as S x C = latentLen x nr_paths to match SCB
                        latentsRowCells = cellfun(@(v) reshape(v, 1, []), ds.all_latent(ds.CurrentIndex, :), 'UniformOutput', false);
                        sampleSC = cat(1, latentsRowCells{:})'; % latentLen x nr_paths (S x C)
                    end
                else
                    % Single path latent: make S x C = latentLen x 1
                    v = ds.all_latent{ds.CurrentIndex};
                    sampleSC = reshape(v, [], 1); % latentLen x 1
                end

                latent_target = double(real(ds.lambda * (log(max(ds.CurrentIndex,1) / max(ds.NumCycles,1)))^(1/ds.k)));

                % Store per-sample (trainnet will assemble along batch dim B)
                inputsBatch{batchCount + 1} = sampleSC;           % S x C
                targetsBatch{batchCount + 1} = reshape(latent_target, 1, 1); % 1 x 1

                ds.CurrentIndex = ds.CurrentIndex + 1;
                batchCount = batchCount + 1;
            end

            % How many observations were actually consumed in this call?
            nRead = ds.CurrentIndex - startIndex;

            % If we filled a full batch, apply overlap; otherwise, end stream cleanly
            if nRead == ds.MiniBatchSize
                ds.CurrentIndex = ds.CurrentIndex - ds.Overlap_size; % Move back by overlap for next window
            else
                % Partial last batch: do NOT overlap, otherwise we'll repeat forever
                ds.CurrentIndex = ds.NumObservations + 1; % force hasdata() to false next time
            end

            % Trim cells to actual nRead
            inputsBatch = inputsBatch(1:nRead);
            targetsBatch = targetsBatch(1:nRead);

            % Create table with nRead rows; trainnet will stack along B (third dim)
            data = table(inputsBatch, targetsBatch);
        end

        function reset(ds)
            ds.CurrentIndex = 1;
        end

        function dsNew = partition(ds, numPart, numWork)
            idx = numWork:numPart:ds.NumCycles;
            dsNew = subset(ds, idx);
        end

        function subds = subset(ds, indices)
            subds = copy(ds);
            if iscell(ds.all_latent) && size(ds.all_latent,2) > 1
                subds.all_latent = ds.all_latent(indices, :);
            else
                subds.all_latent = ds.all_latent(indices);
            end
            subds.NumObservations = numel(indices);
        end

     
        function N = get.NumObservations(ds)
            N = ds.NumCycles;
        end


    
    end

    methods (Access = protected)
        function subds = subsetByReadIndices(ds, indices)
            if iscell(ds.all_latent) && size(ds.all_latent,2) > 1
                subds.all_latent = ds.all_latent(indices, :);
            else
                subds.all_latent = ds.all_latent(indices);
            end
            subds.NumObservations = numel(indices);
            reset(subds);
        end

        function n = maxpartitions(ds)
            n = ds.NumObservations;
        end
    end
end
