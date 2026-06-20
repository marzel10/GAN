classdef all_path_sHI_sin_freq_fc_datastore < matlab.io.Datastore & ...
                                     matlab.io.datastore.MiniBatchable & ...
                                    matlab.io.datastore.Shuffleable & ...
                                    matlab.io.datastore.Subsettable
    properties
        Cycles_array
        CurrentIndex
        MiniBatchSize
        Overlap_size
        NumInputs
        NumCycles
        info
        k
        lambda
        frequency % frequency index to extract
        path % paths to use when only one input is desired
    end

    properties (SetAccess = protected)
        NumObservations
    end

    methods
        function ds = one_path_sHI_sin_freq_fc_datastore(cycleStructArray, numInputs, miniBatchSize, overlap_size, numCycles, frequency, varargin)
            ds.Cycles_array = cycleStructArray;
            ds.CurrentIndex = 1;
            ds.MiniBatchSize = miniBatchSize;  % Set mini batch size
            ds.Overlap_size = overlap_size;    % Set overlap size
            ds.NumCycles = numCycles;          % Set number of cycles
            % Validate overlap: must be in [0, MiniBatchSize-1]
            if overlap_size < 0
                error('Overlap size must be >= 0. Got %d.', overlap_size);
            end
            if overlap_size >= miniBatchSize
                error('Overlap size (%d) must be less than MiniBatchSize (%d).', overlap_size, miniBatchSize);
            end
            ds.NumObservations = numel(ds.Cycles_array);
            ds.NumInputs = numInputs;
            ds.frequency = frequency;
            

            p = inputParser;
            p.FunctionName = 'CyclemultiInputDatastore_separate_sin_freq_fc';
            addParameter(p, 'paths', [], @(x) isnumeric(x) && ds.NumInputs == 1);
            parse(p, varargin{:});
            ds.path = p.Results.paths;
            if ~isempty(ds.path) && ds.NumInputs ~= 1
                error('When providing ''paths'', NumInputs must be 1. Got NumInputs=%d.', ds.NumInputs);
            end
            
            ds.info = "data with " + num2str(ds.NumObservations) + " observations, divided into " + num2str(ds.NumObservations/ds.MiniBatchSize) + " batches";

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
            % Initialize cell arrays for inputs and targets (will preallocate on first sample)
            allData = cell(1, 2 * ds.NumInputs+1);
            batchCount = 0;
            startIndex = ds.CurrentIndex; % Track to compute how many elements were consumed

            while batchCount < ds.MiniBatchSize && ds.hasdata()
                allPairData = ds.extractAllCycleData(ds.Cycles_array(ds.CurrentIndex), ds.path);

                for i = 1:ds.NumInputs+1

                    if i == ds.NumInputs+1
                        % Proxy label
                        latentVal = double(real(ds.lambda * (log(max(ds.CurrentIndex,1) / max(ds.NumCycles,1)))^(1/ds.k)));
                        
                        size_format = allPairData{1};
                        numfeatures = numel(size_format(:)');
                        B = ds.MiniBatchSize;
                        if batchCount == 0
                            allData{i} = zeros(B, numfeatures, 'like', size_format);
                        end
                        allData{ds.NumInputs+1}(batchCount + 1, 1) = latentVal; % add to the targets
                    elseif i <= length(allPairData)
                        cycleData = allPairData{i};
                        
                        % Preallocate all outputs once per batch using first available cycle
                        if batchCount == 0 && i == 1
                            % Flatten to 1D row vector for FC networks
                            flatData = cycleData(:)';  % [800x1] -> [1x800]
                            numFeatures = numel(flatData);
                            B = ds.MiniBatchSize;
                            % Inputs 1..NumInputs
                            for j = 1:ds.NumInputs
                                allData{j} = zeros(B, numFeatures, 'like', cycleData);
                            end
                            
                            % Targets NumInputs+1 .. 2*NumInputs
                            for j = 1:ds.NumInputs
                                allData{ds.NumInputs+1+j} = zeros(B, numFeatures, 'like', cycleData);
                            end
                        end

                        % Fill this batch row (flatten each sample)

                        allData{i}(batchCount + 1, :) = cycleData(:)';  % input i -> [1x800]
                        allData{i + ds.NumInputs + 1}(batchCount + 1, :) = cycleData(:)'; % target i -> [1x800]
   
                    else
                        warning('Not enough pairs of data to split into inputs');
                    end
                end
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

            % Trim allocated arrays to the actual number of rows read (nRead)
            for j = 1:numel(allData)
                if ~isempty(allData{j}) && size(allData{j},1) > nRead
    
                        allData{j} = allData{j}(1:nRead, :);
                    
                end
            end

            % Create table with inputs and targets
            data = table(allData{:});
        end

        function reset(ds)
            ds.CurrentIndex = 1;
        end

        function dsNew = partition(ds, numPart, numWork)
            idx = numWork:numPart:numel(ds.Cycles_array);
            dsNew = subset(ds, idx);
        end

        function subds = subset(ds, indices)
            subds = copy(ds);
            subds.Cycles_array = ds.Cycles_array(indices);
            subds.NumObservations = numel(indices);
        end

        function s = data_size(ds)
            allPairData = ds.extractAllCycleData(ds.Cycles_array(ds.CurrentIndex), ds.path);
            if ~isempty(allPairData)
                cycleData = allPairData{1};
                s = size(cycleData);  % Return size without adding batch dimension
            else
                warning('The data is empty');
            end
        end

        function N = get.NumObservations(ds)
            N = numel(ds.Cycles_array);
        end

        function inputsCell = extractAllCycleData(ds, cycleStruct, paths)
            numPairs = numel(cycleStruct.Frequency(1).Pair_idx);
            numPoints = numel(cycleStruct.Frequency(1).Pair_idx(1).Amplitude);

            if ~isempty(paths)
                numPairs = 1; % override to 1 if paths are provided
            end
            inputsCell = cell(1, numPairs);

            for p = 1:numPairs
                

                if ~isempty(paths)
                    amp   = cycleStruct.Frequency(ds.frequency).Pair_idx(paths).Amplitude;
                else
                    amp   = cycleStruct.Frequency(ds.frequency).Pair_idx(p).Amplitude;
                end 
                
                pairData = zeros(numPoints, 1);
                
                pairData(:, 1) = single(amp);

                inputsCell{p} = pairData;
            end
        end

        function ds = shuffle(ds)
            numCycles = numel(ds.Cycles_array);
            shuffleIdx = randperm(numCycles);
            ds.Cycles_array = ds.Cycles_array(shuffleIdx);
            ds.reset();
        end
    end

    methods (Access = protected)
        function subds = subsetByReadIndices(ds, indices)
            subds.Cycles_array = ds.Cycles_array(indices);
            subds.NumObservations = numel(indices);
            
            reset(subds);
        end

        function n = maxpartitions(ds)
            n = ds.NumObservations;
        end
    end
end
