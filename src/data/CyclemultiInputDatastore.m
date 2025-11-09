classdef CyclemultiInputDatastore < matlab.io.Datastore & ...
                                     matlab.io.datastore.MiniBatchable
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
    end

    properties (SetAccess = protected)
        NumObservations
    end

    methods
        function ds = CyclemultiInputDatastore(cycleStructArray, numInputs, miniBatchSize, overlap_size, numCycles)
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
            ds.info = "data with " + num2str(ds.NumObservations) + " observations, divided into " + num2str(ds.NumObservations/ds.MiniBatchSize) + " batches";
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
            allData = cell(1, 2 * ds.NumInputs + 1);
            batchCount = 0;
            startIndex = ds.CurrentIndex; % Track to compute how many elements were consumed

            while batchCount < ds.MiniBatchSize && ds.hasdata()
                allPairData = ds.extractAllCycleData(ds.Cycles_array(ds.CurrentIndex));

                for i = 1:ds.NumInputs
                    if i <= length(allPairData)
                        cycleData = allPairData{i};
                        % Preallocate all outputs once per batch using first available cycle
                        if batchCount == 0 && i == 1 && isempty(allData{1})
                            [T,S,C] = size(cycleData); % expected 4000x2x6
                            B = ds.MiniBatchSize;
                            % Inputs 1..NumInputs
                            for j = 1:ds.NumInputs
                                allData{j} = zeros(B, T, S, C, 'like', cycleData);
                            end
                            % Latent target slot (NumInputs+1)
                            %allData{ds.NumInputs+1} = zeros(B, T, S, C, 'like', cycleData);
                            allData{ds.NumInputs+1} = zeros(B, 1, 'like', cycleData); % scalar marker
                            % Targets NumInputs+2 .. 2*NumInputs+1
                            for j = 1:ds.NumInputs
                                allData{ds.NumInputs+1+j} = zeros(B, T, S, C, 'like', cycleData);
                            end
                        end

                        % Fill this batch row
                        allData{i}(batchCount + 1, :, :, :) = cycleData;  % input i
                        allData{i + ds.NumInputs + 1}(batchCount + 1, :, :, :) = cycleData; % target i

                        if i == 1
                            % Latent target: keep datastore outputs numeric (no dlarray)
                            latentVal = double(real(ds.lambda * (log(max(ds.CurrentIndex,1) / max(ds.NumCycles,1)))^(1/ds.k)));
                            % allData{ds.NumInputs+1}(batchCount + 1, 1, 1, 1) = latentVal; % scalar marker
                            allData{ds.NumInputs+1}(batchCount + 1, 1) = latentVal; % scalar marker
                        end
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
                    allData{j} = allData{j}(1:nRead, :, :, :);
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
            allPairData = ds.extractAllCycleData(ds.Cycles_array(ds.CurrentIndex));
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

        function inputsCell = extractAllCycleData(ds, cycleStruct)
            numPairs = numel(cycleStruct.Frequency(1).Pair_idx);
            numFreq  = numel(cycleStruct.Frequency);
            numPoints = numel(cycleStruct.Frequency(1).Pair_idx(1).Amplitude);

            inputsCell = cell(1, numPairs);

            for p = 1:numPairs
                pairData = zeros(numPoints, 2, numFreq);

                for f = 1:numFreq
                    amp   = cycleStruct.Frequency(f).Pair_idx(p).Amplitude;
                    bench = cycleStruct.Frequency(f).Pair_idx(p).Banchmark_Amplitude;
                    pairData(:, 1, f) = amp;
                    pairData(:, 2, f) = bench;
                end

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
end
