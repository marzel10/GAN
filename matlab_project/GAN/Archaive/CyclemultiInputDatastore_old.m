classdef CyclemultiInputDatastore < matlab.io.Datastore & ...
                                     matlab.io.datastore.MiniBatchable
    properties
        Cycles_array
        CurrentIndex
        MiniBatchSize
        NumInputs
        info
    end

    properties (SetAccess = protected)
        NumObservations
    end

    methods
        function ds = CyclemultiInputDatastore(cycleStructArray, numInputs, miniBatchSize)
            ds.Cycles_array = cycleStructArray;
            ds.CurrentIndex = 1;
            ds.MiniBatchSize = miniBatchSize;  % Set mini batch size
            ds.NumObservations = numel(ds.Cycles_array);
            ds.NumInputs = numInputs;
            ds.info = "data with " + num2str(ds.NumObservations) + " observations, divided into " + num2str(ds.NumObservations/ds.MiniBatchSize) + " batches";
        end

        function tf = hasdata(ds)
            tf = ds.CurrentIndex <= ds.NumObservations;
        end

        function [data, info] = read(ds)
            info = ds.info;
            % Initialize cell arrays for inputs and targets
            allData = cell(1, 2 * ds.NumInputs+1);
            % inputCell = cell(1, ds.NumInputs);
            % targetCell = cell(1, ds.NumInputs);
            batchCount = 0;
            
            while batchCount < ds.MiniBatchSize && ds.hasdata()
                allPairData = ds.extractAllCycleData(ds.Cycles_array(ds.CurrentIndex));

                for i = 1:ds.NumInputs
                    if i <= length(allPairData)
                        cycleData = allPairData{i};
                        allData{i}(batchCount + 1, :, :, :) = cycleData;  % Add to batch
                        allData{i+ds.NumInputs+1}(batchCount + 1, :, :, :) = cycleData;
                        if i == 1
                            allData{ds.NumInputs+1}(batchCount + 1, :, :, :) = zeros; % Just to have something in the middle
                        end
                        % inputCell{i}(batchCount + 1, :, :, :) = cycleData;  % Add to batch
                        % targetCell{i}(batchCount + 1, :, :, :) = cycleData;

                    else
                        warning('Not enough pairs of data to split into inputs');
                    end
                end

                
                ds.CurrentIndex = ds.CurrentIndex + 1;
                batchCount = batchCount + 1;
            end

            % Create table with inputs and targets
            % allData = [inputCell, targetCell];
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
                    pairStruct = cycleStruct.Frequency(f).Pair_idx(p);
                    if isfield(pairStruct, 'Benchmark_Amplitude')
                        bench = pairStruct.Benchmark_Amplitude;
                    elseif isfield(pairStruct, 'Banchmark_Amplitude')
                        bench = pairStruct.Banchmark_Amplitude;
                    else
                        error('No benchmark field found (Benchmark_Amplitude/Banchmark_Amplitude).');
                    end
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
