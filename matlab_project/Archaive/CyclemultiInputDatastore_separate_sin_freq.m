classdef CyclemultiInputDatastore_separate_sin_freq < matlab.io.Datastore & ...
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
        envelope = false; % Whether to use envelope of signals or raw signals
        frequency % frequency index to extract
        path % paths to use when only one input is desired
        benchmark % whether to include benchmark data, true - include, false - exclude
    end

    properties (SetAccess = protected)
        NumObservations
    end

    methods
        function ds = CyclemultiInputDatastore_separate_sin_freq(cycleStructArray, numInputs, miniBatchSize, overlap_size, numCycles, envelope, frequency, benchmark,varargin)
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
            ds.envelope = envelope;
            ds.frequency = frequency;
            ds.benchmark = benchmark;

            p = inputParser;
            p.FunctionName = 'CyclemultiInputDatastore_separate_sin_freq';
            addParameter(p, 'paths', [], @(x) isnumeric(x) && ds.NumInputs == 1);
            parse(p, varargin{:});
            ds.path = p.Results.paths;
            if ~isempty(ds.path) && ds.NumInputs ~= 1
                error('When providing ''paths'', NumInputs must be 1. Got NumInputs=%d.', ds.NumInputs);
            end
            
            ds.info = "data with " + num2str(ds.NumObservations) + " observations, divided into " + num2str(ds.NumObservations/ds.MiniBatchSize) + " batches";
        end

        function tf = hasdata(ds)
            tf = ds.CurrentIndex <= ds.NumObservations;
        end

        function [data, info] = read(ds)
            info = ds.info;
            % Initialize cell arrays for inputs and targets (will preallocate on first sample)
            allData = cell(1, 2 * ds.NumInputs);
            batchCount = 0;
            startIndex = ds.CurrentIndex; % Track to compute how many elements were consumed

            while batchCount < ds.MiniBatchSize && ds.hasdata()
                allPairData = ds.extractAllCycleData(ds.Cycles_array(ds.CurrentIndex), ds.path);

                for i = 1:ds.NumInputs
                    if i <= length(allPairData)
                        cycleData = allPairData{i};
                        
                       
                            % Preallocate all outputs once per batch using first available cycle
                            if batchCount == 0 && i == 1 && isempty(allData{1})
                                [S1,S2,C] = size(cycleData); % expected 4000x2x6
                                B = ds.MiniBatchSize;
                                % Inputs 1..NumInputs
                                for j = 1:ds.NumInputs
                                    allData{j} = zeros(B, S1, S2, C, 'like', cycleData);
                                end
                                
                                % Targets NumInputs+1 .. 2*NumInputs
                                for j = 1:ds.NumInputs
                                    allData{ds.NumInputs+j} = zeros(B, S1, S2, C, 'like', cycleData);
                                end
                            end

                            % Fill this batch row
                            allData{i}(batchCount + 1, :, :, :) = cycleData;  % input i
                            allData{i + ds.NumInputs }(batchCount + 1, :, :, :) = cycleData; % target i
                        
                       
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
                    if ~ds.benchmark
                        allData{j} = allData{j}(1:nRead, :, :);
                    else
                        allData{j} = allData{j}(1:nRead, :, :, :);
                    end
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
                    bench = cycleStruct.Frequency(ds.frequency).Pair_idx(paths).Benchmark_Amplitude;
                else
                    amp   = cycleStruct.Frequency(ds.frequency).Pair_idx(p).Amplitude;
                    bench = cycleStruct.Frequency(ds.frequency).Pair_idx(p).Benchmark_Amplitude;
                end

                if ds.envelope
                    % bench = abs(hilbert(bench));
                    [amp, ~] = envelope(amp);
                    [bench, ~] = envelope(bench);
                
                end


                
                if ds.benchmark
                    pairData = zeros(numPoints, 2);
                    pairData(:, 2) = single(bench);
                else
                    pairData = zeros(numPoints, 1);
                end
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
