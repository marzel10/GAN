classdef noise_datastore < matlab.io.Datastore & ...
                             matlab.io.datastore.MiniBatchable & ...
                             matlab.io.datastore.Subsettable
    properties
        std
        numSamples
        MiniBatchSize
        CurrentIndex
        seed
        cell_out          % {numSamples x 1} each cell: [15 x 1] raw sample
        k
        lambda
        lstm_mode
        seq_len
    end

    properties (SetAccess = protected)
        NumObservations   % number of *observations* trainnet will iterate over
    end

    methods
        %------------------------------------------------------------------
        function ds = noise_datastore(std, numSamples, MiniBatchSize, varargin)
            ds.std          = std;
            ds.numSamples   = numSamples;
            ds.MiniBatchSize = MiniBatchSize;
            ds.CurrentIndex = 1;
           

            % Optional LSTM parameters
            p = inputParser;
            addParameter(p, 'lstm_mode', false, @(x) islogical(x) || isnumeric(x));
            addParameter(p, 'seq_len',   0,     @(x) isnumeric(x) && x >= 0);
            addParameter(p, 'seed',      42,    @(x) isnumeric(x) && isscalar(x));
            parse(p, varargin{:});
            ds.lstm_mode = logical(p.Results.lstm_mode);
            ds.seq_len   = p.Results.seq_len;
             ds.seed         = p.Results.seed;
            rng(ds.seed);
            if ds.lstm_mode && ds.seq_len <= 0
                error('For LSTM mode, seq_len must be a positive integer. Got %d.', ds.seq_len);
            end

            % Weibull proxy-label parameters (fatigue health index)
            min_stress        = -6.5;
            max_stress        = -65;
            ultimate_strength = -104;
            amp_stress        = abs(max_stress - min_stress) / 2;
            mean_stress       = (max_stress + min_stress) / 2;
            nor_amp_stress    = amp_stress  / ultimate_strength;
            nor_mean_stress   = mean_stress / ultimate_strength;
            ds.k      = -1.17*nor_mean_stress - 19.9*nor_amp_stress + 5.92;
            ds.lambda =  0.0661*nor_mean_stress - 1.05*nor_amp_stress + 0.754;

            % Pre-generate all raw samples (one [15x1] vector per cycle)
            ds.cell_out = cell(numSamples, 1);
            for i = 1:numSamples
                weibull_health = real(ds.lambda * ...
                    (log(max(i,1) / max(numSamples,1)))^(1/ds.k));
                %weibull_health = i/ds.numSamples; % linear health value for testing
                ds.cell_out{i} = normrnd(weibull_health, std, [15, 1]);
            end
        end

        %------------------------------------------------------------------
        function tf = hasdata(ds)
            tf = ds.CurrentIndex <= ds.NumObservations;
        end

        %------------------------------------------------------------------
        function [data, info] = read(ds)
            if ~hasdata(ds)
                error('No more data to read. Call reset() first.');
            end

            if ds.lstm_mode
                % ---- LSTM path -------------------------------------------
                % Each observation is one non-overlapping sequence of seq_len
                % consecutive raw samples stacked into [15 x seq_len].
                % Target: [1 x seq_len] Weibull health values for each step.

                startObs = ds.CurrentIndex;
                endObs   = min(ds.CurrentIndex + ds.MiniBatchSize - 1, ...
                               ds.NumObservations);
                batchSize = endObs - startObs + 1;

                xCell      = cell(batchSize, 1);
                targetCell = cell(batchSize, 1);

                for b = 1:batchSize
                    obsIdx   = startObs + b - 1;          % which sequence (1-based)
                    sampleStart = (obsIdx - 1) * ds.seq_len + 1;
                    sampleEnd   = min(sampleStart + ds.seq_len - 1, ds.numSamples);

                    % Build [15 x actual_len] input matrix
                    seqData = cell2mat(ds.cell_out(sampleStart:sampleEnd)');
                    xCell{b} = seqData;                   % [15 x seq_len]

                    % Build [1 x actual_len] target (Weibull health per step)
                    tgt = zeros(1, sampleEnd - sampleStart + 1);
                    for t = 1:(sampleEnd - sampleStart + 1)
                        idx = sampleStart + t - 1;
                        % tgt(t) = 1/ds.numSamples * (ds.numSamples - idx); % scalar health value
                        tgt(t) = idx;
                    end
                    targetCell{b} = tgt;                  % [1 x seq_len]
                end

                data = table(xCell, targetCell, 'VariableNames', {'input','target'});
                ds.CurrentIndex = endObs + 1;

            else
                % ---- FC path ---------------------------------------------
                % Each observation is one [15 x 1] sample.
                % Target: scalar Weibull health value, broadcast to [1 x 1].

                startIdx = ds.CurrentIndex;
                endIdx   = min(ds.CurrentIndex + ds.MiniBatchSize - 1, ds.numSamples);

                xCell      = ds.cell_out(startIdx:endIdx);   % {batchSize x 1}
                targetCell = cell(endIdx - startIdx + 1, 1);

                for i = 1:(endIdx - startIdx + 1)
                    idx = startIdx + i - 1;
                    % tgt = repmat(1/ds.numSamples * (ds.numSamples - idx), 15, 1); % [15 x currentBatchSize]
                    tgt = idx/ds.numSamples; % scalar health value
                    targetCell{i} = tgt;                  % scalar
                end

                data = table(xCell, targetCell, 'VariableNames', {'input','target'});
                ds.CurrentIndex = endIdx + 1;
            end

            info = struct();
        end

        %------------------------------------------------------------------
        function reset(ds)
            ds.CurrentIndex = 1;
        end

        %------------------------------------------------------------------
        function num = get.NumObservations(ds)
            if ds.lstm_mode
                % Number of complete (non-overlapping) sequences
                num = floor(ds.numSamples / ds.seq_len);
            else
                num = ds.numSamples;
            end
        end

        %------------------------------------------------------------------
        function dsNew = subset(ds, indices)
            % indices refer to observation indices (sequences or samples)
            if any(indices < 1) || any(indices > ds.NumObservations)
                error('Subset indices must be between 1 and %d.', ds.NumObservations);
            end

            if ds.lstm_mode
                % Map observation indices → raw sample indices
                sampleIndices = [];
                for i = 1:numel(indices)
                    obs = indices(i);
                    s = (obs-1)*ds.seq_len + 1;
                    e = min(obs*ds.seq_len, ds.numSamples);
                    sampleIndices = [sampleIndices, s:e]; %#ok<AGROW>
                end
                dsNew = noise_datastore(ds.std, numel(sampleIndices), ...
                    ds.MiniBatchSize, 'lstm_mode', true, 'seq_len', ds.seq_len);
                dsNew.cell_out = ds.cell_out(sampleIndices);
                dsNew.k        = ds.k;
                dsNew.lambda   = ds.lambda;
            else
                dsNew = noise_datastore(ds.std, numel(indices), ds.MiniBatchSize);
                dsNew.cell_out = ds.cell_out(indices);
                dsNew.k        = ds.k;
                dsNew.lambda   = ds.lambda;
            end
        end

        %------------------------------------------------------------------
        function ds = shuffle(ds)
            % Actually shuffle the data instead of just resetting the index
            newOrder = randperm(ds.numSamples);
            ds.cell_out = ds.cell_out(newOrder);
            % If you store targets in a property, shuffle those too.
            ds.CurrentIndex = 1;
        end
    end

    methods (Access = protected)
        %------------------------------------------------------------------
        function subds = subsetByReadIndices(ds, indices)
            subds = subset(ds, indices);
        end

        %------------------------------------------------------------------
        function n = maxpartitions(ds)
            n = ceil(ds.NumObservations / ds.MiniBatchSize);
        end
    end
end
