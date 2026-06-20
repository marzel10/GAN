data = load("data\Cycle_103.mat").Cycle1;

freqArr = data(10).Frequency;
nFreq = numel(freqArr);
fprintf('Plotting %d frequency sets into subplots...\n', nFreq);

if nFreq == 0
    warning('No frequency entries found.');
else
    nRows = ceil(sqrt(nFreq));
    nCols = ceil(nFreq / nRows);
    figure('Name', 'Cycle 10 - All Frequencies', 'NumberTitle', 'off');

    for i = 1:nFreq
        try
            amp = freqArr(i).Pair_idx(1).Amplitude;
            pairStruct = freqArr(i).Pair_idx(1);
            if isfield(pairStruct, 'Benchmark_Amplitude')
                bmk = pairStruct.Benchmark_Amplitude;
                bmkName = 'Benchmark_Amplitude';
            elseif isfield(pairStruct, 'Banchmark_Amplitude')
                bmk = pairStruct.Banchmark_Amplitude;
                bmkName = 'Banchmark_Amplitude';
            else
                error('No benchmark field found (Benchmark_Amplitude/Banchmark_Amplitude).');
            end

            fprintf('Freq %d: amp size %s class %s | bmk size %s class %s\n', ...
                i, mat2str(size(amp)), class(amp), mat2str(size(bmk)), class(bmk));

            if ~isvector(amp), amp = squeeze(amp); end
            if ~isvector(bmk), bmk = squeeze(bmk); end

            subplot(nRows, nCols, i);
            plot(amp, 'LineWidth', 1.2, 'DisplayName', 'Amplitude'); hold on;
            plot(bmk, 'LineWidth', 1.2, 'DisplayName', bmkName);
            grid on; legend('show', 'Location', 'best');
            xlabel('Time step'); ylabel('Amplitude');
            title(sprintf('Frequency %d', i));
        catch ME
            warning('Failed to plot frequency %d: %s', i, ME.message);
        end
    end

    try
        sgtitle('Cycle 10 - All Frequencies');
    catch
        % sgtitle may not be available in older MATLAB versions
    end
    drawnow;
    % Another figure with spectrums of the signals for differnet frequencies
    figure('Name', 'Cycle 10 - Frequency Spectrums', 'NumberTitle', 'off');
    for i = 1:nFreq
        try
            amp = freqArr(i).Pair_idx(1).Amplitude;
            pairStruct = freqArr(i).Pair_idx(1);
            if isfield(pairStruct, 'Benchmark_Amplitude')
                bmk = pairStruct.Benchmark_Amplitude;
                bmkName = 'Benchmark_Amplitude';
            elseif isfield(pairStruct, 'Banchmark_Amplitude')
                bmk = pairStruct.Banchmark_Amplitude;
                bmkName = 'Banchmark_Amplitude';
            else
                error('No benchmark field found (Benchmark_Amplitude/Banchmark_Amplitude).');
            end

            if ~isvector(amp), amp = squeeze(amp); end
            if ~isvector(bmk), bmk = squeeze(bmk); end

            subplot(nRows, nCols, i);
            p1 = plot(abs(fft(amp)), 'LineWidth', 1.2, 'DisplayName', 'Amplitude Spectrum'); hold on;
            p2 = plot(abs(fft(bmk)), 'LineWidth', 1.2, 'DisplayName', [bmkName, ' Spectrum']);
            grid on; legend('show', 'Location', 'best');
            xlabel('Frequency bin'); ylabel('Magnitude');
            title(sprintf('Frequency %d Spectrum', i));
        catch ME
            warning('Failed to plot spectrum for frequency %d: %s', i, ME.message);
        end
    end
end
