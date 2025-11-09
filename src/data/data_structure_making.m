tic;
panel = 109;

% Panel-specific configuration (set once at the beginning)
if panel == 103
    baseDir = 'L103_2019_12_06_17_35_05';
    baseDir1 = 'L103_2019_12_06_14_02_38';
    stateFolder1 = 'State_1_2019_12_06_14_02_38';
    numStates = 32;
    workingDir = 'C:/Users/Maria/Documents/Honours Programme/PZT_data/PZT_L103';
elseif panel == 104
    baseDir = 'L104_2019_12_11_15_29_20';
    baseDir1 = 'L104-PI_2019_12_11_08_31_20';
    stateFolder1 = 'State_1_2019_12_11_08_31_20';
    numStates = 58;
    workingDir = 'C:/Users/Maria/Documents/Honours Programme/PZT_data/PZT_L104';
elseif panel == 105
    baseDir = 'L105_2019_12_16_14_41_56';
    baseDir1 = 'L105-PI_2019_12_16_11_32_10';
    stateFolder1 = 'State_1_2019_12_16_11_32_10';
    numStates = 30;
    workingDir = 'C:/Users/Maria/Documents/Honours Programme/PZT_data/PZT_L105';
elseif panel == 109
    baseDir = 'L109_2019_12_18_17_49_44';
    baseDir1 = 'L109-PI_2019_12_18_14_03_35';
    stateFolder1 = 'State_1_2019_12_18_14_03_35';
    numStates = 28;
    workingDir = 'C:/Users/Maria/Documents/Honours Programme/PZT_data/PZT_L109';
elseif panel == 123
    baseDir = 'L109_2019_12_18_17_49_44';
    baseDir1 = 'L109-PI_2019_12_18_14_03_35';
    stateFolder1 = 'State_1_2019_12_18_14_03_35';
    numStates = 28;
    workingDir = 'C:/Users/Maria/Documents/Honours Programme/PZT_data/PZT_L109';
else
    error('Panel %d not configured. Please add configuration for this panel.', panel);
end

cd(workingDir);
f = [50, 100, 125, 150, 200, 250];
%f = [50, 100]

Amplitude_joinced = struct();
Time_joined = struct();
Banchmark_joined = struct();
numReceiver = 8;

% Generate all unique pairs
SP = nchoosek(1:numReceiver, 2);
numbering = 1:size(SP,1);
SP = cat(2,SP,numbering');

A = repmat(1:8, 8, 1);
mask = A ~= (1:8)';      % logical mask where element ≠ row index
A_new = flipud(reshape(A(mask), 8, 7));


% Example sizes (now automatically set based on panel)
numFreq     = length(f);
numpairs    = size(SP,1);
% numReceiver = 8;
% numTrans    = 8;

% Predefine the base struct for S2
PairTemplate = struct( ...
    'Amplitude', [], ...
    'Time', [], ...
    'Banchmark_Amplitude', [] );

% Wrap inside Frequency
FreqTemplate = struct('Pair_idx', repmat(PairTemplate, 1, numpairs));

% Wrap inside Cycle1
Cycle1 = repmat(struct('Frequency', repmat(FreqTemplate, 1, numFreq)), 1, numStates);

disp('My current working directory is:');
disp(pwd);
for state = 1:numStates
    for freq_idx = 1:length(f)
        %for receiver = 1:numReceiver
            for rep = 1:10
                %for trans = 1:numTrans
                    for pair = 1:size(SP,1)
                        r1 = SP(pair,1);
                        r2 = SP(pair,2);
                        idx_t1 = find(A_new(r1,:) == r2);
                        idx_t2 = find(A_new(r2,:) == r1);
                        
                        freq = f(freq_idx);
                        full_state = sprintf('%d_2019', state);
                        
    
                        % Step 1: Find the matching folder that starts with "State_<state>"
                        stateFolders = dir(fullfile(baseDir, ['State_', full_state, '*']));
                        disp(fullfile(baseDir, ['State_', full_state, '*']))
                        if isempty(stateFolders)
                            error('No folder found for State_%d', state);
                        end
                        
                        %{
                        if isempty(stateFolders1)
                            error('No banchmark folder found for State_1');
                        end
                        %}
                        % Pick the first matching folder (or loop through them if needed)
                        stateFolder = stateFolders(1).name;
                       
    
                        % Step 2: Build full file path using that folder
                        directory1 = fullfile(...
                            baseDir, ...
                            stateFolder, ...
                            sprintf('%dkHz_5cycles', freq), ...
                            sprintf('Actionneur%d', r1), ...
                            sprintf('measured_data_rep_%d.mat', rep) ...
                        );
                        directoryb1 = fullfile(...
                            baseDir1, ...
                            stateFolder1, ...
                            sprintf('%dkHz_5cycles', freq), ...
                            sprintf('Actionneur%d', r1), ...
                            sprintf('measured_data_rep_%d.mat', rep) ...
                        );
    
                        directory2 = fullfile(...
                            baseDir, ...
                            stateFolder, ...
                            sprintf('%dkHz_5cycles', freq), ...
                            sprintf('Actionneur%d', r2), ...
                            sprintf('measured_data_rep_%d.mat', rep) ...
                        );
    
                        directoryb2 = fullfile(...
                            baseDir1, ...
                            stateFolder1, ...
                            sprintf('%dkHz_5cycles', freq), ...
                            sprintf('Actionneur%d', r2), ...
                            sprintf('measured_data_rep_%d.mat', rep) ...
                        );
    
                        % Step 3: Load the data
                        if isfile(directoryb1)
                            datab1 = load(directoryb1);
                            disp('✅ Banchmark 1 Data loaded');
                        else
                            error('❌ Banchmark 1 File not found: %s', directoryb1)
                        end
    
                        % Step 3: Load the data
                        if isfile(directory1)
                            data1 = load(directory1);
                            disp('✅ Data 1 loaded');
                        else
                            error('❌ File 1 not found: %s', directory1)
                        end
    
                        % Step 3: Load the data
                        if isfile(directoryb2)
                            datab2 = load(directoryb2);
                            disp('✅ Banchmark 2 Data loaded');
                        else
                            error('❌ Banchmark 2 File not found: %s', directoryb2)
                        end
    
                        % Step 3: Load the data
                        if isfile(directory2)
                            data2 = load(directory2);
                            disp('✅ Data 2 loaded');
                        else
                            error('❌ File 2 not found: %s', directory2)
                        end
                        
                        
                      Amplitude_joined = cat(1,data1.Time_Response(:, idx_t1+2)/10, data2.Time_Response(:, idx_t2+2)/10);
                      Time_joined = cat(1, data1.Time_Response(:, 1)/10, data2.Time_Response(:, 1)/10);
                      Banchmark_joined = cat(1, datab1.Time_Response(:, idx_t2+2)/10, datab2.Time_Response(:, idx_t2+2)/10);

                        if isempty(Cycle1(state).Frequency(freq_idx).Pair_idx(pair).Amplitude)
                            % Cycle1(state).Frequency(freq_idx).S1(receiver).S2(trans).Amplitude = data1.Time_Response(:, trans+1)/10
                            % Cycle1(state).Frequency(freq_idx).S1(receiver).S2(trans).Time = data1.Time_Response(:, 1)/10
                            % Cycle1(state).Frequency(freq_idx).S1(receiver).S2(trans).Banchmark_Amplitude = datab1.Time_Response(:, trans+1)/10
                            Cycle1(state).Frequency(freq_idx).Pair_idx(pair).Time = Time_joined;
                            Cycle1(state).Frequency(freq_idx).Pair_idx(pair).Amplitude = Amplitude_joined;
                            Cycle1(state).Frequency(freq_idx).Pair_idx(pair).Banchmark_Amplitude = Banchmark_joined;
                        else
    
                            Cycle1(state).Frequency(freq_idx).Pair_idx(pair).Amplitude = Cycle1(state).Frequency(freq_idx).Pair_idx(pair).Amplitude + Amplitude_joined;
                            Cycle1(state).Frequency(freq_idx).Pair_idx(pair).Time = Cycle1(state).Frequency(freq_idx).Pair_idx(pair).Time + Time_joined;
                            Cycle1(state).Frequency(freq_idx).Pair_idx(pair).Banchmark_Amplitude = Cycle1(state).Frequency(freq_idx).Pair_idx(pair).Banchmark_Amplitude + Banchmark_joined;
                        end
                    end
               % end
            end
        %end
    end
end


save(sprintf('C:/Users/Maria/Documents/Honours Programme/PZT_data/PZT_L103/GAN/data/Cycle_%d.mat', panel),'Cycle1');

end_time = toc;
fprintf('Data struct created in %.2f seconds.\n', end_time);

