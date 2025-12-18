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

cd(workingDir); % Going to the directory where panel's data is stored
f = [50, 100, 125, 150, 200,250];
%f = [50, 100]

% Initialize structures to hold joined data
Amplitude_joined = struct();
Time_joined = struct();
Benchmark_joined = struct();
numReceiver = 8;

% Generate all unique pairs
SP = nchoosek(1:numReceiver, 2);
numbering = 1:size(SP,1);
SP = cat(2,SP,numbering');
% Convert SP to a table with named columns and save
try
    SP_table = array2table(SP, 'VariableNames', {'Sensor1_idx','Sensor2_idx','PairIdx'});
    fprintf('SP size: %s, class(SP): %s\n', mat2str(size(SP)), class(SP));
    if exist('debugDataFlow','file') == 2
        debugDataFlow('SP_table', SP_table);
    end
    save('C:/Users/Maria/Documents/Honours Programme/Networks/GAN/data/SP.mat','SP','SP_table','-v7.3');
    fprintf('Saved SP and SP_table to SP.mat\n');
catch ME
    fprintf('Error saving SP table: %s\n', ME.message);
    rethrow(ME);
end

A = repmat(1:8, 8, 1);
mask = A ~= (1:8)';      % logical mask where element ≠ row index, to remove self connections
% Creating matrix which in each row stores the receiver numbers connected to that actuator (act 1 sends to rec in row 1, etc)
A_new = flipud(reshape(A(mask), 8, 7)); % reshape and flip up side down to get desired format

% Example sizes (now automatically set based on panel)
numFreq     = length(f);
numpairs    = size(SP,1);

% Predefine the base struct for one pair data
PairTemplate = struct( ...
    'Amplitude', [], ...
    'Time', [], ...
    'Benchmark_Amplitude', [] );

% Wrap inside Pair
DirTemplate = struct('Dir', repmat(PairTemplate, 1, 2));

% Wrap inside Frequency
FreqTemplate = struct('Pair_idx', repmat(DirTemplate, 1, numpairs));

% Wrap inside State
StateTemplate = struct('Frequency', repmat(FreqTemplate, 1, numFreq));

% Preallocate the full structure array
States = repmat(StateTemplate, 1, numStates);
States_lowpass = repmat(StateTemplate, 1, numStates);
States_downsampled = repmat(StateTemplate, 1, numStates);

disp('My current working directory is:');
disp(pwd);
%Check if saving directory exists
saveDir = 'C:/Users/Maria/Documents/Honours Programme/Networks/GAN/data/';
if ~isfolder(saveDir)
    disp("Saving directory does not exist. Creating now...");
    mkdir(saveDir);
    disp(['Created directory: ', saveDir]);
end

one_state_loadings = numFreq * numpairs * 10; % number of loadings per state (frequencies * pairs * repetitions)
total_loadings = numStates * one_state_loadings;
delete(gcp('nocreate'));
p = gcp('nocreate');
if isempty(p)
    c = parcluster;                   % current default cluster
    disp(c.Profile);
    desired = 14;
    c.NumWorkers = min(desired, feature('numcores'));
    
    % Persist the change to the profile and start a pool
    saveProfile(c);
    parpool(c, c.NumWorkers);
else
    fprintf('Pool has %d workers\n', p.NumWorkers);
end
disp(feature('numcores'))
input('Press enter')
parfor state = 1:numStates
    counter = 0;
    disp(['Processing State: ', num2str(state)]);
    for freq_idx = 1:length(f)
            disp(['  Processing Frequency: ', num2str(f(freq_idx)), ' kHz']);
            for rep = 1:10
                    for pair = 1:size(SP,1)
                        fprintf('%d/%d (%.1f) loadings inside State %d \n', counter, one_state_loadings, (counter/one_state_loadings)*100, state);
                        counter = counter + 1;
                        r1 = SP(pair,1);
                        r2 = SP(pair,2);
                        idx_t1 = find(A_new(r1,:) == r2);
                        idx_t2 = find(A_new(r2,:) == r1);
                        
                        freq = f(freq_idx);
                        full_state = sprintf('%d_2019', state);
                        
    
                        % Step 1: Find the matching folder that starts with "State_<state>"
                        stateFolders = dir(fullfile(baseDir, ['State_', full_state, '*']));
                        
                        if isempty(stateFolders)
                            error('No folder found for State_%d', state);
                        end
                        
                        %{
                        if isempty(stateFolders1)
                            error('No Benchmark folder found for State_1');
                        end
                        %}
                        % Pick the first matching folder (it should be unique)
                        stateFolder = stateFolders(1).name;
                       
    
                        % Step 2: Build full file path using that folder
                        % Normal data sent by actuator r1 to receiver r2
                        directory1 = fullfile(...
                            baseDir, ...
                            stateFolder, ...
                            sprintf('%dkHz_5cycles', freq), ...
                            sprintf('Actionneur%d', r1), ...
                            sprintf('measured_data_rep_%d.mat', rep) ...
                        );
                        % Benchmark data sent by actuator r1 to receiver r2
                        directoryb1 = fullfile(...
                            baseDir1, ...
                            stateFolder1, ...
                            sprintf('%dkHz_5cycles', freq), ...
                            sprintf('Actionneur%d', r1), ...
                            sprintf('measured_data_rep_%d.mat', rep) ...
                        );
                        % Normal data sent by actuator r2 to receiver r1
                        directory2 = fullfile(...
                            baseDir, ...
                            stateFolder, ...
                            sprintf('%dkHz_5cycles', freq), ...
                            sprintf('Actionneur%d', r2), ...
                            sprintf('measured_data_rep_%d.mat', rep) ...
                        );
                        % Benchmark data sent by actuator r2 to receiver r1
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
                            disp('✅ Benchmark 1 Data loaded');
                        else
                            error('❌ Benchmark 1 File not found: %s', directoryb1)
                        end
    
                        if isfile(directory1)
                            data1 = load(directory1);
                            disp('✅ Data 1 loaded');
                        else
                            error('❌ File 1 not found: %s', directory1)
                        end
    
                        if isfile(directoryb2)
                            datab2 = load(directoryb2);
                            disp('✅ Benchmark 2 Data loaded');
                        else
                            error('❌ Benchmark 2 File not found: %s', directoryb2)
                        end
    
                        if isfile(directory2)
                            data2 = load(directory2);
                            disp('✅ Data 2 loaded');
                        else
                            error('❌ File 2 not found: %s', directory2)
                        end
                           
                     
                        Amplitude_dir1 = data1.Time_Response(:, idx_t1+2)/10*1000; % in mV
                        Amplitude_dir2 = data2.Time_Response(:, idx_t2+2)/10*1000; % in mV
                        Time_dir1 = data1.Time_Response(:, 1)/10;   
                        Time_dir2 = data2.Time_Response(:, 1)/10;   
                        Benchmark_dir1 = datab1.Time_Response(:, idx_t1+2)/10*1000; % in mV
                        Benchmark_dir2 = datab2.Time_Response(:, idx_t2+2)/10*1000; % in mV

                        % Prepare to downsampling 
                        samp_rate = 2*10^6; % 2 MHz
                        Nyquist_freq = samp_rate/2;
                        Nyquist_freq_downsampled = Nyquist_freq/5; % After downsampling by 5, new Nyquist frequency

                        % Remove frequencies above Nyquist frequency (without downsampling)
                        Amplitude_lowpass_dir1 = lowpass(Amplitude_dir1, 0.99*Nyquist_freq, samp_rate);
                        Amplitude_lowpass_dir2 = lowpass(Amplitude_dir2, 0.99*Nyquist_freq, samp_rate);
                        Benchmark_lowpass_dir1 = lowpass(Benchmark_dir1, 0.99*Nyquist_freq, samp_rate);
                        Benchmark_lowpass_dir2 = lowpass(Benchmark_dir2, 0.99*Nyquist_freq, samp_rate);

                        % Remove frequencies above new Nyquist frequency (with downsampling)
                        Amplitude_lowpass_downsampled_dir1 = lowpass(Amplitude_dir1, 0.99*Nyquist_freq_downsampled, samp_rate);
                        Amplitude_lowpass_downsampled_dir2 = lowpass(Amplitude_dir2, 0.99*Nyquist_freq_downsampled, samp_rate);
                        Benchmark_lowpass_downsampled_dir1 = lowpass(Benchmark_dir1, 0.99*Nyquist_freq_downsampled, samp_rate);
                        Benchmark_lowpass_downsampled_dir2 = lowpass(Benchmark_dir2, 0.99*Nyquist_freq_downsampled, samp_rate);
                        % Downsample by 5
                        Amplitude_downsampled_dir1 = downsample(Amplitude_lowpass_downsampled_dir1, 5);
                        Amplitude_downsampled_dir2 = downsample(Amplitude_lowpass_downsampled_dir2, 5);
                        Benchmark_downsampled_dir1 = downsample(Benchmark_lowpass_downsampled_dir1, 5);
                        Benchmark_downsampled_dir2 = downsample(Benchmark_lowpass_downsampled_dir2, 5);
                        Time_downsampled_dir1 = downsample(Time_dir1, 5);
                        Time_downsampled_dir2 = downsample(Time_dir2, 5);

                        if isempty(States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude) % First time assignment
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Time = Time_dir1;
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude = Amplitude_dir1;
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude = Benchmark_dir1;   
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Time = Time_dir2;
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude = Amplitude_dir2;
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude = Benchmark_dir2;

                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Time = Time_dir1;
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude = Amplitude_lowpass_dir1;
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude = Benchmark_lowpass_dir1;
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Time = Time_dir2;
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude = Amplitude_lowpass_dir2;
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude = Benchmark_lowpass_dir2;

                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Time = Time_downsampled_dir1;
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude = Amplitude_downsampled_dir1;
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude = Benchmark_downsampled_dir1;
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Time = Time_downsampled_dir2;
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude = Amplitude_downsampled_dir2;
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude = Benchmark_downsampled_dir2;
                        else
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Time = States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Time + Time_dir1;
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude =  States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude + Amplitude_dir1;
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude = States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude + Benchmark_dir1;   
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Time = States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Time + Time_dir2;
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude = States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude + Amplitude_dir2;
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude = States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude + Benchmark_dir2;

                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Time = States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Time + Time_dir1;
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude = States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude + Amplitude_lowpass_dir1;
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude = States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude + Benchmark_lowpass_dir1;
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Time = States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Time + Time_dir2;
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude = States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude + Amplitude_lowpass_dir2;
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude = States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude + Benchmark_lowpass_dir2;

                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Time =  States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Time + Time_downsampled_dir1;
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude = States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude + Amplitude_downsampled_dir1;
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude = States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude + Benchmark_downsampled_dir1;
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Time = States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Time + Time_downsampled_dir2;
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude = States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude + Amplitude_downsampled_dir2;
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude = States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude + Benchmark_downsampled_dir2;
                        end

                        if rep == 10
                            % Standarize over amplitude 
                            m = mean(States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude, 'all');
                            std_dev = std(States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude, 0, 'all');
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude = (States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude - m)/ std_dev;
                            % Standarize over benchmark amplitude
                            m_b = mean(States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude, 'all');
                            std_dev_b = std(States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude, 0, 'all');
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude = (States(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude - m_b)/ std_dev_b;
                        
                            m2 = mean(States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude, 'all');
                            std_dev2 = std(States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude, 0, 'all');
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude = (States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude - m2)/ std_dev2;
                            % Standarize over benchmark amplitude
                            m_b2 = mean(States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude, 'all');
                            std_dev_b2 = std(States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude, 0, 'all');
                            States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude = (States(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude - m_b2)/ std_dev_b2;
                            
                            % Lowpass standarization
                            m_low = mean(States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude, 'all');
                            std_dev_low = std(States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude, 0, 'all');
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude = (States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude - m_low)/ std_dev_low;
                            % Standarize over benchmark amplitude
                            m_b_low = mean(States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude, 'all');
                            std_dev_b_low = std(States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude, 0, 'all');
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude = (States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude - m_b_low)/ std_dev_b_low;
                            
                            m_b2_low = mean(States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude, 'all');
                            std_dev_b2_low = std(States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude, 0, 'all');
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude = (States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude - m_b2_low)/ std_dev_b2_low;
                            % Standarize over benchmark amplitude
                            m_b2b_low = mean(States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude, 'all');
                            std_dev_b2b_low = std(States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude, 0, 'all');
                            States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude = (States_lowpass(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude - m_b2b_low)/ std_dev_b2b_low;
                            
                            % Downsampled standarization
                            m_down = mean(States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude, 'all');
                            std_dev_down = std(States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude, 0, 'all');
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude = (States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Amplitude - m_down)/ std_dev_down;
                            % Standarize over benchmark amplitude
                            m_b_down = mean(States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude, 'all');
                            std_dev_b_down = std(States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude, 0, 'all');
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude = (States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(1).Benchmark_Amplitude - m_b_down)/ std_dev_b_down;
                        
                            m_b2_down = mean(States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude, 'all');
                            std_dev_b2_down = std(States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude, 0, 'all');
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude = (States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Amplitude - m_b2_down)/ std_dev_b2_down;
                            % Standarize over benchmark amplitude   
                            m_b2b_down = mean(States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude, 'all');
                            std_dev_b2b_down = std(States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude, 0, 'all');
                            States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude = (States_downsampled(state).Frequency(freq_idx).Pair_idx(pair).Dir(2).Benchmark_Amplitude - m_b2b_down)/ std_dev_b2b_down;
                        end

                    end
            end
    end
end


save(fullfile(saveDir, sprintf('States_2000_%d.mat', panel)),'States');
save(fullfile(saveDir, sprintf('States_lowpass_2000_%d.mat', panel)),'States_lowpass');
save(fullfile(saveDir, sprintf('States_downsampled_2000_%d.mat', panel)),'States_downsampled');

end_time = toc;
fprintf('Data structs created in %.2f seconds.\n', end_time);

