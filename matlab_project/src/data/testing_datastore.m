% Create single datastore that handles all 28 inputs and targets
num_in = 1;
b_size = 4;
overlap_size = 3;
% 148 instances of data for the training and 28 for validation
envelope = false; % Whether to use envelope of signals or raw signals
freq = 4; % frequency index to extract
path_idx = 1; % path index to extract
benchmark = false; % whether to use benchmark datastore (no random cropping)
fc = true; % whether to use fully connected network


States103 = load(sprintf("data\\States_%d.mat", 103)).States;  % Load the Cycle1 datastore
States104 = load(sprintf("data\\States_%d.mat", 104)).States;  % Load the Cycle2 datastore
States105 = load(sprintf("data\\States_%d.mat", 105)).States;  % Load the Cycle3 datastore
States109 = load(sprintf("data\\States_%d.mat", 109)).States;  % Load the Cycle4 datastore

inputDs1 = one_path_sHI_sin_freq_fc_datastore(States103, num_in, b_size, overlap_size, 32, freq, 'paths', path_idx); % Provide paths only for first datastore
inputDs2 = one_path_sHI_sin_freq_fc_datastore(States104, num_in, b_size, overlap_size, 58, freq, 'paths', path_idx);
inputDs3 = one_path_sHI_sin_freq_fc_datastore(States105, num_in, b_size, overlap_size, 30, freq, 'paths', path_idx);
inputDs4 = one_path_sHI_sin_freq_fc_datastore(States109, num_in, b_size, overlap_size, 28, freq, 'paths', path_idx);

inputDs_train = combine(inputDs1,  inputDs2, inputDs3, ReadOrder="sequential");
inputDs_val = inputDs4;
fprintf('Data preparation complete. Training observations: %d, Validation observations: %d\n', ...
    numpartitions(inputDs_train), numpartitions(inputDs_val));

% Test the datastore
fprintf('Testing datastore...\n');
if hasdata(inputDs4)
    testData = read(inputDs4);
    fprintf('Data table has %d columns (should be %d for %d inputs + %d targets)\n', ...
        width(testData), 2*num_in+1, num_in, num_in+1);
    
    disp(size(testData));  % Display size of the table
    % Check the size of the first input
    firstInput = testData{1,1};  % Get first column, first row
    fprintf('First input size: %s (should be 1x4000)\n', mat2str(size(firstInput)));
    
    plot(firstInput(1,:));

    disp(firstInput(1,1000:1100));  % Display a slice of the first input for verification
    latent_target = testData{1, num_in+1};
    latent_target2 = testData{2, num_in+1};
    disp(size(latent_target));
    hold on;
    plot(latent_target(1,:),'r');
    hold off;
    fprintf('Latent target value: %.4f\n', latent_target(1,1));
    fprintf('Latent2 target value: %.4f\n', latent_target2(1,1));
    
    reset(inputDs4);  % Reset for training
end
fprintf('Datastore test complete.\n');