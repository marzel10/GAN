% Path setup so tiny_architectures_container is visible
scriptDir   = fileparts(mfilename('fullpath'));    % ...\GAN\micro_net_project
projectRoot = fileparts(scriptDir);                % ...\GAN
addpath(fullfile(projectRoot,'src','models'));
addpath(fullfile(projectRoot,'src','models','layers'));  % if you use custom layers later
addpath(fullfile(projectRoot,'src','utils'));
addpath(fullfile(projectRoot,'src','data'));

%% Custom Loss Functions
function L = multiOutputMSE_monotonicity(latent_size, b, c, lstm, predictions, targets)    
    % Compute custom loss; ensures scalar dlarray output so dlgradient works

    if ~lstm && latent_size == 1 && b>0
        mono = monotonicity(predictions,targets);
    elseif lstm && latent_size==1 && b>0
        mono = monotonicity_lstm(predictions,targets);
    else
        mono = 0;
    end

    if ~lstm && latent_size == 1 && c>0
        proxy_loss = proxy_labels_loss(predictions, targets);
    elseif lstm && latent_size==1 && c>0
        proxy_loss = proxy_labels_loss_lstm(predictions, targets);
    else
        proxy_loss = 0;
    end

    L = b*mono + c*proxy_loss;

    % Ensure loss is dlarray scalar
    if ~isa(L, 'dlarray')
        L = dlarray(L);
    end
end

function L = proxy_labels_loss(predicted_latent, weibull_target)
    L = mean(sqrt(sum((predicted_latent - weibull_target(1,:)).^2, 1)), 'all');
end

function L = proxy_labels_loss_lstm(predicted_latent, weibull_target)
    L = mean(sqrt(sum((predicted_latent - weibull_target).^2, 3)), 'all');
end


function L = monotonicity(predicted_latent, ~)
    if size(predicted_latent, 2) < 2
        L = dlarray(0); return;
    end
    % For decreasing trend: penalize positive differences only
    delta = predicted_latent(:, 2:end) - predicted_latent(:, 1:end-1);
    violations = max(delta, dlarray(0));   % only penalize increases
    L = mean(violations .^ 2, 'all');
end

function L = monotonicity_lstm(predicted_latent,targets)
    seqLen = size(predicted_latent, 2); % time dimension
    if seqLen < 2
        L = dlarray(0);
        return;
    end
    diff = (predicted_latent(:, 2:end, :) - predicted_latent(:, 1:end-1, :)) + 0.1;
    L = mean(diff .* diff, 'all');
end

function fc_net = tiny_fully_connected_network(params)
            
    if ~isfield(params, 'input_size'), params.input_size = 15; end
    if ~isfield(params, 'latent_size'), params.latent_size = 1; end
    
    input_size = params.input_size;
    output_size = params.latent_size;
    
    fc_net = dlnetwork;
    layers = [
        inputLayer([input_size, NaN],'CB', "Name","input");
        fullyConnectedLayer( ...
                30, ...
                'Name','tiny_fc_output_1');
        reluLayer('Name', 'relu_output_1');
        fullyConnectedLayer( ...
                20, ...
                'Name','tiny_fc_output_2');
        reluLayer('Name', 'relu_output_2');
        fullyConnectedLayer( ...
                10, ...
                'Name','tiny_fc_output_3');
        reluLayer('Name', 'relu_output_3');
        fullyConnectedLayer( ...
            output_size, ...
            'Name','tiny_fc_output', 'WeightsInitializer', 'glorot');
    ];
    
    % relu_layer = sigmoidLayer('Name', 'relu_output');

    fc_net = addLayers(fc_net, layers);
    fc_net = initialize(fc_net);
end

function lstm_network = tiny_lstm_network(params)
    if ~isfield(params, 'input_size'), params.input_size = 15; end
    if ~isfield(params, 'latent_size'), params.latent_size = 1; end
    
    input_size = params.input_size;
    output_size = params.latent_size;
    
    lstm_network = dlnetwork;
    input_layer = sequenceInputLayer(input_size,'Name','input','Normalization','none');
    lstm_layer = lstmLayer(10,'OutputMode','sequence','Name','tiny_lstm');
    fully_connected_layer = fullyConnectedLayer(output_size,'Name','tiny_fc_output', 'WeightsInitializer', 'glorot');
    
    lstm_network = addLayers(lstm_network, input_layer);
    lstm_network = addLayers(lstm_network, lstm_layer);
    lstm_network = addLayers(lstm_network, fully_connected_layer);
    
    lstm_network = connectLayers(lstm_network, "input", "tiny_lstm");
    lstm_network = connectLayers(lstm_network, "tiny_lstm", "tiny_fc_output");
    
    lstm_network = initialize(lstm_network);
end
batch_size = 64;
lstm = false;
if lstm
    train_data = noise_datastore(0, 10000, batch_size, 'lstm_mode', lstm, 'seq_len', 10, 'seed', 42); % mean=0, std=1, numSamples=10000, batch_size=128
    %train_data_2 = noise_datastore(0, 10000, batch_size, 'lstm_mode', lstm, 'seq_len', 10, 'seed', 43); % mean=0, std=1, numSamples=10000, batch_size=128
    %train_data = combine(train_data_1, train_data_2); % Combine the two datastores for training
    val_data = noise_datastore( 0, 2000, batch_size, 'lstm_mode', lstm, 'seq_len', 10, 'seed', 2);   % mean=0, std=1, numSamples=2000, batch_size=128
else
    train_data = noise_datastore(0, 10000, batch_size, 'seed', 42); % mean=0, std=1, numSamples=10000, batch_size=128
    %train_data_2 = noise_datastore(0, 10000, batch_size, 'seed', 43); % mean=0, std=1, numSamples=10000, batch_size=128
    %train_data = combine(train_data_1, train_data_2); % Combine the two datastores for training
    val_data = noise_datastore( 0, 2000, batch_size, 'seed', 2);   % mean=0, std=1, numSamples=2000, batch_size=128
end

num_in = 1;

if lstm
    format_input =repmat("CTB", 1, num_in);
    format_targets = repmat("CTB", 1, num_in); % the output of the autoencoder and the 1 dimmensional latent space (sHI)
    metrics = {@monotonicity_lstm, @proxy_labels_loss_lstm};
else
    format_input =repmat("CB", 1, num_in);
    format_targets = repmat("CB", 1, num_in); % the output of the autoencoder and the 1 dimmensional latent space (sHI)
    metrics = {@monotonicity, @proxy_labels_loss};
end 

options = trainingOptions("adam", ...
    MaxEpochs=100, ...  
    MiniBatchSize=batch_size, ...  
    InitialLearnRate= 0.001, ... 
    L2Regularization=1e-4, ...      % add weight decay
    Metrics=metrics, ...
    GradientThreshold=1, ...
    Verbose=true, ...
    VerboseFrequency=10, ...
    ValidationPatience=100, ... 
    Plots="training-progress", ...
    InputDataFormats=format_input, ...
    TargetDataFormats=format_targets, ...
    Shuffle="every-epoch",...
    ValidationData= val_data,...
    ValidationFrequency=10);

% Net configuration
fc_params.input_size = 15;
fc_params.latent_size = 1;
% Net transforming the latent space of the big autoencoder into one value (sHI)
if lstm
    tiny_net = tiny_lstm_network(fc_params);
else
    tiny_net = tiny_fully_connected_network(fc_params);
end


% Loss function setup 
[a, b, c] = deal(0.0, 0.0, 1.0); % Weights for MSE, monotonicity, proxy loss
lossFcn = @(Y,T) multiOutputMSE_monotonicity(fc_params.latent_size, b, c, lstm, Y, T); 

% Train the network
[trained_net, training_info] = trainnet(train_data, tiny_net, lossFcn, options);

resultsDir = 'C:\Users\Maria\Documents\Honours Programme\Networks\GAN\micro_net_project';
modelName = sprintf('FC_Net_with_ae_training_%s.mat', datestr(now,'yyyymmdd_HHMMSS'));
modelPath = fullfile(resultsDir, modelName);
save(modelPath, 'trained_net', 'training_info');
fprintf('Training completed. Model saved to %s\n', modelPath);

% Plot the prediction for the test set
if lstm
    test_data = noise_datastore(0, 100000, 1, 'lstm_mode', lstm, 'seq_len', 10); % mean=0, std=1, numSamples=1000, batch_size=1

        %create cell array to store predictions and targets
    num_seq = test_data.numSamples/test_data.seq_len;
    test_inputs = zeros(1, test_data.numSamples);
    test_outputs = zeros(test_data.numSamples, 1);
    test_targets = zeros(test_data.numSamples, 1);
    globa_counter = 1;
    for i=1:(test_data.numSamples/test_data.seq_len)
        test_batch = read(test_data);

        test_inputs(1, globa_counter:globa_counter+test_data.seq_len-1) = test_batch.input{1}(1,:); % Store the entire sequence as input
        test_targets(globa_counter:globa_counter+test_data.seq_len-1) = test_batch.target{1}(1, :, :); % Extract first element only

        % Wrap input as dlarray with explicit CTB format to satisfy sequenceInputLayer
        input_ctb = dlarray(reshape(test_batch.input{1}, size(test_batch.input{1},1), size(test_batch.input{1},2), 1), "CTB");
        test_outputs(globa_counter:globa_counter+test_data.seq_len-1) = predict(trained_net, input_ctb); % Predict using the trained network
        globa_counter = globa_counter + test_data.seq_len;

    end

    % plot predictions vs targets
    figure;
    input = test_inputs;
    predicted = test_outputs;
    actual = test_targets;

    hold on;
    plot(input, 'b-', 'DisplayName', 'Input');
    plot(actual, 'go', 'DisplayName', 'Actual Target');
    plot(predicted, 'rx', 'DisplayName', 'Predicted');
    legend;
    hold off;


else
    test_data = noise_datastore(0, 1000, 1); % mean=0, std=1, numSamples=1000, batch_size=1

    %create cell array to store predictions and targets
    test_inputs = zeros(test_data.numSamples, 1);
    test_outputs = zeros(test_data.numSamples, 1);
    test_targets = zeros(test_data.numSamples, 1);
    for i=1:test_data.numSamples
        test_batch = read(test_data);

        test_inputs(i) = test_batch.input{1}(1);
        test_targets(i) = test_batch.target{1}(1); % Extract first element only
        test_outputs(i) = predict(trained_net, test_batch.input{1}); % Predict using the trained network

    end

    % plot predictions vs targets
    figure;
    input = test_inputs;
    predicted = test_outputs;
    actual = test_targets;

    hold on;
    plot(input, 'b-', 'DisplayName', 'Input');
    plot(actual, 'go', 'DisplayName', 'Actual Target');
    plot(predicted, 'rx', 'DisplayName', 'Predicted');
    legend;
    hold off;

end
