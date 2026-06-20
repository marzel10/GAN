%% MATLAB Path Setup - Run this first!
% Get the directory where THIS script is located
scriptDir = fileparts(mfilename('fullpath'));
fprintf('Script location: %s\n', scriptDir);

% Navigate to project root (2 levels up from src/training)
projectRoot = fullfile(scriptDir, '..', '..');
cd(projectRoot);
fprintf('Changed MATLAB working directory to: %s\n', pwd);

% Remove any conflicting paths from old PZT folder (to avoid conflicts)
if contains(path, 'C:\Users\Maria\Documents\Honours Programme\PZT')
    rmpath('C:\Users\Maria\Documents\Honours Programme\PZT');
    fprintf('🗑️  Removed old PZT folder from path to avoid conflicts\n');
end

% Add all necessary paths relative to project root (LOCAL FIRST for precedence!)
addpath(fullfile(projectRoot, 'src', 'models'), '-begin');  % HIGH PRIORITY - Local classes
addpath(fullfile(projectRoot, 'src', 'data'), '-begin');    % HIGH PRIORITY - Local datastore
addpath(fullfile(projectRoot, 'src', 'utils'), '-begin');   % HIGH PRIORITY - Local utilities  
addpath('C:\Users\Maria\Documents\Honours Programme\PZT', '-end');  % LOW PRIORITY - Original data files

fprintf('✅ All paths configured successfully\n\n');

%% Verify which class files MATLAB will use

fprintf('🔍 Verifying class locations:\n');
ganLocation = which('GAN');
deconLocation = which('deconcatenation');
datastoreLocation = which('CyclemultiInputDatastore');

if contains(ganLocation, 'GAN\src\models')
    fprintf('✅ GAN class: %s (LOCAL - GOOD!)\n', ganLocation);
else
    fprintf('⚠️  GAN class: %s (EXTERNAL - might be wrong!)\n', ganLocation);
end

if contains(deconLocation, 'GAN\src\models')
    fprintf('✅ deconcatenation class: %s (LOCAL - GOOD!)\n', deconLocation);
else
    fprintf('⚠️  deconcatenation class: %s (EXTERNAL - might be wrong!)\n', deconLocation);
end

if contains(datastoreLocation, 'GAN\src\data')
    fprintf('✅ CyclemultiInputDatastore3 class: %s (LOCAL - GOOD!)\n', datastoreLocation);
else
    fprintf('⚠️  CyclemultiInputDatastore3 class: %s (EXTERNAL - might be wrong!)\n', datastoreLocation);
end
fprintf('✅ All paths configured successfully\n\n');


%% Training Configuration



function L = multiOutputMSE_monotonicity(varargin)    
    numOut = numel(varargin)/2; 
    accum = 0; 
   
    for k = 1:numOut 

        Yk = varargin{1,k}; 
        Tk = varargin{1,k+numOut}; 

        if k == 1
            diff = Yk( :, :,2:end, :) - Yk( :, :,1:end-1, :);
            violations = diff .* (diff < 0);  % Differentiable alternative to max(0, ...)
            accum = accum + dlarray(mean(violations, 'all'));
        else
            accum = accum + mean((Yk - Tk).^2,'all'); 
        end
    end 
    L = accum / numOut; 
end

function L = monotonicity(varargin)
    
    % Use only differentiable operations
    % Example: penalize negative differences between consecutive elements
    diff = varargin{1,1}( :, :,2:end, :) - varargin{1,1}( :, :,1:end-1, :);
    violations = diff .* (diff < 0);  % Differentiable alternative to max(0, ...)
    L = dlarray(mean(violations, 'all'));
end

function L = multiOutputMSE(varargin)    
    numOut = numel(varargin)/2-1; 
    accum = 0; 
   
    for k = 1:numOut 

        Yk = varargin{1,k+1}; 
        Tk = varargin{1,k+numOut+1}; 

        accum = accum + mean((Yk - Tk).^2,'all'); 
    end 
    L = accum / numOut; 
end

  

% Create single datastore that handles all 28 inputs and targets
num_in = 28;
Cycle1 = load(sprintf("data\\Cycle_%d.mat", 103)).Cycle1;  % Load the Cycle1 datastore
Cycle2 = load(sprintf("data\\Cycle_%d.mat", 104)).Cycle1;  % Load the Cycle2 datastore
Cycle3 = load(sprintf("data\\Cycle_%d.mat", 105)).Cycle1;  % Load the Cycle3 datastore
Cycle4 = load(sprintf("data\\Cycle_%d.mat", 109)).Cycle1;  % Load the Cycle4 datastore

inputDs1 = CyclemultiInputDatastore(Cycle1, num_in, 4);
inputDs2 = CyclemultiInputDatastore(Cycle2, num_in, 4);
inputDs3 = CyclemultiInputDatastore(Cycle3, num_in, 4);
inputDs4 = CyclemultiInputDatastore(Cycle4, num_in, 4);
inputDs = combine(inputDs1, inputDs2, inputDs3,ReadOrder="sequential");  % Combine datastores
options = trainingOptions("sgdm", ...
    MaxEpochs=2, ...
    MiniBatchSize=4, ...  % Process 4 cycles at a time instead of 128
    Verbose=true, ...
    Plots="training-progress", ...
    Metrics={@monotonicity, @multiOutputMSE}, ...
    InputDataFormats=repmat("BSSC", 1, 28), ...  % 28 inputs
    TargetDataFormats=repmat("BSSC", 1, 29), ...  % 29 targets (including G2/latent_out)
    Shuffle="never",...
    ValidationData=inputDs4,...
    ValidationFrequency=5);  


% Test the datastore
fprintf('Testing datastore...\n');
if hasdata(inputDs1)
    testData = read(inputDs1);
    fprintf('Data table has %d columns (should be %d for %d inputs + %d targets)\n', ...
        width(testData), 2*num_in, num_in, num_in);
    
    disp(size(testData));  % Display size of the table
    % Check the size of the first input
    firstInput = testData{1,1};  % Get first column, first row
    fprintf('First input size: %s (should be 4000×2×6)\n', mat2str(size(firstInput)));
    
    reset(inputDs1);  % Reset for training
end



lossFcn = @(varargin) multiOutputMSE_monotonicity(varargin{:});

% Call the network building function
net = net_builder.build_net_with_28_inputs();

% Train the network
[trained_net, training_info] = trainnet(inputDs, net, lossFcn, options);

save("train_info.mat","training_info")

