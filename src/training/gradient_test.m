% Test gradient flow through your current network

net = load('results/one_path_one_freq_downsampled/path_1_freq_4_net_loss_50.354275.mat','trained_net');
function debugDataFlow(varName, varValue)
% debugDataFlow - unified inspector for structs/objects with robust logging
try
    fprintf("[debug] %s: class=%s, size=%s\n", varName, class(varValue), mat2str(size(varValue)));

    if isstruct(varValue)
        fns = fieldnames(varValue);
        fprintf("[debug] %s fields (%d): %s\n", varName, numel(fns), strjoin(fns.', ", "));
    else
        props = {};
        try
            props = properties(varValue);
        catch, end
        mc = metaclass(varValue);
        metaProps = string({mc.PropertyList.Name});
        allProps = unique([string(props(:)); metaProps(:)]);
        fprintf("[debug] %s properties (%d): %s\n", varName, numel(allProps), strjoin(allProps.', ", "));

        try
            summary(varValue);
        catch ME
            fprintf("[debug] summary unavailable: %s\n", ME.message);
        end
    end
catch ME
    fprintf("[error] debugDataFlow(%s) failed: %s\n", varName, ME.message);
end
end
% ...existing code...
debugDataFlow("net_before", net);
net = net.trained_net;
debugDataFlow("net_after", net);
% ...existing code...


% Build input cells in BSSC format: [B, S(time), S(sensors), C]
nIn = 1;  % number of inputs
nOut = nIn; % number of outputs
X = cell(1, nIn);
for i = 1:nIn
    X{i} = dlarray(randn(1, 800, 2, 1, 'single'), 'BSSC');
end

% Compute loss and gradients with tracing
[loss, gradients, Y] = dlfeval(@modelGradientsAE, net, X);

fprintf('[grad] Loss=%.6f\n', gather(extractdata(loss)));

% Check gradient magnitudes
for i = 1:height(net.Learnables)
    g = gradients.Value{i};
    if isempty(g), gn = 0; else, gn = norm(extractdata(g), 'fro'); end
    if gn < 1e-8
        fprintf('DEAD GRADIENT at %s - %s: %.2e\n', net.Learnables.Layer{i}, net.Learnables.Parameter{i}, gn);
    else
        fprintf('%s - %s: %.2e\n', net.Learnables.Layer{i}, net.Learnables.Parameter{i}, gn);
    end
end

% --- Local functions (must be at end of script) ---
function [loss, gradients, Y] = modelGradientsAE(dlnet, Xin)
% Forward, loss, and gradients for multi-input AE (BSSC)
    nOut = numel(dlnet.OutputNames);
    Y = cell(1, nOut);
    [Y{:}] = forward(dlnet, Xin{:});  % multi-output

    % Sum MSE over outputs; match X and Y by index
    loss = dlarray(0,'CB');  % traced scalar created inside dlfeval
    k = min(nOut, numel(Xin));
    for j = 1:k
        loss = loss + mean((Y{j} - Xin{j}).^2, 'all');
    end
    loss = loss / k;

    % Gradients wrt learnable parameters
    gradients = dlgradient(loss, dlnet.Learnables);
end