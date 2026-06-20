function L = monotonicity(varargin)
   
    predicted_latent = varargin{2};  % Second output (the size is 1xbatchsize)
    disp("Predicted latent size:");
    disp(size(predicted_latent));
    diff = predicted_latent( :, 2:end) -  predicted_latent( :,1:end-1) + 0; % Enforce a minimum increase of 1 unit per cycle
    violations = relu(diff).^2;  % Differentiable alternative to max(0, ...)
    %violations = diff.*diff;
    L = dlarray(mean(violations, 'all'));
end

batchSize = 4;
lossFcn = @(x) monotonicity([], x);

predicted_latent = dlarray(linspace(5,1,batchSize), "CB");

L = dlfeval(lossFcn, predicted_latent);
disp("Loss (decreasing):");
disp(extractdata(L));


predicted_latent = dlarray(linspace(1,5,batchSize), "CB");

L = dlfeval(lossFcn, predicted_latent);
disp("Loss (increasing):");
disp(extractdata(L));


predicted_latent = dlarray(ones(1,batchSize), "CB");

L = dlfeval(lossFcn, predicted_latent);
disp("Loss (flat):");
disp(extractdata(L));

function [L, grads] = modelGradients(x, lossFcn)
    L = lossFcn(x);
    grads = dlgradient(L, x);
end

predicted_latent = dlarray([ 1.0 1.3 1.4 4.0 5.0], "CB");

[L, grads] = dlfeval(@modelGradients, predicted_latent, lossFcn);


disp("Gradients:");
disp(extractdata(grads));

