% this code extracts the MSE from the fc one path model and plot it with respect to the paths index
function MSE_extractor(models_folder, varargin)
    p = inputParser;
    p.addParameter('paths_to_evaluate', 1:28, @(x) isnumeric(x) && isvector(x));
    p.parse(varargin{:});
    paths = p.Results.paths_to_evaluate;

    valLoss = zeros(size(paths));
    for p = paths
        % Find the model file for the given path
        modelname =  dir(fullfile(models_folder, sprintf('%dp*.mat', p)));
        if isempty(modelname)
            warning('No model file found for path %d in folder %s', p, models_folder);
            continue;
        end

        % take the model name and read what is after valLoss_
        underscoreIdx = strfind(modelname(1).name, 'valLoss_');
        if isempty(underscoreIdx)
            warning('Model file name %s does not contain ''valLoss_''.', modelname(1).name);
            continue;
        end
        valLossStr = modelname(1).name(underscoreIdx + length('valLoss_'):end-4); % remove .mat
        valLoss(p) = str2double(valLossStr);
    end
    %Create a table with paths and valLoss
    T = table(paths(:), valLoss(paths)', 'VariableNames', {'PathIndex', 'ValidationMSE'});
    disp(T);

    meanMSE = mean(valLoss(valLoss > 0));
    % Plot MSE vs path index
    figure;
    plot(paths, valLoss(paths), '-o', 'LineWidth', 2);
    hold on;
    yline(meanMSE, 'r--', 'LineWidth', 2);
    xlabel('Path Index');
    ylabel('Validation MSE');
    title('Validation MSE vs Path Index');
    legend('Validation MSE', 'Mean MSE', 'Location', 'Best');
    grid on;

end