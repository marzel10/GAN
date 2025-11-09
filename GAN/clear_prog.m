
%clearvars;       % remove workspace variables
%close all force; % close all figure windows

t_info = load("train_info.mat");
disp(fieldnames(t_info.training_info));
disp(t_info.training_info.TrainingHistory);
disp(t_info.training_info.ValidationHistory);
disp(t_info.training_info.OutputNetworkIteration);
disp(t_info.training_info.StopReason);

figure;
plot(t_info.training_info.TrainingHistory.Loss);
hold on
plot(t_info.training_info.ValidationHistory.Iteration,t_info.training_info.ValidationHistory.Loss);
legend('Training Loss', 'Validation Loss');