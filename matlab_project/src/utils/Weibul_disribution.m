function latentVal = Weibul(CurrentIndex, NumCycles, lambda, k)
% Weibul_disribution computes the Weibull distribution values for input x
% given scale parameter lambda and shape parameter k.
    %dist = lambda * log(x).^(1/k);
    latentVal = double(real(lambda * (log(max(CurrentIndex,1) / max(NumCycles,1))).^(1/k)));
end
min_stress = -6.5; %kN
max_stress = -65; %kN
ultimate_strength = -104; %kN
amp_stress = abs(max_stress - min_stress)/2; %kN
mean_stress = (max_stress + min_stress)/2; %kN
nor_amp_stress = amp_stress/ultimate_strength;
nor_mean_stress = mean_stress/ultimate_strength;

k = -1.17*nor_mean_stress - 19.9*nor_amp_stress + 5.92;
lambda = 0.0661*nor_mean_stress - 1.05*nor_amp_stress + 0.754;

% x = linspace(0,1,100);
% y = Weibul(x, lambda, k);

% disp(y);

% figure; plot(x,y); title(sprintf('Inverse weibul distribution with \\lambda=%.2f, k=%.2f', lambda, k)); xlabel('x'); ylabel('Weibul(x)'); grid on;


% Proxy labels parameters
min_stress = -6.5; %kN
max_stress = -65; %kN
ultimate_strength = -104; %kN
amp_stress = abs(max_stress - min_stress)/2; %kN
mean_stress = (max_stress + min_stress)/2; %kN
nor_amp_stress = amp_stress/ultimate_strength;
nor_mean_stress = mean_stress/ultimate_strength;

k = -1.17*nor_mean_stress - 19.9*nor_amp_stress + 5.92;
lambda = 0.0661*nor_mean_stress - 1.05*nor_amp_stress + 0.754;

y = Weibul(1:30,30, lambda, k);
disp(y(1));
figure; 
plot((1:30)/30, y,"LineWidth",2);
title(sprintf('Inverse weibul distribution with \\lambda=%.2f, k=%.2f', lambda, k)); 
xlabel('Fraction of life gone'); ylabel('Weibul Value'); 
grid on;

