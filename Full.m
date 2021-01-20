clear ; close all; clc

%++++++++++++++++++++++++++++++++++++++++++++++++++++++PLOTTING DATA++++++++++++++++++++++++++++++++++++++++++++++++++++++

fprintf('Plotting Data ...\n')
data = load('MLdata.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); 
% m is the number of training examples

plotData(X, y);
% Called from the function plot data

fprintf('Program paused. Press enter to continue.\n');
pause;

%++++++++++++++++++++++++++++++++++++++++++++++++++COST & GRADIENT DECENT++++++++++++++++++++++++++++++++++++++++++++++++++

X = [ones(m, 1), data(:,1)]; 
% Add a column of ones to x

theta = zeros(2, 1); 
% initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

fprintf('\nTesting the cost function ...\n')
% compute and display initial scores
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);


% Plot the linear fit
hold on; 
% keep previous plot visible

plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off 
% don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;
