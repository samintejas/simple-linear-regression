clear ; close all; clc

%++++++++++++++++++++++++++++++++++++++++++++++++++++++PLOTTING DATA++++++++++++++++++++++++++++++++++++++++++++++++++++++

fprintf('Plotting Data ...\n')
data = load('MLData.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); 
% m is the number of training examples

plotData(X, y);
% Called from the function plot data

fprintf('Program paused. Press enter to continue.\n');
pause;

%++++++++++++++++++++++++++++++++++++++++++++++++++COST FUNCTION++++++++++++++++++++++++++++++++++++++++++++++++++

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

fprintf('Program paused. Press enter to continue.\n');
pause;

%++++++++++++++++++++++++++++++++++++++++++++++++++GRADIENT DECENT++++++++++++++++++++++++++++++++++++++++++++++++++

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

% Predict values for 9.25 hrs/day
predict1 = [1, 9.25] *theta;
fprintf('The score for studying 9.25hrs per day is %f\n',...
    predict1*10000);


fprintf('Program paused. Press enter to continue.\n');
pause;
