clear ; close all; clc

%====================PLOTTING DATA====================

fprintf('Plotting Data ...\n')
data = load('MLData.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); 
% m is the number of training examples

plotData(X, y);
% Called from the function plot data

fprintf('Program paused. Press enter to continue.\n');
pause;

%====================COST FUNCTION====================

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
fprintf('With theta = [0 ; 0]\nCostfunction computed is = %f\n', J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%====================GRADIENT DECENT====================

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
    predict1);


fprintf('Program paused. Press enter to continue.\n');
pause;

%====================COUNTOUR VISUALIZATION====================

%copied from stanford university machine learning cource

fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
