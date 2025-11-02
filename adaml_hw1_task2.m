% Advanced Data Analysis and Machine Learning 
% Period 2 - Exercise 1
% Task 2 - Visualizing with SOM

% Task
% Visualize MNIST-784 handwritten digits dataset with SOM and discuss what you can learn
% from the visualization.

close all; clearvars; clc

data = loadARFF('mnist_784.arff')

% transfer instances from Weka to matlab double
M = weka2matlab(data);

% create subset of the data
rng(0)
X = M(:,1:784); 
N = size(X,1);
idx = randperm(N, 5000); % subset of 5000 samples
X = X(idx,:);

% Standardising 
X = rescale(X,0,1);
X = X'; % transposed for SOM as features x samples

% SOM map training
som = selforgmap([15 15]); % 15x15 ruudukko
som.trainParam.epochs = 20; % iterations
som = train(som, X);

% Plots
plotsomnd(som) % U-matrix (etäisyydet)
figure; plotsomhits(som, X) % Hits (montako pistettä / solmu)
