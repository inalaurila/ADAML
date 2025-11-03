% Advanced Data Analysis and Machine Learning 
% Non-linear dimensionality reduction
% Period 2, week 1 - Homework

close all; clearvars; clc

% Task 1 - Comparing linear and non-linear DR
% Compare PCA and t-SNE methods by visualizing Bike Sharing Rental dataset. Explore how
% the different features are shown in the DR components. Build a simple prediction model (for
% example, MLP or Random Forest) to predict the count of total rental bikes and compare the
% performance of the model with the different DR techniques.

% load data from file with weka
data = loadARFF('dataset.arff');

% finding variable names from Weka
numAttr = data.numAttributes;
varNames = cell(numAttr,1);
for i = 1:numAttr
    varNames{i} = char(data.attribute(i-1).name);  % Java-indeksit alkaa 0:sta
end

X_names = varNames(1:end-1); 

% transfer instances from Weka to matlab double
M = weka2matlab(data);
X_raw = M(:,1:end-1);
% [n, p] = size(M);

%% PCA
Xz = zscore(X_raw);
[coeff, score, ~, ~, expl] = pca(Xz);

figure; scatter(score(:,1), score(:,2));

% % Explained Variance (not needed)
% figure; % How much of the variance do the first 10 components explain
% subplot(1,2,1);
% bar(expl(1:10));
% title('Individual Explained Variance (%)');
% xlabel('Principal Component');
% ylabel('Variance Explained');
% grid on;
% 
% subplot(1,2,2); % How much of the variance do components explain cumulatively
% plot(cumsum(expl), 'ro-', 'LineWidth', 2);
% title('Cumulative Explained Variance');
% xlabel('Principal Component');
% ylabel('Cumulative Variance (%)');
% grid on;


%% biplots & loadings
figure('Name','PCA biplot');
biplot(coeff(:,1:2), 'Scores', score(:,1:2), 'VarLabels', X_names);
title(sprintf('PCA biplot — PC1+PC2 (%.1f%% var)', sum(expl(1:2)))); 
grid on;

% laodings
k = 14; % montako tärkeintä näytetään
[~,i1] = sort(abs(coeff(:,1)),'descend');
[~,i2] = sort(abs(coeff(:,2)),'descend');

figure;
subplot(1,2,1)
bar(coeff(i1(1:k),1))
set(gca,'XTick',1:k,'XTickLabel',X_names(i1(1:k)),'XTickLabelRotation',45)
ylabel('Loading'); title('PC1 loadings (top)')
grid on

subplot(1,2,2)
bar(coeff(i2(1:k),2))
set(gca,'XTick',1:k,'XTickLabel',X_names(i2(1:k)),'XTickLabelRotation',45)
ylabel('Loading'); title('PC2 loadings (top)')
grid on


%% t-SNE - default 
Y = tsne(Xz, 'NumDimensions',2, 'Perplexity',30);
% varNames = {'season','yr','mnth','holiday','weekday','workingday',...
%            'weathersit','temp','atemp','hum','windspeed','cnt'};

figure; scatter(Y(:,1), Y(:,2));
title('t-SNE');

featuresToShow = {'feel_temp', 'humidity','windspeed','season', 'month', 'registered', }; 

figure('Name','t-SNE colored by selected features','Position',[100 100 1200 700])
nCols = 3; nRows = ceil(numel(featuresToShow)/nCols);

for i = 1:numel(featuresToShow)
    featName = featuresToShow{i};
    idx = find(strcmpi(X_names, featName), 1);
    if isempty(idx), continue; end

    subplot(nRows, nCols, i)
    scatter(Y(:,1), Y(:,2), 5, X_raw(:,idx), 'filled'); % käytä M:ää (raaka feature)
    title(featName, 'Interpreter','none'); axis equal off; colormap(turbo); colorbar
end

%% Prediction model 

y = M(:,end); % ennustettava muuttuja

% Train/test jako
cv = cvpartition(size(Xz,1),'HoldOut',0.2);
Xtr = Xz(training(cv),:);
ytr = y(training(cv));
Xte = Xz(test(cv),:);
yte = y(test(cv));

% Random Forest ilman DR:ää 
rf_raw = TreeBagger(200, Xtr, ytr, 'Method','regression', ...
                    'OOBPrediction','on','MinLeafSize',5);
yhat_rf_raw = predict(rf_raw, Xte);

% Random forest / PCA, 95 % selitysaste
cumExpl = cumsum(expl);
k = find(cumExpl>=95,1); if isempty(k), k=10; end
Xpca_tr = Xtr*coeff(:,1:k);
Xpca_te = Xte*coeff(:,1:k);

rf_pca = TreeBagger(200, Xpca_tr, ytr, 'Method','regression', ...
                    'OOBPrediction','on','MinLeafSize',5);
yhat_rf_pca = predict(rf_pca, Xpca_te);

% random forest / t-SNE
Ytr = tsne(Xtr, 'NumDimensions',3, 'Perplexity',30, 'Standardize',false);
Yte = tsne(Xte, 'NumDimensions',3, 'Perplexity',30, 'Standardize',false);

rf_tsne = TreeBagger(200, Ytr, ytr, 'Method','regression', ...
                     'OOBPrediction','on','MinLeafSize',5);
yhat_rf_tsne = predict(rf_tsne, Yte);

% measuring performance
rmse = @(a,b) sqrt(mean((a-b).^2));
r2   = @(a,b) 1 - sum((a-b).^2)/sum((a-mean(a)).^2);

names = {'RF raw','RF PCA','RF t-SNE'};
preds = {yhat_rf_raw, yhat_rf_pca, yhat_rf_tsne};

for i=1:3
    RMSE(i) = rmse(yte,preds{i});
    R2(i)   = r2(yte,preds{i});
end

results = table(names',RMSE',R2','VariableNames',{'Model','RMSE','R2'});
disp(results)
