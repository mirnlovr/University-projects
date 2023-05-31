clear all

nesto = load('mnist-original.mat');

X_data = double(nesto.data);
X_data = X_data / 255; 
y_data = double(nesto.label);

X_train = X_data(:,1:60000);
y_train = y_data(:,1:60000);
X_test = X_data(:,60001:70000);
y_test = y_data(:,60001:70000);

clear X_data;
clear y_data;

figure(1)
imagesc(reshape(X_train(:,333),[28,28])');colormap(gray);

% prvih 60000 je za train (poredane), a zadnjih 10000 za test
% labeli su u y data

% ovdje ce kasnije biti random 2000 slika, a ne prvih 2000 (sve su nule)
m = 30000; % size(X_train,2);

% Set the parameters
totalIndices = 60000;  % Total number of indices
numIndices = m;    % Number of random indices to select

% Generate random indices
randomIndices = randperm(totalIndices);
randomIndices = randomIndices(1:numIndices);

X_train_new = X_train(:, randomIndices);
y_train_new = y_train(:, randomIndices);

G = zeros(m,m);
alpha = 0.2;

for i = 1:m
    for j = 1:m
        G(i,j) = kernel_f(X_train_new(:,i),X_train_new(:,j),alpha);
    end
end

d = 10;

% abels = y_data(:,1:m);
Y = one_hot_encode(y_train_new, d, m);

Z = Y/G;

figure(2)
imagesc(reshape(X_test(:,1),[28,28])');colormap(gray);

x = X_test(:,1);
y = f(X_train_new,Z,x,alpha);
y
y_test(1)

figure(3)
imagesc(reshape(X_test(:,4001),[28,28])');colormap(gray);

x = X_test(:,4001);
y = f(X_train_new,Z,x,alpha);
y
y_test(4001)

figure(4)
imagesc(reshape(X_test(:,9001),[28,28])');colormap(gray);

x = X_test(:,9001);
y = f(X_train_new,Z,x,alpha);
y
y_test(9001)


tic

% totalIndices = 10000;  % Total number of indices
numIndices = 10000;    % Number of random indices to select

% randomIndices = randperm(totalIndices);
% randomIndices = randomIndices(1:numIndices);
% 
% X_test_new = X_test(:, 1:numIndices);
% y_test_new = y_test(:, numIndices);

y_preds = zeros(1, numIndices);

for i = 1:numIndices
    x = X_test(:,i);
    rates = f(X_train_new, Z, x, alpha);
    [~, pred_abs] = max(abs(rates));
    y_preds(1,i) = pred_abs-1;
    
end

% Assuming y_preds and y_test are row vectors of shape (1, numIndices)

% Count the number of correct predictions
numCorrect = sum(y_preds == y_test);


% Calculate the accuracy
accuracy = (numCorrect / numIndices) * 100;

toc

% Display the accuracy
fprintf('Accuracy abs: %.2f%%\n', accuracy);


C = confusionmat(y_test, y_preds);

h = heatmap(C, 'XLabel', 'Predicted Label', 'YLabel', 'True Label');

xLabels = cell(size(h.XData));
yLabels = cell(size(h.YData));
for i = 1:numel(h.XData)
    xLabels{i} = num2str(str2double(h.XData{i}) - 1);
    yLabels{i} = num2str(str2double(h.YData{i}) - 1);
end
h.XDisplayLabels = xLabels;
h.YDisplayLabels = yLabels;

titleStr = sprintf('Confusion Matrix, alpha =%1.2f, m = %d', alpha, m);
title(titleStr);

% Save the image with the same name as the title
imageName = sprintf('confusion_matrix_alpha_%1.2f_m_%d.png', alpha, m);

saveas(gcf, imageName);


