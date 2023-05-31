clear all

nesto1 = csvread('fashion-mnist_train.csv', 1, 0);
nesto2 = csvread('fashion-mnist_test.csv', 1, 0);

X_train = double(nesto1(:, 2:end));
X_train = X_train'/255;
y_train = double(nesto1(:, 1));
y_train = y_train';

X_test = double(nesto2(:, 2:end));
X_test = X_test'/255;
y_test = double(nesto2(:, 1));
y_test = y_test';

clear nesto1;
clear nesto2;


% prvih 60000 je za train (poredane), a zadnjih 10000 za test

% ovdje ce kasnije biti random 2000 slika, a ne prvih 2000 (sve su nule)
m = 30000; % size(X_train,2);

totalIndices = 60000;  
numIndices = m;    

% Generate random indices
randomIndices = randperm(totalIndices);
randomIndices = randomIndices(1:numIndices);

X_train_new = X_train(:, randomIndices);
y_train_new = y_train(:, randomIndices);

G = zeros(m,m);
alpha = 0.22;

for i = 1:m
    for j = 1:m
        G(i,j) = kernel_f(X_train_new(:,i),X_train_new(:,j),alpha);
    end
end

d = 10;

Y = one_hot_encode(y_train_new, d, m);

Z = Y/G;
% 
% figure(2)
% imagesc(reshape(X_test(:,1),[28,28])');colormap(gray);
% 
% x = X_test(:,1);
% y = f(X_train_new,Z,x,alpha);
% y
% y_test(1)
% 
% figure(3)
% imagesc(reshape(X_test(:,4001),[28,28])');colormap(gray);
% 
% x = X_test(:,4001);
% y = f(X_train_new,Z,x,alpha);
% y
% y_test(4001)
% 
% figure(4)
% imagesc(reshape(X_test(:,9001),[28,28])');colormap(gray);
% 
% x = X_test(:,9001);
% y = f(X_train_new,Z,x,alpha);
% y
% y_test(9001)


tic

% Set the parameters
numIndices = 10000;    % Number of random indices to select

y_preds = zeros(1, numIndices);

for i = 1:numIndices
    x = X_test(:,i);
    rates = f(X_train_new, Z, x, alpha);
    [~, pred_abs] = max(abs(rates));
    y_preds(1,i) = pred_abs-1;
    
end


% Count the number of correct predictions
numCorrect = sum(y_preds == y_test);

% Calculate the accuracy
accuracy = (numCorrect / numIndices) * 100;

toc

% Display the accuracy
fprintf('Accuracy: %.2f%%\n', accuracy);


C = confusionmat(y_test, y_preds);

h = heatmap(C, 'XLabel', 'Predicted Label', 'YLabel', 'True Label');

fashionItems = {'Tenisice', 'Hlače', 'Pulover', 'Haljina', 'Kaput', 'Sandale', 'Košulja', 'Tenisica', 'Torba', 'Čizma'};

xLabels = cell(size(h.XData));
yLabels = cell(size(h.YData));
for i = 1:numel(h.XData)
    xLabels{i} = fashionItems{i};
    yLabels{i} = fashionItems{i};
end
h.XDisplayLabels = xLabels;
h.YDisplayLabels = yLabels;

titleStr = sprintf('Confusion Matrix, alpha = %1.2f, m = %d', alpha, m);
title(titleStr);

% Save the image with the same name as the title
imageName = sprintf('confusion_matrix_alpha_%1.2f_m_%d.png', alpha, m);

saveas(gcf, imageName);

